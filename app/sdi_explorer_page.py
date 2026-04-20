"""Cross-sport SDI explorer tied to the final poster story."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from demo_content import SDI_DEFAULT_MIN_ATTEMPTS, SDI_FIGURE_SPECS, SDI_SUMMARY_PATHS
from app_utils import MUTED_TEXT, apply_theme

SCATTER_BG = "#F8F7F2"
SCATTER_TEXT = "#1B1B1B"
SCATTER_GRID = "rgba(0,0,0,0.08)"


@st.cache_data(show_spinner=False)
def load_sdi_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["player"] = df["player"].astype(str)
    return df


def compute_residual_color_limits(
    residuals: pd.Series,
    *,
    lower_floor: float,
    upper_cap: float,
    quantile: float,
) -> tuple[float, float]:
    numeric = pd.to_numeric(residuals, errors="coerce").dropna()
    if numeric.empty:
        return -0.05, 0.05
    limit = float(np.nanpercentile(np.abs(numeric), quantile * 100))
    limit = max(limit, lower_floor)
    limit = min(limit, upper_cap)
    return -limit, limit


def get_sdi_chart_meta(sport: str) -> dict[str, str]:
    return {
        "NBA": {
            "title": "NBA Shot Difficulty vs Actual Scoring Rate (2014-2024)",
            "y_axis_title": "Actual FG%",
        },
        "NHL": {
            "title": "NHL Shot Difficulty vs Actual Scoring Rate (2014-2024)",
            "y_axis_title": "Actual Goal %",
        },
    }[sport]


def build_sdi_scatter(df: pd.DataFrame, sport: str) -> go.Figure:
    meta = get_sdi_chart_meta(sport)
    if sport == "NHL":
        color_min, color_max = compute_residual_color_limits(
            df["residual"],
            lower_floor=0.010,
            upper_cap=0.08,
            quantile=0.92,
        )
    else:
        color_min, color_max = compute_residual_color_limits(
            df["residual"],
            lower_floor=0.015,
            upper_cap=0.12,
            quantile=0.98,
        )

    fig = px.scatter(
        df,
        x="mean_sdi",
        y="actual_rate",
        size="attempts",
        color="residual",
        color_continuous_scale="RdYlGn",
        range_color=[color_min, color_max],
        custom_data=["player"],
        hover_name="player",
        hover_data={
            "attempts": True,
            "mean_sdi": ":.3f",
            "actual_rate": ":.1%",
            "expected_rate": ":.1%",
            "residual": ":.1%",
            "position_group": True,
        },
        size_max=22,
    )
    fig.update_traces(marker=dict(line=dict(color="rgba(0,0,0,0.45)", width=1)))

    valid_points = df[["mean_sdi", "actual_rate"]].apply(pd.to_numeric, errors="coerce").dropna()
    x_values = valid_points["mean_sdi"]
    y_values = valid_points["actual_rate"]
    if len(valid_points) > 1 and x_values.nunique() > 1:
        coeffs = np.polyfit(x_values, y_values, 1)
        trend_x = np.linspace(float(x_values.min()), float(x_values.max()), 250)
        trend_y = coeffs[0] * trend_x + coeffs[1]
        fig.add_trace(
            go.Scatter(
                x=trend_x,
                y=trend_y,
                mode="lines",
                line=dict(color="rgba(80,80,80,0.78)", dash="dash", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_vline(
        x=float(x_values.median()),
        line_dash="dash",
        line_color="rgba(120,120,120,0.35)",
        line_width=2,
    )
    fig.add_hline(
        y=float(y_values.median()),
        line_dash="dash",
        line_color="rgba(120,120,120,0.35)",
        line_width=2,
    )
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"<b>{meta['title']}</b>", font=dict(color=SCATTER_TEXT, size=20)),
        paper_bgcolor=SCATTER_BG,
        plot_bgcolor=SCATTER_BG,
        font=dict(color=SCATTER_TEXT),
        clickmode="event+select",
        margin=dict(l=40, r=20, t=60, b=50),
        coloraxis_colorbar=dict(
            title=dict(
                text="<b>Residual (Actual - Expected)</b>",
                font=dict(color=SCATTER_TEXT, size=14),
            ),
            tickfont=dict(color=SCATTER_TEXT, size=12),
        ),
    )
    fig.update_xaxes(
        title="<b>Average Shot Difficulty Index (SDI)</b>",
        showgrid=True,
        gridcolor=SCATTER_GRID,
        zeroline=False,
        title_font=dict(color=SCATTER_TEXT, size=15),
        tickfont=dict(color=SCATTER_TEXT, size=12),
    )
    fig.update_yaxes(
        title=f"<b>{meta['y_axis_title']}</b>",
        tickformat=".0%",
        showgrid=True,
        gridcolor=SCATTER_GRID,
        zeroline=False,
        title_font=dict(color=SCATTER_TEXT, size=15),
        tickfont=dict(color=SCATTER_TEXT, size=12),
    )
    return fig


def render_poster_reference(sport: str) -> None:
    spec = SDI_FIGURE_SPECS[sport]
    figure_path = Path(spec["path"])
    with st.container(border=True):
        st.markdown("#### Final Poster Figure")
        if not figure_path.exists():
            st.warning(f"Missing poster SDI figure: {figure_path.name}")
            return
        st.image(str(figure_path), use_container_width=True)
        st.caption(str(spec["caption"]))


def render_player_card(player_row: pd.Series | None, sport: str) -> None:
    st.markdown("### Selected Player")
    if player_row is None:
        st.info(f"Click a {sport} player dot to inspect that profile.")
        return

    residual = float(player_row["residual"])
    sign = "+" if residual > 0 else ""

    st.markdown(f"#### {player_row['player']}")
    top = st.columns(2)
    top[0].metric("Attempts", f"{int(player_row['attempts']):,}")
    top[1].metric("Position", str(player_row.get("position_group", "—")))
    middle = st.columns(2)
    middle[0].metric("Mean SDI", f"{float(player_row['mean_sdi']):.3f}")
    middle[1].metric("Actual Rate", f"{float(player_row['actual_rate']):.1%}")
    bottom = st.columns(2)
    bottom[0].metric("Expected Rate", f"{float(player_row['expected_rate']):.1%}")
    bottom[1].metric("Residual", f"{sign}{residual:.1%}")


def render_sport_tab(sport: str, df: pd.DataFrame, key_prefix: str) -> None:
    if df.empty:
        st.warning(f"No {sport} summary data found.")
        return

    render_poster_reference(sport)
    st.markdown("### Interactive Explorer")

    default_min = SDI_DEFAULT_MIN_ATTEMPTS[sport]
    control_cols = st.columns([1, 1.2])
    min_attempts = control_cols[0].slider(
        "Minimum attempts",
        min_value=int(df["attempts"].min()),
        max_value=int(df["attempts"].max()),
        value=min(default_min, int(df["attempts"].max())),
        step=50,
        key=f"{key_prefix}_min_attempts",
    )
    positions = sorted(df["position_group"].dropna().astype(str).unique().tolist())
    selected_positions = control_cols[1].multiselect(
        "Position group",
        options=positions,
        default=positions,
        key=f"{key_prefix}_positions",
    )

    filtered = df[df["attempts"] >= min_attempts].copy()
    if selected_positions:
        filtered = filtered[filtered["position_group"].astype(str).isin(selected_positions)]

    if filtered.empty:
        st.info("No players match the current filters.")
        return

    st.markdown(
        f'<div class="panel-copy" style="margin-top:-0.25rem;">'
        f'Click a dot to inspect the {sport} player profile. The interactive view uses the same summary data as the poster figure, '
        f'with poster-matched axis labels, residual color scaling, and reference lines.'
        f'</div>',
        unsafe_allow_html=True,
    )

    plot_col, card_col = st.columns([2.3, 1], gap="large")
    with plot_col:
        figure = build_sdi_scatter(filtered, sport)
        selection = st.plotly_chart(
            figure,
            use_container_width=True,
            config={"displayModeBar": False},
            theme=None,
            on_select="rerun",
            selection_mode="points",
            key=f"{key_prefix}_scatter",
        )

        selected_points = None
        if isinstance(selection, dict):
            selected_points = selection.get("points", []) or selection.get("selection", {}).get("points", [])
        else:
            selected = getattr(selection, "selection", None)
            if isinstance(selected, dict):
                selected_points = selected.get("points", [])

        state_key = f"{key_prefix}_selected_player"
        if selected_points is not None and len(selected_points) == 0:
            st.session_state.pop(state_key, None)
        elif selected_points:
            st.session_state[state_key] = selected_points[0]["customdata"][0]

    with card_col:
        selected_name = st.session_state.get(f"{key_prefix}_selected_player")
        selected_row = None
        if selected_name:
            matched = filtered[filtered["player"] == selected_name]
            if not matched.empty:
                selected_row = matched.iloc[0]
        render_player_card(selected_row, sport)


def main() -> None:
    apply_theme()
    st.title("📊 SDI Explorer")
    st.markdown(
        f'<div class="panel-copy" style="font-size:1rem;color:{MUTED_TEXT};">'
        f'This page turns the poster SDI result into an interactive cross-sport explorer. '
        f'Use the tabs to compare NBA and NHL player difficulty profiles and click a dot to view the player details.'
        f"</div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["NBA", "NHL"])
    with tabs[0]:
        render_sport_tab("NBA", load_sdi_summary(SDI_SUMMARY_PATHS["NBA"]), "nba_sdi")
    with tabs[1]:
        render_sport_tab("NHL", load_sdi_summary(SDI_SUMMARY_PATHS["NHL"]), "nhl_sdi")


main()
