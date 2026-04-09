"""Cross-sport SDI explorer tied to the final poster story."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from demo_content import SDI_DEFAULT_MIN_ATTEMPTS, SDI_SUMMARY_PATHS
from app_utils import APP_BG, MUTED_TEXT, PANEL_BG, TEXT_COLOR, apply_theme


@st.cache_data(show_spinner=False)
def load_sdi_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["player"] = df["player"].astype(str)
    return df


def build_sdi_scatter(df: pd.DataFrame, sport: str) -> go.Figure:
    color_range = max(float(df["residual"].abs().max()), 0.01)
    fig = px.scatter(
        df,
        x="mean_sdi",
        y="actual_rate",
        size="attempts",
        color="residual",
        color_continuous_scale="RdYlGn",
        range_color=[-color_range, color_range],
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
    fig.update_traces(marker=dict(line=dict(color="rgba(255,255,255,0.25)", width=1)))
    fig.update_layout(
        title=f"{sport} SDI vs Actual Scoring",
        paper_bgcolor=APP_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        clickmode="event+select",
        margin=dict(l=40, r=20, t=60, b=50),
        coloraxis_colorbar=dict(title="Residual"),
    )
    fig.update_xaxes(title="Mean SDI", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(
        title="Actual scoring rate",
        tickformat=".0%",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
    )
    return fig


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
        f'Click a dot to inspect the {sport} player profile. Color shows residual '
        f'(actual rate minus expected rate), and size reflects shot volume.'
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
