"""Player Stats — leaderboard, individual metrics, and POE shot charts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    ACCENT,
    ANALYSIS_DATA_DIR,
    APP_BG,
    COURT_BG,
    COURT_LINE,
    MUTED_TEXT,
    PANEL_BG,
    TEXT_COLOR,
    add_court_traces,
    apply_theme,
    compute_sdi_for_shot,
    load_shots_data,
)

st.set_page_config(page_title="Player Stats", layout="wide")
apply_theme()
st.title("👤 Player Stats")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SUMMARY_PATH = ANALYSIS_DATA_DIR / "player_summary.csv"
ENRICHED_PATH = ANALYSIS_DATA_DIR / "shots_with_xp_2025-26.parquet"


@st.cache_data(show_spinner=False)
def load_player_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_enriched_shots() -> pd.DataFrame:
    if not ENRICHED_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(ENRICHED_PATH)


summary_df = load_player_summary()
enriched_df = load_enriched_shots()

if summary_df.empty:
    st.warning(f"Player summary not found at `{SUMMARY_PATH}`. Run `player_performance_analysis.py` first.")
    st.stop()


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

st.subheader("Player Leaderboard")

with st.sidebar:
    st.header("Player Filters")
    min_att = st.slider("Min Attempts", 50, 500, 200, 25, key="ps_min_att")
    sort_col = st.selectbox("Sort By", [
        "total_poe", "poe_per_100", "fg_pct", "fg_residual_pct", "total_attempts",
    ], index=0, key="ps_sort")
    sort_asc = st.checkbox("Ascending", value=False, key="ps_sort_asc")

filtered = summary_df[summary_df["total_attempts"] >= min_att].copy()
filtered = filtered.sort_values(sort_col, ascending=sort_asc)

display_cols = {
    "PLAYER_NAME": "Player",
    "total_attempts": "FGA",
    "fg_pct": "FG%",
    "expected_fg_pct": "xFG%",
    "fg_residual_pct": "FG Resid %",
    "total_poe": "Total POE",
    "poe_per_100": "POE/100",
}
show_df = filtered[list(display_cols.keys())].rename(columns=display_cols).reset_index(drop=True)
show_df["FG%"] = (show_df["FG%"] * 100).round(1)
show_df["xFG%"] = (show_df["xFG%"] * 100).round(1)
show_df["FG Resid %"] = show_df["FG Resid %"].round(2)
show_df["Total POE"] = show_df["Total POE"].round(1)
show_df["POE/100"] = show_df["POE/100"].round(2)

st.dataframe(show_df, use_container_width=True, height=420)

st.markdown("---")

# ---------------------------------------------------------------------------
# Top / Bottom performers highlight
# ---------------------------------------------------------------------------

col_top, col_bot = st.columns(2)

with col_top:
    st.subheader("🏆 Top 10 by POE")
    top10 = filtered.nlargest(10, "total_poe")[["PLAYER_NAME", "total_poe", "fg_pct", "poe_per_100"]].copy()
    top10["fg_pct"] = (top10["fg_pct"] * 100).round(1)
    top10["total_poe"] = top10["total_poe"].round(1)
    top10["poe_per_100"] = top10["poe_per_100"].round(2)
    top10 = top10.rename(columns={
        "PLAYER_NAME": "Player", "total_poe": "POE", "fg_pct": "FG%", "poe_per_100": "POE/100"
    }).reset_index(drop=True)
    st.dataframe(top10, use_container_width=True, hide_index=True)

with col_bot:
    st.subheader("📉 Bottom 10 by POE")
    bot10 = filtered.nsmallest(10, "total_poe")[["PLAYER_NAME", "total_poe", "fg_pct", "poe_per_100"]].copy()
    bot10["fg_pct"] = (bot10["fg_pct"] * 100).round(1)
    bot10["total_poe"] = bot10["total_poe"].round(1)
    bot10["poe_per_100"] = bot10["poe_per_100"].round(2)
    bot10 = bot10.rename(columns={
        "PLAYER_NAME": "Player", "total_poe": "POE", "fg_pct": "FG%", "poe_per_100": "POE/100"
    }).reset_index(drop=True)
    st.dataframe(bot10, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Player shot chart (POE-colored)
# ---------------------------------------------------------------------------

def _render_shot_card(shot: dict | None):
    """Render a shot detail card."""
    st.markdown("**Selected Shot**")
    if not shot:
        st.markdown(
            f'<div style="padding: 12px; border-radius: 12px; background: {PANEL_BG}; '
            f'border: 1px solid rgba(255,255,255,0.08); color: {MUTED_TEXT};">'
            f'Click a shot on the chart to see detailed info.</div>',
            unsafe_allow_html=True,
        )
        return

    player = shot.get("PLAYER_NAME", "Unknown")
    team = shot.get("TEAM_NAME", "")
    is_make = shot.get("SHOT_MADE_FLAG") == 1
    result = "Make" if is_make else "Miss"
    result_color = ACCENT if is_make else "#FF6B6B"

    distance = shot.get("SHOT_DISTANCE")
    dist_text = f"{int(round(float(distance)))} ft" if distance is not None and not pd.isna(distance) else "N/A"

    period = shot.get("PERIOD")
    period_text = str(int(period)) if period is not None and not pd.isna(period) else "N/A"

    minutes = shot.get("MINUTES_REMAINING")
    seconds = shot.get("SECONDS_REMAINING")
    if minutes is not None and seconds is not None and not pd.isna(minutes) and not pd.isna(seconds):
        clock = f"{int(minutes):02d}:{int(seconds):02d}"
    else:
        clock = "N/A"

    zone = shot.get("SHOT_ZONE_BASIC") or "N/A"
    action = shot.get("ACTION_TYPE") or shot.get("SHOT_TYPE") or "Shot"
    game_date = shot.get("GAME_DATE") or ""
    htm = shot.get("HTM") or ""
    vtm = shot.get("VTM") or ""
    matchup = f"{htm} vs {vtm}".strip() if htm or vtm else ""

    sdi_val = compute_sdi_for_shot(shot)

    info_lines = [
        f'<div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 4px;">{player}</div>',
        f'<div style="color: {MUTED_TEXT}; font-size: 0.85rem; margin-bottom: 6px;">{team}</div>',
    ]
    if game_date or matchup:
        info_lines.append(f'<div style="margin-bottom: 4px; font-size: 0.9rem;">{game_date} {matchup}</div>')
    info_lines += [
        f'<div style="margin-bottom: 4px;">Q{period_text} · {clock}</div>',
        f'<div style="margin-bottom: 4px;">{action}</div>',
        f'<div style="margin-bottom: 4px;">{zone} · {dist_text}</div>',
        f'<div style="color: {MUTED_TEXT}; font-size: 0.9rem; margin-bottom: 4px;">'
        f'SDI: <b style="color: {TEXT_COLOR}">{sdi_val:.3f}</b></div>',
        f'<div style="font-weight: 600; color: {result_color};">{result}</div>',
    ]

    st.markdown(
        f'<div style="padding: 12px; border-radius: 12px; background: {PANEL_BG}; '
        f'border: 1px solid rgba(255,255,255,0.08);">'
        + "".join(info_lines) +
        '</div>',
        unsafe_allow_html=True,
    )

st.subheader("Player Shot Chart (POE)")

if enriched_df.empty:
    st.info("Enriched shot data not available. Run `expected_points_analysis.py` to generate it.")
else:
    players = sorted(enriched_df["PLAYER_NAME"].dropna().unique().tolist())
    selected_player = st.selectbox("Select Player", players, key="ps_player_select")

    if selected_player:
        player_shots = enriched_df[enriched_df["PLAYER_NAME"] == selected_player].copy()
        player_shots["LOC_X"] = pd.to_numeric(player_shots["LOC_X"], errors="coerce")
        player_shots["LOC_Y"] = pd.to_numeric(player_shots["LOC_Y"], errors="coerce")
        player_shots = player_shots.dropna(subset=["LOC_X", "LOC_Y"])

        if player_shots.empty:
            st.warning(f"No shot data for {selected_player}.")
        else:
            # Stats card
            p_row = summary_df[summary_df["PLAYER_NAME"] == selected_player]
            if not p_row.empty:
                p = p_row.iloc[0]
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("FGA", f"{int(p['total_attempts']):,}")
                mc2.metric("FG%", f"{p['fg_pct'] * 100:.1f}%")
                mc3.metric("Total POE", f"{p['total_poe']:.1f}")
                mc4.metric("POE/100", f"{p['poe_per_100']:.2f}")

            # Interactive Player Shot Chart
            st.markdown("**Player Shot Chart (POE)** — _click any shot for details_")

            chart_col, detail_col = st.columns([2.5, 1])

            with chart_col:
                if len(player_shots) > 5000:
                    player_shots = player_shots.sample(5000, random_state=42)

                player_shots["result"] = player_shots["SHOT_MADE_FLAG"].map({1: "Made", 0: "Missed"})
                # Add unique index for reliable click matching
                player_shots["plotly_idx"] = np.arange(len(player_shots))
                
                fig = px.scatter(
                    player_shots, x="LOC_X", y="LOC_Y",
                    color="POE",
                    color_continuous_scale="RdYlGn",
                    range_color=[-2, 2],
                    symbol="result",
                    symbol_map={"Made": "star", "Missed": "circle"},
                    custom_data=["plotly_idx"],
                    opacity=0.8, width=650, height=500,
                    hover_data={"SHOT_ZONE_BASIC": True, "ACTION_TYPE": True, "POE": ":.2f"},
                )
                fig.update_traces(
                    marker=dict(size=10, line=dict(width=0.5, color="black")),
                    selected=dict(marker=dict(opacity=1.0, size=14)),
                    unselected=dict(marker=dict(opacity=0.5)),
                )
                add_court_traces(fig, full_court=False, line_color=COURT_LINE)
                fig.update_xaxes(range=[-250, 250], showgrid=False, zeroline=False, showticklabels=False)
                fig.update_yaxes(range=[-52, 418], showgrid=False, zeroline=False,
                                 showticklabels=False, scaleanchor="x", scaleratio=1)
                fig.update_layout(
                    autosize=False, margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor=COURT_BG, paper_bgcolor=APP_BG,
                    coloraxis_colorbar=dict(title="POE", tickformat="+.1f"),
                    font=dict(color=TEXT_COLOR),
                    legend=dict(font=dict(color=TEXT_COLOR)),
                    clickmode="event+select",
                )

                selection = st.plotly_chart(
                    fig, use_container_width=False,
                    config={"displayModeBar": False},
                    on_select="rerun", selection_mode="points",
                )

                # Handle click selection
                selected_points = None
                if isinstance(selection, dict):
                    selected_points = selection.get("points", []) or selection.get("selection", {}).get("points", [])
                else:
                    sel = getattr(selection, "selection", None)
                    if isinstance(sel, dict):
                        selected_points = sel.get("points", [])

                if selected_points is not None and len(selected_points) == 0:
                    st.session_state.pop("player_selected_shot", None)
                elif selected_points:
                    pt = selected_points[0]
                    cd = pt.get("customdata")
                    if cd:
                        if isinstance(cd, dict):
                            idx = cd.get("0") or cd.get(0)
                        elif isinstance(cd, list) and len(cd) > 0:
                            idx = cd[0]
                        else:
                            idx = None
                            
                        if idx is not None:
                            match = player_shots[player_shots["shot_id"].astype(str) == str(idx)]
                        if not match.empty:
                            shot_data = match.iloc[0].to_dict()
                            if "shot_id" in shot_data and pd.notna(shot_data.get("shot_id")):
                                full_df = load_shots_data()
                                if "shot_id" in full_df.columns:
                                    full_match = full_df[full_df["shot_id"].astype(str) == str(shot_data["shot_id"])]
                                    if not full_match.empty:
                                        shot_data = full_match.iloc[0].to_dict()
                            st.session_state["player_selected_shot"] = shot_data

            with detail_col:
                shot = st.session_state.get("player_selected_shot")
                _render_shot_card(shot)

