"""Team Stats — team-level shooting breakdowns and shot charts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    ACCENT,
    APP_BG,
    COURT_BG,
    COURT_LINE,
    MUTED_TEXT,
    PANEL_BG,
    SHOTS_DATA_PATH,
    TEXT_COLOR,
    add_court_traces,
    apply_theme,
    compute_sdi_for_shot,
    compute_summary,
    load_shots_data,
    get_teams,
)

st.set_page_config(page_title="Team Stats", layout="wide")
apply_theme()
st.title("🏟️ Team Stats")

# Hardcoded — single season in dataset
SEASON = "2025-26"
SEASON_TYPE = "regular"

if not SHOTS_DATA_PATH.exists():
    st.error("Shot data not found.")
    st.stop()

teams = get_teams(str(SHOTS_DATA_PATH), SEASON, SEASON_TYPE)
if not teams:
    st.error("No teams found.")
    st.stop()

with st.sidebar:
    st.header("Team Filters")
    st.caption(f"**Season:** {SEASON}  ·  **Type:** {SEASON_TYPE}")
    team_choice = st.selectbox("Team", teams, key="ts_team")
    compare_mode = st.checkbox("Compare with another team", key="ts_compare")
    team_choice_2 = None
    if compare_mode:
        other_teams = [t for t in teams if t != team_choice]
        if other_teams:
            team_choice_2 = st.selectbox("Compare Team", other_teams, key="ts_team2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_team_shots(team_name: str) -> pd.DataFrame:
    df = load_shots_data()
    f = df[df["season"].astype(str) == SEASON]
    if "season_type" in f.columns:
        f = f[f["season_type"].astype(str) == SEASON_TYPE]
    if "SHOT_ATTEMPTED_FLAG" in f.columns:
        f = f[f["SHOT_ATTEMPTED_FLAG"] == 1]
    f = f[f["TEAM_NAME"].astype(str) == str(team_name)]
    return f


def render_team_block(team_name: str, shots_raw: pd.DataFrame, chart_key: str):
    """Render a full-width team stats block with interactive shot chart."""
    minimal = shots_raw[["LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "SHOT_TYPE", "SHOT_ZONE_BASIC"]].copy()
    summary = compute_summary(minimal)

    st.subheader(team_name)

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("FGA", f"{summary['total_fga']:,}")
    c2.metric("FGM", f"{summary['total_fgm']:,}")
    c3.metric("FG%", f"{summary['total_fg'] * 100:.1f}%")
    c4.metric("2PT%", f"{summary['two_fg'] * 100:.1f}%")
    c5.metric("3PT%", f"{summary['three_fg'] * 100:.1f}%")

    # Charts row: zone breakdown + shot type side-by-side
    col_chart, col_types = st.columns(2)

    with col_chart:
        st.markdown("**Shot Zone Breakdown**")
        zone_table = summary["zone_table"].copy()
        zone_table["share_pct"] = zone_table["share_pct"] * 100
        zone_table["fg_pct"] = zone_table["fg_pct"] * 100
        zone_table = zone_table.sort_values("share_pct")

        fig_bar, ax = plt.subplots(figsize=(6, 3.2))
        fig_bar.patch.set_facecolor(PANEL_BG)
        ax.set_facecolor(PANEL_BG)
        ax.barh(zone_table["SHOT_ZONE_BASIC"], zone_table["share_pct"], color=ACCENT, alpha=0.85)
        ax.set_xlabel("Shot Share (%)", color=MUTED_TEXT, fontsize=9)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.grid(axis="x", color="white", alpha=0.08)
        for spine in ax.spines.values():
            spine.set_visible(False)
        mx = max(zone_table["share_pct"].max(), 1)
        ax.set_xlim(0, mx * 1.25)
        for idx, share in enumerate(zone_table["share_pct"]):
            ax.text(share + mx * 0.03, idx, f"{share:.1f}%", va="center", ha="left",
                    color=TEXT_COLOR, fontsize=8)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    with col_types:
        st.markdown("**Shot Type Distribution**")
        if "ACTION_TYPE" in shots_raw.columns:
            action = shots_raw["ACTION_TYPE"].fillna("Unknown").str.lower()
            cats = []
            for a in action:
                if "dunk" in a:
                    cats.append("Dunk")
                elif "layup" in a or "finger roll" in a:
                    cats.append("Layup")
                elif "hook" in a:
                    cats.append("Hook")
                elif "float" in a:
                    cats.append("Floater")
                elif any(kw in a for kw in ["pullup", "step back", "fadeaway", "turnaround"]):
                    cats.append("Pull-Up Jump Shot")
                elif "jump shot" in a:
                    cats.append("Catch & Shoot")
                else:
                    cats.append("Other")
            type_counts = pd.Series(cats).value_counts()
            colors = ["#F5C84C", "#FF6B6B", "#4ECDC4", "#A78BFA", "#60A5FA", "#F97316", "#94A3B8"]
            fig_pie = px.pie(
                values=type_counts.values, names=type_counts.index,
                color_discrete_sequence=colors,
            )
            fig_pie.update_layout(
                paper_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(font=dict(size=10)),
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                                   textfont_size=10)
            st.plotly_chart(fig_pie, use_container_width=True)

    # Interactive Team Shot Chart
    st.markdown("**Team Shot Chart** — _click any shot for details_")

    chart_col, detail_col = st.columns([2.5, 1])

    with chart_col:
        # Prepare shot data with full detail columns for click events
        detail_cols = [
            "shot_id", "PLAYER_NAME", "TEAM_NAME", "LOC_X", "LOC_Y",
            "SHOT_MADE_FLAG", "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA",
            "SHOT_ZONE_RANGE", "SHOT_DISTANCE", "ACTION_TYPE",
            "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
            "GAME_DATE", "HTM", "VTM",
        ]
        available = [c for c in detail_cols if c in shots_raw.columns]
        loc_df = shots_raw[available].copy()
        loc_df["LOC_X"] = pd.to_numeric(loc_df["LOC_X"], errors="coerce")
        loc_df["LOC_Y"] = pd.to_numeric(loc_df["LOC_Y"], errors="coerce")
        loc_df = loc_df.dropna(subset=["LOC_X", "LOC_Y", "SHOT_MADE_FLAG"])
        loc_df["result"] = loc_df["SHOT_MADE_FLAG"].map({1: "Make", 0: "Miss"}).fillna("Miss")

        if len(loc_df) > 8000:
            loc_df = loc_df.sample(8000, random_state=42)

        custom_data_cols = ["shot_id"] if "shot_id" in loc_df.columns else []

        fig = px.scatter(
            loc_df, x="LOC_X", y="LOC_Y", color="result",
            color_discrete_map={"Make": ACCENT, "Miss": "#FF6B6B"},
            category_orders={"result": ["Miss", "Make"]},
            custom_data=custom_data_cols if custom_data_cols else None,
            opacity=0.8, width=620, height=460,
        )
        fig.update_traces(
            marker=dict(size=8),
            selected=dict(marker=dict(opacity=1.0, size=12)),
            unselected=dict(marker=dict(opacity=0.7)),
            selectedpoints=[],
            selector=dict(mode="markers"),
        )
        add_court_traces(fig, full_court=False, line_color=COURT_LINE)
        fig.update_xaxes(range=[-250, 250], showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(range=[-52, 418], showgrid=False, zeroline=False,
                         showticklabels=False, scaleanchor="x", scaleratio=1)
        fig.update_layout(
            autosize=False, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True, plot_bgcolor=COURT_BG, paper_bgcolor=APP_BG,
            legend=dict(font=dict(color=TEXT_COLOR)),
            clickmode="event+select",
        )

        selection = st.plotly_chart(
            fig, use_container_width=False,
            config={"displayModeBar": False},
            on_select="rerun", selection_mode="points",
            key=f"team_chart_{chart_key}",
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
            st.session_state.pop(f"team_selected_shot_{chart_key}", None)
        elif selected_points:
            pt = selected_points[0]
            cd = pt.get("customdata")
            if cd:
                # Handle cd as dict or list
                if isinstance(cd, dict):
                    idx = cd.get("0") or cd.get(0)
                elif isinstance(cd, list) and len(cd) > 0:
                    idx = cd[0]
                else:
                    idx = None
                    
                if idx is not None:
                    match = loc_df[loc_df["shot_id"].astype(str) == str(idx)]
                if not match.empty:
                    shot_data = match.iloc[0].to_dict()
                    # Look up full detail from source if possible
                    if "shot_id" in shot_data and pd.notna(shot_data.get("shot_id")):
                        full_df = load_shots_data()
                        if "shot_id" in full_df.columns:
                            full_match = full_df[full_df["shot_id"].astype(str) == str(shot_data["shot_id"])]
                            if not full_match.empty:
                                shot_data = full_match.iloc[0].to_dict()
                    st.session_state[f"team_selected_shot_{chart_key}"] = shot_data

    with detail_col:
        shot = st.session_state.get(f"team_selected_shot_{chart_key}")
        _render_shot_card(shot)


def _render_shot_card(shot: dict | None):
    """Render a shot detail card (shared by team and comparison charts)."""
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


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

shots_1 = load_team_shots(team_choice)
if shots_1.empty:
    st.warning(f"No shots found for {team_choice}.")
else:
    # Always render full-width (stacked for comparison, not side-by-side)
    render_team_block(team_choice, shots_1, chart_key="team1")

    if compare_mode and team_choice_2:
        st.markdown("---")
        shots_2 = load_team_shots(team_choice_2)
        if shots_2.empty:
            st.warning(f"No shots for {team_choice_2}.")
        else:
            render_team_block(team_choice_2, shots_2, chart_key="team2")
