"""NBA Shot Map Dashboard — Home / Shot Map page."""

import sys
from pathlib import Path

# Ensure the app directory is importable (for utils)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    ACCENT,
    APP_BG,
    COURT_BG,
    COURT_LINE,
    MUTED_TEXT,
    PANEL_BG,
    PLOT_WIDTH,
    SHOTS_DATA_PATH,
    TEXT_COLOR,
    apply_theme,
    compute_sdi_for_shot,
    compute_summary,
    compute_xfg_for_shot,
    get_players,
    get_season_types,
    get_seasons,
    get_shot_detail,
    get_shot_points,
    get_teams,
    load_shots,
    plot_scatter_plotly,
    sample_shots_with_far,
)


def render_selected_shot(selected_shot: dict | None) -> None:
    """Render the shot-detail card below the chart."""
    st.subheader("Selected Shot")
    if not selected_shot:
        st.markdown(
            f"""
            <div style="width: {PLOT_WIDTH}px; max-width: 100%; margin: 0;">
              <div style="padding: 12px; border-radius: 12px; background: {PANEL_BG};
                   border: 1px solid rgba(255,255,255,0.08); color: {MUTED_TEXT};">
                Click a shot on the chart to see details.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    period = selected_shot.get("PERIOD")
    minutes = selected_shot.get("MINUTES_REMAINING")
    seconds = selected_shot.get("SECONDS_REMAINING")
    if minutes is None or seconds is None or pd.isna(minutes) or pd.isna(seconds):
        clock = "N/A"
    else:
        clock = f"{int(minutes):02d}:{int(seconds):02d}"
    period_text = "N/A" if period is None or pd.isna(period) else str(int(period))

    is_make = selected_shot.get("SHOT_MADE_FLAG") == 1
    result = "Make" if is_make else "Miss"
    result_color = ACCENT if is_make else "#FF6B6B"

    shot_distance = selected_shot.get("SHOT_DISTANCE")
    distance_text = "N/A" if shot_distance is None or pd.isna(shot_distance) else f"{int(round(float(shot_distance)))} ft"

    zone_basic = selected_shot.get("SHOT_ZONE_BASIC") or "N/A"
    zone_area = selected_shot.get("SHOT_ZONE_AREA") or "N/A"
    zone_range = selected_shot.get("SHOT_ZONE_RANGE") or "N/A"
    game_date = selected_shot.get("GAME_DATE") or "N/A"
    htm = selected_shot.get("HTM") or ""
    vtm = selected_shot.get("VTM") or ""
    matchup = f"{htm} vs {vtm}".strip()

    xfg_val = selected_shot.get("xFG")
    if xfg_val is None or (isinstance(xfg_val, float) and pd.isna(xfg_val)):
        xfg_text = "N/A"
    else:
        try:
            xfg_text = f"{float(xfg_val):.1%}"
        except (TypeError, ValueError):
            xfg_text = "N/A"

    sdi_val = compute_sdi_for_shot(selected_shot)

    st.markdown(
        f"""
        <div style="width: {PLOT_WIDTH}px; max-width: 100%; margin: 0;">
          <div style="padding: 12px; border-radius: 12px; background: {PANEL_BG};
               border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-weight: 600; margin-bottom: 8px;">{selected_shot.get("PLAYER_NAME", "Unknown")}</div>
            <div style="color: {MUTED_TEXT}; font-size: 0.9rem; margin-bottom: 6px;">
              {selected_shot.get("TEAM_NAME", "Unknown Team")}
            </div>
            <div style="margin-bottom: 6px;">{game_date} · {matchup}</div>
            <div style="margin-bottom: 6px;">Period {period_text} · {clock}</div>
            <div style="margin-bottom: 6px;">{selected_shot.get("SHOT_TYPE", "Shot")} · {distance_text}</div>
            <div style="margin-bottom: 6px;">{zone_basic} / {zone_area} / {zone_range}</div>
            <div style="display: flex; gap: 12px; margin-bottom: 6px; font-size: 0.9rem; color: {MUTED_TEXT};">
                <span>SDI: <b style="color: {TEXT_COLOR}">{sdi_val:.3f}</b></span>
                <span>xFG: <b style="color: {TEXT_COLOR}">{xfg_text}</b></span>
            </div>
            <div style="font-weight: 600; color: {result_color};">{result}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="NBA Shot Analytics Dashboard",
        layout="wide",
    )
    apply_theme()

    st.title("🏀 NBA Shot Map Dashboard")

    # Hardcoded — only one season / type in the dataset
    db_path = str(SHOTS_DATA_PATH)
    season = "2025-26"
    season_type = "regular"

    with st.sidebar:
        st.header("Filters")
        if not SHOTS_DATA_PATH.exists():
            st.error(f"Sample data not found: {SHOTS_DATA_PATH}")
            st.stop()

        st.caption(f"**Season:** {season}  ·  **Type:** {season_type}")

        teams = get_teams(db_path, season, season_type)
        team_options = ["All Teams"] + teams
        if st.session_state.get("team_select") not in team_options:
            st.session_state["team_select"] = "All Teams"
        team_choice = st.selectbox("Team", team_options, key="team_select")
        selected_team = None if team_choice == "All Teams" else team_choice

        players = get_players(db_path, season, season_type, selected_team)
        player_options = ["All Players"] + players

        qp_player = st.query_params.get("player")
        if qp_player and qp_player in player_options:
            st.session_state["player_select"] = qp_player
            st.query_params.pop("player", None)
        elif st.session_state.get("player_select") not in player_options:
            st.session_state["player_select"] = "All Players"

        player_choice = st.selectbox("Player", player_options, key="player_select")
        selected_player = None if player_choice == "All Players" else player_choice

        if st.session_state.get("court_view_radio") not in ["Full court", "Half court"]:
            st.session_state["court_view_radio"] = "Half court"
        court_view = st.radio("Court View", ["Full court", "Half court"], key="court_view_radio")

        if st.session_state.get("max_points_slider") is None:
            st.session_state["max_points_slider"] = 40000
        max_points = st.slider("Max points", 5000, 100000, step=5000, key="max_points_slider")

        if st.session_state.get("far_threshold_slider") is None:
            st.session_state["far_threshold_slider"] = 40
        far_threshold = st.slider("Always include shots beyond (ft)", 30, 60, step=5, key="far_threshold_slider")

    # -----------------------------------------------------------------------
    # Main content
    # -----------------------------------------------------------------------
    subtitle_parts = [season, season_type]
    if selected_team:
        subtitle_parts.append(selected_team)
    if selected_player:
        subtitle_parts.append(selected_player)
    st.caption(" | ".join(subtitle_parts))

    shots = load_shots(db_path, season, season_type, selected_team, selected_player)

    filter_signature = (
        season, season_type,
        selected_team or "ALL_TEAMS",
        selected_player or "ALL_PLAYERS",
    )
    if st.session_state.get("filter_signature") != filter_signature:
        st.session_state["filter_signature"] = filter_signature
        st.session_state.pop("selected_shot", None)

    shots_points = get_shot_points(db_path, season, season_type, selected_team, selected_player)
    raw_count = len(shots_points)
    shots_points = sample_shots_with_far(shots_points, max_points or 40000, far_threshold or 40)
    omitted = max(0, raw_count - len(shots_points))
    if omitted > 0:
        st.caption(
            f"Showing {len(shots_points):,} points. Omitted {omitted:,} for speed "
            f"(all shots ≥ {far_threshold} ft included)."
        )

    if shots.empty:
        available = get_season_types(db_path, season)
        st.warning(
            f"No shots found for {season} {season_type}. "
            f"Available season types: {', '.join(available)}"
        )
        st.stop()

    summary = compute_summary(shots)

    col_plot, col_side = st.columns([2.2, 1])

    with col_plot:
        fig = plot_scatter_plotly(shots_points, court_view)
        selection = st.plotly_chart(
            fig, use_container_width=False,
            config={"displayModeBar": False},
            on_select="rerun", selection_mode="points",
        )
        selected_points = None
        if isinstance(selection, dict):
            selected_points = selection.get("points", []) or selection.get("selection", {}).get("points", [])
        else:
            sel = getattr(selection, "selection", None)
            if isinstance(sel, dict):
                selected_points = sel.get("points", [])

        if selected_points is not None and len(selected_points) == 0:
            st.session_state.pop("selected_shot", None)
        elif selected_points:
            shot_id = selected_points[0]["customdata"][0]
            detail = get_shot_detail(db_path, str(shot_id))
            if detail:
                detail["xFG"] = compute_xfg_for_shot(detail, season)
                st.session_state["selected_shot"] = detail

        render_selected_shot(st.session_state.get("selected_shot"))

    with col_side:
        st.markdown(
            '<div style="margin-top:-18px; margin-bottom: 8px; font-size: 1.35rem; font-weight: 700;">Summary</div>',
            unsafe_allow_html=True,
        )
        m1 = st.columns(2)
        m1[0].metric("Field Goal Attempts", f"{summary['total_fga']:,}")
        m1[1].metric("Field Goal %", f"{summary['total_fg'] * 100:.1f}%")
        m2 = st.columns(2)
        m2[0].metric("2PT Field Goal %", f"{summary['two_fg'] * 100:.1f}%")
        m2[1].metric("3PT Field Goal %", f"{summary['three_fg'] * 100:.1f}%")

        st.subheader("Top Shot Zones")
        zone_table = summary["zone_table"].copy()
        zone_table["share_pct"] = zone_table["share_pct"] * 100
        zone_table["fg_pct"] = zone_table["fg_pct"] * 100
        zone_table = zone_table.sort_values("share_pct")

        fig_bar, ax_bar = plt.subplots(figsize=(4, 3.4))
        fig_bar.patch.set_facecolor(PANEL_BG)
        ax_bar.set_facecolor(PANEL_BG)
        ax_bar.barh(zone_table["SHOT_ZONE_BASIC"], zone_table["share_pct"], color=ACCENT, alpha=0.85)
        ax_bar.set_xlabel("Shot Share (%)", color=MUTED_TEXT, fontsize=9)
        ax_bar.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax_bar.grid(axis="x", color="white", alpha=0.08, linewidth=1)
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        max_share = max(zone_table["share_pct"].max(), 1)
        ax_bar.set_xlim(0, max_share * 1.25)
        for idx, share in enumerate(zone_table["share_pct"]):
            ax_bar.text(share + max_share * 0.03, idx, f"{share:.1f}%",
                        va="center", ha="left", color=TEXT_COLOR, fontsize=9)
        st.pyplot(fig_bar, use_container_width=True)

        made_count = int((shots["SHOT_MADE_FLAG"] == 1).sum())
        miss_count = int((shots["SHOT_MADE_FLAG"] == 0).sum())
        st.markdown(
            f"""
            <div style="margin-top: 8px; margin-bottom: 4px; font-weight: 600;">Legend</div>
            <div style="display: grid; grid-template-columns: 16px auto auto; gap: 8px 10px; align-items: center;">
              <span style="width: 10px; height: 10px; border-radius: 50%; background: {ACCENT}; display: inline-block;"></span>
              <span>Make</span>
              <span style="color: {MUTED_TEXT}; text-align: right;">{made_count:,}</span>
              <span style="width: 10px; height: 10px; border-radius: 2px; background: #FF6B6B; display: inline-block;"></span>
              <span>Miss</span>
              <span style="color: {MUTED_TEXT}; text-align: right;">{miss_count:,}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
