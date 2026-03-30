"""NHL Shot Map Dashboard — NHL Shot Map page."""

import sys
from pathlib import Path

# Ensure the app directory is importable (for utils and nhl_utils)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import (
    ACCENT, APP_BG, MUTED_TEXT, PANEL_BG, PLOT_WIDTH, TEXT_COLOR, apply_theme
)
from nhl_utils import (
    get_nhl_seasons, get_nhl_teams, get_nhl_players, load_nhl_shots,
    get_nhl_shot_points, get_nhl_shot_detail, sample_nhl_shots_with_far,
    plot_nhl_scatter_plotly, compute_nhl_summary, NHL_SHOTS_DATA_PATH
)

def render_nhl_selected_shot(selected_shot: dict | None) -> None:
    st.subheader("Selected Shot")
    if not selected_shot:
        st.markdown(
            f"""
            <div style="width: {PLOT_WIDTH}px; max-width: 100%; margin: 0;">
              <div style="padding: 12px; border-radius: 12px; background: {PANEL_BG};
                   border: 1px solid rgba(255,255,255,0.08); color: {MUTED_TEXT};">
                Click a shot on the rink chart to see details.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    is_goal = selected_shot.get("goal") == 1
    result = "Goal" if is_goal else "No Goal"
    result_color = ACCENT if is_goal else "#FF6B6B"

    shot_distance = selected_shot.get("shotDistance")
    distance_text = "N/A" if shot_distance is None or pd.isna(shot_distance) else f"{int(round(float(shot_distance)))} ft"
    
    # Format xGoal if present
    xfg_val = selected_shot.get("xGoal")
    if xfg_val is None or (isinstance(xfg_val, float) and pd.isna(xfg_val)):
        xfg_text = "N/A"
    else:
        try:
            xfg_text = f"{float(xfg_val):.1%}"
        except (TypeError, ValueError):
            xfg_text = "N/A"
            
    sdi_val = selected_shot.get("SDI")
    sdi_text = f"{float(sdi_val):.3f}" if (sdi_val is not None and not pd.isna(sdi_val)) else "N/A"

    st.markdown(
        f"""
        <div style="width: {PLOT_WIDTH}px; max-width: 100%; margin: 0;">
          <div style="padding: 12px; border-radius: 12px; background: {PANEL_BG};
               border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-weight: 600; margin-bottom: 8px;">{selected_shot.get("shooterName", "Unknown")}</div>
            <div style="color: {MUTED_TEXT}; font-size: 0.9rem; margin-bottom: 6px;">
              Team: {selected_shot.get("teamCode", "Unknown") if not pd.isna(selected_shot.get("teamCode")) else "Unknown"}
            </div>
            <div style="margin-bottom: 6px;">Event: {selected_shot.get("event", "Shot")} · {distance_text}</div>
            <div style="margin-bottom: 6px;">Type: {selected_shot.get("shotType", "N/A")}</div>
            <div style="display: flex; gap: 12px; margin-bottom: 6px; font-size: 0.9rem; color: {MUTED_TEXT};">
                <span>SDI: <b style="color: {TEXT_COLOR}">{sdi_text}</b></span>
                <span>xGoal: <b style="color: {TEXT_COLOR}">{xfg_text}</b></span>
            </div>
            <div style="font-weight: 600; color: {result_color};">{result}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(
        page_title="NHL Shot Map Dashboard",
        layout="wide",
    )
    apply_theme()

    st.title("🏒 NHL Shot Map Dashboard")

    if not NHL_SHOTS_DATA_PATH.exists():
        st.error(f"Sample data not found: {NHL_SHOTS_DATA_PATH}. Have you run the data preprocessor?")
        st.stop()

    seasons = get_nhl_seasons()
    if not seasons:
        st.error("No valid seasons found in data.")
        st.stop()

    with st.sidebar:
        st.header("Filters")
        
        season = st.selectbox("Season", seasons, index=len(seasons)-1, key="nhl_season_select")
        
        teams = get_nhl_teams(season)
        team_options = ["All Teams"] + teams
        team_choice = st.selectbox("Team", team_options, key="nhl_team_select")
        selected_team = None if team_choice == "All Teams" else team_choice

        players = get_nhl_players(season, selected_team)
        player_options = ["All Players"] + players
        player_choice = st.selectbox("Player", player_options, key="nhl_player_select")
        selected_player = None if player_choice == "All Players" else player_choice

        max_points = st.slider("Max points", 5000, 100000, value=20000, step=5000, key="nhl_max_points_slider")
        far_threshold = st.slider("Always include shots beyond (ft)", 30, 100, value=50, step=10, key="nhl_far_threshold_slider")

    shots = load_nhl_shots(season, selected_team, selected_player)

    filter_signature = (season, selected_team or "ALL_TEAMS", selected_player or "ALL_PLAYERS")
    if st.session_state.get("nhl_filter_signature") != filter_signature:
        st.session_state["nhl_filter_signature"] = filter_signature
        st.session_state.pop("nhl_selected_shot", None)

    shots_points = get_nhl_shot_points(season, selected_team, selected_player)
    raw_count = len(shots_points)
    shots_points = sample_nhl_shots_with_far(shots_points, max_points, far_threshold)
    omitted = max(0, raw_count - len(shots_points))
    if omitted > 0:
        st.caption(
            f"Showing {len(shots_points):,} points. Omitted {omitted:,} for speed "
            f"(all shots ≥ {far_threshold} ft included)."
        )

    if shots.empty:
        st.warning(f"No shots found for the selected criteria.")
        st.stop()

    summary = compute_nhl_summary(shots)

    col_plot, col_side = st.columns([2.2, 1])

    with col_plot:
        fig = plot_nhl_scatter_plotly(shots_points)
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
            st.session_state.pop("nhl_selected_shot", None)
        elif selected_points:
            shot_id = selected_points[0]["customdata"][0]
            detail = get_nhl_shot_detail(str(shot_id))
            if detail:
                st.session_state["nhl_selected_shot"] = detail

        render_nhl_selected_shot(st.session_state.get("nhl_selected_shot"))

    with col_side:
        st.markdown(
            '<div style="margin-top:-18px; margin-bottom: 8px; font-size: 1.35rem; font-weight: 700;">Summary</div>',
            unsafe_allow_html=True,
        )
        m1 = st.columns(2)
        m1[0].metric("Total Shots", f"{summary['total_shots']:,}")
        m1[1].metric("Goal %", f"{summary['goal_pct'] * 100:.1f}%")

        st.subheader("Top Shot Types")
        type_table = summary.get("type_table")
        if type_table is not None and not type_table.empty:
            type_table["share_pct"] = type_table["share_pct"] * 100
            type_table = type_table.sort_values("share_pct")

            fig_bar, ax_bar = plt.subplots(figsize=(4, 3.4))
            fig_bar.patch.set_facecolor(PANEL_BG)
            ax_bar.set_facecolor(PANEL_BG)
            
            # Using shotType for y-axis
            y_labels = [str(x) for x in type_table["shotType"]]
            ax_bar.barh(y_labels, type_table["share_pct"], color=ACCENT, alpha=0.85)
            ax_bar.set_xlabel("Shot Share (%)", color=MUTED_TEXT, fontsize=9)
            ax_bar.tick_params(colors=TEXT_COLOR, labelsize=9)
            ax_bar.grid(axis="x", color="white", alpha=0.08, linewidth=1)
            for spine in ax_bar.spines.values():
                spine.set_visible(False)
            max_share = max(type_table["share_pct"].max(), 1)
            ax_bar.set_xlim(0, max_share * 1.25)
            for idx, share in enumerate(type_table["share_pct"]):
                ax_bar.text(share + max_share * 0.03, idx, f"{share:.1f}%",
                            va="center", ha="left", color=TEXT_COLOR, fontsize=9)
            st.pyplot(fig_bar, use_container_width=True)

        made_count = int(summary["total_goals"])
        miss_count = int(summary["total_shots"] - summary["total_goals"])
        st.markdown(
            f"""
            <div style="margin-top: 8px; margin-bottom: 4px; font-weight: 600;">Legend</div>
            <div style="display: grid; grid-template-columns: 16px auto auto; gap: 8px 10px; align-items: center;">
              <span style="width: 10px; height: 10px; border-radius: 50%; background: {ACCENT}; display: inline-block;"></span>
              <span>Goal</span>
              <span style="color: {MUTED_TEXT}; text-align: right;">{made_count:,}</span>
              <span style="width: 10px; height: 10px; border-radius: 2px; background: #FF6B6B; display: inline-block;"></span>
              <span>No Goal</span>
              <span style="color: {MUTED_TEXT}; text-align: right;">{miss_count:,}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
