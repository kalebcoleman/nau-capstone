"""SDI Explorer – Shot Difficulty Index vs Actual FG%."""

import sys
from pathlib import Path

# Allow imports from the app directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    ACCENT,
    APP_BG,
    MUTED_TEXT,
    PANEL_BG,
    SHOTS_DATA_PATH,
    TEXT_COLOR,
    apply_theme,
    load_shots_data,
    load_usage_data,
)

st.set_page_config(page_title="SDI Explorer", layout="wide")
apply_theme()

# Hardcoded — single season in dataset
SEASON = "2025-26"
SEASON_TYPE = "regular"


# =============================================================================
# SDI computation
# =============================================================================

@st.cache_data(show_spinner="Computing SDI…")
def compute_player_sdi_for_season(min_shots: int = 50) -> pd.DataFrame:
    shots_df = load_shots_data()
    df = shots_df[shots_df["season"].astype(str) == SEASON].copy()
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str) == SEASON_TYPE]
    if "SHOT_ATTEMPTED_FLAG" in df.columns:
        df = df[df["SHOT_ATTEMPTED_FLAG"] == 1]

    cols = [
        "PLAYER_NAME", "PLAYER_ID", "TEAM_NAME", "SHOT_DISTANCE",
        "LOC_X", "LOC_Y", "MINUTES_REMAINING", "SECONDS_REMAINING",
        "PERIOD", "ACTION_TYPE", "SHOT_ZONE_BASIC", "SHOT_TYPE", "SHOT_MADE_FLAG",
    ]
    df = df[cols].copy()
    if df.empty:
        return pd.DataFrame()

    # Distance
    df["shot_distance_feet"] = pd.to_numeric(df["SHOT_DISTANCE"], errors="coerce").fillna(0)
    df["sdi_distance"] = df["shot_distance_feet"].clip(0, 35) / 35.0

    # Clock pressure
    df["seconds_in_period"] = (
        pd.to_numeric(df["MINUTES_REMAINING"], errors="coerce").fillna(0) * 60
        + pd.to_numeric(df["SECONDS_REMAINING"], errors="coerce").fillna(0)
    )
    df["sdi_clock"] = 1 - (df["seconds_in_period"].clip(0, 720) / 720.0)

    # Shot type difficulty
    action = df["ACTION_TYPE"].str.lower().fillna("")
    df["sdi_shot_type"] = 0.3
    df.loc[action.str.contains("pullup|step back|fadeaway|turnaround"), "sdi_shot_type"] = 0.8
    df.loc[action.str.contains("driving|running"), "sdi_shot_type"] = 0.6
    df.loc[action.str.contains("dunk"), "sdi_shot_type"] = 0.1
    df.loc[action.str.contains("layup") & ~action.str.contains("driving"), "sdi_shot_type"] = 0.2

    # Zone difficulty
    zone_difficulty = {
        "Restricted Area": 0.1, "In The Paint (Non-RA)": 0.4,
        "Mid-Range": 0.7, "Left Corner 3": 0.5, "Right Corner 3": 0.5,
        "Above the Break 3": 0.6, "Backcourt": 0.9,
    }
    df["sdi_zone"] = df["SHOT_ZONE_BASIC"].map(zone_difficulty).fillna(0.5)

    # Angle difficulty
    df["LOC_X"] = pd.to_numeric(df["LOC_X"], errors="coerce").fillna(0)
    df["LOC_Y"] = pd.to_numeric(df["LOC_Y"], errors="coerce").fillna(0)
    df["shot_angle"] = np.arctan2(df["LOC_X"], df["LOC_Y"].clip(lower=1))
    df["sdi_angle"] = np.abs(df["shot_angle"]) / (np.pi / 2)

    # Weighted combination
    df["SDI"] = (
        0.30 * df["sdi_distance"]
        + 0.20 * df["sdi_clock"]
        + 0.20 * df["sdi_shot_type"]
        + 0.15 * df["sdi_zone"]
        + 0.15 * df["sdi_angle"]
    )

    # Pull-up rate
    df["is_jump_shot"] = (
        df["ACTION_TYPE"].str.lower()
        .str.contains("jump shot|pullup|step back|fadeaway", na=False)
        .astype(int)
    )

    # Aggregate by player
    player_sdi = (
        df.groupby(["PLAYER_NAME", "PLAYER_ID"])
        .agg({
            "SDI": "mean",
            "SHOT_MADE_FLAG": ["mean", "count"],
            "TEAM_NAME": "first",
            "is_jump_shot": "mean",
        })
        .reset_index()
    )
    player_sdi.columns = [
        "PLAYER_NAME", "PLAYER_ID", "avg_SDI", "actual_FG_pct",
        "attempts", "TEAM_NAME", "pullup_rate",
    ]

    # Estimate xFG
    league_fg = df["SHOT_MADE_FLAG"].mean()
    player_sdi["avg_xFG"] = league_fg + (0.5 - player_sdi["avg_SDI"]) * 0.3
    player_sdi["avg_xFG"] = player_sdi["avg_xFG"].clip(0.25, 0.75)

    player_sdi = player_sdi[player_sdi["attempts"] >= min_shots]

    # Merge usage
    usage_df = load_usage_data().copy()
    if usage_df.empty:
        usage_df = pd.DataFrame(columns=["PLAYER_ID", "usage_pct"])
    else:
        usage_df["PLAYER_ID"] = usage_df.get("personId", usage_df.get("PLAYER_ID")).astype(str)
        usage_df["usage_pct"] = pd.to_numeric(usage_df.get("usagePercentage"), errors="coerce")
        usage_df = usage_df.groupby("PLAYER_ID", as_index=False)["usage_pct"].mean().fillna(0)

    player_sdi["PLAYER_ID"] = player_sdi["PLAYER_ID"].astype(str)
    usage_df["PLAYER_ID"] = usage_df["PLAYER_ID"].astype(str)
    player_sdi = player_sdi.merge(usage_df, on="PLAYER_ID", how="left")
    player_sdi["usage_pct"] = player_sdi["usage_pct"].fillna(0)
    player_sdi["archetype"] = "—"

    return player_sdi.sort_values("avg_SDI", ascending=False)


# =============================================================================
# Scatter plot
# =============================================================================

def plot_sdi_scatter(df: pd.DataFrame, min_attempts: int = 100) -> go.Figure:
    filtered = df[df["attempts"] >= min_attempts].copy()
    if filtered.empty:
        return go.Figure().update_layout(
            title="No players meet the minimum attempts threshold",
            paper_bgcolor=APP_BG, plot_bgcolor=PANEL_BG,
        )

    filtered["residual"] = filtered["actual_FG_pct"] - filtered["avg_xFG"]
    median_sdi = filtered["avg_SDI"].median()
    median_fg = filtered["actual_FG_pct"].median()

    fig = px.scatter(
        filtered,
        x="avg_SDI", y="actual_FG_pct",
        size="attempts", color="residual",
        color_continuous_scale="RdYlGn",
        range_color=[-0.10, 0.10],
        custom_data=["PLAYER_NAME", "attempts", "actual_FG_pct", "avg_xFG", "residual", "archetype", "usage_pct"],
        hover_name="PLAYER_NAME",
        hover_data={
            "avg_SDI": ":.3f", "actual_FG_pct": ":.1%",
            "avg_xFG": ":.1%", "residual": ":.1%",
            "attempts": True, "archetype": True,
        },
        size_max=20,
    )

    fig.add_hline(y=median_fg, line_dash="dash", line_color=MUTED_TEXT, opacity=0.5,
                  annotation_text="Median FG%", annotation_position="right")
    fig.add_vline(x=median_sdi, line_dash="dash", line_color=MUTED_TEXT, opacity=0.5,
                  annotation_text="Median SDI", annotation_position="top")

    x_min, x_max = filtered["avg_SDI"].min(), filtered["avg_SDI"].max()
    y_min, y_max = filtered["actual_FG_pct"].min(), filtered["actual_FG_pct"].max()
    x_pad = (x_max - x_min) * 0.02
    y_pad = (y_max - y_min) * 0.02
    for x, y, text, xa, ya in [
        (x_min + x_pad, y_max - y_pad, "Elite Finishers", "left", "top"),
        (x_max - x_pad, y_max - y_pad, "Elite Shot-Makers", "right", "top"),
        (x_min + x_pad, y_min + y_pad, "Inefficient", "left", "bottom"),
        (x_max - x_pad, y_min + y_pad, "Volume Shooters", "right", "bottom"),
    ]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                           font=dict(size=10, color=MUTED_TEXT),
                           xanchor=xa, yanchor=ya, opacity=0.7)

    elite = filtered[(filtered["residual"] >= 0.07) & (filtered["usage_pct"] >= 0.25)]
    for _, row in elite.iterrows():
        fig.add_annotation(x=row["avg_SDI"], y=row["actual_FG_pct"],
                           text=row["PLAYER_NAME"], showarrow=False,
                           font=dict(size=9, color=TEXT_COLOR), yshift=12)

    fig.update_layout(
        title=dict(text=f"Shot Difficulty vs Actual Efficiency ({len(filtered)} players)",
                   font=dict(color=TEXT_COLOR)),
        xaxis_title="Average Shot Difficulty Index (SDI)",
        yaxis_title="Actual Field Goal %",
        paper_bgcolor=APP_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        coloraxis_colorbar=dict(title="FG% Residual", tickformat="+.0%"),
        margin=dict(l=60, r=20, t=50, b=60),
        clickmode="event+select",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickformat=".0%")
    return fig


def render_sdi_player_card(player_data) -> str | None:
    st.subheader("Selected Player")
    if player_data is None or (hasattr(player_data, "empty") and player_data.empty):
        st.info("Click a player on the scatter plot to see their details.")
        return None

    player_name = player_data.get("PLAYER_NAME", "Unknown")
    attempts = int(player_data.get("attempts", 0))
    avg_sdi = player_data.get("avg_SDI", 0)
    avg_xfg = player_data.get("avg_xFG", 0)
    actual_fg = player_data.get("actual_FG_pct", 0)
    usage = player_data.get("usage_pct", 0)
    pullup_rate = player_data.get("pullup_rate", 0)
    fg_residual = actual_fg - avg_xfg
    sign = "+" if fg_residual > 0 else ""

    st.markdown(f"### {player_name}")

    c1, c2 = st.columns(2)
    c1.metric("Attempts", f"{attempts:,}")
    c2.metric("FG%", f"{actual_fg:.1%}")

    c3, c4 = st.columns(2)
    c3.metric("Expected FG%", f"{avg_xfg:.1%}")
    c4.metric("FG Residual", f"{sign}{fg_residual:.1%}")

    c5, c6 = st.columns(2)
    c5.metric("Shot Difficulty", f"{avg_sdi:.3f}")
    c6.metric("Usage %", f"{usage:.2%}")

    st.metric("Pull-up Rate", f"{pullup_rate:.2%}")
    return player_name


# =============================================================================
# Main page
# =============================================================================

st.title("📊 SDI Explorer")

if not SHOTS_DATA_PATH.exists():
    st.error(f"Data not found: {SHOTS_DATA_PATH}")
    st.stop()

with st.sidebar:
    st.header("SDI Filters")
    st.caption(f"**Season:** {SEASON}  ·  **Type:** {SEASON_TYPE}")
    min_attempts_sdi = st.slider("Min Attempts", 50, 500, 100, 25, key="sdi_min_att",
                                  help="Filter players by minimum shot attempts")

st.markdown(
    f'<div style="color: {MUTED_TEXT}; margin-bottom: 16px;">'
    f'Explore Shot Difficulty Index (SDI) vs Actual FG% for <b>{SEASON} {SEASON_TYPE}</b>. '
    f'Click a player to view their detailed stats.</div>',
    unsafe_allow_html=True,
)

sdi_df = compute_player_sdi_for_season(min_shots=50)

if sdi_df.empty:
    st.warning(f"No shot data for {SEASON} {SEASON_TYPE}.")
else:
    sdi_plot_col, sdi_card_col = st.columns([2.5, 1])

    with sdi_plot_col:
        sdi_fig = plot_sdi_scatter(sdi_df, min_attempts=min_attempts_sdi)
        sdi_selection = st.plotly_chart(
            sdi_fig, use_container_width=True,
            config={"displayModeBar": False},
            on_select="rerun", selection_mode="points",
            key="sdi_scatter_chart",
        )

        sdi_selected_points = None
        if isinstance(sdi_selection, dict):
            sdi_selected_points = sdi_selection.get("points", []) or sdi_selection.get("selection", {}).get("points", [])
        else:
            sel = getattr(sdi_selection, "selection", None)
            if isinstance(sel, dict):
                sdi_selected_points = sel.get("points", [])

        if sdi_selected_points is not None and len(sdi_selected_points) == 0:
            st.session_state.pop("sdi_selected_player", None)
        elif sdi_selected_points:
            st.session_state["sdi_selected_player"] = sdi_selected_points[0]["customdata"][0]

    with sdi_card_col:
        selected_name = st.session_state.get("sdi_selected_player")
        if selected_name:
            row = sdi_df[sdi_df["PLAYER_NAME"] == selected_name]
            if not row.empty:
                render_sdi_player_card(row.iloc[0])
            else:
                render_sdi_player_card(None)
        else:
            render_sdi_player_card(None)
