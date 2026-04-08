"""Shared utilities for the NHL Shot Analytics within the dashboard."""

from pathlib import Path

import matplotlib.pyplot as plt
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
    PLOT_WIDTH,
    TEXT_COLOR,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
NHL_DATA_DIR = REPO_ROOT / "analysis" / "nhl" / "data" / "app_data"
NHL_SHOTS_DATA_PATH = NHL_DATA_DIR / "nhl_shots_2024.csv.gz"


@st.cache_data(show_spinner=False)
def load_nhl_shots_data() -> pd.DataFrame:
    if not NHL_SHOTS_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(NHL_SHOTS_DATA_PATH)


@st.cache_data(show_spinner=False)
def get_nhl_seasons(db_path: str = "") -> list[str]:
    df = load_nhl_shots_data()
    if df.empty or "season" not in df.columns:
        return []
    return sorted(df["season"].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def get_nhl_teams(season: str) -> list[str]:
    df = load_nhl_shots_data()
    if df.empty:
        return []
    filtered = df[df["season"].astype(str) == str(season)]
    return sorted(filtered["teamCode"].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def get_nhl_players(season: str, team_name: str | None) -> list[str]:
    df = load_nhl_shots_data()
    if df.empty:
        return []
    filtered = df[df["season"].astype(str) == str(season)]
    if team_name and team_name != "All Teams":
        filtered = filtered[filtered["teamCode"].astype(str) == str(team_name)]
    return sorted(filtered["shooterName"].dropna().astype(str).unique().tolist())


def load_nhl_shots(
    season: str,
    team_name: str | None,
    player_name: str | None,
) -> pd.DataFrame:
    df = load_nhl_shots_data()
    if df.empty:
        return df
    filtered = df[df["season"].astype(str) == str(season)]
    if team_name and team_name != "All Teams":
        filtered = filtered[filtered["teamCode"].astype(str) == str(team_name)]
    if player_name and player_name != "All Players":
        filtered = filtered[filtered["shooterName"].astype(str) == str(player_name)]
    return filtered[["xCord", "yCord", "goal", "shotType", "shotDistance"]].copy()


@st.cache_data(show_spinner=False)
def get_nhl_shot_points(
    season: str,
    team_name: str | None,
    player_name: str | None,
) -> pd.DataFrame:
    df = load_nhl_shots_data()
    if df.empty:
        return df
    filtered = df[df["season"].astype(str) == str(season)]
    if team_name and team_name != "All Teams":
        filtered = filtered[filtered["teamCode"].astype(str) == str(team_name)]
    if player_name and player_name != "All Players":
        filtered = filtered[filtered["shooterName"].astype(str) == str(player_name)]
    return filtered[["shotID", "xCord", "yCord", "goal"]].copy()


@st.cache_data(show_spinner=False)
def get_nhl_shot_detail(shot_id: str) -> dict | None:
    if not shot_id:
        return None
    df = load_nhl_shots_data()
    if df.empty:
        return None
    detail = df[df["shotID"].astype(str) == str(shot_id)]
    if detail.empty:
        return None
    return detail.iloc[0].to_dict()


def sample_nhl_shots_with_far(
    shots: pd.DataFrame, max_points: int, far_threshold: int
) -> pd.DataFrame:
    shots = shots.copy()
    shots["xCord"] = pd.to_numeric(shots["xCord"], errors="coerce")
    shots["yCord"] = pd.to_numeric(shots["yCord"], errors="coerce")
    shots = shots.dropna(subset=["xCord", "yCord", "goal"])

    shots["shot_distance_ft"] = (
        ((shots["xCord"].abs() - 89) ** 2 + shots["yCord"] ** 2) ** 0.5
    )

    if len(shots) <= max_points:
        return shots
    far = shots[shots["shot_distance_ft"] >= far_threshold].copy()
    near = shots[shots["shot_distance_ft"] < far_threshold].copy()
    if len(far) >= max_points:
        return far.sort_values("shot_distance_ft", ascending=False).head(max_points)
    remaining = max_points - len(far)
    if len(near) > remaining:
        near = near.sample(n=remaining, random_state=42)
    return pd.concat([far, near], ignore_index=True)


def add_rink_traces(fig, line_color="#FFFFFF"):
    outer = dict(
        type="rect",
        x0=-100,
        y0=-42.5,
        x1=100,
        y1=42.5,
        line=dict(color=line_color, width=2),
    )
    center = dict(
        type="line",
        x0=0,
        y0=-42.5,
        x1=0,
        y1=42.5,
        line=dict(color="red", width=2),
    )
    blue1 = dict(
        type="line",
        x0=-25,
        y0=-42.5,
        x1=-25,
        y1=42.5,
        line=dict(color="blue", width=2),
    )
    blue2 = dict(
        type="line",
        x0=25,
        y0=-42.5,
        x1=25,
        y1=42.5,
        line=dict(color="blue", width=2),
    )
    goal1 = dict(
        type="line",
        x0=-89,
        y0=-42.5,
        x1=-89,
        y1=42.5,
        line=dict(color="red", width=1),
    )
    goal2 = dict(
        type="line",
        x0=89,
        y0=-42.5,
        x1=89,
        y1=42.5,
        line=dict(color="red", width=1),
    )

    fig.add_shape(outer)
    fig.add_shape(center)
    fig.add_shape(blue1)
    fig.add_shape(blue2)
    fig.add_shape(goal1)
    fig.add_shape(goal2)
    fig.add_shape(
        type="circle",
        x0=-15,
        y0=-15,
        x1=15,
        y1=15,
        line_color="blue",
        line_width=1,
    )


def plot_nhl_scatter_plotly(shots: pd.DataFrame) -> go.Figure:
    shots = shots.copy()
    shots["result"] = shots["goal"].map({1: "Goal", 0: "No Goal"}).fillna("No Goal")
    fig = px.scatter(
        shots,
        x="xCord",
        y="yCord",
        color="result",
        color_discrete_map={"Goal": ACCENT, "No Goal": "#FF6B6B"},
        category_orders={"result": ["No Goal", "Goal"]},
        custom_data=["shotID"],
        opacity=0.9,
        width=PLOT_WIDTH,
        height=320,
    )
    fig.update_traces(
        marker=dict(size=7),
        selected=dict(marker=dict(opacity=0.98, size=8)),
        unselected=dict(marker=dict(opacity=0.6)),
        selectedpoints=[],
        selector=dict(mode="markers"),
    )
    add_rink_traces(fig, line_color=COURT_LINE)
    fig.update_xaxes(
        range=[-100, 100],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title_text="",
    )
    fig.update_yaxes(
        range=[-42.5, 42.5],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
        title_text="",
    )
    fig.update_layout(
        autosize=False,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        plot_bgcolor=COURT_BG,
        paper_bgcolor=APP_BG,
        clickmode="event+select",
    )
    return fig


def compute_nhl_summary(df: pd.DataFrame) -> dict:
    total_shots = len(df)
    total_goals = int(df["goal"].sum()) if total_shots else 0
    goal_pct = total_goals / total_shots if total_shots else 0

    if "shotType" in df.columns:
        type_counts = (
            df.groupby("shotType")["goal"]
            .agg(shots="size", goals="sum")
            .reset_index()
        )
        type_counts["share_pct"] = type_counts["shots"] / type_counts["shots"].sum()
        type_counts["goal_pct"] = type_counts["goals"] / type_counts["shots"]
        type_counts = type_counts.sort_values("shots", ascending=False).head(5)
    else:
        type_counts = pd.DataFrame()

    return {
        "total_shots": total_shots,
        "total_goals": total_goals,
        "goal_pct": goal_pct,
        "type_table": type_counts,
    }
