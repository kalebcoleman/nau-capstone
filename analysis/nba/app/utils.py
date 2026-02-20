"""Shared utilities for the NBA Shot Analytics Dashboard."""

import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.patches import Arc, Circle, Rectangle

# =============================================================================
# Paths
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
MODEL_DIR = REPO_ROOT / "models"
SDI_DATA_PATH = ANALYSIS_DATA_DIR / "player_clusters.csv"
SHOTS_DATA_PATH = ANALYSIS_DATA_DIR / "nba_shots_2025-26.csv.gz"
USAGE_DATA_PATH = ANALYSIS_DATA_DIR / "player_box_usage_2025-26.csv"

# =============================================================================
# Theme constants
# =============================================================================
APP_BG = "#0B0F1A"
PANEL_BG = "#121826"
COURT_BG = "#23272B"
COURT_LINE = "#FFFFFF"
TEXT_COLOR = "#E6E8EE"
MUTED_TEXT = "#98A1B3"
ACCENT = "#F5C84C"
PLOT_WIDTH = 620


def apply_theme() -> None:
    """Inject CSS theme for consistent look across pages."""
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            background: {APP_BG};
            color: {TEXT_COLOR};
            font-family: "Space Grotesk", "IBM Plex Sans", "SF Pro Display",
                         "Segoe UI", sans-serif;
        }}
        .stApp {{
            background: {APP_BG};
        }}
        [data-testid="stSidebar"] {{
            background: {PANEL_BG};
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {{
            color: {TEXT_COLOR};
        }}
        h1, h2, h3, h4 {{
            color: {TEXT_COLOR};
            letter-spacing: 0.02em;
        }}
        p, span, label {{
            color: {TEXT_COLOR};
        }}
        .stMetric {{
            background: {PANEL_BG};
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px;
        }}
        .stMetric div {{
            overflow: visible !important;
            text-overflow: unset !important;
            white-space: normal !important;
        }}
        .stMetric label {{
            color: {MUTED_TEXT};
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(show_spinner=False)
def load_shots_data() -> pd.DataFrame:
    df = pd.read_csv(SHOTS_DATA_PATH)
    if "season_type" in df.columns:
        df["season_type"] = df["season_type"].astype(str).str.lower()
    return df


@st.cache_data(show_spinner=False)
def load_usage_data() -> pd.DataFrame:
    if not USAGE_DATA_PATH.exists():
        return pd.DataFrame(columns=["personId", "usagePercentage"])
    return pd.read_csv(USAGE_DATA_PATH)


@st.cache_data(show_spinner=False)
def get_seasons(db_path: str) -> list[str]:
    df = load_shots_data()
    return sorted(df["season"].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def get_season_types(db_path: str, season: str) -> list[str]:
    df = load_shots_data()
    filtered = df[df["season"].astype(str) == str(season)]
    if "season_type" not in filtered.columns:
        return ["regular"]
    return sorted(filtered["season_type"].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def get_teams(db_path: str, season: str, season_type: str) -> list[str]:
    df = load_shots_data()
    filtered = df[df["season"].astype(str) == str(season)]
    if "season_type" in filtered.columns:
        filtered = filtered[filtered["season_type"].astype(str) == str(season_type)]
    return sorted(filtered["TEAM_NAME"].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def get_players(
    db_path: str,
    season: str,
    season_type: str,
    team_name: str | None,
) -> list[str]:
    df = load_shots_data()
    filtered = df[df["season"].astype(str) == str(season)]
    if "season_type" in filtered.columns:
        filtered = filtered[filtered["season_type"].astype(str) == str(season_type)]
    if team_name:
        filtered = filtered[filtered["TEAM_NAME"].astype(str) == str(team_name)]
    return sorted(filtered["PLAYER_NAME"].dropna().astype(str).unique().tolist())


def load_shots(
    db_path: str,
    season: str,
    season_type: str,
    team_name: str | None,
    player_name: str | None,
) -> pd.DataFrame:
    df = load_shots_data()
    filtered = df[df["season"].astype(str) == str(season)]
    if "season_type" in filtered.columns:
        filtered = filtered[filtered["season_type"].astype(str) == str(season_type)]
    if "SHOT_ATTEMPTED_FLAG" in filtered.columns:
        filtered = filtered[filtered["SHOT_ATTEMPTED_FLAG"] == 1]
    if team_name:
        filtered = filtered[filtered["TEAM_NAME"].astype(str) == str(team_name)]
    if player_name:
        filtered = filtered[filtered["PLAYER_NAME"].astype(str) == str(player_name)]
    return filtered[
        ["LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "SHOT_TYPE", "SHOT_ZONE_BASIC"]
    ].copy()


@st.cache_data(show_spinner=False)
def get_shot_points(
    db_path: str,
    season: str,
    season_type: str,
    team_name: str | None,
    player_name: str | None,
) -> pd.DataFrame:
    df = load_shots_data()
    filtered = df[df["season"].astype(str) == str(season)]
    if "season_type" in filtered.columns:
        filtered = filtered[filtered["season_type"].astype(str) == str(season_type)]
    if "SHOT_ATTEMPTED_FLAG" in filtered.columns:
        filtered = filtered[filtered["SHOT_ATTEMPTED_FLAG"] == 1]
    if team_name:
        filtered = filtered[filtered["TEAM_NAME"].astype(str) == str(team_name)]
    if player_name:
        filtered = filtered[filtered["PLAYER_NAME"].astype(str) == str(player_name)]
    return filtered[["shot_id", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG"]].copy()


@st.cache_data(show_spinner=False)
def get_shot_detail(db_path: str, shot_id: str) -> dict | None:
    if not shot_id:
        return None
    df = load_shots_data()
    detail = df[df["shot_id"].astype(str) == str(shot_id)]
    if detail.empty:
        return None
    return detail.iloc[0].to_dict()


@st.cache_resource(show_spinner=False)
def load_xfg_model(season: str):
    model_path = MODEL_DIR / f"xp_model_{season}.joblib"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


# =============================================================================
# xFG / SDI computation helpers
# =============================================================================

def build_xfg_features(shot: dict) -> pd.DataFrame | None:
    if not shot:
        return None

    def is_missing(value) -> bool:
        return (
            value is None
            or (isinstance(value, float) and pd.isna(value))
            or pd.isna(value)
        )

    def to_float(value):
        if is_missing(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    loc_x = to_float(shot.get("LOC_X"))
    loc_y = to_float(shot.get("LOC_Y"))
    period = to_float(shot.get("PERIOD"))
    minutes = to_float(shot.get("MINUTES_REMAINING"))
    seconds = to_float(shot.get("SECONDS_REMAINING"))

    if any(v is None for v in [loc_x, loc_y, period, minutes, seconds]):
        return None

    shot_distance = to_float(shot.get("SHOT_DISTANCE"))
    if shot_distance is None:
        shot_distance = ((loc_x**2 + loc_y**2) ** 0.5) / 10.0

    shot_zone_basic = shot.get("SHOT_ZONE_BASIC")
    shot_zone_area = shot.get("SHOT_ZONE_AREA")
    if is_missing(shot_zone_basic) or is_missing(shot_zone_area):
        return None

    shot_angle = math.atan2(loc_x, max(loc_y, 1.0))
    seconds_in_period = minutes * 60 + seconds
    is_clutch = int(period >= 4 and seconds_in_period <= 120)

    action = str(shot.get("ACTION_TYPE") or "").lower()
    is_layup = int("layup" in action or "finger roll" in action)
    is_dunk = int("dunk" in action)
    is_jump_shot = int(
        "jump shot" in action
        or "pullup" in action
        or "step back" in action
        or "fadeaway" in action
    )
    is_hook = int("hook" in action)
    is_floater = int("float" in action)

    features = {
        "LOC_X": loc_x,
        "LOC_Y": loc_y,
        "shot_distance_feet": shot_distance,
        "shot_angle": shot_angle,
        "PERIOD": period,
        "seconds_in_period": seconds_in_period,
        "is_clutch": is_clutch,
        "is_layup": is_layup,
        "is_dunk": is_dunk,
        "is_jump_shot": is_jump_shot,
        "is_hook": is_hook,
        "is_floater": is_floater,
        "SHOT_ZONE_BASIC": shot_zone_basic,
        "SHOT_ZONE_AREA": shot_zone_area,
    }

    return pd.DataFrame([features])


def compute_xfg_for_shot(shot: dict, season: str) -> float | None:
    model = load_xfg_model(season)
    if model is None:
        return None
    features = build_xfg_features(shot)
    if features is None:
        return None
    try:
        proba = model.predict_proba(features)
        return float(proba[0, 1])
    except Exception:
        return None


def compute_sdi_for_shot(shot: dict) -> float:
    """Compute the Shot Difficulty Index for a single shot."""
    try:
        d_ft = float(shot.get("SHOT_DISTANCE", 0))
        s_dist = min(d_ft, 35) / 35.0

        m_rem = float(shot.get("MINUTES_REMAINING", 0))
        s_rem = float(shot.get("SECONDS_REMAINING", 0))
        sec_total = m_rem * 60 + s_rem
        s_clock = 1 - (min(sec_total, 720) / 720.0)

        act = str(shot.get("ACTION_TYPE", "")).lower()
        s_type = 0.3
        if any(x in act for x in ["pullup", "step back", "fadeaway", "turnaround"]):
            s_type = 0.8
        elif any(x in act for x in ["driving", "running"]):
            s_type = 0.6
        elif "dunk" in act:
            s_type = 0.1
        elif "layup" in act and "driving" not in act:
            s_type = 0.2

        zon = shot.get("SHOT_ZONE_BASIC", "")
        z_map = {
            "Restricted Area": 0.1,
            "In The Paint (Non-RA)": 0.4,
            "Mid-Range": 0.7,
            "Left Corner 3": 0.5,
            "Right Corner 3": 0.5,
            "Above the Break 3": 0.6,
            "Backcourt": 0.9,
        }
        s_zone = z_map.get(zon, 0.5)

        lx = float(shot.get("LOC_X", 0))
        ly = float(shot.get("LOC_Y", 0))
        ang = np.arctan2(lx, max(ly, 1))
        s_ang = np.abs(ang) / (np.pi / 2)

        return float(
            0.3 * s_dist + 0.2 * s_clock + 0.2 * s_type + 0.15 * s_zone + 0.15 * s_ang
        )
    except Exception:
        return 0.0


# =============================================================================
# Summary helpers
# =============================================================================

def compute_summary(df: pd.DataFrame) -> dict:
    total_fga = len(df)
    total_fgm = int(df["SHOT_MADE_FLAG"].sum()) if total_fga else 0
    total_fg = total_fgm / total_fga if total_fga else 0

    is_three = df["SHOT_TYPE"].fillna("").str.contains("3PT", case=False, na=False)
    threes = df[is_three]
    twos = df[~is_three]

    three_fga = len(threes)
    three_fgm = int(threes["SHOT_MADE_FLAG"].sum()) if three_fga else 0
    three_fg = three_fgm / three_fga if three_fga else 0

    two_fga = len(twos)
    two_fgm = int(twos["SHOT_MADE_FLAG"].sum()) if two_fga else 0
    two_fg = two_fgm / two_fga if two_fga else 0

    zone = (
        df.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"]
        .agg(fga="size", fgm="sum")
        .reset_index()
    )
    zone["share_pct"] = zone["fga"] / zone["fga"].sum()
    zone["fg_pct"] = zone["fgm"] / zone["fga"]
    zone = zone.sort_values("fga", ascending=False).head(5)

    return {
        "total_fga": total_fga,
        "total_fgm": total_fgm,
        "total_fg": total_fg,
        "two_fga": two_fga,
        "two_fgm": two_fgm,
        "two_fg": two_fg,
        "three_fga": three_fga,
        "three_fgm": three_fgm,
        "three_fg": three_fg,
        "zone_table": zone,
    }


# =============================================================================
# Court drawing (matplotlib)
# =============================================================================

def draw_court(
    ax=None,
    color="white",
    lw=2,
    bg="#0b0f1a",
    show_axis=False,
    full_court=False,
    draw_corner_threes=True,
    zorder=1,
    pad_x=0,
    pad_y=0,
):
    if ax is None:
        ax = plt.gca()

    ax.set_facecolor(bg)
    baseline_y = -52
    halfcourt_y = 418
    corner_x = 220
    arc_radius = 237.5
    corner_y = (arc_radius**2 - corner_x**2) ** 0.5

    def add_half_court(y_flip=False):
        def y(val):
            return 2 * halfcourt_y - val if y_flip else val

        def add_rect(x, y0, w, h):
            y0f = y(y0)
            y1f = y(y0 + h)
            y_min = min(y0f, y1f)
            height = abs(y1f - y0f)
            ax.add_patch(
                Rectangle((x, y_min), w, height, fill=False, linewidth=lw, color=color)
            )

        ax.add_patch(
            Circle((0, y(0)), 7.5, fill=False, linewidth=lw, color=color, zorder=zorder)
        )
        ax.add_patch(
            Rectangle((-30, y(-12.5)), 60, 0, linewidth=lw, color=color, zorder=zorder)
        )

        add_rect(-80, baseline_y, 160, 190)
        add_rect(-60, baseline_y, 120, 190)

        if not y_flip:
            ax.add_patch(
                Arc(
                    (0, y(142.5)), 120, 120, theta1=0, theta2=180,
                    linewidth=lw, color=color, zorder=zorder,
                )
            )
            ax.add_patch(
                Arc(
                    (0, y(142.5)), 120, 120, theta1=180, theta2=360,
                    linewidth=lw, color=color, linestyle="dashed", zorder=zorder,
                )
            )
            ax.add_patch(
                Arc(
                    (0, y(0)), 80, 80, theta1=0, theta2=180,
                    linewidth=lw, color=color, zorder=zorder,
                )
            )
            if draw_corner_threes:
                ax.plot(
                    [-corner_x, -corner_x], [y(baseline_y), y(corner_y)],
                    linewidth=lw, color=color, zorder=zorder,
                )
                ax.plot(
                    [corner_x, corner_x], [y(baseline_y), y(corner_y)],
                    linewidth=lw, color=color, zorder=zorder,
                )
            ax.add_patch(
                Arc(
                    (0, y(0)), 2 * arc_radius, 2 * arc_radius, theta1=22, theta2=158,
                    linewidth=lw, color=color, zorder=zorder,
                )
            )
        else:
            ax.add_patch(
                Arc(
                    (0, y(142.5)), 120, 120, theta1=180, theta2=360,
                    linewidth=lw, color=color, zorder=zorder,
                )
            )
            ax.add_patch(
                Arc(
                    (0, y(142.5)), 120, 120, theta1=0, theta2=180,
                    linewidth=lw, color=color, linestyle="dashed", zorder=zorder,
                )
            )
            ax.add_patch(
                Arc(
                    (0, y(0)), 80, 80, theta1=180, theta2=360,
                    linewidth=lw, color=color, zorder=zorder,
                )
            )
            if draw_corner_threes:
                ax.plot(
                    [-corner_x, -corner_x], [y(baseline_y), y(corner_y)],
                    linewidth=lw, color=color, zorder=zorder,
                )
                ax.plot(
                    [corner_x, corner_x], [y(baseline_y), y(corner_y)],
                    linewidth=lw, color=color, zorder=zorder,
                )
            ax.add_patch(
                Arc(
                    (0, y(0)), 2 * arc_radius, 2 * arc_radius, theta1=202, theta2=338,
                    linewidth=lw, color=color, zorder=zorder,
                )
            )

    add_half_court(y_flip=False)
    if full_court:
        add_half_court(y_flip=True)
        ax.plot(
            [-250, 250], [halfcourt_y, halfcourt_y],
            color=color, linewidth=lw, zorder=zorder,
        )
        ax.add_patch(
            Circle(
                (0, halfcourt_y), 60, fill=False,
                linewidth=lw, color=color, zorder=zorder,
            )
        )
        ax.add_patch(
            Rectangle(
                (-250, baseline_y), 500, 940, fill=False,
                linewidth=lw, color=color, zorder=zorder,
            )
        )
        ax.set_xlim(-250, 250)
        ax.set_ylim(-52, 888)
    else:
        ax.add_patch(
            Rectangle(
                (-250 - pad_x, baseline_y), 500 + (2 * pad_x),
                (halfcourt_y - baseline_y) + pad_y, fill=False,
                linewidth=lw, color=color, zorder=zorder,
            )
        )
        ax.plot(
            [-250 - pad_x, 250 + pad_x], [halfcourt_y, halfcourt_y],
            color=color, linewidth=lw, zorder=zorder,
        )
        ax.add_patch(
            Arc(
                (0, halfcourt_y), 120, 120, theta1=180, theta2=360,
                linewidth=lw, color=color, zorder=zorder,
            )
        )
        ax.set_xlim(-250, 250)
        ax.set_ylim(-52, 418)
    ax.set_aspect("equal")
    if show_axis:
        ax.tick_params(colors=color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(color)
        ax.set_xlabel("Court X (NBA units; 1 unit = 0.1 ft)", color=color)
        ax.set_ylabel("Court Y (NBA units; 1 unit = 0.1 ft)", color=color)
        ax.set_xticks(range(-250, 251, 50))
        if full_court:
            ax.set_yticks(range(-50, 901, 100))
        else:
            ax.set_yticks(range(-50, 451, 50))
        for label in ax.get_yticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment("right")
    else:
        ax.axis("off")
    return ax


# =============================================================================
# Court drawing (Plotly)
# =============================================================================

def arc_points(cx, cy, r, theta1, theta2, n=120):
    angles = [theta1 + (theta2 - theta1) * i / (n - 1) for i in range(n)]
    x = [cx + r * math.cos(math.radians(a)) for a in angles]
    y = [cy + r * math.sin(math.radians(a)) for a in angles]
    return x, y


def add_court_traces(fig, full_court: bool, line_color: str = "#FFFFFF"):
    baseline_y = -52
    halfcourt_y = 418
    corner_x = 220
    arc_radius = 237.5
    corner_y = (arc_radius**2 - corner_x**2) ** 0.5

    def _line(x, y, dash=None):
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color=line_color, width=2, dash=dash),
                hoverinfo="skip", showlegend=False,
            )
        )

    def add_half(y_flip=False):
        def y(val):
            return 2 * halfcourt_y - val if y_flip else val

        # Paint
        _line([-80, 80, 80, -80, -80],
              [y(baseline_y), y(baseline_y), y(baseline_y + 190), y(baseline_y + 190), y(baseline_y)])
        _line([-60, 60, 60, -60, -60],
              [y(baseline_y), y(baseline_y), y(baseline_y + 190), y(baseline_y + 190), y(baseline_y)])

        # Hoop
        hx, hy = arc_points(0, y(0), 7.5, 0, 360, n=90)
        _line(hx, hy)
        _line([-30, 30], [y(-12.5), y(-12.5)])

        # FT circle
        ft_solid = arc_points(0, y(142.5), 60, 0, 180, n=90)
        ft_dashed = arc_points(0, y(142.5), 60, 180, 360, n=90)
        if not y_flip:
            _line(ft_solid[0], ft_solid[1])
            _line(ft_dashed[0], ft_dashed[1], dash="dash")
        else:
            _line(ft_dashed[0], ft_dashed[1])
            _line(ft_solid[0], ft_solid[1], dash="dash")

        # Restricted area
        ra = arc_points(0, y(0), 40, 0, 180, n=90)
        ra_alt = arc_points(0, y(0), 40, 180, 360, n=90)
        _line(*(ra_alt if y_flip else ra))

        # Corner 3s
        _line([-corner_x, -corner_x], [y(baseline_y), y(corner_y)])
        _line([corner_x, corner_x], [y(baseline_y), y(corner_y)])

        # 3-pt arc
        t1, t2 = (202, 338) if y_flip else (22, 158)
        three_arc = arc_points(0, y(0), arc_radius, t1, t2, n=150)
        _line(three_arc[0], three_arc[1])

    add_half(y_flip=False)
    if full_court:
        add_half(y_flip=True)
        _line([-250, 250], [halfcourt_y, halfcourt_y])
        center = arc_points(0, halfcourt_y, 60, 0, 360, n=120)
        _line(center[0], center[1])
        _line([-250, 250, 250, -250, -250], [baseline_y, baseline_y, 888, 888, baseline_y])
    else:
        _line([-250, 250, 250, -250, -250],
              [baseline_y, baseline_y, halfcourt_y, halfcourt_y, baseline_y])


# =============================================================================
# Shot sampling + scatter
# =============================================================================

def sample_shots_with_far(
    shots: pd.DataFrame, max_points: int, far_threshold: int
) -> pd.DataFrame:
    shots = shots.copy()
    shots["LOC_X"] = pd.to_numeric(shots["LOC_X"], errors="coerce")
    shots["LOC_Y"] = pd.to_numeric(shots["LOC_Y"], errors="coerce")
    shots = shots.dropna(subset=["LOC_X", "LOC_Y", "SHOT_MADE_FLAG"])
    shots["shot_distance_ft"] = ((shots["LOC_X"] ** 2 + shots["LOC_Y"] ** 2) ** 0.5) / 10.0
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


def plot_scatter_plotly(
    shots: pd.DataFrame,
    court_view: str,
) -> go.Figure:
    shots = shots.copy()
    shots["result"] = shots["SHOT_MADE_FLAG"].map({1: "Make", 0: "Miss"}).fillna("Miss")
    fig = px.scatter(
        shots,
        x="LOC_X",
        y="LOC_Y",
        color="result",
        color_discrete_map={"Make": ACCENT, "Miss": "#FF6B6B"},
        category_orders={"result": ["Miss", "Make"]},
        custom_data=["shot_id"],
        opacity=0.9,
        width=PLOT_WIDTH,
        height=520 if court_view == "Full court" else 460,
    )
    fig.update_traces(
        marker=dict(size=9),
        selected=dict(marker=dict(opacity=0.98, size=9)),
        unselected=dict(marker=dict(opacity=0.98)),
        selectedpoints=[],
        selector=dict(mode="markers"),
    )
    add_court_traces(fig, full_court=court_view == "Full court", line_color=COURT_LINE)
    fig.update_xaxes(range=[-250, 250], showgrid=False, zeroline=False, showticklabels=False)
    y_range = [-52, 888] if court_view == "Full court" else [-52, 418]
    fig.update_yaxes(
        range=y_range, showgrid=False, zeroline=False,
        showticklabels=False, scaleanchor="x", scaleratio=1,
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
