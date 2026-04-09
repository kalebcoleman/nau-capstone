"""Build matched NBA/NHL comparison outputs for the 2014-2024 story."""

from __future__ import annotations

import csv
import gzip
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

NBA_MIN_ATTEMPTS = 150
NHL_MIN_ATTEMPTS = 150
BOOTSTRAP_SAMPLES = 8

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.nba.gam_analysis import (
    add_shot_type_features as add_nba_gam_shot_type_features,
    fit_gam as fit_nba_full_gam,
)
from analysis.gam_explorer_config import GAM_EXPLORER_FIGURES, WINDOW_LABEL
from analysis.nhl.modeling import (
    NHL_EXPORT_PATH,
    NHL_RAW_PATH,
    add_shot_type_features,
    build_feature_matrix,
    build_full_model_effect_frame,
    export_nhl_historical,
    fit_expected_goal_gam,
    load_nhl_modeling_sample,
    score_expected_goal_rate,
)

DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
NHL_DATA_DIR = SCRIPT_DIR.parent / "nhl" / "data"
NHL_FIGURES_DIR = SCRIPT_DIR.parent / "nhl" / "figures"
SPATIAL_SPORTS_DB = Path(
    os.environ.get(
        "SPATIAL_SPORTS_DB",
        REPO_ROOT.parent / "spatialSportsR" / "data" / "parsed" / "nba.sqlite",
    )
)

NBA_EXPORT_PATH = DATA_DIR / "nba_shots_2014_2024.csv.gz"
NBA_SUMMARY_PATH = DATA_DIR / "nba_player_summary_2014_2024.csv"
NHL_SUMMARY_PATH = NHL_DATA_DIR / "nhl_player_summary_2014_2024.csv"
NBA_POSITION_PATH = DATA_DIR / "nba_position_summary_2014_2024.csv"
NHL_POSITION_PATH = NHL_DATA_DIR / "nhl_position_summary_2014_2024.csv"

NBA_SDI_FIGURE = FIGURES_DIR / "nba_sdi_vs_actual_2014_2024.png"
NHL_SDI_FIGURE = NHL_FIGURES_DIR / "nhl_sdi_vs_actual_2014_2024.png"
NBA_POSITION_FIGURE = FIGURES_DIR / "nba_sdi_by_position_2014_2024.png"
NHL_POSITION_FIGURE = NHL_FIGURES_DIR / "nhl_sdi_by_position_2014_2024.png"

DISTANCE_PLOT_MAX = {
    "NBA": 60.0,
    "NHL": 100.0,
}

POSITION_COLORS = {
    "NBA": {"G": "#2A6F97", "F": "#D17A22", "C": "#3F8F5F"},
    "NHL": {"C": "#2A6F97", "W": "#D17A22", "D": "#3F8F5F"},
}

EXPLORER_FIGURES = {
    (str(fig["sport"]), str(fig["factor_key"]), str(fig["plot_type"])): fig
    for fig in GAM_EXPLORER_FIGURES
}
CONTINUOUS_FACTOR_KEYS = ("distance", "angle", "clock", "period")
DISCRETE_FACTOR_KEYS = ("shot_type", "clutch", "rebound", "rush", "goalie_froze", "empty_net")
EFFECT_Y_LABEL = "Marginal log-odds contribution"
LINE_COLOR = "#2A6F97"
COURT_LANDMARKS = {
    sport: [
        (str(marker["label"]), float(marker["value"]), str(marker["color"]))
        for marker in EXPLORER_FIGURES[(sport, "distance", "continuous_pdp")]["markers"]
    ]
    for sport in ("NBA", "NHL")
}

NBA_CONTINUOUS_SPECS = {
    "distance": {
        "feature_col": "shot_distance_feet",
        "term": 1,
        "x_label": "Shot Distance (feet)",
        "x_max": DISTANCE_PLOT_MAX["NBA"],
        "x_min": 0.0,
    },
    "angle": {
        "feature_col": "shot_angle",
        "term": 2,
        "x_label": "Shot Angle (radians)",
        "x_min": -1.6,
        "x_max": 1.6,
    },
    "clock": {
        "feature_col": "seconds_in_period",
        "term": 3,
        "x_label": "Seconds Remaining in Period",
        "x_min": 720.0,
        "x_max": 0.0,
    },
    "period": {
        "feature_col": "PERIOD",
        "term": 4,
        "x_label": "Period",
        "x_min": 1.0,
        "x_max": 6.0,
    },
}

NHL_CONTINUOUS_SPECS = {
    "distance": {
        "feature_col": "shotDistance",
        "term": 1,
        "x_label": "Shot Distance (feet)",
        "x_min": 0.0,
        "x_max": DISTANCE_PLOT_MAX["NHL"],
    },
    "angle": {
        "feature_col": "shotAngle",
        "term": 2,
        "x_label": "Shot Angle (degrees)",
        "x_min": -100.0,
        "x_max": 100.0,
    },
    "clock": {
        "feature_col": "period_seconds_remaining",
        "term": 3,
        "x_label": "Seconds Remaining in Period",
        "x_min": 1200.0,
        "x_max": 0.0,
    },
    "period": {
        "feature_col": "period",
        "term": 4,
        "x_label": "Period",
        "x_min": 1.0,
        "x_max": 6.0,
    },
}

NBA_SPLINE_DISTANCE_SPEC = {
    "distance_col": "shot_distance_feet",
    "y_col": "SHOT_MADE_FLAG",
    "numeric_controls": [
        "shot_angle",
        "seconds_in_period",
        "PERIOD",
        "is_clutch",
        "is_jump_shot",
        "is_dunk",
        "is_layup",
    ],
    "categorical_controls": [],
    "x_label": "Shot Distance (feet)",
}

NHL_SPLINE_DISTANCE_SPEC = {
    "distance_col": "shotDistance",
    "y_col": "goal",
    "numeric_controls": [
        "shotAngle",
        "period_seconds_remaining",
        "period",
        "shotRebound",
        "shotGoalieFroze",
        "shotRush",
        "shotOnEmptyNet",
    ],
    "categorical_controls": [],
    "x_label": "Shot Distance (feet)",
}

NBA_DISCRETE_SPECS = {
    "clutch": {
        "x_label": "State",
        "categories": [("Not Clutch", "is_clutch", 0), ("Clutch", "is_clutch", 1)],
    },
    "shot_type": {
        "x_label": "Shot Type",
        "categories": [
            ("Other", None, 0),
            ("Dunk", "is_dunk", 1),
            ("Layup", "is_layup", 1),
            ("Hook", "is_hook", 1),
            ("Floater", "is_floater", 1),
            ("2PT Jump", "is_jump_shot_2", 1),
            ("3PT Jump", "is_jump_shot_3", 1),
        ],
    },
}

NHL_DISCRETE_SPECS = {
    "rebound": {
        "x_label": "State",
        "categories": [("No", "shotRebound", 0), ("Yes", "shotRebound", 1)],
    },
    "rush": {
        "x_label": "State",
        "categories": [("No", "shotRush", 0), ("Yes", "shotRush", 1)],
    },
    "goalie_froze": {
        "x_label": "State",
        "categories": [("No", "shotGoalieFroze", 0), ("Yes", "shotGoalieFroze", 1)],
    },
    "empty_net": {
        "x_label": "State",
        "categories": [("No", "shotOnEmptyNet", 0), ("Yes", "shotOnEmptyNet", 1)],
    },
    "shot_type": {
        "x_label": "Shot Type",
        "categories": [
            ("Other", None, 0),
            ("Wrist", "is_wrist_shot", 1),
            ("Snap", "is_snap_shot", 1),
            ("Slap", "is_slap_shot", 1),
            ("Backhand", "is_backhand", 1),
        ],
    },
}

@dataclass
class RunningPlayerTotals:
    attempts: int = 0
    actual_sum: float = 0.0
    expected_sum: float = 0.0
    sdi_sum: float = 0.0


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    NHL_DATA_DIR.mkdir(exist_ok=True)
    NHL_FIGURES_DIR.mkdir(exist_ok=True)


def plot_nhl_distance_effect(gam_df, output_path: Path) -> None:
    """Keep NHL distance plots visually aligned with the NBA comparison figure."""
    plot_gam_distance(gam_df, "NHL", output_path)


def season_start_in_window(season: str) -> bool:
    try:
        start_year = int(str(season).split("-")[0])
    except (TypeError, ValueError, IndexError):
        return False
    return 2014 <= start_year <= 2023


def export_nba_historical() -> None:
    """Export regular-season NBA shots for the 2014-15 through 2023-24 window."""
    if NBA_EXPORT_PATH.exists():
        print(f"Using existing NBA export: {NBA_EXPORT_PATH}")
        return
    if not SPATIAL_SPORTS_DB.exists():
        raise FileNotFoundError(f"Missing spatialSportsR SQLite database: {SPATIAL_SPORTS_DB}")

    print(f"Exporting historical NBA shots to {NBA_EXPORT_PATH} ...")
    query = """
        SELECT
            GRID_TYPE, GAME_ID, GAME_EVENT_ID, PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_NAME,
            PERIOD, MINUTES_REMAINING, SECONDS_REMAINING, EVENT_TYPE, ACTION_TYPE, SHOT_TYPE,
            SHOT_ZONE_BASIC, SHOT_ZONE_AREA, SHOT_ZONE_RANGE, SHOT_DISTANCE, LOC_X, LOC_Y,
            SHOT_ATTEMPTED_FLAG, SHOT_MADE_FLAG, GAME_DATE, HTM, VTM, season, season_type,
            league, source, event_num, shot_id
        FROM nba_stats_shots
        WHERE season_type = 'regular'
          AND CAST(substr(season, 1, 4) AS INTEGER) BETWEEN 2014 AND 2023
    """

    with sqlite3.connect(SPATIAL_SPORTS_DB) as con:
        chunk_idx = 0
        for chunk in pd.read_sql_query(query, con, chunksize=100_000):
            chunk_idx += 1
            chunk.to_csv(
                NBA_EXPORT_PATH,
                mode="a",
                index=False,
                header=chunk_idx == 1,
                compression="gzip",
            )
            if chunk_idx % 5 == 0:
                print(f"  wrote NBA export chunk {chunk_idx}")


def engineer_nba_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "LOC_X",
        "LOC_Y",
        "SHOT_DISTANCE",
        "PERIOD",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
        "SHOT_MADE_FLAG",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out["SHOT_ATTEMPTED_FLAG"].fillna(1).astype(int) == 1].copy()
    out.dropna(
        subset=["LOC_X", "LOC_Y", "SHOT_DISTANCE", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "SHOT_MADE_FLAG"],
        inplace=True,
    )

    out["shot_distance_feet"] = out["SHOT_DISTANCE"].fillna(
        np.sqrt(out["LOC_X"] ** 2 + out["LOC_Y"] ** 2) / 10.0
    )
    out["shot_angle"] = np.arctan2(out["LOC_X"], out["LOC_Y"].clip(lower=1))
    out["seconds_in_period"] = out["MINUTES_REMAINING"] * 60 + out["SECONDS_REMAINING"]
    out["is_clutch"] = ((out["PERIOD"] >= 4) & (out["seconds_in_period"] <= 120)).astype(int)

    action = out["ACTION_TYPE"].astype(str).str.lower()
    out["is_layup"] = action.str.contains("layup|finger roll").astype(int)
    out["is_dunk"] = action.str.contains("dunk").astype(int)
    out["is_jump_shot"] = action.str.contains("jump shot|pullup|step back|fadeaway").astype(int)
    out["is_hook"] = action.str.contains("hook").astype(int)
    out["is_floater"] = action.str.contains("float").astype(int)

    out["shot_type_family"] = np.select(
        [
            out["is_dunk"] == 1,
            out["is_layup"] == 1,
            out["is_hook"] == 1,
            out["is_floater"] == 1,
            out["is_jump_shot"] == 1,
        ],
        ["dunk", "layup", "hook", "floater", "jump_shot"],
        default="other",
    )

    zone_difficulty = {
        "Restricted Area": 0.1,
        "In The Paint (Non-RA)": 0.4,
        "Mid-Range": 0.7,
        "Left Corner 3": 0.5,
        "Right Corner 3": 0.5,
        "Above the Break 3": 0.6,
        "Backcourt": 0.95,
    }
    out["sdi_distance"] = out["shot_distance_feet"].clip(0, 35) / 35.0
    out["sdi_clock"] = 1 - (out["seconds_in_period"].clip(0, 720) / 720.0)
    out["sdi_zone"] = out["SHOT_ZONE_BASIC"].map(zone_difficulty).fillna(0.5)
    out["sdi_angle"] = np.abs(out["shot_angle"]) / (np.pi / 2)
    out["sdi_type"] = 0.3
    out.loc[action.str.contains("pullup|step back|fadeaway|turnaround"), "sdi_type"] = 0.8
    out.loc[action.str.contains("driving|running"), "sdi_type"] = 0.6
    out.loc[action.str.contains("dunk"), "sdi_type"] = 0.1
    out.loc[action.str.contains("layup") & ~action.str.contains("driving"), "sdi_type"] = 0.2
    out["SDI"] = (
        0.30 * out["sdi_distance"]
        + 0.20 * out["sdi_clock"]
        + 0.20 * out["sdi_type"]
        + 0.15 * out["sdi_zone"]
        + 0.15 * out["sdi_angle"]
    )
    return out


def build_nba_expected_model(sample_df: pd.DataFrame) -> Pipeline:
    numeric = [
        "shot_distance_feet",
        "shot_angle",
        "PERIOD",
        "seconds_in_period",
        "is_clutch",
        "is_layup",
        "is_dunk",
        "is_jump_shot",
        "is_hook",
        "is_floater",
    ]
    categorical = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "shot_type_family"]
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    model.fit(sample_df[numeric + categorical], sample_df["SHOT_MADE_FLAG"].astype(int))
    return model


def load_nba_position_map() -> pd.DataFrame:
    query = """
        SELECT
            CAST(player_id AS TEXT) AS player_id,
            TRIM(COALESCE(firstName, '') || ' ' || COALESCE(familyName, '')) AS player,
            position,
            COUNT(*) AS n_games
        FROM nba_stats_player_box_traditional
        WHERE season_type = 'regular'
          AND CAST(substr(season, 1, 4) AS INTEGER) BETWEEN 2014 AND 2023
          AND position IN ('G', 'F', 'C')
        GROUP BY 1, 2, 3
    """
    with sqlite3.connect(SPATIAL_SPORTS_DB) as con:
        df = pd.read_sql_query(query, con)
    df = df.sort_values(["player_id", "n_games"], ascending=[True, False])
    df = df.drop_duplicates(subset=["player_id"], keep="first")
    return df.rename(columns={"position": "position_group"})[
        ["player_id", "player", "position_group"]
    ]


def load_nhl_position_map() -> pd.DataFrame:
    counts: dict[tuple[str, str], int] = {}
    for chunk in pd.read_csv(
        NHL_RAW_PATH,
        usecols=["shooterName", "playerPositionThatDidEvent", "season"],
        chunksize=200_000,
    ):
        chunk = chunk[pd.to_numeric(chunk["season"], errors="coerce").between(2014, 2024)].copy()
        if chunk.empty:
            continue
        chunk["playerPositionThatDidEvent"] = chunk["playerPositionThatDidEvent"].replace(
            {"L": "W", "R": "W"}
        )
        chunk = chunk[chunk["playerPositionThatDidEvent"].isin(["C", "W", "D"])]
        grouped = chunk.groupby(["shooterName", "playerPositionThatDidEvent"]).size()
        for (player, pos), n_rows in grouped.items():
            key = (str(player), str(pos))
            counts[key] = counts.get(key, 0) + int(n_rows)
    rows = [
        {"player": player, "position_group": pos, "n_events": n_events}
        for (player, pos), n_events in counts.items()
    ]
    df = pd.DataFrame(rows).sort_values(["player", "n_events"], ascending=[True, False])
    return df.drop_duplicates(subset=["player"], keep="first")[["player", "position_group"]]


def sample_nba_for_models(sample_size: int | None = None) -> pd.DataFrame:
    print("Loading NBA rows for model fitting ...")
    df = pd.read_csv(NBA_EXPORT_PATH)
    df = df[df["season"].map(season_start_in_window)].copy()
    df = engineer_nba_features(df)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


NBA_FULL_GAM_FEATURE_COLS = [
    "LOC_X",
    "LOC_Y",
    "shot_distance_feet",
    "shot_angle",
    "seconds_in_period",
    "PERIOD",
    "is_dunk",
    "is_layup",
    "is_hook",
    "is_floater",
    "is_jump_shot_2",
    "is_jump_shot_3",
    "is_clutch",
]


def build_nba_full_model_effect_frame(
    gam,
    reference_df: pd.DataFrame,
    *,
    feature_col: str = "shot_distance_feet",
    term: int = 1,
    plot_min: float = 0.0,
    plot_max: float | None = None,
    n_points: int = 200,
) -> pd.DataFrame:
    """Build a one-dimensional effect frame from the full NBA PyGAM model."""
    base = {
        "LOC_X": float(pd.to_numeric(reference_df["LOC_X"], errors="coerce").median()),
        "LOC_Y": float(pd.to_numeric(reference_df["LOC_Y"], errors="coerce").median()),
        "shot_distance_feet": float(
            pd.to_numeric(reference_df["shot_distance_feet"], errors="coerce").median()
        ),
        "shot_angle": float(
            pd.to_numeric(reference_df["shot_angle"], errors="coerce").median()
        ),
        "seconds_in_period": float(
            pd.to_numeric(reference_df["seconds_in_period"], errors="coerce").median()
        ),
        "PERIOD": float(pd.to_numeric(reference_df["PERIOD"], errors="coerce").median()),
        "is_dunk": int(reference_df["is_dunk"].mode().iloc[0]),
        "is_layup": int(reference_df["is_layup"].mode().iloc[0]),
        "is_hook": int(reference_df["is_hook"].mode().iloc[0]),
        "is_floater": int(reference_df["is_floater"].mode().iloc[0]),
        "is_jump_shot_2": int(reference_df["is_jump_shot_2"].mode().iloc[0]),
        "is_jump_shot_3": int(reference_df["is_jump_shot_3"].mode().iloc[0]),
        "is_clutch": int(reference_df["is_clutch"].mode().iloc[0]),
    }
    series = pd.to_numeric(reference_df[feature_col], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No valid NBA values found for {feature_col}.")
    if plot_max is None:
        plot_max = float(series.max())
    baseline_value = float(series.median())

    value_grid = np.linspace(float(plot_min), float(plot_max), n_points)
    plot_df = pd.DataFrame({"shot_distance_feet": np.repeat(base["shot_distance_feet"], n_points)})
    for col, value in base.items():
        plot_df[col] = value
    plot_df[feature_col] = value_grid

    X_grid = plot_df[NBA_FULL_GAM_FEATURE_COLS].to_numpy(dtype=float)
    effect = gam.partial_dependence(term=term, X=X_grid)
    conf = gam.partial_dependence(term=term, X=X_grid, width=0.95)[1]

    baseline_df = plot_df.iloc[[0]].copy()
    baseline_df[feature_col] = baseline_value
    baseline_effect = float(
        gam.partial_dependence(
            term=term, X=baseline_df[NBA_FULL_GAM_FEATURE_COLS].to_numpy(dtype=float)
        )[0]
    )

    return pd.DataFrame(
        {
            "x_value": value_grid,
            "fitted_effect": effect - baseline_effect,
            "lower_ci": conf[:, 0] - baseline_effect,
            "upper_ci": conf[:, 1] - baseline_effect,
            "sport": "NBA",
            "effect_label": feature_col,
            "season_window": WINDOW_LABEL,
            "baseline_value": baseline_value,
        }
    )


def finalize_effect_frame(
    frame: pd.DataFrame,
    *,
    sport: str,
    factor_key: str,
    plot_type: str,
    target_label: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["sport"] = sport
    out["model_family"] = "LogisticGAM"
    out["target_label"] = target_label
    out["factor_key"] = factor_key
    out["plot_type"] = plot_type
    out["season_window"] = WINDOW_LABEL
    if "baseline_value" not in out.columns:
        out["baseline_value"] = np.nan
    ordered_cols = [
        "sport",
        "model_family",
        "target_label",
        "factor_key",
        "plot_type",
        "x_value",
        "fitted_effect",
        "lower_ci",
        "upper_ci",
        "baseline_value",
        "season_window",
    ]
    extra_cols = [col for col in out.columns if col not in ordered_cols]
    return out[ordered_cols + extra_cols]


def add_marker_lines(ax, markers: list[dict[str, object]], max_x: float | None = None) -> None:
    for marker in markers:
        value = float(marker["value"])
        if max_x is not None and value > max_x:
            continue
        ax.axvline(
            value,
            color=str(marker["color"]),
            linestyle=str(marker.get("linestyle", ":")),
            linewidth=2,
            alpha=0.95,
            label=str(marker["label"]),
        )


def plot_continuous_effect(effect_df: pd.DataFrame, spec: dict[str, object], x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.plot(effect_df["x_value"], effect_df["fitted_effect"], color=LINE_COLOR, linewidth=2.5, label="Effect")
    ax.fill_between(
        effect_df["x_value"],
        effect_df["lower_ci"],
        effect_df["upper_ci"],
        color=LINE_COLOR,
        alpha=0.2,
        label="95% CI",
    )
    ax.axhline(0, color="#7A7A7A", linestyle="--", alpha=0.5, label="Baseline")
    add_marker_lines(ax, list(spec["markers"]), max_x=float(np.nanmax(effect_df["x_value"])))
    baseline_value = float(effect_df["baseline_value"].iloc[0])
    if np.isfinite(baseline_value):
        ax.axvline(
            baseline_value,
            color="#4CAF50",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="Median Baseline",
        )
    ax.set_title(str(spec["title"]), fontsize=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(EFFECT_Y_LABEL, fontsize=12)
    if str(spec["factor_key"]) == "period":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(spec["figure_path"], dpi=220, bbox_inches="tight")
    plt.close()


def plot_discrete_summary(effect_df: pd.DataFrame, spec: dict[str, object], x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    labels = effect_df["level_label"].tolist()
    values = effect_df["fitted_effect"].to_numpy(dtype=float)
    colors = ["#2E8B57" if value >= 0 else "#8B1E3F" for value in values]
    bars = ax.bar(labels, values, color=colors, alpha=0.78, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="#7A7A7A", linestyle="--", alpha=0.5)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )
    ax.set_title(str(spec["title"]), fontsize=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(EFFECT_Y_LABEL, fontsize=12)
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(spec["figure_path"], dpi=220, bbox_inches="tight")
    plt.close()


def plot_spatial_surface(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    effect: np.ndarray,
    *,
    sport: str,
    spec: dict[str, object],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    lim = np.nanpercentile(np.abs(effect), 98)
    mesh = ax.pcolormesh(
        grid_x,
        grid_y,
        effect,
        cmap="coolwarm",
        shading="auto",
        alpha=0.9,
        vmin=-lim,
        vmax=lim,
    )
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(EFFECT_Y_LABEL, fontsize=10)
    ax.set_title(str(spec["title"]), fontsize=15)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_aspect("equal")
    if sport == "NBA":
        ax.set_xlim(-250, 250)
        ax.set_ylim(-50, 420)
    else:
        ax.set_xlim(-100, 100)
        ax.set_ylim(-10, 100)
    ax.grid(alpha=0.12)
    plt.tight_layout()
    plt.savefig(spec["figure_path"], dpi=220, bbox_inches="tight")
    plt.close()


def build_nba_discrete_effect_frame(
    gam,
    reference_df: pd.DataFrame,
    *,
    factor_key: str,
) -> pd.DataFrame:
    base = {
        "LOC_X": float(pd.to_numeric(reference_df["LOC_X"], errors="coerce").median()),
        "LOC_Y": float(pd.to_numeric(reference_df["LOC_Y"], errors="coerce").median()),
        "shot_distance_feet": float(pd.to_numeric(reference_df["shot_distance_feet"], errors="coerce").median()),
        "shot_angle": float(pd.to_numeric(reference_df["shot_angle"], errors="coerce").median()),
        "seconds_in_period": float(pd.to_numeric(reference_df["seconds_in_period"], errors="coerce").median()),
        "PERIOD": float(pd.to_numeric(reference_df["PERIOD"], errors="coerce").median()),
        "is_dunk": 0,
        "is_layup": 0,
        "is_hook": 0,
        "is_floater": 0,
        "is_jump_shot_2": 0,
        "is_jump_shot_3": 0,
        "is_clutch": 0,
    }
    rows = []
    for order, (label, field, value) in enumerate(NBA_DISCRETE_SPECS[factor_key]["categories"]):
        row = base.copy()
        if field is not None:
            row[field] = value
        X = pd.DataFrame([row])[NBA_FULL_GAM_FEATURE_COLS].to_numpy(dtype=float)
        pred = float(logit(np.array([gam.predict_mu(X)[0]], dtype=float))[0])
        rows.append(
            {
                "x_value": float(order),
                "fitted_effect": pred,
                "lower_ci": np.nan,
                "upper_ci": np.nan,
                "baseline_value": np.nan,
                "level_key": field or "baseline",
                "level_label": label,
            }
        )
    out = pd.DataFrame(rows)
    out["fitted_effect"] = out["fitted_effect"] - float(out["fitted_effect"].iloc[0])
    return out


def build_nhl_discrete_effect_frame(
    gam,
    reference_df: pd.DataFrame,
    *,
    factor_key: str,
) -> pd.DataFrame:
    base = {
        "xCord": float(pd.to_numeric(reference_df["xCord"], errors="coerce").median()),
        "yCord": float(pd.to_numeric(reference_df["yCord"], errors="coerce").median()),
        "shotDistance": float(pd.to_numeric(reference_df["shotDistance"], errors="coerce").median()),
        "shotAngle": float(pd.to_numeric(reference_df["shotAngle"], errors="coerce").median()),
        "period_seconds_remaining": float(
            pd.to_numeric(reference_df["period_seconds_remaining"], errors="coerce").median()
        ),
        "period": float(pd.to_numeric(reference_df["period"], errors="coerce").median()),
        "shotRebound": 0,
        "shotGoalieFroze": 0,
        "shotRush": 0,
        "shotOnEmptyNet": 0,
        "is_wrist_shot": 0,
        "is_snap_shot": 0,
        "is_slap_shot": 0,
        "is_backhand": 0,
    }
    rows = []
    for order, (label, field, value) in enumerate(NHL_DISCRETE_SPECS[factor_key]["categories"]):
        row = base.copy()
        if field is not None:
            row[field] = value
        X = build_feature_matrix(pd.DataFrame([row]))
        pred = float(logit(np.array([gam.predict_mu(X)[0]], dtype=float))[0])
        rows.append(
            {
                "x_value": float(order),
                "fitted_effect": pred,
                "lower_ci": np.nan,
                "upper_ci": np.nan,
                "baseline_value": np.nan,
                "level_key": field or "baseline",
                "level_label": label,
            }
        )
    out = pd.DataFrame(rows)
    out["fitted_effect"] = out["fitted_effect"] - float(out["fitted_effect"].iloc[0])
    return out


def build_spatial_effect_frame(
    gam,
    reference_df: pd.DataFrame,
    *,
    sport: str,
    grid_size: int = 70,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if sport == "NBA":
        x_range = np.linspace(-250, 250, grid_size)
        y_range = np.linspace(-50, 420, grid_size)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_df = pd.DataFrame(
            {
                "LOC_X": grid_x.ravel(),
                "LOC_Y": grid_y.ravel(),
                "shot_distance_feet": np.sqrt(grid_x.ravel() ** 2 + grid_y.ravel() ** 2) / 10.0,
                "shot_angle": np.arctan2(grid_x.ravel(), np.clip(grid_y.ravel(), 1, None)),
                "seconds_in_period": float(
                    pd.to_numeric(reference_df["seconds_in_period"], errors="coerce").median()
                ),
                "PERIOD": float(pd.to_numeric(reference_df["PERIOD"], errors="coerce").median()),
                "is_dunk": int(reference_df["is_dunk"].mode().iloc[0]),
                "is_layup": int(reference_df["is_layup"].mode().iloc[0]),
                "is_hook": int(reference_df["is_hook"].mode().iloc[0]),
                "is_floater": int(reference_df["is_floater"].mode().iloc[0]),
                "is_jump_shot_2": int(reference_df["is_jump_shot_2"].mode().iloc[0]),
                "is_jump_shot_3": int(reference_df["is_jump_shot_3"].mode().iloc[0]),
                "is_clutch": int(reference_df["is_clutch"].mode().iloc[0]),
            }
        )
        X = grid_df[NBA_FULL_GAM_FEATURE_COLS].to_numpy(dtype=float)
    else:
        x_range = np.linspace(-100, 100, grid_size)
        y_range = np.linspace(-10, 100, grid_size)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_df = pd.DataFrame(
            {
                "xCord": grid_x.ravel(),
                "yCord": grid_y.ravel(),
                "shotDistance": np.sqrt(grid_x.ravel() ** 2 + grid_y.ravel() ** 2),
                "shotAngle": np.degrees(np.arctan2(np.abs(grid_x.ravel()), np.clip(grid_y.ravel(), 1, None))),
                "period_seconds_remaining": float(
                    pd.to_numeric(reference_df["period_seconds_remaining"], errors="coerce").median()
                ),
                "period": float(pd.to_numeric(reference_df["period"], errors="coerce").median()),
                "shotRebound": int(reference_df["shotRebound"].mode().iloc[0]),
                "shotGoalieFroze": int(reference_df["shotGoalieFroze"].mode().iloc[0]),
                "shotRush": int(reference_df["shotRush"].mode().iloc[0]),
                "shotOnEmptyNet": int(reference_df["shotOnEmptyNet"].mode().iloc[0]),
                "is_wrist_shot": int(reference_df["is_wrist_shot"].mode().iloc[0]),
                "is_snap_shot": int(reference_df["is_snap_shot"].mode().iloc[0]),
                "is_slap_shot": int(reference_df["is_slap_shot"].mode().iloc[0]),
                "is_backhand": int(reference_df["is_backhand"].mode().iloc[0]),
            }
        )
        X = build_feature_matrix(grid_df)
    effect = gam.partial_dependence(term=0, X=X).reshape(grid_x.shape)
    frame = pd.DataFrame(
        {
            "sport": sport,
            "model_family": "LogisticGAM",
            "target_label": EXPLORER_FIGURES[(sport, "spatial", "spatial_surface")]["target_label"],
            "factor_key": "spatial",
            "plot_type": "spatial_surface",
            "x_value": grid_x.ravel(),
            "y_value": grid_y.ravel(),
            "fitted_effect": effect.ravel(),
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "baseline_value": np.nan,
            "season_window": WINDOW_LABEL,
        }
    )
    return frame, grid_x, grid_y, effect


def save_continuous_artifact(effect_df: pd.DataFrame, spec: dict[str, object], x_label: str) -> None:
    Path(spec["data_path"]).parent.mkdir(parents=True, exist_ok=True)
    effect_df.to_csv(spec["data_path"], index=False)
    print(f"Saved: {spec['data_path']}")
    plot_continuous_effect(effect_df, spec, x_label)
    print(f"Saved: {spec['figure_path']}")


def save_discrete_artifact(effect_df: pd.DataFrame, spec: dict[str, object], x_label: str) -> None:
    Path(spec["data_path"]).parent.mkdir(parents=True, exist_ok=True)
    effect_df.to_csv(spec["data_path"], index=False)
    print(f"Saved: {spec['data_path']}")
    plot_discrete_summary(effect_df, spec, x_label)
    print(f"Saved: {spec['figure_path']}")


def save_spatial_artifact(
    effect_df: pd.DataFrame,
    spec: dict[str, object],
    *,
    sport: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    effect: np.ndarray,
) -> None:
    Path(spec["data_path"]).parent.mkdir(parents=True, exist_ok=True)
    effect_df.to_csv(spec["data_path"], index=False)
    print(f"Saved: {spec['data_path']}")
    plot_spatial_surface(grid_x, grid_y, effect, sport=sport, spec=spec)
    print(f"Saved: {spec['figure_path']}")


def build_distance_spline_model(
    df: pd.DataFrame,
    *,
    distance_col: str,
    y_col: str,
    numeric_controls: list[str],
    categorical_controls: list[str] | None = None,
) -> Pipeline:
    categorical_controls = categorical_controls or []
    preprocessor = ColumnTransformer(
        [
            (
                "distance_spline",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("spline", SplineTransformer(n_knots=8, degree=3, include_bias=False)),
                    ]
                ),
                [distance_col],
            ),
            ("numeric", StandardScaler(), numeric_controls),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_controls),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    feature_cols = [distance_col] + numeric_controls + categorical_controls
    model.fit(df[feature_cols], df[y_col].astype(int))
    return model


def bootstrap_distance_effect(
    df: pd.DataFrame,
    *,
    distance_col: str,
    y_col: str,
    numeric_controls: list[str],
    categorical_controls: list[str] | None,
    sport: str,
    distance_max: float | None = None,
) -> pd.DataFrame:
    categorical_controls = categorical_controls or []
    plot_max = distance_max if distance_max is not None else DISTANCE_PLOT_MAX.get(
        sport, float(df[distance_col].max())
    )
    distance_grid = np.linspace(0, plot_max, 200)
    base_row = {}
    for col in numeric_controls:
        base_row[col] = float(df[col].median())
    for col in categorical_controls:
        mode = df[col].mode(dropna=True)
        base_row[col] = mode.iloc[0] if not mode.empty else ""

    curves = []
    fit_df = df.copy()
    if len(fit_df) > 120_000:
        fit_df = fit_df.sample(n=120_000, random_state=42)
    print(f"Fitting {sport} spline-logistic distance model ...")
    fit_model = build_distance_spline_model(
        fit_df,
        distance_col=distance_col,
        y_col=y_col,
        numeric_controls=numeric_controls,
        categorical_controls=categorical_controls,
    )

    def predict_curve(model: Pipeline) -> np.ndarray:
        grid = pd.DataFrame({distance_col: distance_grid})
        for key, value in base_row.items():
            grid[key] = value
        probs = model.predict_proba(grid[[distance_col] + numeric_controls + categorical_controls])[:, 1]
        logits = logit(probs)
        center_idx = len(logits) // 2
        return logits - logits[center_idx]

    main_curve = predict_curve(fit_model)
    curves.append(main_curve)
    for seed in range(BOOTSTRAP_SAMPLES):
        print(f"  {sport} bootstrap {seed + 1}/{BOOTSTRAP_SAMPLES}")
        boot = fit_df.sample(n=len(fit_df), replace=True, random_state=seed)
        model = build_distance_spline_model(
            boot,
            distance_col=distance_col,
            y_col=y_col,
            numeric_controls=numeric_controls,
            categorical_controls=categorical_controls,
        )
        curves.append(predict_curve(model))

    curve_arr = np.vstack(curves)
    return pd.DataFrame(
        {
            "x_value": distance_grid,
            "fitted_effect": curve_arr[0],
            "lower_ci": np.quantile(curve_arr[1:], 0.025, axis=0),
            "upper_ci": np.quantile(curve_arr[1:], 0.975, axis=0),
            "sport": sport,
            "effect_label": "Shot Distance",
            "season_window": WINDOW_LABEL,
        }
    )


def build_nba_outputs() -> None:
    print("Building NBA comparison outputs ...")
    sample_df = sample_nba_for_models(sample_size=None)
    model = build_nba_expected_model(sample_df)
    print("Scoring full NBA export into player summary ...")
    gam_sample_df = add_nba_gam_shot_type_features(sample_df.copy())
    gam_sample_df = gam_sample_df.dropna(
        subset=NBA_FULL_GAM_FEATURE_COLS + ["SHOT_MADE_FLAG"]
    ).copy()
    nba_full_gam = fit_nba_full_gam(
        gam_sample_df[NBA_FULL_GAM_FEATURE_COLS].to_numpy(dtype=float),
        gam_sample_df["SHOT_MADE_FLAG"].astype(int).to_numpy(),
    )
    print("Generating centralized NBA explorer GAM artifacts...")
    for factor_key, settings in NBA_CONTINUOUS_SPECS.items():
        spec = EXPLORER_FIGURES[("NBA", factor_key, "continuous_pdp")]
        effect_df = build_nba_full_model_effect_frame(
            nba_full_gam,
            gam_sample_df,
            feature_col=str(settings["feature_col"]),
            term=int(settings["term"]),
            plot_min=float(settings["x_min"]),
            plot_max=float(settings["x_max"]),
        )
        effect_df = finalize_effect_frame(
            effect_df,
            sport="NBA",
            factor_key=factor_key,
            plot_type="continuous_pdp",
            target_label=str(spec["target_label"]),
        )
        save_continuous_artifact(effect_df, spec, x_label=str(settings["x_label"]))

    nba_spline_spec = EXPLORER_FIGURES[("NBA", "distance_spline", "continuous_pdp")]
    nba_spline_df = bootstrap_distance_effect(
        sample_df,
        distance_col=str(NBA_SPLINE_DISTANCE_SPEC["distance_col"]),
        y_col=str(NBA_SPLINE_DISTANCE_SPEC["y_col"]),
        numeric_controls=list(NBA_SPLINE_DISTANCE_SPEC["numeric_controls"]),
        categorical_controls=list(NBA_SPLINE_DISTANCE_SPEC["categorical_controls"]),
        sport="NBA",
        distance_max=DISTANCE_PLOT_MAX["NBA"],
    )
    nba_spline_df = finalize_effect_frame(
        nba_spline_df,
        sport="NBA",
        factor_key="distance_spline",
        plot_type="continuous_pdp",
        target_label=str(nba_spline_spec["target_label"]),
    )
    save_continuous_artifact(
        nba_spline_df,
        nba_spline_spec,
        x_label=str(NBA_SPLINE_DISTANCE_SPEC["x_label"]),
    )

    for factor_key, settings in NBA_DISCRETE_SPECS.items():
        spec = EXPLORER_FIGURES[("NBA", factor_key, "discrete_summary")]
        effect_df = build_nba_discrete_effect_frame(
            nba_full_gam,
            gam_sample_df,
            factor_key=factor_key,
        )
        effect_df = finalize_effect_frame(
            effect_df,
            sport="NBA",
            factor_key=factor_key,
            plot_type="discrete_summary",
            target_label=str(spec["target_label"]),
        )
        save_discrete_artifact(effect_df, spec, x_label=str(settings["x_label"]))

    spatial_spec = EXPLORER_FIGURES[("NBA", "spatial", "spatial_surface")]
    spatial_df, grid_x, grid_y, spatial_effect = build_spatial_effect_frame(
        nba_full_gam,
        gam_sample_df,
        sport="NBA",
    )
    save_spatial_artifact(
        spatial_df,
        spatial_spec,
        sport="NBA",
        grid_x=grid_x,
        grid_y=grid_y,
        effect=spatial_effect,
    )

    totals: dict[tuple[str, str], RunningPlayerTotals] = {}
    raw_df = pd.read_csv(NBA_EXPORT_PATH, chunksize=150_000)
    numeric_and_cat = [
        "shot_distance_feet",
        "shot_angle",
        "PERIOD",
        "seconds_in_period",
        "is_clutch",
        "is_layup",
        "is_dunk",
        "is_jump_shot",
        "is_hook",
        "is_floater",
        "SHOT_ZONE_BASIC",
        "SHOT_ZONE_AREA",
        "shot_type_family",
    ]
    for chunk_idx, chunk in enumerate(raw_df, start=1):
        chunk = chunk[chunk["season"].map(season_start_in_window)].copy()
        if chunk.empty:
            continue
        chunk = engineer_nba_features(chunk)
        if chunk.empty:
            continue
        chunk["expected_rate"] = model.predict_proba(chunk[numeric_and_cat])[:, 1]
        grouped = chunk.groupby(["PLAYER_ID", "PLAYER_NAME"]).agg(
            attempts=("SHOT_MADE_FLAG", "size"),
            actual_sum=("SHOT_MADE_FLAG", "sum"),
            expected_sum=("expected_rate", "sum"),
            sdi_sum=("SDI", "sum"),
        )
        for (player_id, player_name), row in grouped.iterrows():
            entry = totals.setdefault((str(player_id), str(player_name)), RunningPlayerTotals())
            entry.attempts += int(row["attempts"])
            entry.actual_sum += float(row["actual_sum"])
            entry.expected_sum += float(row["expected_sum"])
            entry.sdi_sum += float(row["sdi_sum"])
        if chunk_idx % 5 == 0:
            print(f"  processed NBA summary chunk {chunk_idx}")

    rows = []
    for (player_id, player_name), vals in totals.items():
        if vals.attempts < NBA_MIN_ATTEMPTS:
            continue
        actual_rate = vals.actual_sum / vals.attempts
        expected_rate = vals.expected_sum / vals.attempts
        rows.append(
            {
                "player_id": player_id,
                "player": player_name,
                "season_window": WINDOW_LABEL,
                "attempts": vals.attempts,
                "mean_sdi": vals.sdi_sum / vals.attempts,
                "actual_rate": actual_rate,
                "expected_rate": expected_rate,
                "residual": actual_rate - expected_rate,
                "sport": "NBA",
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(["attempts", "residual"], ascending=[False, False])
    summary_df = summary_df.merge(load_nba_position_map(), how="left", on=["player_id", "player"])
    summary_df["position_group"] = summary_df["position_group"].fillna("Unknown")
    summary_df.to_csv(NBA_SUMMARY_PATH, index=False)
    print(f"Saved: {NBA_SUMMARY_PATH}")
    summary_df.to_csv(NBA_POSITION_PATH, index=False)
    print(f"Saved: {NBA_POSITION_PATH}")

    plot_sdi_scatter(summary_df, "NBA", "Actual FG%", NBA_SDI_FIGURE)
    plot_position_sdi(summary_df, "NBA", "Actual FG%", NBA_POSITION_FIGURE)


def build_nhl_outputs() -> None:
    print("Building NHL comparison outputs ...")
    sample_df = load_nhl_modeling_sample(sample_size=None)
    model = fit_expected_goal_gam(sample_df)
    print("Generating centralized NHL explorer GAM artifacts...")
    for factor_key, settings in NHL_CONTINUOUS_SPECS.items():
        spec = EXPLORER_FIGURES[("NHL", factor_key, "continuous_pdp")]
        effect_df = build_full_model_effect_frame(
            model,
            sample_df,
            feature_col=str(settings["feature_col"]),
            term=int(settings["term"]),
            plot_min=float(settings["x_min"]),
            plot_max=float(settings["x_max"]),
        )
        effect_df = finalize_effect_frame(
            effect_df,
            sport="NHL",
            factor_key=factor_key,
            plot_type="continuous_pdp",
            target_label=str(spec["target_label"]),
        )
        save_continuous_artifact(effect_df, spec, x_label=str(settings["x_label"]))

    nhl_spline_spec = EXPLORER_FIGURES[("NHL", "distance_spline", "continuous_pdp")]
    nhl_spline_df = bootstrap_distance_effect(
        sample_df,
        distance_col=str(NHL_SPLINE_DISTANCE_SPEC["distance_col"]),
        y_col=str(NHL_SPLINE_DISTANCE_SPEC["y_col"]),
        numeric_controls=list(NHL_SPLINE_DISTANCE_SPEC["numeric_controls"]),
        categorical_controls=list(NHL_SPLINE_DISTANCE_SPEC["categorical_controls"]),
        sport="NHL",
        distance_max=DISTANCE_PLOT_MAX["NHL"],
    )
    nhl_spline_df = finalize_effect_frame(
        nhl_spline_df,
        sport="NHL",
        factor_key="distance_spline",
        plot_type="continuous_pdp",
        target_label=str(nhl_spline_spec["target_label"]),
    )
    save_continuous_artifact(
        nhl_spline_df,
        nhl_spline_spec,
        x_label=str(NHL_SPLINE_DISTANCE_SPEC["x_label"]),
    )

    for factor_key, settings in NHL_DISCRETE_SPECS.items():
        spec = EXPLORER_FIGURES[("NHL", factor_key, "discrete_summary")]
        effect_df = build_nhl_discrete_effect_frame(
            model,
            sample_df,
            factor_key=factor_key,
        )
        effect_df = finalize_effect_frame(
            effect_df,
            sport="NHL",
            factor_key=factor_key,
            plot_type="discrete_summary",
            target_label=str(spec["target_label"]),
        )
        save_discrete_artifact(effect_df, spec, x_label=str(settings["x_label"]))

    spatial_spec = EXPLORER_FIGURES[("NHL", "spatial", "spatial_surface")]
    spatial_df, grid_x, grid_y, spatial_effect = build_spatial_effect_frame(
        model,
        sample_df,
        sport="NHL",
    )
    save_spatial_artifact(
        spatial_df,
        spatial_spec,
        sport="NHL",
        grid_x=grid_x,
        grid_y=grid_y,
        effect=spatial_effect,
    )

    totals: dict[str, RunningPlayerTotals] = {}
    for chunk_idx, chunk in enumerate(pd.read_csv(NHL_EXPORT_PATH, chunksize=150_000), start=1):
        chunk = add_shot_type_features(chunk)
        chunk["expected_rate"] = score_expected_goal_rate(model, chunk)
        grouped = chunk.groupby("shooterName").agg(
            attempts=("goal", "size"),
            actual_sum=("goal", "sum"),
            expected_sum=("expected_rate", "sum"),
            sdi_sum=("SDI", "sum"),
        )
        for player, row in grouped.iterrows():
            entry = totals.setdefault(player, RunningPlayerTotals())
            entry.attempts += int(row["attempts"])
            entry.actual_sum += float(row["actual_sum"])
            entry.expected_sum += float(row["expected_sum"])
            entry.sdi_sum += float(row["sdi_sum"])
        if chunk_idx % 5 == 0:
            print(f"  processed NHL summary chunk {chunk_idx}")

    rows = []
    for player, vals in totals.items():
        if vals.attempts < NHL_MIN_ATTEMPTS:
            continue
        actual_rate = vals.actual_sum / vals.attempts
        expected_rate = vals.expected_sum / vals.attempts
        rows.append(
            {
                "player": player,
                "season_window": WINDOW_LABEL,
                "attempts": vals.attempts,
                "mean_sdi": vals.sdi_sum / vals.attempts,
                "actual_rate": actual_rate,
                "expected_rate": expected_rate,
                "residual": actual_rate - expected_rate,
                "sport": "NHL",
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(["attempts", "residual"], ascending=[False, False])
    summary_df = summary_df.merge(load_nhl_position_map(), how="left", on="player")
    summary_df["position_group"] = summary_df["position_group"].fillna("Unknown")
    summary_df.to_csv(NHL_SUMMARY_PATH, index=False)
    print(f"Saved: {NHL_SUMMARY_PATH}")
    summary_df.to_csv(NHL_POSITION_PATH, index=False)
    print(f"Saved: {NHL_POSITION_PATH}")

    plot_sdi_scatter(summary_df, "NHL", "Actual Goal %", NHL_SDI_FIGURE)
    plot_position_sdi(summary_df, "NHL", "Actual Goal %", NHL_POSITION_FIGURE)


def label_extremes(summary_df: pd.DataFrame) -> pd.DataFrame:
    attempts_cutoff = summary_df["attempts"].median()
    high_attempts = summary_df[summary_df["attempts"] >= attempts_cutoff]
    candidates = pd.concat(
        [
            summary_df.nlargest(3, "residual"),
            summary_df.nsmallest(3, "residual"),
            high_attempts.nlargest(2, "mean_sdi"),
            summary_df.nlargest(2, "attempts"),
        ],
        ignore_index=True,
    )
    return candidates.drop_duplicates(subset=["player"]).head(10)


def shorten_player_labels(players: list[str]) -> dict[str, str]:
    last_names: dict[str, list[str]] = {}
    for player in players:
        parts = str(player).split()
        last = parts[-1] if parts else str(player)
        last_names.setdefault(last, []).append(str(player))
    labels = {}
    for player in players:
        parts = str(player).split()
        last = parts[-1] if parts else str(player)
        labels[player] = last if len(last_names[last]) == 1 else str(player)
    return labels


def annotate_selected_players(
    ax,
    label_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    text_map: dict[str, str],
) -> None:
    offsets = [
        (10, 8),
        (-10, 8),
        (10, -8),
        (-10, -8),
        (14, 0),
        (-14, 0),
        (0, 12),
        (0, -12),
        (16, 10),
        (-16, 10),
    ]
    for idx, (_, row) in enumerate(label_df.iterrows()):
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(
            text_map.get(row["player"], row["player"]),
            xy=(row[x_col], row[y_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            alpha=0.95,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.72),
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6, alpha=0.6),
        )


def select_position_exemplars(summary_df: pd.DataFrame, sport: str) -> pd.DataFrame:
    rows = []
    for position in POSITION_COLORS[sport].keys():
        group = summary_df[summary_df["position_group"] == position].copy()
        if group.empty:
            continue
        residual_cutoff = group["residual"].median()
        preferred = group[group["residual"] >= residual_cutoff].sort_values(
            ["attempts", "residual"], ascending=[False, False]
        )
        chosen = preferred.head(1)
        if chosen.empty:
            chosen = group.sort_values("residual", ascending=False).head(1)
        rows.append(chosen)
    if not rows:
        return pd.DataFrame(columns=summary_df.columns)
    return pd.concat(rows, ignore_index=True)


def plot_sdi_scatter(summary_df: pd.DataFrame, sport: str, y_label: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    scatter = ax.scatter(
        summary_df["mean_sdi"],
        summary_df["actual_rate"] * 100,
        s=np.clip(summary_df["attempts"] / 7, 18, 260),
        c=summary_df["residual"],
        cmap="RdYlGn",
        vmin=-0.1,
        vmax=0.1,
        alpha=0.62,
        edgecolors="black",
        linewidths=0.35,
    )

    z = np.polyfit(summary_df["mean_sdi"], summary_df["actual_rate"] * 100, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(summary_df["mean_sdi"].to_numpy())
    ax.plot(x_sorted, p(x_sorted), linestyle="--", color="#2A6F97", linewidth=2)
    ax.axhline((summary_df["actual_rate"] * 100).median(), color="#9A9A9A", linestyle="--", alpha=0.25)
    ax.axvline(summary_df["mean_sdi"].median(), color="#9A9A9A", linestyle="--", alpha=0.25)

    labels_df = label_extremes(summary_df).copy()
    labels_df["actual_rate_pct"] = labels_df["actual_rate"] * 100
    label_map = shorten_player_labels(labels_df["player"].tolist())
    annotate_selected_players(
        ax,
        labels_df,
        x_col="mean_sdi",
        y_col="actual_rate_pct",
        text_map=label_map,
    )

    ax.set_title(f"{sport} Shot Difficulty vs Actual Scoring Rate ({WINDOW_LABEL})", fontsize=15)
    ax.text(
        0.01,
        0.98,
        "Size = volume, Color = actual - expected",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
    )
    ax.set_xlabel("Average Shot Difficulty Index (SDI)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Residual (Actual - Expected)", fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def add_group_ellipse(ax, x: pd.Series, y: pd.Series, color: str) -> None:
    if len(x) < 8:
        return
    cov = np.cov(x, y)
    if cov.shape != (2, 2) or np.linalg.det(cov) <= 0:
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals) * 1.8
    ellipse = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=width,
        height=height,
        angle=theta,
        facecolor=color,
        edgecolor=color,
        alpha=0.08,
        linewidth=2.2,
    )
    ax.add_patch(ellipse)


def plot_position_sdi(summary_df: pd.DataFrame, sport: str, y_label: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    palette = POSITION_COLORS[sport]
    df = summary_df[summary_df["position_group"].isin(palette.keys())].copy()
    for position, color in palette.items():
        group = df[df["position_group"] == position].copy()
        if group.empty:
            continue
        ax.scatter(
            group["mean_sdi"],
            group["actual_rate"] * 100,
            s=np.clip(group["attempts"] / 7, 16, 230),
            color=color,
            alpha=0.66,
            edgecolors="black",
            linewidths=0.35,
            label=position,
        )
        add_group_ellipse(ax, group["mean_sdi"], group["actual_rate"] * 100, color)
        centroid_x = float(group["mean_sdi"].mean())
        centroid_y = float((group["actual_rate"] * 100).mean())
        ax.annotate(
            position,
            (centroid_x, centroid_y),
            fontsize=10,
            weight="bold",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=color,
                alpha=0.85,
            ),
        )

    exemplar_df = select_position_exemplars(df, sport).copy()
    exemplar_df["actual_rate_pct"] = exemplar_df["actual_rate"] * 100
    exemplar_map = shorten_player_labels(exemplar_df["player"].tolist())
    annotate_selected_players(
        ax,
        exemplar_df,
        x_col="mean_sdi",
        y_col="actual_rate_pct",
        text_map=exemplar_map,
    )

    ax.set_title(f"{sport} SDI by Position Cluster ({WINDOW_LABEL})", fontsize=15)
    ax.set_xlabel("Average Shot Difficulty Index (SDI)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", title="Position")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_gam_distance(
    gam_df: pd.DataFrame,
    sport: str,
    output_path: Path,
    *,
    model_label: str = "GAM",
    x_max: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.plot(
        gam_df["x_value"],
        gam_df["fitted_effect"],
        color="#2A6F97",
        linewidth=2.5,
        label="Distance Effect",
    )
    ax.fill_between(
        gam_df["x_value"],
        gam_df["lower_ci"],
        gam_df["upper_ci"],
        color="#2A6F97",
        alpha=0.2,
        label="95% CI",
    )
    ax.axhline(0, color="#7A7A7A", linestyle="--", alpha=0.5, label="Baseline")
    for label, x_value, color in COURT_LANDMARKS.get(sport, []):
        ax.axvline(
            x_value,
            color=color,
            linestyle=":",
            linewidth=2,
            alpha=0.95,
            label=label,
        )
    baseline_col = None
    if "baseline_distance" in gam_df.columns:
        baseline_col = "baseline_distance"
    elif "baseline_value" in gam_df.columns:
        baseline_col = "baseline_value"
    if baseline_col is not None:
        ax.axvline(
            float(gam_df[baseline_col].iloc[0]),
            color="#4CAF50",
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            label="Median Distance",
        )
    ax.set_title(
        f"{sport} {model_label} Distance Effect with 95% CI ({WINDOW_LABEL})",
        fontsize=15,
    )
    ax.set_xlabel("Shot Distance (feet)", fontsize=12)
    ax.set_ylabel("Marginal log-odds contribution", fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    if x_max is not None:
        ax.set_xlim(0, x_max)
    elif sport == "NHL":
        ax.set_xlim(0, DISTANCE_PLOT_MAX["NHL"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_all_outputs() -> None:
    ensure_dirs()
    export_nba_historical()
    export_nhl_historical()
    build_nba_outputs()
    build_nhl_outputs()
    print("Cross-sport comparison outputs complete.")


if __name__ == "__main__":
    build_all_outputs()
