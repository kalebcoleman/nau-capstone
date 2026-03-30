"""Build matched NBA/NHL comparison outputs for the 2014-2024 story."""

from __future__ import annotations

import csv
import gzip
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

WINDOW_LABEL = "2014-2024"
NBA_MIN_ATTEMPTS = 150
NHL_MIN_ATTEMPTS = 150
NBA_SAMPLE_SIZE = 300_000
NHL_SAMPLE_SIZE = 300_000
BOOTSTRAP_SAMPLES = 8

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
NHL_DATA_DIR = SCRIPT_DIR.parent / "nhl" / "data"
NHL_APP_DATA_DIR = NHL_DATA_DIR / "app_data"
SPATIAL_SPORTS_DB = Path(
    os.environ.get(
        "SPATIAL_SPORTS_DB",
        REPO_ROOT.parent / "spatialSportsR" / "data" / "parsed" / "nba.sqlite",
    )
)

NBA_EXPORT_PATH = DATA_DIR / "nba_shots_2014_2024.csv.gz"
NHL_EXPORT_PATH = NHL_APP_DATA_DIR / "nhl_shots_2014_2024.csv.gz"
NBA_SUMMARY_PATH = DATA_DIR / "nba_player_summary_2014_2024.csv"
NHL_SUMMARY_PATH = DATA_DIR / "nhl_player_summary_2014_2024.csv"
NBA_GAM_PATH = DATA_DIR / "nba_gam_distance_2014_2024.csv"
NHL_GAM_PATH = DATA_DIR / "nhl_gam_distance_2014_2024.csv"
NBA_POSITION_PATH = DATA_DIR / "nba_position_summary_2014_2024.csv"
NHL_POSITION_PATH = DATA_DIR / "nhl_position_summary_2014_2024.csv"

NBA_SDI_FIGURE = FIGURES_DIR / "nba_sdi_vs_actual_2014_2024.png"
NHL_SDI_FIGURE = FIGURES_DIR / "nhl_sdi_vs_actual_2014_2024.png"
NBA_GAM_FIGURE = FIGURES_DIR / "nba_gam_distance_2014_2024.png"
NHL_GAM_FIGURE = FIGURES_DIR / "nhl_gam_distance_2014_2024.png"
NBA_POSITION_FIGURE = FIGURES_DIR / "nba_sdi_by_position_2014_2024.png"
NHL_POSITION_FIGURE = FIGURES_DIR / "nhl_sdi_by_position_2014_2024.png"

COURT_LANDMARKS = {
    "NBA": [
        ("Restricted Area", 4.0, "#2E8B57"),
        ("Corner 3", 22.0, "#C97C00"),
        ("Arc 3", 23.75, "#8B1E3F"),
    ],
    "NHL": [
        ("Crease Edge", 6.0, "#2E8B57"),
        ("High Slot", 25.0, "#C97C00"),
        ("Blue Line", 60.0, "#8B1E3F"),
    ],
}

DISTANCE_PLOT_MAX = {
    "NBA": 60.0,
    "NHL": 100.0,
}

POSITION_COLORS = {
    "NBA": {"G": "#2A6F97", "F": "#D17A22", "C": "#3F8F5F"},
    "NHL": {"C": "#2A6F97", "W": "#D17A22", "D": "#3F8F5F"},
}

NHL_RAW_PATH = NHL_DATA_DIR / "shots_2007-2024.csv"


@dataclass
class RunningPlayerTotals:
    attempts: int = 0
    actual_sum: float = 0.0
    expected_sum: float = 0.0
    sdi_sum: float = 0.0


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    NHL_APP_DATA_DIR.mkdir(exist_ok=True)


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


def sample_nba_for_models() -> pd.DataFrame:
    print("Sampling NBA rows for model fitting ...")
    df = pd.read_csv(NBA_EXPORT_PATH)
    df = df[df["season"].map(season_start_in_window)].copy()
    df = engineer_nba_features(df)
    if len(df) > NBA_SAMPLE_SIZE:
        df = df.sample(n=NBA_SAMPLE_SIZE, random_state=42)
    return df


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


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
) -> pd.DataFrame:
    categorical_controls = categorical_controls or []
    distance_max = DISTANCE_PLOT_MAX.get(sport, float(df[distance_col].max()))
    distance_grid = np.linspace(0, distance_max, 200)
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
    sample_df = sample_nba_for_models()
    model = build_nba_expected_model(sample_df)
    print("Scoring full NBA export into player summary ...")
    gam_df = bootstrap_distance_effect(
        sample_df,
        distance_col="shot_distance_feet",
        y_col="SHOT_MADE_FLAG",
        numeric_controls=[
            "shot_angle",
            "seconds_in_period",
            "PERIOD",
            "is_clutch",
            "is_jump_shot",
            "is_dunk",
            "is_layup",
        ],
        categorical_controls=[],
        sport="NBA",
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

    gam_df.to_csv(NBA_GAM_PATH, index=False)
    print(f"Saved: {NBA_GAM_PATH}")

    plot_sdi_scatter(summary_df, "NBA", "Actual FG%", NBA_SDI_FIGURE)
    plot_position_sdi(summary_df, "NBA", "Actual FG%", NBA_POSITION_FIGURE)
    plot_gam_distance(gam_df, "NBA", NBA_GAM_FIGURE)


def compute_nhl_scalers() -> tuple[float, float]:
    max_dist = 0.0
    max_angle = 0.0
    for chunk in pd.read_csv(NHL_RAW_PATH, chunksize=200_000):
        chunk = chunk[pd.to_numeric(chunk["season"], errors="coerce").between(2014, 2024)].copy()
        if chunk.empty:
            continue
        distances = pd.to_numeric(chunk["shotDistance"], errors="coerce")
        angles = pd.to_numeric(chunk["shotAngle"], errors="coerce").abs()
        if distances.notna().any():
            max_dist = max(max_dist, float(distances.max()))
        if angles.notna().any():
            max_angle = max(max_angle, float(angles.max()))
    if max_dist <= 0 or max_angle <= 0:
        raise ValueError("Unable to compute NHL distance/angle scalers.")
    return max_dist, max_angle


def prepare_nhl_chunk(chunk: pd.DataFrame, max_dist: float, max_angle: float) -> pd.DataFrame:
    keep_cols = [
        "shotID",
        "season",
        "game_id",
        "team",
        "teamCode",
        "goal",
        "shotGoalieFroze",
        "shotRebound",
        "shotRush",
        "period",
        "xCord",
        "yCord",
        "shotAngle",
        "shotDistance",
        "shotType",
        "shooterName",
        "xGoal",
        "shotWasOnGoal",
    ]
    current_cols = [col for col in keep_cols if col in chunk.columns]
    out = chunk[current_cols].copy()
    for col in ["goal", "shotGoalieFroze", "shotRebound", "shotRush", "period", "shotAngle", "shotDistance", "xGoal"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out.dropna(subset=["goal", "shotDistance", "shotAngle", "xGoal", "shooterName"], inplace=True)

    out["difficulty_distance"] = (out["shotDistance"] / max_dist) * 100
    out["difficulty_angle"] = (out["shotAngle"].abs() / max_angle) * 100
    out["difficulty_rebound"] = np.where(out["shotRebound"].fillna(0) == 1, 30, 0)
    out["difficulty_goalie_froze"] = np.where(out["shotGoalieFroze"].fillna(0) == 1, 20, 0)
    out["SDI"] = (
        out["difficulty_distance"] * 0.4
        + out["difficulty_angle"] * 0.3
        + out["difficulty_rebound"] * 0.2
        + out["difficulty_goalie_froze"] * 0.1
    )
    return out


def export_nhl_historical() -> None:
    if NHL_EXPORT_PATH.exists():
        print(f"Using existing NHL export: {NHL_EXPORT_PATH}")
        return
    print(f"Exporting historical NHL shots to {NHL_EXPORT_PATH} ...")
    max_dist, max_angle = compute_nhl_scalers()
    for chunk_idx, chunk in enumerate(pd.read_csv(NHL_RAW_PATH, chunksize=200_000), start=1):
            season_num = pd.to_numeric(chunk["season"], errors="coerce")
            chunk = chunk[season_num.between(2014, 2024)].copy()
            if chunk.empty:
                continue
            chunk = prepare_nhl_chunk(chunk, max_dist, max_angle)
            chunk.to_csv(
                NHL_EXPORT_PATH,
                mode="a",
                index=False,
                header=chunk_idx == 1,
                compression="gzip",
            )
            if chunk_idx % 5 == 0:
                print(f"  wrote NHL export chunk {chunk_idx}")


def build_nhl_outputs() -> None:
    print("Building NHL comparison outputs ...")
    totals: dict[str, RunningPlayerTotals] = {}
    sample_chunks = []
    sampled_rows = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(NHL_EXPORT_PATH, chunksize=150_000), start=1):
        grouped = chunk.groupby("shooterName").agg(
            attempts=("goal", "size"),
            actual_sum=("goal", "sum"),
            expected_sum=("xGoal", "sum"),
            sdi_sum=("SDI", "sum"),
        )
        for player, row in grouped.iterrows():
            entry = totals.setdefault(player, RunningPlayerTotals())
            entry.attempts += int(row["attempts"])
            entry.actual_sum += float(row["actual_sum"])
            entry.expected_sum += float(row["expected_sum"])
            entry.sdi_sum += float(row["sdi_sum"])

        if sampled_rows < NHL_SAMPLE_SIZE:
            take = min(50_000, len(chunk), NHL_SAMPLE_SIZE - sampled_rows)
            sample_chunks.append(chunk.sample(n=take, random_state=42))
            sampled_rows += take
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

    sample_df = pd.concat(sample_chunks, ignore_index=True)
    for col in ["shotDistance", "shotAngle", "period", "shotRebound", "shotGoalieFroze", "shotRush", "goal"]:
        sample_df[col] = pd.to_numeric(sample_df[col], errors="coerce").fillna(0)
    gam_df = bootstrap_distance_effect(
        sample_df,
        distance_col="shotDistance",
        y_col="goal",
        numeric_controls=["shotAngle", "period", "shotRebound", "shotGoalieFroze", "shotRush"],
        categorical_controls=[],
        sport="NHL",
    )
    gam_df.to_csv(NHL_GAM_PATH, index=False)
    print(f"Saved: {NHL_GAM_PATH}")

    plot_sdi_scatter(summary_df, "NHL", "Actual Goal %", NHL_SDI_FIGURE)
    plot_position_sdi(summary_df, "NHL", "Actual Goal %", NHL_POSITION_FIGURE)
    plot_gam_distance(gam_df, "NHL", NHL_GAM_FIGURE)


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


def plot_gam_distance(gam_df: pd.DataFrame, sport: str, output_path: Path) -> None:
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
    ax.set_title(f"{sport} GAM Distance Effect with 95% CI ({WINDOW_LABEL})", fontsize=15)
    ax.set_xlabel("Shot Distance", fontsize=12)
    ax.set_ylabel("Marginal Log-Odds Contribution", fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    if sport == "NHL":
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
