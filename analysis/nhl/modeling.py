"""NHL shot-model helpers for expected goals and GAM-style diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pygam import LogisticGAM, l, s, te
    PYGAM_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    LogisticGAM = None
    l = s = te = None
    PYGAM_IMPORT_ERROR = exc

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
MODELS_DIR = SCRIPT_DIR / "models"
NHL_RAW_PATH = DATA_DIR / "shots_2007-2024.csv"
NHL_EXPORT_PATH = DATA_DIR / "app_data" / "nhl_shots_2014_2024.csv.gz"

SEED = 42
NHL_MODEL_SAMPLE_SIZE = 300_000
NHL_DISTANCE_PLOT_SAMPLE_SIZE = 200_000
DISTANCE_PLOT_MAX = 100.0

SHOT_TYPE_FAMILIES = ("wrist", "snap", "slap", "backhand")
NUMERIC_COLS = [
    "goal",
    "shotGoalieFroze",
    "shotRebound",
    "shotRush",
    "period",
    "xCord",
    "yCord",
    "shotAngle",
    "shotDistance",
    "xGoal",
]
KEEP_COLS = [
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


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    NHL_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def compute_nhl_scalers(raw_path: Path = NHL_RAW_PATH) -> tuple[float, float]:
    max_dist = 0.0
    max_angle = 0.0
    for chunk in pd.read_csv(raw_path, chunksize=200_000):
        season_num = pd.to_numeric(chunk["season"], errors="coerce")
        chunk = chunk[season_num.between(2014, 2024)].copy()
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


def add_shot_type_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "shotType" not in out.columns:
        out["shot_type_family"] = "other"
        out["is_wrist_shot"] = 0
        out["is_snap_shot"] = 0
        out["is_slap_shot"] = 0
        out["is_backhand"] = 0
        return out

    shot_type = out["shotType"].fillna("").astype(str).str.lower()
    out["shot_type_family"] = np.select(
        [
            shot_type.str.contains("wrist"),
            shot_type.str.contains("snap"),
            shot_type.str.contains("slap"),
            shot_type.str.contains("backhand"),
        ],
        SHOT_TYPE_FAMILIES,
        default="other",
    )
    out["is_wrist_shot"] = (out["shot_type_family"] == "wrist").astype(int)
    out["is_snap_shot"] = (out["shot_type_family"] == "snap").astype(int)
    out["is_slap_shot"] = (out["shot_type_family"] == "slap").astype(int)
    out["is_backhand"] = (out["shot_type_family"] == "backhand").astype(int)
    return out


def prepare_nhl_chunk(
    chunk: pd.DataFrame,
    max_dist: float,
    max_angle: float,
) -> pd.DataFrame:
    current_cols = [col for col in KEEP_COLS if col in chunk.columns]
    out = chunk[current_cols].copy()
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out.dropna(
        subset=["goal", "shotDistance", "shotAngle", "xCord", "yCord", "shooterName"],
        inplace=True,
    )

    out["difficulty_distance"] = (out["shotDistance"] / max_dist) * 100
    out["difficulty_angle"] = (out["shotAngle"].abs() / max_angle) * 100
    out["difficulty_rebound"] = np.where(out["shotRebound"].fillna(0) == 1, 30, 0)
    out["difficulty_goalie_froze"] = np.where(
        out["shotGoalieFroze"].fillna(0) == 1, 20, 0
    )
    out["difficulty_rush"] = np.where(out["shotRush"].fillna(0) == 1, 20, 0)
    out["SDI"] = (
        out["difficulty_distance"] * 0.35
        + out["difficulty_angle"] * 0.25
        + out["difficulty_rebound"] * 0.2
        + out["difficulty_goalie_froze"] * 0.1
        + out["difficulty_rush"] * 0.1
    )
    out = add_shot_type_features(out)
    return out


def export_nhl_historical() -> None:
    ensure_dirs()
    if NHL_EXPORT_PATH.exists():
        print(f"Using existing NHL export: {NHL_EXPORT_PATH}")
        return
    print(f"Exporting historical NHL shots to {NHL_EXPORT_PATH} ...")
    max_dist, max_angle = compute_nhl_scalers()
    for chunk_idx, chunk in enumerate(
        pd.read_csv(NHL_RAW_PATH, chunksize=200_000), start=1
    ):
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


def load_nhl_modeling_sample(sample_size: int = NHL_MODEL_SAMPLE_SIZE) -> pd.DataFrame:
    export_nhl_historical()
    df = pd.read_csv(NHL_EXPORT_PATH)
    if "shot_type_family" not in df.columns or "is_wrist_shot" not in df.columns:
        df = add_shot_type_features(df)
    df = df.dropna(
        subset=[
            "xCord",
            "yCord",
            "shotDistance",
            "shotAngle",
            "period",
            "goal",
        ]
    ).copy()
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=SEED)
    return df


def load_nhl_distance_plot_data(
    sample_size: int = NHL_DISTANCE_PLOT_SAMPLE_SIZE,
    *,
    exclude_empty_net: bool = True,
) -> pd.DataFrame:
    cols = ["season", "shotDistance", "goal", "shotOnEmptyNet"]
    chunks = []
    kept = 0
    for chunk in pd.read_csv(NHL_RAW_PATH, usecols=cols, chunksize=200_000):
        season_num = pd.to_numeric(chunk["season"], errors="coerce")
        chunk = chunk[season_num.between(2014, 2024)].copy()
        if chunk.empty:
            continue
        chunk["shotDistance"] = pd.to_numeric(chunk["shotDistance"], errors="coerce")
        chunk["goal"] = pd.to_numeric(chunk["goal"], errors="coerce")
        chunk["shotOnEmptyNet"] = pd.to_numeric(chunk["shotOnEmptyNet"], errors="coerce")
        chunk = chunk.dropna(subset=["shotDistance", "goal"])
        if exclude_empty_net:
            chunk = chunk[chunk["shotOnEmptyNet"].fillna(0) != 1].copy()
        if chunk.empty:
            continue
        if kept < sample_size:
            take = min(len(chunk), sample_size - kept)
            if take < len(chunk):
                chunk = chunk.sample(n=take, random_state=SEED)
            chunks.append(chunk[["shotDistance", "goal", "shotOnEmptyNet"]])
            kept += len(chunk)
        if kept >= sample_size:
            break
    if not chunks:
        raise ValueError("No NHL distance-plot data found.")
    return pd.concat(chunks, ignore_index=True)


FEATURE_COLS = [
    "xCord",
    "yCord",
    "shotDistance",
    "shotAngle",
    "period",
    "shotRebound",
    "shotGoalieFroze",
    "shotRush",
    "is_wrist_shot",
    "is_snap_shot",
    "is_slap_shot",
    "is_backhand",
]


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    if "shot_type_family" not in out.columns or "is_wrist_shot" not in out.columns:
        out = add_shot_type_features(out)
    for col in FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out[FEATURE_COLS].to_numpy(dtype=float)


def fit_expected_goal_gam(df: pd.DataFrame) -> LogisticGAM:
    if LogisticGAM is None:
        raise ModuleNotFoundError(
            "pygam is required for the NHL expected-goal GAM. Install it with "
            "`pip install -r analysis/nhl/requirements.txt` or the repo root requirements."
        ) from PYGAM_IMPORT_ERROR
    X = build_feature_matrix(df)
    y = pd.to_numeric(df["goal"], errors="coerce").fillna(0).astype(int).to_numpy()
    gam = LogisticGAM(
        te(0, 1, n_splines=10)
        + s(2, n_splines=14)
        + s(3, n_splines=10)
        + s(4, n_splines=5)
        + l(5)
        + l(6)
        + l(7)
        + l(8)
        + l(9)
        + l(10)
        + l(11)
    )
    gam.fit(X, y)
    return gam


def fit_distance_only_gam(df: pd.DataFrame) -> LogisticGAM:
    if LogisticGAM is None:
        raise ModuleNotFoundError(
            "pygam is required for the NHL GAM plots. Install it with "
            "`pip install -r analysis/nhl/requirements.txt` or the repo root requirements."
        ) from PYGAM_IMPORT_ERROR
    distance = (
        pd.to_numeric(df["shotDistance"], errors="coerce").fillna(0).to_numpy(dtype=float)
    )
    y = pd.to_numeric(df["goal"], errors="coerce").fillna(0).astype(int).to_numpy()
    gam = LogisticGAM(s(0, n_splines=9))
    gam.gridsearch(
        distance.reshape(-1, 1),
        y,
        lam=np.logspace(0, 4, 9),
        progress=False,
    )
    return gam


def score_expected_goal_rate(gam: LogisticGAM, df: pd.DataFrame) -> np.ndarray:
    return gam.predict_proba(build_feature_matrix(df))


def build_distance_effect_frame(
    gam: LogisticGAM,
    reference_df: pd.DataFrame,
    *,
    plot_max: float = DISTANCE_PLOT_MAX,
    n_points: int = 200,
) -> pd.DataFrame:
    base = {
        "xCord": float(pd.to_numeric(reference_df["xCord"], errors="coerce").median()),
        "yCord": float(pd.to_numeric(reference_df["yCord"], errors="coerce").median()),
        "shotDistance": float(
            pd.to_numeric(reference_df["shotDistance"], errors="coerce").median()
        ),
        "shotAngle": float(pd.to_numeric(reference_df["shotAngle"], errors="coerce").median()),
        "period": float(pd.to_numeric(reference_df["period"], errors="coerce").median()),
        "shotRebound": int(
            pd.to_numeric(reference_df["shotRebound"], errors="coerce").fillna(0).mode().iloc[0]
        ),
        "shotGoalieFroze": int(
            pd.to_numeric(reference_df["shotGoalieFroze"], errors="coerce").fillna(0).mode().iloc[0]
        ),
        "shotRush": int(
            pd.to_numeric(reference_df["shotRush"], errors="coerce").fillna(0).mode().iloc[0]
        ),
        "is_wrist_shot": int(reference_df["is_wrist_shot"].mode().iloc[0]),
        "is_snap_shot": int(reference_df["is_snap_shot"].mode().iloc[0]),
        "is_slap_shot": int(reference_df["is_slap_shot"].mode().iloc[0]),
        "is_backhand": int(reference_df["is_backhand"].mode().iloc[0]),
    }
    distance_grid = np.linspace(0, plot_max, n_points)
    plot_df = pd.DataFrame({"shotDistance": distance_grid})
    for col, value in base.items():
        if col != "shotDistance":
            plot_df[col] = value

    X_grid = build_feature_matrix(plot_df)
    effect = gam.partial_dependence(term=1, X=X_grid)
    conf = gam.partial_dependence(term=1, X=X_grid, width=0.95)[1]
    baseline_distance = base["shotDistance"]
    baseline_df = plot_df.iloc[[0]].copy()
    baseline_df["shotDistance"] = baseline_distance
    baseline_effect = float(
        gam.partial_dependence(term=1, X=build_feature_matrix(baseline_df))[0]
    )

    return pd.DataFrame(
        {
            "x_value": distance_grid,
            "fitted_effect": effect - baseline_effect,
            "lower_ci": conf[:, 0] - baseline_effect,
            "upper_ci": conf[:, 1] - baseline_effect,
            "sport": "NHL",
            "effect_label": "Shot Distance",
            "season_window": "2014-2024",
            "baseline_distance": baseline_distance,
        }
    )


def build_full_model_effect_frame(
    gam: LogisticGAM,
    reference_df: pd.DataFrame,
    *,
    feature_col: str,
    term: int,
    plot_max: float | None = None,
    plot_min: float = 0.0,
    n_points: int = 200,
) -> pd.DataFrame:
    base = {
        "xCord": float(pd.to_numeric(reference_df["xCord"], errors="coerce").median()),
        "yCord": float(pd.to_numeric(reference_df["yCord"], errors="coerce").median()),
        "shotDistance": float(
            pd.to_numeric(reference_df["shotDistance"], errors="coerce").median()
        ),
        "shotAngle": float(pd.to_numeric(reference_df["shotAngle"], errors="coerce").median()),
        "period": float(pd.to_numeric(reference_df["period"], errors="coerce").median()),
        "shotRebound": int(
            pd.to_numeric(reference_df["shotRebound"], errors="coerce").fillna(0).mode().iloc[0]
        ),
        "shotGoalieFroze": int(
            pd.to_numeric(reference_df["shotGoalieFroze"], errors="coerce").fillna(0).mode().iloc[0]
        ),
        "shotRush": int(
            pd.to_numeric(reference_df["shotRush"], errors="coerce").fillna(0).mode().iloc[0]
        ),
        "is_wrist_shot": int(reference_df["is_wrist_shot"].mode().iloc[0]),
        "is_snap_shot": int(reference_df["is_snap_shot"].mode().iloc[0]),
        "is_slap_shot": int(reference_df["is_slap_shot"].mode().iloc[0]),
        "is_backhand": int(reference_df["is_backhand"].mode().iloc[0]),
    }
    series = pd.to_numeric(reference_df[feature_col], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No valid values found for {feature_col}.")
    if plot_max is None:
        plot_max = float(series.max())
    baseline_value = float(series.median())

    grid_values = np.linspace(plot_min, float(plot_max), n_points)
    plot_df = pd.DataFrame({"shotDistance": np.repeat(base["shotDistance"], n_points)})
    for col, value in base.items():
        plot_df[col] = value
    plot_df[feature_col] = grid_values

    X_grid = build_feature_matrix(plot_df)
    effect = gam.partial_dependence(term=term, X=X_grid)
    conf = gam.partial_dependence(term=term, X=X_grid, width=0.95)[1]

    baseline_df = plot_df.iloc[[0]].copy()
    baseline_df[feature_col] = baseline_value
    baseline_effect = float(gam.partial_dependence(term=term, X=build_feature_matrix(baseline_df))[0])

    return pd.DataFrame(
        {
            "x_value": grid_values,
            "fitted_effect": effect - baseline_effect,
            "lower_ci": conf[:, 0] - baseline_effect,
            "upper_ci": conf[:, 1] - baseline_effect,
            "sport": "NHL",
            "effect_label": feature_col,
            "season_window": "2014-2024",
            "baseline_value": baseline_value,
        }
    )


def build_distance_only_effect_frame(
    gam: LogisticGAM,
    reference_df: pd.DataFrame,
    *,
    plot_max: float | None = None,
    n_points: int = 200,
) -> pd.DataFrame:
    distances = pd.to_numeric(reference_df["shotDistance"], errors="coerce").dropna()
    if distances.empty:
        raise ValueError("No valid NHL shot distances found for GAM plotting.")

    observed_max = float(distances.max())
    if plot_max is None:
        plot_max = observed_max

    distance_grid = np.linspace(0, float(plot_max), n_points)
    grid = distance_grid.reshape(-1, 1)
    effect = gam.partial_dependence(term=0, X=grid)
    conf = gam.partial_dependence(term=0, X=grid, width=0.95)[1]

    baseline_distance = float(distances.median())
    baseline_grid = np.array([[baseline_distance]])
    baseline_effect = float(gam.partial_dependence(term=0, X=baseline_grid)[0])

    return pd.DataFrame(
        {
            "x_value": distance_grid,
            "fitted_effect": effect - baseline_effect,
            "lower_ci": conf[:, 0] - baseline_effect,
            "upper_ci": conf[:, 1] - baseline_effect,
            "sport": "NHL",
            "effect_label": "Shot Distance",
            "season_window": "2014-2024",
            "baseline_distance": baseline_distance,
            "observed_max_distance": observed_max,
        }
    )


def compute_farthest_made_goal_distance(
    export_path: Path = NHL_EXPORT_PATH,
) -> float:
    max_goal_distance = 0.0
    for chunk in pd.read_csv(export_path, usecols=["shotDistance", "goal"], chunksize=200_000):
        chunk["shotDistance"] = pd.to_numeric(chunk["shotDistance"], errors="coerce")
        chunk["goal"] = pd.to_numeric(chunk["goal"], errors="coerce")
        made = chunk.loc[chunk["goal"] == 1, "shotDistance"].dropna()
        if not made.empty:
            max_goal_distance = max(max_goal_distance, float(made.max()))
    if max_goal_distance <= 0:
        raise ValueError("Unable to determine farthest made NHL goal distance.")
    return max_goal_distance


def compute_farthest_non_empty_made_goal_distance() -> float:
    max_goal_distance = 0.0
    cols = ["season", "shotDistance", "goal", "shotOnEmptyNet"]
    for chunk in pd.read_csv(NHL_RAW_PATH, usecols=cols, chunksize=200_000):
        season_num = pd.to_numeric(chunk["season"], errors="coerce")
        chunk = chunk[season_num.between(2014, 2024)].copy()
        if chunk.empty:
            continue
        chunk["shotDistance"] = pd.to_numeric(chunk["shotDistance"], errors="coerce")
        chunk["goal"] = pd.to_numeric(chunk["goal"], errors="coerce")
        chunk["shotOnEmptyNet"] = pd.to_numeric(chunk["shotOnEmptyNet"], errors="coerce")
        made = chunk.loc[
            (chunk["goal"] == 1) & (chunk["shotOnEmptyNet"].fillna(0) != 1),
            "shotDistance",
        ].dropna()
        if not made.empty:
            max_goal_distance = max(max_goal_distance, float(made.max()))
    if max_goal_distance <= 0:
        raise ValueError("Unable to determine farthest non-empty-net NHL goal distance.")
    return max_goal_distance
