"""Quick matched-window GAM holdout metrics for the 2014-2024 NBA/NHL comparison.

This is intentionally a fast single-split evaluation for slide support.
It is not cross-validation and should not be presented as CV.
"""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from analysis.nba.cross_sport_comparison import (
    NBA_EXPORT_PATH,
    NHL_EXPORT_PATH,
    WINDOW_LABEL,
    add_nba_gam_shot_type_features,
    fit_nba_full_gam,
    load_nhl_modeling_sample,
    sample_nba_for_models,
)
from analysis.nhl.modeling import fit_expected_goal_gam, score_expected_goal_rate

DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "matched_window_gam_metrics_2014_2024.csv"
SEED = 42
TEST_SIZE = 0.2

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


def count_rows(path: Path) -> int:
    rows = 0
    for chunk in pd.read_csv(path, usecols=[0], chunksize=200_000):
        rows += len(chunk)
    return rows


def metric_row(
    *,
    sport: str,
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_train: int,
    n_eval: int,
    archive_rows: int,
    sampled_rows: int,
) -> dict[str, float | int | str]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "sport": sport,
        "model": model_name,
        "sample_window": WINDOW_LABEL,
        "eval_mode": "single random holdout (not CV)",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision_at_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_eval": float(np.mean(y_true)),
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "sampled_rows_for_fit": int(sampled_rows),
        "archive_rows_in_window": int(archive_rows),
    }


def evaluate_nba() -> dict[str, float | int | str]:
    sample_df = sample_nba_for_models()
    sample_df = add_nba_gam_shot_type_features(sample_df.copy())
    sample_df = sample_df.dropna(
        subset=NBA_FULL_GAM_FEATURE_COLS + ["SHOT_MADE_FLAG"]
    ).copy()

    train_df, test_df = train_test_split(
        sample_df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=sample_df["SHOT_MADE_FLAG"],
    )
    X_train = train_df[NBA_FULL_GAM_FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df["SHOT_MADE_FLAG"].astype(int).to_numpy()
    X_test = test_df[NBA_FULL_GAM_FEATURE_COLS].to_numpy(dtype=float)
    y_test = test_df["SHOT_MADE_FLAG"].astype(int).to_numpy()

    model = fit_nba_full_gam(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    archive_rows = count_rows(NBA_EXPORT_PATH)
    return metric_row(
        sport="NBA",
        model_name="GAM (PyGAM)",
        y_true=y_test,
        y_prob=y_prob,
        n_train=len(X_train),
        n_eval=len(X_test),
        archive_rows=archive_rows,
        sampled_rows=len(sample_df),
    )


def evaluate_nhl() -> dict[str, float | int | str]:
    sample_df = load_nhl_modeling_sample()
    train_df, test_df = train_test_split(
        sample_df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=sample_df["goal"],
    )
    model = fit_expected_goal_gam(train_df)
    y_test = pd.to_numeric(test_df["goal"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_prob = score_expected_goal_rate(model, test_df)
    archive_rows = count_rows(NHL_EXPORT_PATH)
    return metric_row(
        sport="NHL",
        model_name="GAM (PyGAM)",
        y_true=y_test,
        y_prob=y_prob,
        n_train=len(train_df),
        n_eval=len(test_df),
        archive_rows=archive_rows,
        sampled_rows=len(sample_df),
    )


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rows = [evaluate_nba(), evaluate_nhl()]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_PATH, index=False)
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
