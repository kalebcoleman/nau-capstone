"""Fit an NHL GAM from shot data and export the distance-effect artifacts."""

from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.nhl.modeling import (
    DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    build_distance_only_effect_frame,
    build_full_model_effect_frame,
    compute_farthest_non_empty_made_goal_distance,
    compute_farthest_made_goal_distance,
    fit_distance_only_gam,
    fit_expected_goal_gam,
    load_nhl_distance_plot_data,
    load_nhl_modeling_sample,
    score_expected_goal_rate,
)

GAM_OUTPUT_PATH = DATA_DIR / "nhl_gam_distance_2014_2024.csv"
FIGURE_OUTPUT_PATH = FIGURES_DIR / "nhl_gam_distance_2014_2024.png"
FULL_MODEL_ANGLE_OUTPUT_PATH = DATA_DIR / "nhl_gam_angle_2014_2024.csv"
FULL_MODEL_ANGLE_FIGURE_OUTPUT_PATH = FIGURES_DIR / "nhl_gam_angle_2014_2024.png"
DIST_ONLY_NON_EMPTY_OUTPUT_PATH = DATA_DIR / "nhl_gam_distance_non_empty_net_distance_only_2014_2024.csv"
DIST_ONLY_NON_EMPTY_FIGURE_OUTPUT_PATH = FIGURES_DIR / "nhl_gam_distance_non_empty_net_distance_only_2014_2024.png"
ALL_SHOTS_OUTPUT_PATH = DATA_DIR / "nhl_gam_distance_all_shots_distance_only_2014_2024.csv"
ALL_SHOTS_FIGURE_OUTPUT_PATH = FIGURES_DIR / "nhl_gam_distance_all_shots_distance_only_2014_2024.png"
COMPARISON_OUTPUT_PATH = DATA_DIR / "nhl_gam_distance_empty_net_comparison_distance_only_2014_2024.csv"
COMPARISON_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "nhl_gam_distance_empty_net_comparison_distance_only_2014_2024.png"
)
ALL_SHOTS_LANDMARK_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "nhl_gam_distance_all_shots_2014_2024_landmarks.png"
)
MODEL_OUTPUT_PATH = MODELS_DIR / "nhl_expected_goal_gam_2014_2024.pkl"
METRICS_OUTPUT_PATH = DATA_DIR / "nhl_model_metrics_gam.csv"

NHL_DISTANCE_LANDMARKS = [
    ("Crease edge", 6.0, "#2E8B57"),
    ("Top circles", 33.0, "#E09F3E"),
    ("Blue line", 60.0, "#8B1E3F"),
    ("Center red line", 89.0, "#6A4C93"),
]
POSTER_EXPORT_DPI = 400


def add_landmark_lines(ax, *, max_x: float) -> None:
    for label, x_value, color in NHL_DISTANCE_LANDMARKS:
        if x_value > max_x:
            continue
        ax.axvline(
            x_value,
            color=color,
            linestyle=":",
            label=label,
            linewidth=2,
            alpha=0.95,
        )


def plot_effect(
    gam_df,
    *,
    title: str,
    output_path: Path,
    baseline_label: str,
    show_landmarks: bool = False,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gam_df["x_value"], gam_df["fitted_effect"], color="#1f77b4", linewidth=2)
    ax.fill_between(
        gam_df["x_value"],
        gam_df["lower_ci"],
        gam_df["upper_ci"],
        alpha=0.2,
        color="#1f77b4",
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(
        float(gam_df.get("baseline_distance", gam_df.get("baseline_value")).iloc[0]),
        color="green",
        linestyle="--",
        alpha=0.7,
        label=baseline_label,
    )
    ax.set_xlim(0, float(gam_df["x_value"].max()))
    ax.set_xlabel("Shot Distance (feet)")
    ax.set_ylabel("Marginal log-odds contribution")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if show_landmarks:
        add_landmark_lines(ax, max_x=float(gam_df["x_value"].max()))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=POSTER_EXPORT_DPI)
    plt.close()


def plot_distance_comparison(all_df, non_empty_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        all_df["x_value"],
        all_df["fitted_effect"],
        color="#C06C2B",
        linewidth=2,
        label="All shots",
    )
    ax.fill_between(
        all_df["x_value"],
        all_df["lower_ci"],
        all_df["upper_ci"],
        alpha=0.16,
        color="#C06C2B",
    )

    ax.plot(
        non_empty_df["x_value"],
        non_empty_df["fitted_effect"],
        color="#1f77b4",
        linewidth=2,
        label="Non-empty-net shots",
    )
    ax.fill_between(
        non_empty_df["x_value"],
        non_empty_df["lower_ci"],
        non_empty_df["upper_ci"],
        alpha=0.16,
        color="#1f77b4",
    )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(
        float(non_empty_df["baseline_distance"].iloc[0]),
        color="green",
        linestyle="--",
        alpha=0.7,
        label="Non-empty median baseline",
    )
    ax.axvline(
        float(all_df["baseline_distance"].iloc[0]),
        color="#C06C2B",
        linestyle="--",
        alpha=0.45,
        label="All-shots median baseline",
    )
    ax.set_xlim(0, max(float(all_df["x_value"].max()), float(non_empty_df["x_value"].max())))
    ax.set_xlabel("Shot Distance (feet)")
    ax.set_ylabel("Marginal log-odds contribution")
    ax.set_title("NHL GAM Distance Effect: All Shots vs Non-Empty-Net Shots")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(COMPARISON_FIGURE_OUTPUT_PATH, dpi=POSTER_EXPORT_DPI)
    plt.close()


def evaluate_full_model(model_df: pd.DataFrame) -> pd.DataFrame:
    train_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=42,
        stratify=model_df["goal"],
    )
    eval_model = fit_expected_goal_gam(train_df)
    y_true = pd.to_numeric(test_df["goal"], errors="coerce").fillna(0).astype(int)
    y_prob = score_expected_goal_rate(eval_model, test_df)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "NHL GAM (PyGAM)",
                "sample_window": "2014-2024",
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "auc_roc": float(roc_auc_score(y_true, y_prob)),
                "average_precision": float(average_precision_score(y_true, y_prob)),
                "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
                "brier_score": float(brier_score_loss(y_true, y_prob)),
                "precision_at_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall_at_0_5": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
                "goal_rate_eval": float(y_true.mean()),
                "n_train": int(len(train_df)),
                "n_eval": int(len(test_df)),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    )
    metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
    print(f"Saved: {METRICS_OUTPUT_PATH}")
    return metrics_df


def main():
    full_model_df = load_nhl_modeling_sample()
    evaluate_full_model(full_model_df)
    full_model = fit_expected_goal_gam(full_model_df)
    full_distance_df = build_full_model_effect_frame(
        full_model,
        full_model_df,
        feature_col="shotDistance",
        term=1,
        plot_max=compute_farthest_made_goal_distance(),
    )
    full_angle_df = build_full_model_effect_frame(
        full_model,
        full_model_df,
        feature_col="shotAngle",
        term=2,
        plot_min=float(pd.to_numeric(full_model_df["shotAngle"], errors="coerce").min()),
        plot_max=float(pd.to_numeric(full_model_df["shotAngle"], errors="coerce").max()),
    )

    sample_df = load_nhl_distance_plot_data()
    gam = fit_distance_only_gam(sample_df)
    plot_max = compute_farthest_non_empty_made_goal_distance()
    gam_df = build_distance_only_effect_frame(gam, sample_df, plot_max=plot_max)

    all_sample_df = load_nhl_distance_plot_data(exclude_empty_net=False)
    all_gam = fit_distance_only_gam(all_sample_df)
    all_plot_max = compute_farthest_made_goal_distance()
    all_gam_df = build_distance_only_effect_frame(all_gam, all_sample_df, plot_max=all_plot_max)
    all_gam_df["curve"] = "all_shots"
    comparison_non_empty_df = gam_df.copy()
    comparison_non_empty_df["curve"] = "non_empty_net"

    GAM_OUTPUT_PATH.parent.mkdir(exist_ok=True)
    full_distance_df.to_csv(GAM_OUTPUT_PATH, index=False)
    full_angle_df.to_csv(FULL_MODEL_ANGLE_OUTPUT_PATH, index=False)
    gam_df.to_csv(DIST_ONLY_NON_EMPTY_OUTPUT_PATH, index=False)
    all_gam_df.to_csv(ALL_SHOTS_OUTPUT_PATH, index=False)
    pd.concat([all_gam_df, comparison_non_empty_df], ignore_index=True).to_csv(
        COMPARISON_OUTPUT_PATH, index=False
    )
    with open(MODEL_OUTPUT_PATH, "wb") as fh:
        pickle.dump(full_model, fh)
    plot_effect(
        full_distance_df,
        title="NHL Full-Model GAM Distance Effect on Goal Probability",
        output_path=FIGURE_OUTPUT_PATH,
        baseline_label="Median distance baseline",
    )
    plot_effect(
        full_angle_df,
        title="NHL Full-Model GAM Angle Effect on Goal Probability",
        output_path=FULL_MODEL_ANGLE_FIGURE_OUTPUT_PATH,
        baseline_label="Median angle baseline",
    )
    plot_effect(
        gam_df,
        title="NHL Distance-Only GAM (Non-Empty-Net Shots)",
        output_path=DIST_ONLY_NON_EMPTY_FIGURE_OUTPUT_PATH,
        baseline_label="Non-empty median baseline",
    )
    plot_effect(
        all_gam_df,
        title="NHL Distance-Only GAM (All Shots)",
        output_path=ALL_SHOTS_FIGURE_OUTPUT_PATH,
        baseline_label="All-shots median baseline",
    )
    plot_effect(
        all_gam_df,
        title="NHL Distance-Only GAM (All Shots) with Rink Landmarks",
        output_path=ALL_SHOTS_LANDMARK_FIGURE_OUTPUT_PATH,
        baseline_label="All-shots median baseline",
        show_landmarks=True,
    )
    plot_distance_comparison(all_gam_df, comparison_non_empty_df)

    print(f"Saved: {GAM_OUTPUT_PATH}")
    print(f"Saved: {FIGURE_OUTPUT_PATH}")
    print(f"Saved: {FULL_MODEL_ANGLE_OUTPUT_PATH}")
    print(f"Saved: {FULL_MODEL_ANGLE_FIGURE_OUTPUT_PATH}")
    print(f"Saved: {DIST_ONLY_NON_EMPTY_OUTPUT_PATH}")
    print(f"Saved: {DIST_ONLY_NON_EMPTY_FIGURE_OUTPUT_PATH}")
    print(f"Saved: {ALL_SHOTS_OUTPUT_PATH}")
    print(f"Saved: {ALL_SHOTS_FIGURE_OUTPUT_PATH}")
    print(f"Saved: {ALL_SHOTS_LANDMARK_FIGURE_OUTPUT_PATH}")
    print(f"Saved: {COMPARISON_OUTPUT_PATH}")
    print(f"Saved: {COMPARISON_FIGURE_OUTPUT_PATH}")
    print(f"Saved: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
