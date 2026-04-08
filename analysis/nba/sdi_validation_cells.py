"""Notebook-style cells for validating and extending Shot Difficulty Index (SDI)."""

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 7)


# %%
# Register analyses that already exist in the current notebook/script and should
# not be recreated here. The residual-vs-SDI plot was explicitly skipped.
EXISTING_ANALYSES = {
    "sdi_vs_fg_scatter",
    "residual_vs_sdi",
}


# %%
def _first_present(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Missing required column. Expected one of: {candidates}")
    return None


def standardize_sdi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map a generic shot-level dataset onto a consistent SDI analysis schema."""
    out = df.copy()

    sdi_col = _first_present(out, ["sdi", "SDI"], required=True)
    fg_made_col = _first_present(out, ["fg_made", "SHOT_MADE_FLAG", "goal"], required=True)
    shot_value_col = _first_present(out, ["shot_value", "SHOT_VALUE", "shot_points"])
    player_col = _first_present(
        out,
        ["player", "PLAYER_NAME", "PLAYER_ID", "shooterName", "shooter", "player_id"],
        required=True,
    )
    shot_type_col = _first_present(out, ["shot_type", "SHOT_TYPE", "ACTION_TYPE", "EVENT_TYPE"])
    defender_distance_col = _first_present(out, ["defender_distance", "CLOSE_DEF_DIST", "close_def_dist"])
    distance_col = _first_present(out, ["distance", "SHOT_DISTANCE", "shot_distance_feet", "shot_distance"])
    location_x_col = _first_present(out, ["location_x", "LOC_X", "loc_x"])
    location_y_col = _first_present(out, ["location_y", "LOC_Y", "loc_y"])

    if shot_value_col is None and "SHOT_TYPE" in out.columns:
        shot_type_text = out["SHOT_TYPE"].astype(str).str.upper()
        out["shot_value"] = np.where(shot_type_text.str.contains("3PT"), 3, 2)
        shot_value_col = "shot_value"

    if shot_value_col is None:
        out["shot_value"] = 1
        shot_value_col = "shot_value"

    rename_map = {
        sdi_col: "sdi",
        fg_made_col: "fg_made",
        shot_value_col: "shot_value",
        player_col: "player",
    }
    if shot_type_col:
        rename_map[shot_type_col] = "shot_type"
    if defender_distance_col:
        rename_map[defender_distance_col] = "defender_distance"
    if distance_col:
        rename_map[distance_col] = "distance"
    if location_x_col:
        rename_map[location_x_col] = "location_x"
    if location_y_col:
        rename_map[location_y_col] = "location_y"

    out = out.rename(columns=rename_map)
    out["fg_made"] = pd.to_numeric(out["fg_made"], errors="coerce")
    out["shot_value"] = pd.to_numeric(out["shot_value"], errors="coerce")
    out["sdi"] = pd.to_numeric(out["sdi"], errors="coerce")
    out = out.dropna(subset=["sdi", "fg_made", "shot_value", "player"]).copy()

    if "shot_type" in out.columns:
        out["shot_type"] = out["shot_type"].astype(str).fillna("Unknown")

    out["points_per_shot"] = out["fg_made"] * out["shot_value"]
    return out


def fit_expected_fg_model(df: pd.DataFrame) -> tuple[LogisticRegression, pd.DataFrame, dict[str, float | None]]:
    """Fit fg_made ~ sdi logistic regression and attach predicted probabilities."""
    model = LogisticRegression(max_iter=1000)
    x = df[["sdi"]]
    y = df["fg_made"].astype(int)
    model.fit(x, y)

    out = df.copy()
    out["predicted_fg"] = model.predict_proba(x)[:, 1]
    out["shot_value_added"] = out["fg_made"] - out["predicted_fg"]
    out["expected_points"] = out["predicted_fg"] * out["shot_value"]
    out["points_over_expected"] = out["points_per_shot"] - out["expected_points"]

    metrics: dict[str, float | None] = {
        "attempts": float(len(out)),
        "fg_pct": float(out["fg_made"].mean()),
        "avg_sdi": float(out["sdi"].mean()),
        "avg_points_per_shot": float(out["points_per_shot"].mean()),
        "avg_predicted_fg": float(out["predicted_fg"].mean()),
        "brier_score": float(brier_score_loss(y, out["predicted_fg"])),
        "log_loss": float(log_loss(y, out["predicted_fg"], labels=[0, 1])),
        "auc": None,
        "model_intercept": float(model.intercept_[0]),
        "model_sdi_coef": float(model.coef_[0][0]),
    }

    if y.nunique() > 1:
        metrics["auc"] = float(roc_auc_score(y, out["predicted_fg"]))

    return model, out, metrics


def build_sdi_bins(
    df: pd.DataFrame,
    bins: int = 12,
    min_attempts_per_bin: int = 1,
) -> pd.DataFrame:
    """Aggregate attempts and outcomes into SDI buckets."""
    out = df.copy()
    out["sdi_bucket"] = pd.qcut(out["sdi"], q=min(bins, out["sdi"].nunique()), duplicates="drop")

    bucketed = (
        out.groupby("sdi_bucket", observed=False)
        .agg(
            sdi_mean=("sdi", "mean"),
            attempts=("fg_made", "size"),
            actual_fg=("fg_made", "mean"),
            predicted_fg=("predicted_fg", "mean"),
            avg_points_per_shot=("points_per_shot", "mean"),
            avg_shot_value_added=("shot_value_added", "mean"),
            avg_points_over_expected=("points_over_expected", "mean"),
        )
        .reset_index()
    )
    return bucketed[bucketed["attempts"] >= min_attempts_per_bin].copy()


def compute_player_overlay(df: pd.DataFrame, top_n: int = 8, min_attempts: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top-volume player subset and per-player above-expected summary."""
    player_summary = (
        df.groupby("player")
        .agg(
            attempts=("fg_made", "size"),
            mean_sdi=("sdi", "mean"),
            actual_fg=("fg_made", "mean"),
            predicted_fg=("predicted_fg", "mean"),
            avg_points_per_shot=("points_per_shot", "mean"),
            avg_shot_value_added=("shot_value_added", "mean"),
            avg_points_over_expected=("points_over_expected", "mean"),
        )
        .reset_index()
    )
    player_summary = player_summary[player_summary["attempts"] >= min_attempts].copy()
    player_summary["above_expected_fg"] = player_summary["actual_fg"] - player_summary["predicted_fg"]
    player_summary = player_summary.sort_values(["attempts", "above_expected_fg"], ascending=[False, False])
    overlay = df[df["player"].isin(player_summary.head(top_n)["player"])].copy()
    return overlay, player_summary


def _finalize_plot(title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    

def _save_or_show(fig: plt.Figure, save_path: str | Path | None = None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# %%
def plot_points_per_shot_vs_sdi(bucketed: pd.DataFrame, save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots()
    sns.scatterplot(data=bucketed, x="sdi_mean", y="avg_points_per_shot", size="attempts", ax=ax, legend=False)
    sns.regplot(
        data=bucketed,
        x="sdi_mean",
        y="avg_points_per_shot",
        scatter=False,
        ci=None,
        line_kws={"color": "crimson", "linewidth": 2},
        ax=ax,
    )
    _finalize_plot("Points Per Shot vs SDI", "SDI", "Average Points Per Shot")
    _save_or_show(fig, save_path)


def plot_expected_fg_overlay(bucketed: pd.DataFrame, save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots()
    sns.lineplot(data=bucketed, x="sdi_mean", y="actual_fg", marker="o", linewidth=2.5, label="Actual FG%", ax=ax)
    sns.lineplot(
        data=bucketed,
        x="sdi_mean",
        y="predicted_fg",
        marker="o",
        linewidth=2.5,
        label="Predicted FG%",
        ax=ax,
    )
    _finalize_plot("Actual vs Predicted FG% by SDI", "SDI", "FG%")
    _save_or_show(fig, save_path)


def plot_player_skill_overlay(
    player_overlay_df: pd.DataFrame,
    top_n: int,
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.scatterplot(
        data=player_overlay_df,
        x="sdi",
        y="fg_made",
        hue="player",
        alpha=0.45,
        s=55,
        ax=ax,
    )
    trend = (
        player_overlay_df.assign(
            sdi_bucket=pd.qcut(
                player_overlay_df["sdi"],
                q=min(12, player_overlay_df["sdi"].nunique()),
                duplicates="drop",
            )
        )
        .groupby("sdi_bucket", observed=False)
        .agg(sdi_mean=("sdi", "mean"), fg_rate=("fg_made", "mean"))
        .reset_index()
    )
    sns.lineplot(
        data=trend,
        x="sdi_mean",
        y="fg_rate",
        color="black",
        linewidth=2,
        marker="o",
        label="Top-player average FG%",
        ax=ax,
    )
    _finalize_plot(
        f"Player Skill Overlay: Top {top_n} Players by Attempts",
        "SDI",
        "FG Made / FG%",
    )
    _save_or_show(fig, save_path)


def plot_usage_vs_difficulty(bucketed: pd.DataFrame, save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots()
    sns.barplot(data=bucketed, x="sdi_bucket", y="attempts", color="#4C78A8", ax=ax)
    ax.tick_params(axis="x", rotation=45)
    _finalize_plot("Usage vs Difficulty", "SDI Bucket", "Shot Frequency")
    _save_or_show(fig, save_path)


def plot_shot_type_breakdown(
    df: pd.DataFrame,
    top_k: int = 6,
    save_path: str | Path | None = None,
) -> None:
    if "shot_type" not in df.columns:
        print("Skipping shot type breakdown: 'shot_type' column not available.")
        return

    top_types = df["shot_type"].value_counts().head(top_k).index
    subset = df[df["shot_type"].isin(top_types)].copy()
    subset["sdi_bucket"] = pd.qcut(subset["sdi"], q=min(10, subset["sdi"].nunique()), duplicates="drop")
    grouped = (
        subset.groupby(["shot_type", "sdi_bucket"], observed=False)
        .agg(sdi_mean=("sdi", "mean"), actual_fg=("fg_made", "mean"), attempts=("fg_made", "size"))
        .reset_index()
    )
    grouped = grouped[grouped["attempts"] > 0].copy()

    fig, ax = plt.subplots(figsize=(13, 8))
    sns.lineplot(data=grouped, x="sdi_mean", y="actual_fg", hue="shot_type", marker="o", ax=ax)
    _finalize_plot("SDI vs FG% by Shot Type", "SDI", "FG%")
    _save_or_show(fig, save_path)


def plot_sdi_distribution(
    df: pd.DataFrame,
    bucketed: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(df["sdi"], bins=30, kde=True, color="#4C78A8", ax=axes[0])
    axes[0].set_title("SDI Distribution")
    axes[0].set_xlabel("SDI")
    axes[0].set_ylabel("Shot Count")

    sns.lineplot(data=bucketed, x="sdi_mean", y="actual_fg", marker="o", ax=axes[1])
    axes[1].set_title("FG% per SDI Bin")
    axes[1].set_xlabel("SDI")
    axes[1].set_ylabel("FG%")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_defender_distance_validation(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    if "defender_distance" not in df.columns:
        print("Skipping defender distance validation: 'defender_distance' column not available.")
        return

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="sdi", y="defender_distance", alpha=0.2, s=25, ax=ax)
    sns.regplot(
        data=df,
        x="sdi",
        y="defender_distance",
        scatter=False,
        ci=None,
        line_kws={"color": "crimson", "linewidth": 2},
        ax=ax,
    )
    _finalize_plot("SDI vs Defender Distance", "SDI", "Defender Distance")
    _save_or_show(fig, save_path)


def plot_shot_value_added(bucketed: pd.DataFrame, save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots()
    sns.lineplot(data=bucketed, x="sdi_mean", y="avg_shot_value_added", marker="o", linewidth=2.5, ax=ax)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    _finalize_plot("Shot Value Added vs SDI", "SDI", "Actual Make - Predicted Probability")
    _save_or_show(fig, save_path)


# %%
def run_sdi_validation_analysis(
    raw_df: pd.DataFrame,
    *,
    top_n_players: int = 8,
    min_player_attempts: int = 25,
    sdi_bins: int = 12,
    existing_analyses: set[str] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the missing SDI validation analyses and return reusable outputs."""
    existing_analyses = existing_analyses or set()
    df = standardize_sdi_columns(raw_df)
    model, modeled_df, metrics = fit_expected_fg_model(df)
    bucketed = build_sdi_bins(modeled_df, bins=sdi_bins)
    overlay_df, player_summary = compute_player_overlay(
        modeled_df,
        top_n=top_n_players,
        min_attempts=min_player_attempts,
    )

    save_dir = Path(save_dir) if save_dir is not None else None
    analysis_paths = {
        "points_per_shot_vs_sdi": None if save_dir is None else save_dir / "sdi_points_per_shot.png",
        "expected_fg_model": None if save_dir is None else save_dir / "sdi_actual_vs_predicted_fg.png",
        "player_skill_overlay": None if save_dir is None else save_dir / "sdi_player_skill_overlay.png",
        "usage_vs_difficulty": None if save_dir is None else save_dir / "sdi_usage_vs_difficulty.png",
        "shot_type_breakdown": None if save_dir is None else save_dir / "sdi_shot_type_breakdown.png",
        "sdi_distribution": None if save_dir is None else save_dir / "sdi_distribution.png",
        "defender_distance_validation": None if save_dir is None else save_dir / "sdi_defender_distance_validation.png",
        "shot_value_added": None if save_dir is None else save_dir / "sdi_shot_value_added.png",
    }

    analyses = {
        "points_per_shot_vs_sdi": lambda: plot_points_per_shot_vs_sdi(bucketed, analysis_paths["points_per_shot_vs_sdi"]),
        "expected_fg_model": lambda: plot_expected_fg_overlay(bucketed, analysis_paths["expected_fg_model"]),
        "player_skill_overlay": lambda: plot_player_skill_overlay(overlay_df, top_n_players, analysis_paths["player_skill_overlay"]),
        "usage_vs_difficulty": lambda: plot_usage_vs_difficulty(bucketed, analysis_paths["usage_vs_difficulty"]),
        "shot_type_breakdown": lambda: plot_shot_type_breakdown(modeled_df, save_path=analysis_paths["shot_type_breakdown"]),
        "sdi_distribution": lambda: plot_sdi_distribution(modeled_df, bucketed, analysis_paths["sdi_distribution"]),
        "defender_distance_validation": lambda: plot_defender_distance_validation(modeled_df, analysis_paths["defender_distance_validation"]),
        "shot_value_added": lambda: plot_shot_value_added(bucketed, analysis_paths["shot_value_added"]),
    }

    for name, plot_fn in analyses.items():
        if name in existing_analyses:
            print(f"Skipping existing analysis: {name}")
            continue
        plot_fn()

    return {
        "model": model,
        "modeled_df": modeled_df,
        "bucketed_summary": bucketed,
        "player_summary": player_summary,
        "metrics": pd.Series(metrics, name="value"),
    }


# %%
# Example usage:
#
# 1. Replace `shots_df` with your shot-level DataFrame.
# 2. Keep `EXISTING_ANALYSES` as-is to avoid recreating the residual-vs-SDI plot.
#
# shots_df = pd.read_csv("path/to/your_shots.csv")
# results = run_sdi_validation_analysis(
#     shots_df,
#     top_n_players=8,
#     min_player_attempts=25,
#     sdi_bins=12,
#     existing_analyses=EXISTING_ANALYSES,
# )
# results["metrics"]
# results["player_summary"].head(15)
