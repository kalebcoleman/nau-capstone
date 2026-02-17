import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ANALYSIS_DIR = Path(__file__).parent
DATA_DIR = ANALYSIS_DIR / "data"
FIGURES_DIR = ANALYSIS_DIR / "figures"

DISTANCE_BINS = [0, 4, 8, 14, 22, 30, np.inf]
DISTANCE_LABELS = ["0-4 ft", "4-8 ft", "8-14 ft", "14-22 ft", "22-30 ft", "30+ ft"]


def find_enriched_shots_path(season=None):
    if season:
        path = DATA_DIR / f"shots_with_xp_{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing enriched shots file: {path}. Run expected_points_analysis.py first."
            )
        return path

    files = list(DATA_DIR.glob("shots_with_xp_*.parquet"))
    if not files:
        raise FileNotFoundError(
            "No shots_with_xp parquet files found in analysis/data/. "
            "Run expected_points_analysis.py first."
        )
    return max(files, key=lambda p: p.stat().st_mtime)


def season_from_path(path):
    return path.stem.replace("shots_with_xp_", "")


def load_enriched_shots(season=None):
    data_path = find_enriched_shots_path(season)
    resolved_season = season_from_path(data_path)
    return pd.read_parquet(data_path), resolved_season, data_path


def add_distance_band(df):
    out = df.copy()
    distances = pd.to_numeric(out["shot_distance_feet"], errors="coerce")
    out["distance_band"] = pd.cut(
        distances,
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        include_lowest=True,
        right=False,
    )
    out["distance_band"] = out["distance_band"].astype("string").fillna("unknown")
    return out


def add_shot_type_family(df):
    out = df.copy()
    action = out.get("ACTION_TYPE", pd.Series("", index=out.index))
    action = action.astype(str).str.lower().fillna("")

    shot_family = np.select(
        [
            action.str.contains("dunk"),
            action.str.contains("layup|finger roll"),
            action.str.contains("hook"),
            action.str.contains("float"),
            action.str.contains("jump shot|pullup|step back|fadeaway"),
        ],
        ["dunk", "layup", "hook", "floater", "jump_shot"],
        default="other",
    )
    out["shot_type_family"] = shot_family
    return out


def compute_reliability_table(y_true, y_prob, n_bins=15):
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)

    if y_true_arr.size == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "bin_left",
                "bin_right",
                "n_shots",
                "mean_pred",
                "mean_actual",
            ]
        )

    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    binned = pd.DataFrame(
        {"bin_id": bin_ids, "y_true": y_true_arr, "y_prob": y_prob_arr}
    )
    grouped = (
        binned.groupby("bin_id", as_index=False)
        .agg(
            n_shots=("y_true", "size"),
            mean_pred=("y_prob", "mean"),
            mean_actual=("y_true", "mean"),
        )
        .sort_values("bin_id")
    )

    grouped["bin_left"] = grouped["bin_id"].map(lambda i: edges[i])
    grouped["bin_right"] = grouped["bin_id"].map(lambda i: edges[i + 1])
    grouped["bin"] = grouped.apply(
        lambda row: f"[{row['bin_left']:.2f}, {row['bin_right']:.2f})", axis=1
    )

    return grouped[
        ["bin", "bin_left", "bin_right", "n_shots", "mean_pred", "mean_actual"]
    ]


def compute_ece_mce(reliability_df):
    if reliability_df.empty:
        return np.nan, np.nan

    errors = (reliability_df["mean_actual"] - reliability_df["mean_pred"]).abs()
    total = reliability_df["n_shots"].sum()
    if total <= 0:
        return np.nan, np.nan

    weights = reliability_df["n_shots"] / total
    ece = float((weights * errors).sum())
    mce = float(errors.max())
    return ece, mce


def compute_calibration_slope_intercept(y_true, y_prob):
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)

    if np.unique(y_true_arr).size < 2:
        return np.nan, np.nan

    logit_scores = np.log(y_prob_arr / (1 - y_prob_arr)).reshape(-1, 1)

    try:
        model = LogisticRegression(solver="lbfgs")
        model.fit(logit_scores, y_true_arr)
    except ValueError:
        return np.nan, np.nan

    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0][0])
    return intercept, slope


def safe_auc(y_true, y_prob):
    y_true_arr = np.asarray(y_true, dtype=int)
    if np.unique(y_true_arr).size < 2:
        return np.nan
    return float(roc_auc_score(y_true_arr, y_prob))


def compute_calibration_summary(y_true, y_prob, n_bins=15):
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)

    reliability_df = compute_reliability_table(y_true_arr, y_prob_arr, n_bins=n_bins)
    ece, mce = compute_ece_mce(reliability_df)
    intercept, slope = compute_calibration_slope_intercept(y_true_arr, y_prob_arr)

    return {
        "n_shots": int(y_true_arr.size),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "log_loss": float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1])),
        "auc_roc": safe_auc(y_true_arr, y_prob_arr),
        "ece": ece,
        "mce": mce,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def summarize_group(
    df, probability_col, label_col, group_col, n_bins=15, min_group_shots=300
):
    rows = []
    reliability_parts = []

    for group_name, group_df in df.groupby(group_col, dropna=False):
        if len(group_df) < min_group_shots:
            continue

        y_true = group_df[label_col].astype(int).to_numpy()
        y_prob = group_df[probability_col].to_numpy()
        summary = compute_calibration_summary(y_true, y_prob, n_bins=n_bins)
        summary.update({"scope": group_col, "group": str(group_name)})
        rows.append(summary)

        reliability = compute_reliability_table(y_true, y_prob, n_bins=n_bins)
        if reliability.empty:
            continue
        reliability = reliability.copy()
        reliability["scope"] = group_col
        reliability["group"] = str(group_name)
        reliability_parts.append(reliability)

    return rows, reliability_parts


def compile_reports(
    df,
    season,
    probability_col="xP_prob",
    label_col="SHOT_MADE_FLAG",
    n_bins=15,
    min_group_shots=300,
):
    required = {
        probability_col,
        label_col,
        "shot_distance_feet",
        "SHOT_ZONE_BASIC",
        "ACTION_TYPE",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Input dataframe missing required columns for calibration diagnostics: "
            f"{', '.join(sorted(missing))}"
        )

    clean = df.copy()
    clean[label_col] = pd.to_numeric(clean[label_col], errors="coerce")
    clean[probability_col] = pd.to_numeric(clean[probability_col], errors="coerce")
    clean = clean[clean[label_col].isin([0, 1])]
    clean = clean[clean[probability_col].notna()]

    clean = add_distance_band(clean)
    clean = add_shot_type_family(clean)

    summary_rows = []
    reliability_frames = []

    y_true_all = clean[label_col].astype(int).to_numpy()
    y_prob_all = clean[probability_col].to_numpy()
    overall = compute_calibration_summary(y_true_all, y_prob_all, n_bins=n_bins)
    overall.update({"scope": "overall", "group": "all"})
    summary_rows.append(overall)

    reliability_all = compute_reliability_table(y_true_all, y_prob_all, n_bins=n_bins)
    reliability_all["scope"] = "overall"
    reliability_all["group"] = "all"
    reliability_frames.append(reliability_all)

    for subgroup_col in ["distance_band", "SHOT_ZONE_BASIC", "shot_type_family"]:
        rows, rel_parts = summarize_group(
            clean,
            probability_col=probability_col,
            label_col=label_col,
            group_col=subgroup_col,
            n_bins=n_bins,
            min_group_shots=min_group_shots,
        )
        summary_rows.extend(rows)
        reliability_frames.extend(rel_parts)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.insert(0, "season", season)
    summary_df.insert(1, "probability_source", probability_col)
    summary_df = summary_df.sort_values(["scope", "group"]).reset_index(drop=True)

    reliability_df = pd.concat(reliability_frames, ignore_index=True)
    reliability_df.insert(0, "season", season)
    reliability_df.insert(1, "probability_source", probability_col)

    return summary_df, reliability_df


def plot_overall_reliability(reliability_df, output_path):
    overall = reliability_df[
        (reliability_df["scope"] == "overall") & (reliability_df["group"] == "all")
    ].copy()
    if overall.empty:
        return

    overall = overall.sort_values("bin_left")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Perfect calibration",
    )
    ax.plot(
        overall["mean_pred"],
        overall["mean_actual"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Observed",
    )
    ax.scatter(
        overall["mean_pred"],
        overall["mean_actual"],
        s=overall["n_shots"] / 35,
        alpha=0.8,
        color="#1f77b4",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Make Probability")
    ax.set_ylabel("Observed Make Rate")
    ax.set_title("xFG Reliability Curve")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_distance_reliability(reliability_df, output_path, min_shots=600):
    subset = reliability_df[reliability_df["scope"] == "distance_band"].copy()
    if subset.empty:
        return

    totals = subset.groupby("group", as_index=False)["n_shots"].sum()
    keep = totals[totals["n_shots"] >= min_shots]["group"].tolist()
    subset = subset[subset["group"].isin(keep)]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.2,
        label="Perfect calibration",
    )

    for group_name, group_df in subset.groupby("group"):
        group_df = group_df.sort_values("bin_left")
        ax.plot(
            group_df["mean_pred"],
            group_df["mean_actual"],
            marker="o",
            linewidth=1.8,
            label=group_name,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Make Probability")
    ax.set_ylabel("Observed Make Rate")
    ax.set_title("xFG Reliability by Shot Distance")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate xFG calibration diagnostics")
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season like 2025-26 (default: latest parquet)",
    )
    parser.add_argument(
        "--bins", type=int, default=15, help="Number of reliability bins"
    )
    parser.add_argument(
        "--min-group-shots",
        type=int,
        default=300,
        help="Minimum subgroup sample size for subgroup summaries",
    )
    parser.add_argument(
        "--prob-col", type=str, default="xP_prob", help="Probability column name"
    )
    parser.add_argument(
        "--label-col", type=str, default="SHOT_MADE_FLAG", help="Outcome label column"
    )
    return parser.parse_args()


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    args = parse_args()

    print("Loading enriched shot data...")
    shots_df, resolved_season, data_path = load_enriched_shots(args.season)
    print(f"Loaded {len(shots_df):,} rows from {data_path.name}")

    summary_df, reliability_df = compile_reports(
        shots_df,
        season=resolved_season,
        probability_col=args.prob_col,
        label_col=args.label_col,
        n_bins=args.bins,
        min_group_shots=args.min_group_shots,
    )

    summary_path = DATA_DIR / f"calibration_summary_{resolved_season}.csv"
    reliability_path = DATA_DIR / f"calibration_reliability_{resolved_season}.csv"
    summary_df.to_csv(summary_path, index=False)
    reliability_df.to_csv(reliability_path, index=False)

    reliability_fig = FIGURES_DIR / f"calibration_reliability_{resolved_season}.png"
    distance_fig = FIGURES_DIR / f"calibration_by_distance_{resolved_season}.png"
    plot_overall_reliability(reliability_df, reliability_fig)
    plot_distance_reliability(reliability_df, distance_fig)

    print(f"Saved: {summary_path}")
    print(f"Saved: {reliability_path}")
    print(f"Saved: {reliability_fig}")
    print(f"Saved: {distance_fig}")

    overall = summary_df[
        (summary_df["scope"] == "overall") & (summary_df["group"] == "all")
    ]
    if not overall.empty:
        row = overall.iloc[0]
        print(
            "Overall calibration | "
            f"Brier={row['brier_score']:.4f}, "
            f"ECE={row['ece']:.4f}, "
            f"AUC={row['auc_roc']:.4f}, "
            f"Slope={row['calibration_slope']:.3f}, "
            f"Intercept={row['calibration_intercept']:.3f}"
        )
