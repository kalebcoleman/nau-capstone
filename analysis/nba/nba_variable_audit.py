"""NBA variable audit and effect diagnostics using the full historical shot archive."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pygam import LogisticGAM, s
except ImportError:  # pragma: no cover - local fallback when pygam is unavailable
    LogisticGAM = None
    s = None

from feature_spec import (
    ANALYSIS_ARCHIVE_PATH,
    CURRENT_FEATURE_USAGE,
    ENGINEERED_FEATURE_METADATA,
    RAW_DISCOVERY_PRIORITY_COLUMNS,
    TOP_LEVEL_ARCHIVE_PATH,
    current_usage_rows,
    engineer_nba_features,
)

ANALYSIS_DIR = Path(__file__).resolve().parent
DATA_DIR = ANALYSIS_DIR / "data"
FIGURES_DIR = ANALYSIS_DIR / "figures" / "variable_audit"

AUDIT_SAMPLE_SIZE = 75_000
SEED = 42

CONTINUOUS_AUDIT_VARIABLES = [
    "shot_distance_feet",
    "shot_angle",
    "abs_shot_angle",
    "seconds_in_period",
    "period_seconds_elapsed",
    "period_elapsed_pct",
    "game_minutes_elapsed",
    "PERIOD",
]

CATEGORICAL_AUDIT_VARIABLES = [
    "shot_type_cat",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_RANGE",
    "is_corner_three",
    "is_above_break_three",
    "is_midrange",
    "is_restricted_area",
    "is_end_of_period",
    "is_end_of_game",
    "home_indicator",
]


def load_full_archive() -> tuple[pd.DataFrame, Path]:
    for path in [TOP_LEVEL_ARCHIVE_PATH, ANALYSIS_ARCHIVE_PATH]:
        if path.exists():
            usecols = [
                "TEAM_NAME",
                "PERIOD",
                "MINUTES_REMAINING",
                "SECONDS_REMAINING",
                "EVENT_TYPE",
                "ACTION_TYPE",
                "SHOT_TYPE",
                "SHOT_ZONE_BASIC",
                "SHOT_ZONE_RANGE",
                "SHOT_DISTANCE",
                "LOC_X",
                "LOC_Y",
                "SHOT_ATTEMPTED_FLAG",
                "SHOT_MADE_FLAG",
                "GAME_DATE",
                "HTM",
                "VTM",
                "season",
                "season_type",
                "shot_id",
            ]
            df = pd.read_csv(path, usecols=usecols)
            return df, path
    raise FileNotFoundError("NBA full archive not found in top-level data/ or analysis/nba/data/.")


def archive_schema(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.astype(str).tolist()


def build_raw_column_catalog(raw_columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in raw_columns:
        rows.append(
            {
                "variable_name": column,
                "variable_kind": "raw",
                "raw_source_columns": column,
                "engineered_from": "",
                "current_formula": "",
                "used_in_xfg": column in CURRENT_FEATURE_USAGE["xFG"],
                "used_in_gam": column in CURRENT_FEATURE_USAGE["GAM"],
                "used_in_sdi": column in CURRENT_FEATURE_USAGE["SDI"],
                "used_in_app": column in CURRENT_FEATURE_USAGE["app"],
                "intended_sdi_direction": "not applicable",
                "notes": "Raw archive column discovered from full NBA shot archive.",
            }
        )
    return rows


def build_engineered_feature_catalog() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in ENGINEERED_FEATURE_METADATA:
        rows.append(
            {
                "variable_name": item["variable_name"],
                "variable_kind": "engineered",
                "raw_source_columns": "",
                "engineered_from": item["engineered_from"],
                "current_formula": item["current_formula"],
                "used_in_xfg": item["used_in_xfg"],
                "used_in_gam": item["used_in_gam"],
                "used_in_sdi": item["used_in_sdi"],
                "used_in_app": item["used_in_app"],
                "intended_sdi_direction": item["intended_sdi_direction"],
                "notes": item["notes"],
            }
        )
    rows.append(
        {
            "variable_name": "shot_clock_remaining",
            "variable_kind": "missing_candidate",
            "raw_source_columns": "",
            "engineered_from": "",
            "current_formula": "",
            "used_in_xfg": False,
            "used_in_gam": False,
            "used_in_sdi": False,
            "used_in_app": False,
            "intended_sdi_direction": "lower = harder",
            "notes": "Not present in current archive. Future SDI transform should be max_shot_clock - shot_clock_remaining.",
        }
    )
    return rows


def build_candidate_feature_table() -> pd.DataFrame:
    engineered = pd.DataFrame(ENGINEERED_FEATURE_METADATA)
    engineered["currently_used_anywhere"] = (
        engineered["used_in_xfg"]
        | engineered["used_in_gam"]
        | engineered["used_in_sdi"]
        | engineered["used_in_app"]
    )
    engineered["candidate_status"] = np.where(
        engineered["currently_used_anywhere"],
        "active_or_audit",
        "candidate_only",
    )
    missing = pd.DataFrame(
        [
            {
                "variable_name": "shot_clock_remaining",
                "engineered_from": "external shot-clock source",
                "current_formula": "",
                "used_in_xfg": False,
                "used_in_gam": False,
                "used_in_sdi": False,
                "used_in_app": False,
                "intended_sdi_direction": "lower = harder",
                "notes": "Missing from archive; future SDI transform should be max_shot_clock - shot_clock_remaining.",
                "currently_used_anywhere": False,
                "candidate_status": "missing_high_value",
            }
        ]
    )
    cols = [
        "variable_name",
        "engineered_from",
        "current_formula",
        "used_in_xfg",
        "used_in_gam",
        "used_in_sdi",
        "used_in_app",
        "intended_sdi_direction",
        "candidate_status",
        "notes",
    ]
    return pd.concat([engineered[cols], missing[cols]], ignore_index=True)


def prepare_audit_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SHOT_ATTEMPTED_FLAG" in out.columns:
        out = out[out["SHOT_ATTEMPTED_FLAG"] == 1].copy()
    out = engineer_nba_features(out)
    out["SHOT_MADE_FLAG"] = pd.to_numeric(out["SHOT_MADE_FLAG"], errors="coerce")
    out = out[out["SHOT_MADE_FLAG"].isin([0, 1])].copy()
    return out


def sample_for_effects(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    if len(df) <= AUDIT_SAMPLE_SIZE:
        return df.copy()
    return df.sample(AUDIT_SAMPLE_SIZE, random_state=seed).copy()


def summarize_continuous_direction(df: pd.DataFrame, variable: str) -> dict[str, object]:
    clean = df[[variable, "SHOT_MADE_FLAG"]].copy()
    clean[variable] = pd.to_numeric(clean[variable], errors="coerce")
    clean = clean.dropna(subset=[variable, "SHOT_MADE_FLAG"])
    if clean.empty or clean[variable].nunique() < 4:
        return {
            "variable_name": variable,
            "variable_type": "continuous",
            "observed_direction": "insufficient_data",
            "low_bin_fg": np.nan,
            "high_bin_fg": np.nan,
            "effect_size": np.nan,
            "notes": "Not enough variation for direction review.",
        }

    clean["bucket"] = pd.qcut(clean[variable], q=min(10, clean[variable].nunique()), duplicates="drop")
    grouped = (
        clean.groupby("bucket", observed=False)
        .agg(variable_mean=(variable, "mean"), fg_pct=("SHOT_MADE_FLAG", "mean"), attempts=("SHOT_MADE_FLAG", "size"))
        .reset_index(drop=True)
        .sort_values("variable_mean")
    )
    low_fg = float(grouped.iloc[0]["fg_pct"])
    high_fg = float(grouped.iloc[-1]["fg_pct"])
    if high_fg < low_fg:
        direction = "higher_value_lower_fg"
    elif high_fg > low_fg:
        direction = "higher_value_higher_fg"
    else:
        direction = "flat"
    return {
        "variable_name": variable,
        "variable_type": "continuous",
        "observed_direction": direction,
        "low_bin_fg": low_fg,
        "high_bin_fg": high_fg,
        "effect_size": high_fg - low_fg,
        "notes": "Direction based on lowest vs highest quantile FG% bins.",
    }


def plot_continuous_effect(df: pd.DataFrame, variable: str, output_path: Path) -> None:
    clean = df[[variable, "SHOT_MADE_FLAG"]].copy()
    clean[variable] = pd.to_numeric(clean[variable], errors="coerce")
    clean = clean.dropna(subset=[variable, "SHOT_MADE_FLAG"])
    if clean.empty or clean[variable].nunique() < 4:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if LogisticGAM is not None:
        x = clean[[variable]].to_numpy()
        y = clean["SHOT_MADE_FLAG"].astype(int).to_numpy()
        gam = LogisticGAM(s(0, n_splines=12)).fit(x, y)
        grid = np.linspace(clean[variable].min(), clean[variable].max(), 200)
        grid_x = grid.reshape(-1, 1)
        probs = gam.predict_mu(grid_x)
        conf = gam.prediction_intervals(grid_x, width=0.95)
        ax.plot(grid, probs, color="#1f77b4", linewidth=2, label="Univariate GAM")
        ax.fill_between(grid, conf[:, 0], conf[:, 1], color="#1f77b4", alpha=0.2, label="95% interval")
    else:
        grouped = clean.assign(
            bucket=pd.qcut(clean[variable], q=min(20, clean[variable].nunique()), duplicates="drop")
        )
        grouped = (
            grouped.groupby("bucket", observed=False)
            .agg(variable_mean=(variable, "mean"), fg_pct=("SHOT_MADE_FLAG", "mean"))
            .reset_index(drop=True)
            .sort_values("variable_mean")
        )
        ax.plot(grouped["variable_mean"], grouped["fg_pct"], color="#1f77b4", linewidth=2, marker="o", label="Binned FG%")

    ax.set_title(f"NBA Effect Review: {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Make probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_categorical_direction(df: pd.DataFrame, variable: str) -> tuple[dict[str, object], pd.DataFrame]:
    clean = df[[variable, "SHOT_MADE_FLAG"]].copy()
    clean[variable] = clean[variable].astype("string").fillna("Unknown")
    clean = clean.dropna(subset=["SHOT_MADE_FLAG"])
    grouped = (
        clean.groupby(variable, dropna=False)
        .agg(attempts=("SHOT_MADE_FLAG", "size"), fg_pct=("SHOT_MADE_FLAG", "mean"))
        .reset_index()
        .sort_values("attempts", ascending=False)
    )
    if grouped.empty:
        summary = {
            "variable_name": variable,
            "variable_type": "categorical",
            "observed_direction": "insufficient_data",
            "low_bin_fg": np.nan,
            "high_bin_fg": np.nan,
            "effect_size": np.nan,
            "notes": "No categorical groups available.",
        }
        return summary, grouped

    top = grouped.iloc[0]
    bottom = grouped.sort_values("fg_pct").iloc[0]
    summary = {
        "variable_name": variable,
        "variable_type": "categorical",
        "observed_direction": f"top_group={top[variable]}",
        "low_bin_fg": float(bottom["fg_pct"]),
        "high_bin_fg": float(grouped["fg_pct"].max()),
        "effect_size": float(grouped["fg_pct"].max() - grouped["fg_pct"].min()),
        "notes": "Categorical spread across group FG% values.",
    }
    return summary, grouped


def plot_categorical_effect(grouped: pd.DataFrame, variable: str, output_path: Path) -> None:
    if grouped.empty:
        return
    plot_df = grouped.copy()
    plot_df[variable] = plot_df[variable].astype(str)
    plot_df = plot_df.sort_values("fg_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(plot_df[variable], plot_df["fg_pct"], color="#4C78A8", alpha=0.85)
    ax.set_title(f"NBA Category Effect Review: {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("FG%")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, attempts in zip(bars, plot_df["attempts"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(attempts):,}", ha="center", va="bottom", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def angle_conflict_note(df: pd.DataFrame) -> str:
    needed = {"SHOT_ZONE_BASIC", "abs_shot_angle", "SHOT_MADE_FLAG"}
    if not needed.issubset(df.columns):
        return "Angle conflict could not be evaluated."
    summary = (
        df[df["SHOT_ZONE_BASIC"].isin(["Left Corner 3", "Right Corner 3", "Above the Break 3"])]
        .groupby("SHOT_ZONE_BASIC")
        .agg(mean_abs_angle=("abs_shot_angle", "mean"), fg_pct=("SHOT_MADE_FLAG", "mean"))
    )
    if {"Left Corner 3", "Right Corner 3", "Above the Break 3"}.issubset(summary.index):
        corner_angle = float(summary.loc[["Left Corner 3", "Right Corner 3"], "mean_abs_angle"].mean())
        corner_fg = float(summary.loc[["Left Corner 3", "Right Corner 3"], "fg_pct"].mean())
        atb_angle = float(summary.loc["Above the Break 3", "mean_abs_angle"])
        atb_fg = float(summary.loc["Above the Break 3", "fg_pct"])
        return (
            f"Corner 3 mean abs angle={corner_angle:.3f}, FG%={corner_fg:.3f}; "
            f"Above-the-break mean abs angle={atb_angle:.3f}, FG%={atb_fg:.3f}. "
            "Higher raw angle does not map cleanly to harder shots."
        )
    return "Angle conflict groups were not all present."


def build_direction_review(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for variable in CONTINUOUS_AUDIT_VARIABLES:
        if variable not in df.columns:
            continue
        row = summarize_continuous_direction(df, variable)
        if variable in {"shot_angle", "abs_shot_angle"}:
            row["notes"] = angle_conflict_note(df)
        rows.append(row)
        plot_continuous_effect(df, variable, FIGURES_DIR / f"gam_effect_{variable}.png")

    for variable in CATEGORICAL_AUDIT_VARIABLES:
        if variable not in df.columns:
            continue
        row, grouped = summarize_categorical_direction(df, variable)
        rows.append(row)
        plot_categorical_effect(grouped, variable, FIGURES_DIR / f"effect_{variable}.png")

    usage_lookup = pd.DataFrame(current_usage_rows()).set_index("variable_name")
    direction = pd.DataFrame(rows)
    if direction.empty:
        return direction

    direction["used_in_xfg"] = direction["variable_name"].map(usage_lookup["used_in_xfg"]).eq(True)
    direction["used_in_gam"] = direction["variable_name"].map(usage_lookup["used_in_gam"]).eq(True)
    direction["used_in_sdi"] = direction["variable_name"].map(usage_lookup["used_in_sdi"]).eq(True)
    direction["used_in_app"] = direction["variable_name"].map(usage_lookup["used_in_app"]).eq(True)
    return direction.sort_values(["variable_type", "variable_name"]).reset_index(drop=True)


def save_outputs(catalog: pd.DataFrame, usage: pd.DataFrame, direction: pd.DataFrame, candidates: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(DATA_DIR / "nba_variable_catalog.csv", index=False)
    usage.to_csv(DATA_DIR / "nba_variable_usage_matrix.csv", index=False)
    direction.to_csv(DATA_DIR / "nba_variable_direction_review.csv", index=False)
    candidates.to_csv(DATA_DIR / "nba_candidate_engineering_features.csv", index=False)


def main() -> None:
    print("Loading full NBA archive for variable audit...")
    full_df, source_path = load_full_archive()
    print(f"Using archive: {source_path}")
    raw_columns = archive_schema(source_path)
    print("Engineering shared NBA features...")
    audit_df = prepare_audit_frame(full_df)
    print(f"Prepared {len(audit_df):,} attempted shots for audit.")
    effect_df = sample_for_effects(audit_df)
    print(f"Running effect diagnostics on {len(effect_df):,} sampled shots.")

    catalog = pd.DataFrame(build_raw_column_catalog(raw_columns) + build_engineered_feature_catalog())
    usage = pd.DataFrame(current_usage_rows()).sort_values("variable_name").reset_index(drop=True)
    direction = build_direction_review(effect_df)
    candidates = build_candidate_feature_table()
    save_outputs(catalog, usage, direction, candidates)

    print(f"Variable catalog saved to: {DATA_DIR / 'nba_variable_catalog.csv'}")
    print(f"Usage matrix saved to: {DATA_DIR / 'nba_variable_usage_matrix.csv'}")
    print(f"Direction review saved to: {DATA_DIR / 'nba_variable_direction_review.csv'}")
    print(f"Candidate features saved to: {DATA_DIR / 'nba_candidate_engineering_features.csv'}")
    print(f"Effect figures saved under: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
