from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from player_clustering import (
    POSITION_CONFIG,
    USE_POSITIONS,
    load_assist_pct_from_csv,
    load_position_from_csv,
    load_usage_from_csv,
    run_player_clustering,
)
from sdi_analysis import run_sdi_analysis
from utils.court_utils import draw_half_court, setup_shot_chart_axes

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
MIN_SHOTS = 200


def compute_residuals(df):
    out = df.copy()
    out["residual"] = out["SHOT_MADE_FLAG"] - out["xP_prob"]
    return out


def aggregate_player_residuals(df, min_shots=MIN_SHOTS):
    player_stats = (
        df.groupby("PLAYER_NAME")
        .agg(
            {
                "xP_prob": "mean",
                "SHOT_MADE_FLAG": ["mean", "sum", "count"],
                "residual": "mean",
            }
        )
        .reset_index()
    )

    player_stats.columns = [
        "player",
        "avg_xFG",
        "actual_fg_pct",
        "makes",
        "attempts",
        "avg_residual",
    ]
    player_stats = player_stats[player_stats["attempts"] >= min_shots]
    player_stats["residual_fg_pct"] = (
        player_stats["actual_fg_pct"] - player_stats["avg_xFG"]
    )
    return player_stats.sort_values("residual_fg_pct", ascending=False)


def aggregate_zone_residuals(df):
    zone_stats = (
        df.groupby("SHOT_ZONE_BASIC")
        .agg(
            {
                "xP_prob": "mean",
                "SHOT_MADE_FLAG": "mean",
                "residual": ["mean", "count"],
            }
        )
        .reset_index()
    )

    zone_stats.columns = [
        "zone",
        "avg_xFG",
        "actual_fg_pct",
        "avg_residual",
        "attempts",
    ]
    zone_stats["residual_fg_pct"] = zone_stats["actual_fg_pct"] - zone_stats["avg_xFG"]
    return zone_stats.sort_values("attempts", ascending=False)


def plot_residual_heatmap(df, player_name, output_path):
    player_df = df[df["PLAYER_NAME"] == player_name].copy()
    if len(player_df) < 50:
        print(f"Not enough shots for {player_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 11))
    draw_half_court(ax, outer_lines=True)

    scatter = ax.scatter(
        player_df["LOC_X"],
        player_df["LOC_Y"],
        c=player_df["residual"],
        cmap="RdYlGn",
        vmin=-0.5,
        vmax=0.5,
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )

    setup_shot_chart_axes(ax)
    ax.set_title(
        f"{player_name} - Shot Residuals (Green=Overperform, Red=Underperform)",
        fontsize=14,
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Residual (Actual - Expected)", fontsize=12)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def run_residual_analysis(df):
    print("\n" + "=" * 60)
    print("PHASE 1: SHOT QUALITY (RESIDUAL ANALYSIS)")
    print("=" * 60)

    df_with_residuals = compute_residuals(df)
    player_residuals = aggregate_player_residuals(df_with_residuals)

    print("\nTOP 15 OVERPERFORMERS (Positive Residuals):")
    print(
        player_residuals.head(15)[
            ["player", "avg_xFG", "actual_fg_pct", "residual_fg_pct", "attempts"]
        ].to_string(index=False)
    )

    print("\nBOTTOM 15 UNDERPERFORMERS (Negative Residuals):")
    print(
        player_residuals.tail(15)[
            ["player", "avg_xFG", "actual_fg_pct", "residual_fg_pct", "attempts"]
        ].to_string(index=False)
    )

    zone_residuals = aggregate_zone_residuals(df_with_residuals)
    print("\nRESIDUALS BY ZONE:")
    print(zone_residuals.to_string(index=False))

    top_player = player_residuals.iloc[0]["player"]
    plot_residual_heatmap(
        df_with_residuals,
        top_player,
        FIGURES_DIR / f"{top_player.replace(' ', '_')}_residual_heatmap.png",
    )

    player_residuals.to_csv(DATA_DIR / "player_residuals.csv", index=False)
    print(f"Saved: {DATA_DIR / 'player_residuals.csv'}")

    return df_with_residuals


def validate_required_columns(df):
    required_cols = {
        "PLAYER_NAME",
        "SHOT_MADE_FLAG",
        "SHOT_ZONE_BASIC",
        "ACTION_TYPE",
        "xP_prob",
        "shot_distance_feet",
        "seconds_in_period",
        "shot_angle",
        "is_jump_shot",
    }
    missing = required_cols - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"shots_with_xp file missing required columns: {missing_list}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Loading enriched shot data...")
    current_season = "2025-26"
    current_season_type = "regular"
    shots_path = DATA_DIR / f"shots_with_xp_{current_season}.parquet"
    if not shots_path.exists():
        raise FileNotFoundError(
            f"Missing {shots_path}. Run expected_points_analysis.py first."
        )

    shots_df = pd.read_parquet(shots_path)
    print(f"Loaded {len(shots_df):,} shots")

    validate_required_columns(shots_df)

    usage_df = load_usage_from_csv(current_season, current_season_type)
    position_df = None
    assist_df = None
    if USE_POSITIONS:
        position_df = load_position_from_csv(current_season, current_season_type)
        assist_df = load_assist_pct_from_csv(current_season, current_season_type)

    residual_df = run_residual_analysis(shots_df)
    sdi_df, _player_sdi, _elite = run_sdi_analysis(
        residual_df,
        FIGURES_DIR,
        min_shots=MIN_SHOTS,
    )
    run_player_clustering(
        sdi_df,
        FIGURES_DIR,
        DATA_DIR,
        usage_df=usage_df,
        position_df=position_df,
        assist_df=assist_df,
        use_positions=USE_POSITIONS,
        position_config=POSITION_CONFIG,
        season_label=current_season,
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Data saved to: {DATA_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
