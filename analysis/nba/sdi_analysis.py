"""Shot Difficulty Index analysis for NBA shot data."""

import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MIN_SHOTS = 200


def compute_sdi(df):
    """
    Compute Shot Difficulty Index per shot.

    SDI is a weighted combination of:
    - Distance (30%): farther = harder
    - Shot clock pressure (20%): less time = harder
    - Shot type difficulty (20%): pull-up > catch-and-shoot
    - Zone difficulty (15%): mid-range harder than paint
    - Angle difficulty (15%): extreme angles harder
    """
    df = df.copy()

    df["sdi_distance"] = df["shot_distance_feet"].clip(0, 35) / 35.0
    df["sdi_clock"] = 1 - (df["seconds_in_period"].clip(0, 720) / 720.0)

    action = df["ACTION_TYPE"].str.lower().fillna("")
    df["sdi_shot_type"] = 0.3
    df.loc[
        action.str.contains("pullup|step back|fadeaway|turnaround"), "sdi_shot_type"
    ] = 0.8
    df.loc[action.str.contains("driving|running"), "sdi_shot_type"] = 0.6
    df.loc[action.str.contains("dunk"), "sdi_shot_type"] = 0.1
    df.loc[
        action.str.contains("layup") & ~action.str.contains("driving"), "sdi_shot_type"
    ] = 0.2

    zone_difficulty = {
        "Restricted Area": 0.1,
        "In The Paint (Non-RA)": 0.4,
        "Mid-Range": 0.7,
        "Left Corner 3": 0.5,
        "Right Corner 3": 0.5,
        "Above the Break 3": 0.6,
        "Backcourt": 0.9,
    }
    df["sdi_zone"] = df["SHOT_ZONE_BASIC"].map(zone_difficulty).fillna(0.5)

    df["sdi_angle"] = np.abs(df["shot_angle"]) / (np.pi / 2)

    df["SDI"] = (
        0.30 * df["sdi_distance"]
        + 0.20 * df["sdi_clock"]
        + 0.20 * df["sdi_shot_type"]
        + 0.15 * df["sdi_zone"]
        + 0.15 * df["sdi_angle"]
    )
    return df


def aggregate_player_sdi(df, min_shots=MIN_SHOTS):
    """Aggregate SDI by player."""
    player_sdi = (
        df.groupby("PLAYER_NAME")
        .agg({"SDI": "mean", "xP_prob": "mean", "SHOT_MADE_FLAG": ["mean", "count"]})
        .reset_index()
    )
    player_sdi.columns = ["player", "avg_sdi", "avg_xFG", "actual_fg_pct", "attempts"]
    player_sdi = player_sdi[player_sdi["attempts"] >= min_shots]
    return player_sdi.sort_values("avg_sdi", ascending=False)


def _normalize_name(name):
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().replace(".", "").replace(",", "").strip()


def plot_sdi_vs_xfg(player_sdi, output_path):
    """Scatter plot of average SDI vs actual FG%."""
    fig, ax = plt.subplots(figsize=(12, 10))

    player_sdi = player_sdi.copy()
    player_sdi["residual"] = player_sdi["actual_fg_pct"] - player_sdi["avg_xFG"]

    scatter = ax.scatter(
        player_sdi["avg_sdi"],
        player_sdi["actual_fg_pct"],
        s=player_sdi["attempts"] / 5,
        alpha=0.6,
        c=player_sdi["residual"],
        cmap="RdYlGn",
        vmin=-0.10,
        vmax=0.10,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.axhline(
        player_sdi["actual_fg_pct"].median(), color="gray", linestyle="--", alpha=0.5
    )
    ax.axvline(player_sdi["avg_sdi"].median(), color="gray", linestyle="--", alpha=0.5)

    highlight_names = {
        _normalize_name("giannis antetokounmpo"),
        _normalize_name("shai gilgeous-alexander"),
        _normalize_name("nikola jokic"),
        _normalize_name("nikola jokić"),
    }

    for _, row in player_sdi.iterrows():
        if _normalize_name(row["player"]) in highlight_names:
            ax.annotate(
                row["player"],
                (row["avg_sdi"], row["actual_fg_pct"]),
                fontsize=9,
                alpha=0.95,
            )

    ax.set_xlabel("Average Shot Difficulty Index (SDI)", fontsize=12)
    ax.set_ylabel("Actual Field Goal %", fontsize=12)
    ax.set_title(
        "Shot Difficulty vs Actual Efficiency\n(Size = Volume, Color = FG% Residual)",
        fontsize=14,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("FG% Residual (Actual - Expected)", fontsize=10)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def summarize_elite_shot_makers(player_sdi):
    """Return high-SDI overperformers for quick reporting."""
    median_sdi = player_sdi["avg_sdi"].median()
    return player_sdi[
        (player_sdi["avg_sdi"] > median_sdi)
        & (player_sdi["actual_fg_pct"] > player_sdi["avg_xFG"])
    ]


def run_sdi_analysis(df, figures_dir, min_shots=MIN_SHOTS):
    """Run SDI phase and return transformed shot data and player summaries."""
    figures_dir = Path(figures_dir)

    print("\n" + "=" * 60)
    print("PHASE 2: SHOT DIFFICULTY INDEX (SDI)")
    print("=" * 60)

    df_with_sdi = compute_sdi(df)
    player_sdi = aggregate_player_sdi(df_with_sdi, min_shots=min_shots)

    print("\nTOP 15 PLAYERS BY SHOT DIFFICULTY:")
    print(
        player_sdi.head(15)[
            ["player", "avg_sdi", "avg_xFG", "actual_fg_pct", "attempts"]
        ].to_string(index=False)
    )

    plot_sdi_vs_xfg(player_sdi, figures_dir / "sdi_vs_xfg_scatter.png")

    elite = summarize_elite_shot_makers(player_sdi)
    print("\nELITE SHOT MAKERS (High SDI + Positive Residual):")
    print(
        elite.head(10)[["player", "avg_sdi", "actual_fg_pct", "avg_xFG"]].to_string(
            index=False
        )
    )

    return df_with_sdi, player_sdi, elite
