"""NBA-style normalized SDI analysis for NHL shot data.

This script is intentionally separate from the existing NHL SDI workflow.
It mirrors the NBA SDI structure more closely by:

- using normalized 0-1 components
- keeping weights that sum to 1
- aggregating average SDI by player
- comparing player SDI against actual goal rate and expected goal rate

The goal is not to replace the original NHL SDI. It provides an alternate
version so the shape of the SDI-vs-actual plot can be compared directly
against the NBA-style construction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_SHOTS = 100

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
APP_DATA_DIR = DATA_DIR / "app_data"
FIGURES_DIR = SCRIPT_DIR / "figures"
RAW_DATA_PATH = DATA_DIR / "shots_2007-2024.csv"
EXPORT_DATA_PATH = APP_DATA_DIR / "nhl_shots_2014_2024.csv.gz"
OUTPUT_SUMMARY_PATH = SCRIPT_DIR / "data" / "nhl_player_sdi_nba_style_2014_2024.csv"
OUTPUT_FIGURE_PATH = FIGURES_DIR / "nhl_sdi_nba_style_vs_actual_2014_2024.png"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUT_SUMMARY_PATH.parent.mkdir(exist_ok=True)
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_export_from_raw() -> Path:
    """Create the historical NHL export if only the raw MoneyPuck file exists."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing NHL export at {EXPORT_DATA_PATH} and raw file at {RAW_DATA_PATH}."
        )

    cols_to_keep = [
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

    chunks = []
    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=200_000):
        season_num = pd.to_numeric(chunk["season"], errors="coerce")
        chunk = chunk[season_num.between(2014, 2024)].copy()
        if chunk.empty:
            continue
        current_cols = [col for col in cols_to_keep if col in chunk.columns]
        chunks.append(chunk[current_cols])

    if not chunks:
        raise ValueError("No NHL rows found for seasons 2014-2024.")

    out = pd.concat(chunks, ignore_index=True)
    out.to_csv(EXPORT_DATA_PATH, index=False, compression="gzip")
    return EXPORT_DATA_PATH


def load_shot_data() -> pd.DataFrame:
    """Load the historical NHL export, building it first if needed."""
    ensure_dirs()
    if not EXPORT_DATA_PATH.exists():
        build_export_from_raw()
    return pd.read_csv(EXPORT_DATA_PATH, compression="gzip")


def compute_sdi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute an NBA-style normalized SDI for NHL shots."""
    out = df.copy()

    numeric_cols = [
        "goal",
        "xGoal",
        "shotDistance",
        "shotAngle",
        "shotRebound",
        "shotRush",
        "shotGoalieFroze",
        "period",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(
        subset=["goal", "xGoal", "shotDistance", "shotAngle", "period", "shooterName"]
    ).copy()

    shot_type = out["shotType"].fillna("").astype(str).str.upper()

    shot_type_difficulty = {
        "TIP": 0.20,
        "DEFL": 0.25,
        "BACK": 0.35,
        "SNAP": 0.40,
        "WRIST": 0.45,
        "WRAP": 0.55,
        "SLAP": 0.75,
    }

    out["sdi_distance"] = out["shotDistance"].clip(0, 90) / 90.0
    out["sdi_angle"] = out["shotAngle"].abs().clip(0, 90) / 90.0
    out["sdi_shot_type"] = shot_type.map(shot_type_difficulty).fillna(0.50)

    # Hockey has no shot clock, so this substitutes a possession-context term.
    # Rebounds and rush chances are usually easier looks, while goalie-frozen
    # situations tend to be more settled and harder.
    out["sdi_context"] = (
        0.55
        + 0.25 * out["shotGoalieFroze"].fillna(0)
        - 0.20 * out["shotRebound"].fillna(0)
        - 0.10 * out["shotRush"].fillna(0)
    ).clip(0, 1)

    # Late periods stand in for a light pressure/context term to keep the NHL
    # recipe parallel to the NBA script's extra temporal component.
    out["sdi_period"] = ((out["period"].clip(1, 4) - 1) / 3.0).clip(0, 1)

    out["SDI"] = (
        0.30 * out["sdi_distance"]
        + 0.20 * out["sdi_angle"]
        + 0.20 * out["sdi_shot_type"]
        + 0.15 * out["sdi_context"]
        + 0.15 * out["sdi_period"]
    )
    return out


def aggregate_player_sdi(df: pd.DataFrame, min_shots: int = MIN_SHOTS) -> pd.DataFrame:
    """Aggregate normalized SDI and outcome metrics by player."""
    player_sdi = (
        df.groupby("shooterName")
        .agg({"SDI": "mean", "xGoal": "mean", "goal": ["mean", "count"]})
        .reset_index()
    )
    player_sdi.columns = ["player", "avg_sdi", "avg_xG", "actual_goal_pct", "attempts"]
    player_sdi = player_sdi[player_sdi["attempts"] >= min_shots].copy()
    player_sdi["residual"] = player_sdi["actual_goal_pct"] - player_sdi["avg_xG"]
    return player_sdi.sort_values("avg_sdi", ascending=False)


def select_labels(player_sdi: pd.DataFrame) -> pd.DataFrame:
    """Select a small set of informative player labels for the scatter plot."""
    candidates = pd.concat(
        [
            player_sdi.nlargest(3, "residual"),
            player_sdi.nsmallest(3, "residual"),
            player_sdi.nlargest(3, "avg_sdi"),
            player_sdi.nlargest(3, "attempts"),
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

    labels: dict[str, str] = {}
    for player in players:
        parts = str(player).split()
        last = parts[-1] if parts else str(player)
        labels[player] = last if len(last_names[last]) == 1 else str(player)
    return labels


def plot_sdi_vs_actual(player_sdi: pd.DataFrame, output_path: Path) -> None:
    """Create an NBA-style SDI vs actual goal rate scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        player_sdi["avg_sdi"],
        player_sdi["actual_goal_pct"] * 100,
        s=np.clip(player_sdi["attempts"] / 6, 18, 260),
        alpha=0.62,
        c=player_sdi["residual"],
        cmap="RdYlGn",
        vmin=-0.10,
        vmax=0.10,
        edgecolors="black",
        linewidths=0.45,
    )

    z = np.polyfit(player_sdi["avg_sdi"], player_sdi["actual_goal_pct"] * 100, 1)
    x_sorted = np.sort(player_sdi["avg_sdi"].to_numpy())
    ax.plot(x_sorted, np.poly1d(z)(x_sorted), linestyle="--", color="#3A6EA5", linewidth=2)

    ax.axhline(
        player_sdi["actual_goal_pct"].median() * 100,
        color="gray",
        linestyle="--",
        alpha=0.35,
    )
    ax.axvline(player_sdi["avg_sdi"].median(), color="gray", linestyle="--", alpha=0.35)

    label_df = select_labels(player_sdi)
    text_map = shorten_player_labels(label_df["player"].tolist())
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
            xy=(row["avg_sdi"], row["actual_goal_pct"] * 100),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            alpha=0.95,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.72,
            ),
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6, alpha=0.6),
        )

    ax.set_xlabel("Average Shot Difficulty Index (NBA-Style NHL SDI)", fontsize=12)
    ax.set_ylabel("Actual Goal %", fontsize=12)
    ax.set_title(
        "NHL Shot Difficulty vs Actual Goal Rate\n"
        "(Normalized NBA-Style SDI, Size = Volume, Color = Goal% Residual)",
        fontsize=14,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Residual (Actual - Expected)", fontsize=10)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_sdi_analysis(min_shots: int = MIN_SHOTS) -> pd.DataFrame:
    """Run the alternate NHL SDI workflow and persist outputs."""
    df = load_shot_data()
    df_with_sdi = compute_sdi(df)
    player_sdi = aggregate_player_sdi(df_with_sdi, min_shots=min_shots)
    player_sdi.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    plot_sdi_vs_actual(player_sdi, OUTPUT_FIGURE_PATH)

    print(f"Saved summary: {OUTPUT_SUMMARY_PATH}")
    print(f"Saved figure: {OUTPUT_FIGURE_PATH}")
    print(player_sdi.head(15).to_string(index=False))
    return player_sdi


if __name__ == "__main__":
    run_sdi_analysis()
