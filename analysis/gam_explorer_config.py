"""Shared configuration for canonical GAM explorer artifacts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
NBA_ANALYSIS_DIR = ANALYSIS_DIR / "nba"
NHL_ANALYSIS_DIR = ANALYSIS_DIR / "nhl"

WINDOW_LABEL = "2014-2024"

SPORT_META = {
    "NBA": {
        "target_label": "Expected FG",
        "short_target": "expected_fg",
        "data_dir": NBA_ANALYSIS_DIR / "data",
        "figures_dir": NBA_ANALYSIS_DIR / "figures",
    },
    "NHL": {
        "target_label": "Expected Goal",
        "short_target": "expected_goal",
        "data_dir": NHL_ANALYSIS_DIR / "data",
        "figures_dir": NHL_ANALYSIS_DIR / "figures",
    },
}

FACTOR_LABELS = {
    "distance": "Distance",
    "angle": "Angle",
    "clock": "Clock",
    "period": "Period",
    "shot_type": "Shot Type",
    "clutch": "Clutch",
    "rebound": "Rebound",
    "rush": "Rush",
    "goalie_froze": "Goalie Froze",
    "empty_net": "Empty Net",
    "spatial": "Spatial Surface",
}

PLOT_GROUPS = {
    "core": "Core Distance",
    "continuous": "Continuous Context",
    "discrete": "Discrete Context",
    "spatial": "Spatial Surface",
}

GROUP_BLURBS = {
    "core": "The headline expected FG/xG distance curves for the demo.",
    "continuous": "One-dimensional partial dependence plots for the continuous or ordered GAM factors.",
    "discrete": "Compact summaries for binary and categorical controls kept in the full GAM.",
    "spatial": "Two-dimensional location effects learned by the full model.",
}

FACTOR_SPECS = [
    {
        "sport": "NBA",
        "factor_key": "distance",
        "plot_type": "continuous_pdp",
        "group": "core",
        "display_order": 10,
        "caption": "Full-window expected FG GAM partial dependence for shot distance with basketball landmarks.",
    },
    {
        "sport": "NHL",
        "factor_key": "distance",
        "plot_type": "continuous_pdp",
        "group": "core",
        "display_order": 20,
        "caption": "Full-window expected goal GAM partial dependence for all-shot distance with rink landmarks.",
    },
    {
        "sport": "NBA",
        "factor_key": "angle",
        "plot_type": "continuous_pdp",
        "group": "continuous",
        "display_order": 30,
        "caption": "Expected FG shift as attempts move off-center relative to the basket.",
    },
    {
        "sport": "NBA",
        "factor_key": "clock",
        "plot_type": "continuous_pdp",
        "group": "continuous",
        "display_order": 40,
        "caption": "Expected FG effect across the shot clock context within each period.",
    },
    {
        "sport": "NBA",
        "factor_key": "period",
        "plot_type": "continuous_pdp",
        "group": "continuous",
        "display_order": 50,
        "caption": "Expected FG effect across regulation and overtime periods.",
    },
    {
        "sport": "NHL",
        "factor_key": "angle",
        "plot_type": "continuous_pdp",
        "group": "continuous",
        "display_order": 60,
        "caption": "Expected goal shift as shooting angle moves away from a straight-on look.",
    },
    {
        "sport": "NHL",
        "factor_key": "clock",
        "plot_type": "continuous_pdp",
        "group": "continuous",
        "display_order": 70,
        "caption": "Expected goal effect across the time remaining within a period.",
    },
    {
        "sport": "NHL",
        "factor_key": "period",
        "plot_type": "continuous_pdp",
        "group": "continuous",
        "display_order": 80,
        "caption": "Expected goal effect across periods after holding shot context fixed.",
    },
    {
        "sport": "NBA",
        "factor_key": "shot_type",
        "plot_type": "discrete_summary",
        "group": "discrete",
        "display_order": 90,
        "caption": "Model-implied log-odds shifts for the mutually exclusive NBA shot type families.",
    },
    {
        "sport": "NBA",
        "factor_key": "clutch",
        "plot_type": "discrete_summary",
        "group": "discrete",
        "display_order": 100,
        "caption": "Expected FG change for late-game clutch context relative to non-clutch attempts.",
    },
    {
        "sport": "NHL",
        "factor_key": "shot_type",
        "plot_type": "discrete_summary",
        "group": "discrete",
        "display_order": 110,
        "caption": "Model-implied log-odds shifts for NHL shot type families in the full expected goal GAM.",
    },
    {
        "sport": "NHL",
        "factor_key": "rebound",
        "plot_type": "discrete_summary",
        "group": "discrete",
        "display_order": 120,
        "caption": "Expected goal change for rebound shots relative to settled attempts.",
    },
    {
        "sport": "NHL",
        "factor_key": "rush",
        "plot_type": "discrete_summary",
        "group": "discrete",
        "display_order": 130,
        "caption": "Expected goal change for rush chances relative to non-rush attempts.",
    },
    {
        "sport": "NHL",
        "factor_key": "goalie_froze",
        "plot_type": "discrete_summary",
        "group": "discrete",
        "display_order": 140,
        "caption": "Expected goal change when the preceding state is tagged as goalie-froze context.",
    },
    {
        "sport": "NHL",
        "factor_key": "empty_net",
        "plot_type": "discrete_summary",
        "group": "discrete",
        "display_order": 150,
        "caption": "Expected goal change for empty-net attempts in the all-shots full model.",
    },
    {
        "sport": "NBA",
        "factor_key": "spatial",
        "plot_type": "spatial_surface",
        "group": "spatial",
        "display_order": 160,
        "caption": "Full-model spatial log-odds surface across the half court after controlling for other shot context.",
    },
    {
        "sport": "NHL",
        "factor_key": "spatial",
        "plot_type": "spatial_surface",
        "group": "spatial",
        "display_order": 170,
        "caption": "Full-model spatial log-odds surface across the offensive zone after controlling for other shot context.",
    },
]

MARKERS = {
    ("NBA", "distance"): [
        {"label": "Restricted Area", "value": 4.0, "color": "#2E8B57", "linestyle": ":"},
        {"label": "Corner 3", "value": 22.0, "color": "#C97C00", "linestyle": ":"},
        {"label": "Arc 3", "value": 23.75, "color": "#8B1E3F", "linestyle": ":"},
    ],
    ("NBA", "angle"): [
        {"label": "Straight On", "value": 0.0, "color": "#6A4C93", "linestyle": ":"},
    ],
    ("NBA", "clock"): [
        {"label": "Clutch Time", "value": 120.0, "color": "#8B1E3F", "linestyle": ":"},
    ],
    ("NHL", "distance"): [
        {"label": "Crease Edge", "value": 6.0, "color": "#2E8B57", "linestyle": ":"},
        {"label": "Top Circles", "value": 33.0, "color": "#E09F3E", "linestyle": ":"},
        {"label": "Blue Line", "value": 60.0, "color": "#8B1E3F", "linestyle": ":"},
        {"label": "Center Red Line", "value": 89.0, "color": "#6A4C93", "linestyle": ":"},
    ],
    ("NHL", "angle"): [
        {"label": "Straight On", "value": 0.0, "color": "#6A4C93", "linestyle": ":"},
    ],
    ("NHL", "clock"): [
        {"label": "Start of Period", "value": 1200.0, "color": "#2E8B57", "linestyle": ":"},
        {"label": "Mid Period", "value": 600.0, "color": "#E09F3E", "linestyle": ":"},
        {"label": "Final Minute", "value": 60.0, "color": "#8B1E3F", "linestyle": ":"},
    ],
}


def _build_title(sport: str, factor_key: str, plot_type: str) -> str:
    target_label = SPORT_META[sport]["target_label"]
    factor_label = FACTOR_LABELS[factor_key]
    if plot_type == "spatial_surface":
        return f"{sport} {target_label} {factor_label} ({WINDOW_LABEL})"
    if plot_type == "discrete_summary":
        return f"{sport} {target_label} {factor_label} Summary ({WINDOW_LABEL})"
    return f"{sport} {target_label} {factor_label} Effect with 95% CI ({WINDOW_LABEL})"


def _build_stem(sport: str, factor_key: str, plot_type: str) -> str:
    short_target = SPORT_META[sport]["short_target"]
    if plot_type == "spatial_surface":
        suffix = "spatial_surface"
    elif plot_type == "discrete_summary":
        suffix = f"{factor_key}_summary"
    else:
        suffix = f"{factor_key}_pdp"
    return f"{sport.lower()}_{short_target}_{suffix}_{WINDOW_LABEL.replace('-', '_')}"


def build_figure_spec(raw_spec: dict[str, object]) -> dict[str, object]:
    sport = str(raw_spec["sport"])
    factor_key = str(raw_spec["factor_key"])
    plot_type = str(raw_spec["plot_type"])
    stem = _build_stem(sport, factor_key, plot_type)
    base = SPORT_META[sport]
    return {
        **raw_spec,
        "title": _build_title(sport, factor_key, plot_type),
        "target_label": base["target_label"],
        "season_window": WINDOW_LABEL,
        "data_path": Path(base["data_dir"]) / f"{stem}.csv",
        "figure_path": Path(base["figures_dir"]) / f"{stem}.png",
        "markers": MARKERS.get((sport, factor_key), []),
        "group_title": PLOT_GROUPS[str(raw_spec["group"])],
        "group_blurb": GROUP_BLURBS[str(raw_spec["group"])],
    }


GAM_EXPLORER_FIGURES = tuple(
    build_figure_spec(spec)
    for spec in sorted(FACTOR_SPECS, key=lambda item: int(item["display_order"]))
)

