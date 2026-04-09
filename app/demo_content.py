"""Poster-demo content and configuration for the Streamlit app."""

from __future__ import annotations

from path_setup import NBA_ANALYSIS_DIR, NHL_ANALYSIS_DIR
from analysis.gam_explorer_config import GAM_EXPLORER_FIGURES


NBA_DATA_DIR = NBA_ANALYSIS_DIR / "data"
NBA_FIGURES_DIR = NBA_ANALYSIS_DIR / "figures"
NHL_DATA_DIR = NHL_ANALYSIS_DIR / "data"
NHL_FIGURES_DIR = NHL_ANALYSIS_DIR / "figures"

APP_TITLE = "Cross-Sport Shot Analysis"
APP_SUBTITLE = (
    "Poster companion for the NAU capstone project comparing Shot Difficulty Index "
    "and GAM-based scoring difficulty across the NBA and NHL."
)
ABSTRACT_TEXT = (
    "This demo mirrors the final poster: a matched 2014–2024 cross-sport view of shot "
    "difficulty, expected scoring, and player efficiency relative to expectation."
)

NAVIGATION_PAGES = [
    {"path": "overview_page.py", "title": "Overview", "icon": "🏠", "default": True},
    {"path": "sdi_explorer_page.py", "title": "SDI Explorer", "icon": "📊", "default": False},
    {"path": "gam_explorer_page.py", "title": "GAM Explorer", "icon": "📈", "default": False},
]

MODEL_MEASURE_BULLETS = [
    "SDI summarizes how difficult a player's average shot attempts are.",
    "The GAM figures isolate how specific shot characteristics shift scoring odds while holding other context fixed.",
    "The cross-sport comparison focuses on shared ideas: distance, angle, timing, shot type, and situational context.",
]

MAIN_RESULT_BULLETS = [
    "Distance lowers scoring odds in both sports, but the shape of that decline differs between NBA shots and NHL shots.",
    "SDI separates high-difficulty volume shooters from efficient finishers and lets us compare actual scoring to expected scoring.",
    "Position-based clusters show that shot difficulty profiles are structured by role in both leagues.",
]

SDI_SUMMARY_PATHS = {
    "NBA": NBA_DATA_DIR / "nba_player_summary_2014_2024.csv",
    "NHL": NHL_DATA_DIR / "nhl_player_summary_2014_2024.csv",
}

SDI_DEFAULT_MIN_ATTEMPTS = {
    "NBA": 500,
    "NHL": 250,
}

GAM_CORE_FIGURES = [
    {
        "title": fig["title"],
        "path": fig["figure_path"],
        "caption": fig["caption"],
        "sport": fig["sport"],
        "factor_key": fig["factor_key"],
    }
    for fig in GAM_EXPLORER_FIGURES
    if fig["group"] == "core"
]

GAM_EXTRA_GROUPS = {"NBA": [], "NHL": []}
for sport in ("NBA", "NHL"):
    sport_figures = [fig for fig in GAM_EXPLORER_FIGURES if fig["sport"] == sport and fig["group"] != "core"]
    seen_groups: set[str] = set()
    for fig in sport_figures:
        group_key = str(fig["group"])
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        grouped = [item for item in sport_figures if item["group"] == group_key]
        GAM_EXTRA_GROUPS[sport].append(
            {
                "group_title": grouped[0]["group_title"],
                "blurb": grouped[0]["group_blurb"],
                "figures": [
                    {
                        "title": item["title"],
                        "path": item["figure_path"],
                        "caption": item["caption"],
                        "factor_key": item["factor_key"],
                        "plot_type": item["plot_type"],
                    }
                    for item in grouped
                ],
            }
        )
