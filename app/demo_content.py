"""Poster-demo content and configuration for the Streamlit app."""

from __future__ import annotations

from path_setup import NBA_ANALYSIS_DIR, NHL_ANALYSIS_DIR


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
GITHUB_REPO_URL = "https://github.com/kalebcoleman/nau-capstone"

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
    "Distance lowers scoring odds in both sports, but the headline distance fit differs by sport: NBA uses the spline view while NHL uses the full all-shot GAM.",
    "SDI separates high-difficulty volume shooters from efficient finishers and lets us compare actual scoring to expected scoring.",
    "Position-based clusters show that shot difficulty profiles are structured by role in both leagues.",
]

SDI_SUMMARY_PATHS = {
    "NBA": NBA_DATA_DIR / "nba_player_summary_2014_2024.csv",
    "NHL": NHL_DATA_DIR / "nhl_player_summary_2014_2024.csv",
}

SDI_FIGURE_SPECS = {
    "NBA": {
        "path": NBA_FIGURES_DIR / "nba_sdi_vs_actual_2014_2024.png",
        "caption": "Final NBA SDI poster scatter: volume in marker size and residual in color.",
    },
    "NHL": {
        "path": NHL_FIGURES_DIR / "nhl_sdi_vs_actual_2014_2024_opt1_raw_scaled.png",
        "caption": "Final NHL SDI poster scatter using the preferred opt1 residual color scale.",
    },
}

SDI_DEFAULT_MIN_ATTEMPTS = {
    "NBA": 500,
    "NHL": 250,
}

def _figure(
    title: str,
    path,
    caption: str,
    *,
    sport: str,
    factor_key: str,
    **extras: object,
) -> dict[str, object]:
    return {
        "title": title,
        "path": path,
        "caption": caption,
        "sport": sport,
        "factor_key": factor_key,
        **extras,
    }


def _group(group_title: str, blurb: str, figures: list[dict[str, object]]) -> dict[str, object]:
    return {
        "group_title": group_title,
        "blurb": blurb,
        "figures": figures,
    }


# Keep the explorer pinned to a curated GAM-only figure set instead of the
# broader regenerated manifest that mixes in unstable or misleading panels.
GAM_CORE_FIGURES = [
    _figure(
        "NBA Expected FG Distance Spline with 95% CI (2014-2024)",
        NBA_FIGURES_DIR / "nba_expected_fg_distance_spline_pdp_2014_2024.png",
        "Current NBA spline distance headline figure from the newer expected-FG export set.",
        sport="NBA",
        factor_key="distance_spline",
    ),
    _figure(
        "NHL Expected Goal Distance Effect with 95% CI (2014-2024)",
        NHL_FIGURES_DIR / "nhl_expected_goal_distance_pdp_2014_2024.png",
        "Current NHL headline GAM distance panel, kept alongside empty-net comparison support views.",
        sport="NHL",
        factor_key="distance",
    ),
]

GAM_EXTRA_GROUPS = {
    "NBA": [
        _group(
            "Distance Variants",
            "Stable legacy NBA distance panels kept for comparison against the spline headline figure.",
            [
                _figure(
                    "NBA GAM Distance Effect with 95% CI (2014-2024)",
                    NBA_FIGURES_DIR / "nba_gam_distance_2014_2024.png",
                    "Legacy full-model NBA GAM distance panel used as the alternate support view.",
                    sport="NBA",
                    factor_key="distance",
                ),
            ],
        ),
        _group(
            "Continuous Context",
            "Legacy NBA continuous-effect panels for angle, clock, and period context.",
            [
                _figure(
                    "NBA GAM Angle Effect (Legacy)",
                    NBA_FIGURES_DIR / "nba_gam_angle_2014_2024.png",
                    "Legacy angle effect panel from the poster-era NBA GAM workflow.",
                    sport="NBA",
                    factor_key="angle",
                ),
                _figure(
                    "NBA GAM Clock Effect (Legacy)",
                    NBA_FIGURES_DIR / "nba_gam_clock_2014_2024.png",
                    "Legacy game-clock effect panel from the poster-era NBA GAM workflow.",
                    sport="NBA",
                    factor_key="clock",
                ),
                _figure(
                    "NBA GAM Period Effect (Legacy)",
                    NBA_FIGURES_DIR / "nba_gam_period_2014_2024.png",
                    "Legacy period effect panel from the poster-era NBA GAM workflow.",
                    sport="NBA",
                    factor_key="period",
                ),
            ],
        ),
    ],
    "NHL": [
        _group(
            "Distance Variants",
            "NHL support panels use the current expected-goal distance exports: an overlay that shows how the all-shot tail diverges from the non-empty-net refit, the standalone non-empty-net panel, and the spline view for cross-sport comparison.",
            [
                _figure(
                    "NHL Expected Goal Distance Comparison: All Shots vs Non-Empty-Net Shots (2014-2024)",
                    None,
                    "Overlay of the current full-model NHL distance effects so the long-distance empty-net tail is visible in one panel.",
                    sport="NHL",
                    factor_key="distance",
                    renderer="nhl_distance_comparison",
                    data_paths={
                        "all_shots": NHL_DATA_DIR / "nhl_expected_goal_distance_pdp_2014_2024.csv",
                        "non_empty_net": NHL_DATA_DIR / "nhl_expected_goal_distance_non_empty_net_pdp_2014_2024.csv",
                    },
                    markers=[
                        {"label": "Crease Edge", "value": 6.0, "color": "#2E8B57", "linestyle": ":"},
                        {"label": "Top Circles", "value": 33.0, "color": "#E09F3E", "linestyle": ":"},
                        {"label": "Blue Line", "value": 60.0, "color": "#8B1E3F", "linestyle": ":"},
                        {"label": "Center Red Line", "value": 89.0, "color": "#6A4C93", "linestyle": ":"},
                    ],
                ),
                _figure(
                    "NHL Expected Goal Distance Effect (Non-Empty-Net Shots) with 95% CI (2014-2024)",
                    NHL_FIGURES_DIR / "nhl_expected_goal_distance_non_empty_net_pdp_2014_2024.png",
                    "Full-model NHL distance GAM refit after removing empty-net attempts so the long-distance tail stays comparable to the headline panel.",
                    sport="NHL",
                    factor_key="distance",
                ),
                _figure(
                    "NHL Expected Goal Distance Spline with 95% CI (2014-2024)",
                    NHL_FIGURES_DIR / "nhl_expected_goal_distance_spline_pdp_2014_2024.png",
                    "Spline-only NHL distance view kept beside the NBA spline for matched cross-sport comparison.",
                    sport="NHL",
                    factor_key="distance_spline",
                ),
            ],
        ),
        _group(
            "Continuous Context",
            "NHL continuous GAM panels for angle, clock, and period context from the newer expected-goal export set.",
            [
                _figure(
                    "NHL Expected Goal Angle Effect with 95% CI (2014-2024)",
                    NHL_FIGURES_DIR / "nhl_expected_goal_angle_pdp_2014_2024.png",
                    "NHL full-model angle effect from the newer expected-goal GAM export set.",
                    sport="NHL",
                    factor_key="angle",
                ),
                _figure(
                    "NHL Expected Goal Clock Effect with 95% CI (2014-2024)",
                    NHL_FIGURES_DIR / "nhl_expected_goal_clock_pdp_2014_2024.png",
                    "NHL full-model within-period time effect from the newer expected-goal GAM export set.",
                    sport="NHL",
                    factor_key="clock",
                ),
                _figure(
                    "NHL Expected Goal Period Effect with 95% CI (2014-2024)",
                    NHL_FIGURES_DIR / "nhl_expected_goal_period_pdp_2014_2024.png",
                    "NHL full-model period effect from the newer expected-goal GAM export set.",
                    sport="NHL",
                    factor_key="period",
                ),
            ],
        ),
    ],
}
