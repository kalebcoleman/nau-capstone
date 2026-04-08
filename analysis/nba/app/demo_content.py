"""Poster-demo content and configuration for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

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
        "title": "NBA Distance GAM",
        "path": NBA_FIGURES_DIR / "nba_gam_distance_2014_2024.png",
        "caption": "Distance is the cleanest shared difficulty signal in basketball. The fitted curve shows how scoring odds change as attempts move away from the basket.",
    },
    {
        "title": "NHL Distance GAM",
        "path": NHL_FIGURES_DIR / "nhl_gam_distance_2014_2024.png",
        "caption": "The hockey distance curve drops even more sharply near the net, reflecting how quickly scoring probability decays with shooting distance.",
    },
]

GAM_EXTRA_GROUPS = {
    "NBA": [
        {
            "group_title": "NBA Context GAMs",
            "blurb": "These figures show which extra shot contexts mattered most once distance and court location were already in the model.",
            "figures": [
                {
                    "title": "Angle Effect",
                    "path": NBA_FIGURES_DIR / "gam_effect_angle.png",
                    "caption": "Shot angle captures how side-angle looks differ from straight-on attempts.",
                },
                {
                    "title": "Clock Effect",
                    "path": NBA_FIGURES_DIR / "gam_effect_clock.png",
                    "caption": "Clock pressure helps separate late-clock shots from easier early-possession looks.",
                },
                {
                    "title": "Period Effect",
                    "path": NBA_FIGURES_DIR / "gam_effect_period.png",
                    "caption": "Quarter effects capture broad game-state differences beyond location alone.",
                },
                {
                    "title": "Shot Type Effect",
                    "path": NBA_FIGURES_DIR / "gam_effect_shot_types.png",
                    "caption": "The model differentiates dunks, layups, hooks, floaters, and jumper families.",
                },
            ],
        },
        {
            "group_title": "NBA Spatial GAMs",
            "blurb": "These visualizations summarize the court surface learned by the model rather than a single one-dimensional effect.",
            "figures": [
                {
                    "title": "Spatial Probability Surface",
                    "path": NBA_FIGURES_DIR / "gam_spatial_probability.png",
                    "caption": "Predicted scoring probability across the half court.",
                },
                {
                    "title": "Spatial Tensor Surface",
                    "path": NBA_FIGURES_DIR / "gam_spatial_tensor.png",
                    "caption": "Tensor-product smooth showing how location contributes to the GAM.",
                },
            ],
        },
    ],
    "NHL": [
        {
            "group_title": "NHL Angle and Alternative Distance Views",
            "blurb": "These figures expand the hockey side beyond the main poster distance curve.",
            "figures": [
                {
                    "title": "Angle Effect",
                    "path": NHL_FIGURES_DIR / "nhl_gam_angle_2014_2024.png",
                    "caption": "Angle changes difficulty sharply in hockey because off-angle shots lose net visibility and clean shooting windows.",
                },
                {
                    "title": "Non-Empty-Net Distance",
                    "path": NHL_FIGURES_DIR / "nhl_gam_distance_non_empty_net_distance_only_2014_2024.png",
                    "caption": "Distance-only view restricted to non-empty-net shots.",
                },
                {
                    "title": "All-Shots Distance",
                    "path": NHL_FIGURES_DIR / "nhl_gam_distance_all_shots_distance_only_2014_2024.png",
                    "caption": "Distance-only view including all shots.",
                },
            ],
        },
        {
            "group_title": "NHL Comparison and Spline Variants",
            "blurb": "These extra semester outputs show how the distance story changes under alternate modeling cuts.",
            "figures": [
                {
                    "title": "Empty-Net Comparison",
                    "path": NHL_FIGURES_DIR / "nhl_gam_distance_empty_net_comparison_distance_only_2014_2024.png",
                    "caption": "Comparison between all-shot and non-empty-net distance fits.",
                },
                {
                    "title": "Spline Distance View",
                    "path": NHL_FIGURES_DIR / "nhl_spline_logistic_distance_2014_2024.png",
                    "caption": "Spline-based distance view used as a model comparison artifact.",
                },
                {
                    "title": "100-Foot Spline View",
                    "path": NHL_FIGURES_DIR / "nhl_spline_logistic_distance_2014_2024_100ft_view.png",
                    "caption": "The same spline fit cropped to the most interpretable scoring range.",
                },
            ],
        },
    ],
}
