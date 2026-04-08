"""Shared NBA feature definitions, engineering helpers, and audit metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
TOP_LEVEL_ARCHIVE_PATH = REPO_ROOT / "data" / "nba_shots_2014_2024.csv.gz"
ANALYSIS_ARCHIVE_PATH = ANALYSIS_DIR / "data" / "nba_shots_2014_2024.csv.gz"
TRAINING_DATA_PATH = ANALYSIS_DIR / "data" / "nba_shots_training.csv.gz"
PLAYER_SUMMARY_PATH = ANALYSIS_DIR / "data" / "player_summary.csv"

REGULATION_PERIOD_SECONDS = 720
OVERTIME_PERIOD_SECONDS = 300
REGULATION_GAME_MINUTES = 48.0

RAW_DISCOVERY_PRIORITY_COLUMNS = [
    "GRID_TYPE",
    "GAME_ID",
    "GAME_EVENT_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_NAME",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "EVENT_TYPE",
    "ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
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
    "league",
    "source",
    "event_num",
    "shot_id",
]

XFG_NUMERIC_FEATURES = [
    "LOC_X",
    "LOC_Y",
    "shot_distance_feet",
    "shot_angle",
    "PERIOD",
    "seconds_in_period",
    "is_clutch",
    "is_layup",
    "is_dunk",
    "is_jump_shot",
    "is_hook",
    "is_floater",
]
XFG_CATEGORICAL_FEATURES = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA"]
XFG_FEATURES = XFG_NUMERIC_FEATURES + XFG_CATEGORICAL_FEATURES

GAM_FEATURES = [
    "LOC_X",
    "LOC_Y",
    "shot_distance_feet",
    "shot_angle",
    "seconds_in_period",
    "PERIOD",
    "is_dunk",
    "is_layup",
    "is_hook",
    "is_floater",
    "is_jump_shot_2",
    "is_jump_shot_3",
    "is_clutch",
]

APP_USAGE_FEATURES = [
    "shot_distance_feet",
    "game_minutes_elapsed",
    "shot_type_cat",
    "SHOT_ZONE_BASIC",
    "shot_angle",
    "abs_shot_angle",
]

SDI_COMPONENT_WEIGHTS = {
    "sdi_distance": 0.30,
    "sdi_game_clock": 0.20,
    "sdi_shot_type": 0.20,
    "sdi_zone": 0.15,
}
SDI_ACTIVE_COMPONENTS = list(SDI_COMPONENT_WEIGHTS.keys())
SDI_AUDIT_ONLY_COMPONENTS = ["sdi_angle"]

CURRENT_FEATURE_USAGE = {
    "xFG": set(XFG_FEATURES),
    "GAM": set(GAM_FEATURES),
    "SDI": {
        "shot_distance_feet",
        "game_minutes_elapsed",
        "shot_type_cat",
        "SHOT_ZONE_BASIC",
    },
    "app": {
        "shot_distance_feet",
        "game_minutes_elapsed",
        "shot_type_cat",
        "SHOT_ZONE_BASIC",
        "shot_angle",
        "abs_shot_angle",
        "xP_prob",
    },
}

POSTER_MODEL_SNAPSHOT = [
    {
        "concept": "Spatial location",
        "nba_variables": "LOC_X, LOC_Y (GAM)",
        "nhl_variables": "xCord, yCord (GAM)",
        "why_it_matters": "Lets each GAM learn where shot quality changes across the floor or rink.",
    },
    {
        "concept": "Shot distance",
        "nba_variables": "shot_distance_feet (GAM + SDI)",
        "nhl_variables": "shotDistance (GAM + SDI)",
        "why_it_matters": "Primary shared difficulty signal; farther attempts are harder in both sports.",
    },
    {
        "concept": "Shot angle",
        "nba_variables": "shot_angle (GAM)",
        "nhl_variables": "shotAngle (GAM + SDI)",
        "why_it_matters": "Captures shooting geometry beyond distance alone.",
    },
    {
        "concept": "Game timing",
        "nba_variables": "seconds_in_period, PERIOD, game_minutes_elapsed",
        "nhl_variables": "period",
        "why_it_matters": "Separates shot quality from late-game or late-period pressure.",
    },
    {
        "concept": "Shot type",
        "nba_variables": "shot_type_cat; dunk/layup/hook/floater/jump indicators",
        "nhl_variables": "shotType; wrist/snap/slap/backhand indicators",
        "why_it_matters": "Distinguishes easy finishes from harder attempt families.",
    },
    {
        "concept": "Extra context",
        "nba_variables": "SHOT_ZONE_BASIC, is_clutch",
        "nhl_variables": "shotRebound, shotGoalieFroze, shotRush",
        "why_it_matters": "Keeps the final index/model aware of context that changes difficulty without changing distance.",
    },
]

TEAM_NAME_TO_TRICODE = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

ENGINEERED_FEATURE_METADATA = [
    {
        "variable_name": "shot_distance_feet",
        "engineered_from": "SHOT_DISTANCE, LOC_X, LOC_Y",
        "current_formula": "SHOT_DISTANCE else sqrt(LOC_X^2 + LOC_Y^2) / 10",
        "used_in_xfg": True,
        "used_in_gam": True,
        "used_in_sdi": True,
        "used_in_app": True,
        "intended_sdi_direction": "higher = harder",
        "notes": "Primary distance difficulty signal.",
    },
    {
        "variable_name": "shot_angle",
        "engineered_from": "LOC_X, LOC_Y",
        "current_formula": "atan2(LOC_X, clip(LOC_Y, lower=1))",
        "used_in_xfg": True,
        "used_in_gam": True,
        "used_in_sdi": False,
        "used_in_app": True,
        "intended_sdi_direction": "audit only",
        "notes": "0 = straight on; corner shots approach +/- pi/2.",
    },
    {
        "variable_name": "abs_shot_angle",
        "engineered_from": "shot_angle",
        "current_formula": "abs(shot_angle)",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": True,
        "intended_sdi_direction": "audit only",
        "notes": "High values collide with corner-three efficiency; do not use in SDI by default.",
    },
    {
        "variable_name": "period_seconds_remaining",
        "engineered_from": "MINUTES_REMAINING, SECONDS_REMAINING",
        "current_formula": "MINUTES_REMAINING * 60 + SECONDS_REMAINING",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "lower = harder",
        "notes": "Current pipeline aliases this as seconds_in_period.",
    },
    {
        "variable_name": "period_seconds_elapsed",
        "engineered_from": "PERIOD, MINUTES_REMAINING, SECONDS_REMAINING",
        "current_formula": "period_length_seconds - period_seconds_remaining",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "higher = harder",
        "notes": "Preferred quarter-progression variable for SDI review.",
    },
    {
        "variable_name": "period_elapsed_pct",
        "engineered_from": "period_seconds_elapsed, period_length_seconds",
        "current_formula": "period_seconds_elapsed / period_length_seconds",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "higher = harder",
        "notes": "Normalized period progression.",
    },
    {
        "variable_name": "game_seconds_elapsed",
        "engineered_from": "PERIOD, period_seconds_elapsed",
        "current_formula": "sum(completed periods) + period_seconds_elapsed",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "higher = harder",
        "notes": "Captures full-game progression including overtime.",
    },
    {
        "variable_name": "game_minutes_elapsed",
        "engineered_from": "game_seconds_elapsed",
        "current_formula": "game_seconds_elapsed / 60",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": True,
        "used_in_app": True,
        "intended_sdi_direction": "higher = harder",
        "notes": "Default SDI game clock variable running from 0 to 48+.",
    },
    {
        "variable_name": "seconds_in_period",
        "engineered_from": "period_seconds_remaining",
        "current_formula": "legacy alias of period_seconds_remaining",
        "used_in_xfg": True,
        "used_in_gam": True,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "lower = harder",
        "notes": "Kept for xFG/GAM compatibility.",
    },
    {
        "variable_name": "is_clutch",
        "engineered_from": "PERIOD, seconds_in_period",
        "current_formula": "int(PERIOD >= 4 and seconds_in_period <= 120)",
        "used_in_xfg": True,
        "used_in_gam": True,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "not applicable",
        "notes": "Current late-game context flag.",
    },
    {
        "variable_name": "shot_type_cat",
        "engineered_from": "ACTION_TYPE, SHOT_TYPE",
        "current_formula": "exclusive action-family labels including dunk, layup, hook, floater, 2pt_jump, 3pt_jump",
        "used_in_xfg": False,
        "used_in_gam": True,
        "used_in_sdi": True,
        "used_in_app": True,
        "intended_sdi_direction": "category-dependent",
        "notes": "Used for SDI difficulty scoring and shot-type diagnostics.",
    },
    {
        "variable_name": "is_corner_three",
        "engineered_from": "SHOT_ZONE_BASIC",
        "current_formula": "int(SHOT_ZONE_BASIC in {'Left Corner 3', 'Right Corner 3'})",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "category-dependent",
        "notes": "Used to audit the angle sign reversal.",
    },
    {
        "variable_name": "is_above_break_three",
        "engineered_from": "SHOT_ZONE_BASIC",
        "current_formula": "int(SHOT_ZONE_BASIC == 'Above the Break 3')",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "category-dependent",
        "notes": "Reference group for angle/corner comparisons.",
    },
    {
        "variable_name": "is_midrange",
        "engineered_from": "SHOT_ZONE_BASIC",
        "current_formula": "int(SHOT_ZONE_BASIC == 'Mid-Range')",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "category-dependent",
        "notes": "Candidate SDI indicator.",
    },
    {
        "variable_name": "is_restricted_area",
        "engineered_from": "SHOT_ZONE_BASIC",
        "current_formula": "int(SHOT_ZONE_BASIC == 'Restricted Area')",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "lower = harder",
        "notes": "Easy-shot reference flag.",
    },
    {
        "variable_name": "distance_bucket",
        "engineered_from": "shot_distance_feet",
        "current_formula": "cut shot_distance_feet into ordered basketball distance bins",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "higher bucket = harder",
        "notes": "Candidate grouped diagnostic feature.",
    },
    {
        "variable_name": "zone_range_bucket",
        "engineered_from": "SHOT_ZONE_RANGE",
        "current_formula": "normalized text bucket from SHOT_ZONE_RANGE",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "category-dependent",
        "notes": "More granular than SHOT_ZONE_BASIC for discovery.",
    },
    {
        "variable_name": "is_end_of_period",
        "engineered_from": "period_seconds_remaining",
        "current_formula": "int(period_seconds_remaining <= 30)",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "higher = harder",
        "notes": "Candidate late-clock pressure flag.",
    },
    {
        "variable_name": "is_end_of_game",
        "engineered_from": "PERIOD, seconds_in_period",
        "current_formula": "int(PERIOD >= 4 and seconds_in_period <= 120)",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "higher = harder",
        "notes": "Game-context pressure flag; parallels clutch.",
    },
    {
        "variable_name": "home_indicator",
        "engineered_from": "TEAM_NAME, HTM, VTM",
        "current_formula": "int(team tricode from TEAM_NAME matches HTM)",
        "used_in_xfg": False,
        "used_in_gam": False,
        "used_in_sdi": False,
        "used_in_app": False,
        "intended_sdi_direction": "audit only",
        "notes": "Candidate context feature from the full archive.",
    },
]


def period_length_seconds(period: pd.Series) -> pd.Series:
    period_num = pd.to_numeric(period, errors="coerce")
    return pd.Series(
        np.where(period_num.fillna(0).le(4), REGULATION_PERIOD_SECONDS, OVERTIME_PERIOD_SECONDS),
        index=period.index,
        dtype=float,
    )


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _derive_home_indicator(team_name: pd.Series, htm: pd.Series, vtm: pd.Series) -> pd.Series:
    tricode = team_name.astype(str).map(TEAM_NAME_TO_TRICODE)
    home = np.where(
        tricode.notna() & htm.astype(str).eq(tricode),
        1.0,
        np.where(tricode.notna() & vtm.astype(str).eq(tricode), 0.0, np.nan),
    )
    return pd.Series(home, index=team_name.index, dtype=float)


def engineer_nba_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add shared NBA engineered fields without dropping original columns."""
    out = _coerce_numeric(
        df,
        [
            "LOC_X",
            "LOC_Y",
            "SHOT_MADE_FLAG",
            "SHOT_DISTANCE",
            "PERIOD",
            "MINUTES_REMAINING",
            "SECONDS_REMAINING",
            "SHOT_ATTEMPTED_FLAG",
        ],
    )

    loc_x = out.get("LOC_X", pd.Series(np.nan, index=out.index))
    loc_y = out.get("LOC_Y", pd.Series(np.nan, index=out.index))

    if "shot_distance_feet" not in out.columns:
        out["shot_distance_feet"] = out.get("SHOT_DISTANCE", pd.Series(np.nan, index=out.index)).fillna(
            np.sqrt(loc_x.pow(2) + loc_y.pow(2)) / 10.0
        )

    out["shot_angle"] = np.arctan2(loc_x, loc_y.clip(lower=1))
    out["abs_shot_angle"] = out["shot_angle"].abs()

    out["period_length_seconds"] = period_length_seconds(out.get("PERIOD", pd.Series(np.nan, index=out.index)))
    out["period_seconds_remaining"] = (
        out.get("MINUTES_REMAINING", pd.Series(0, index=out.index)).fillna(0) * 60
        + out.get("SECONDS_REMAINING", pd.Series(0, index=out.index)).fillna(0)
    )
    out["seconds_in_period"] = out["period_seconds_remaining"]
    out["period_seconds_elapsed"] = (out["period_length_seconds"] - out["period_seconds_remaining"]).clip(lower=0)
    out["period_elapsed_pct"] = np.where(
        out["period_length_seconds"].gt(0),
        out["period_seconds_elapsed"] / out["period_length_seconds"],
        np.nan,
    )

    regulation_complete = np.clip(out.get("PERIOD", pd.Series(1, index=out.index)).fillna(1) - 1, 0, 4)
    overtime_complete = np.maximum(out.get("PERIOD", pd.Series(1, index=out.index)).fillna(1) - 5, 0)
    out["game_seconds_elapsed"] = (
        regulation_complete * REGULATION_PERIOD_SECONDS
        + overtime_complete * OVERTIME_PERIOD_SECONDS
        + out["period_seconds_elapsed"]
    )
    out["game_minutes_elapsed"] = out["game_seconds_elapsed"] / 60.0

    out["is_clutch"] = (
        (out.get("PERIOD", pd.Series(np.nan, index=out.index)).fillna(0) >= 4)
        & (out["seconds_in_period"] <= 120)
    ).astype(int)
    out["is_end_of_period"] = (out["period_seconds_remaining"] <= 30).astype(int)
    out["is_end_of_game"] = (
        (out.get("PERIOD", pd.Series(np.nan, index=out.index)).fillna(0) >= 4)
        & (out["seconds_in_period"] <= 120)
    ).astype(int)

    action = out.get("ACTION_TYPE", pd.Series("", index=out.index)).astype(str).str.lower().fillna("")
    shot_type = out.get("SHOT_TYPE", pd.Series("", index=out.index)).astype(str).str.upper().fillna("")

    out["is_layup"] = action.str.contains("layup|finger roll").astype(int)
    out["is_dunk"] = action.str.contains("dunk").astype(int)
    out["is_jump_shot"] = action.str.contains("jump shot|pullup|step back|fadeaway").astype(int)
    out["is_hook"] = action.str.contains("hook").astype(int)
    out["is_floater"] = action.str.contains("float").astype(int)

    is_three = shot_type.str.contains("3PT", na=False)
    shot_type_cat = np.select(
        [
            out["is_dunk"].eq(1),
            out["is_layup"].eq(1),
            out["is_hook"].eq(1),
            out["is_floater"].eq(1),
            out["is_jump_shot"].eq(1) & (~is_three),
            out["is_jump_shot"].eq(1) & is_three,
        ],
        ["dunk", "layup", "hook", "floater", "2pt_jump", "3pt_jump"],
        default="other",
    )
    out["shot_type_cat"] = shot_type_cat
    out["is_jump_shot_2"] = (out["shot_type_cat"] == "2pt_jump").astype(int)
    out["is_jump_shot_3"] = (out["shot_type_cat"] == "3pt_jump").astype(int)
    out["shot_value"] = np.where(is_three, 3, 2)

    zone_basic = out.get("SHOT_ZONE_BASIC", pd.Series("", index=out.index)).astype(str)
    zone_range = out.get("SHOT_ZONE_RANGE", pd.Series("", index=out.index)).astype(str)
    out["zone_range_bucket"] = zone_range.replace("", "Unknown")
    out["is_corner_three"] = zone_basic.isin(["Left Corner 3", "Right Corner 3"]).astype(int)
    out["is_above_break_three"] = zone_basic.eq("Above the Break 3").astype(int)
    out["is_midrange"] = zone_basic.eq("Mid-Range").astype(int)
    out["is_restricted_area"] = zone_basic.eq("Restricted Area").astype(int)

    out["distance_bucket"] = pd.cut(
        out["shot_distance_feet"],
        bins=[-0.001, 4, 8, 14, 22, 30, np.inf],
        labels=["0-4 ft", "4-8 ft", "8-14 ft", "14-22 ft", "22-30 ft", "30+ ft"],
    ).astype("string")

    if {"TEAM_NAME", "HTM", "VTM"}.issubset(out.columns):
        out["home_indicator"] = _derive_home_indicator(out["TEAM_NAME"], out["HTM"], out["VTM"])
    else:
        out["home_indicator"] = np.nan

    return out


def compute_sdi_components(df: pd.DataFrame) -> pd.DataFrame:
    """Attach SDI component scores using the current audited NBA convention."""
    out = engineer_nba_features(df)

    out["sdi_distance"] = out["shot_distance_feet"].clip(lower=0, upper=35) / 35.0
    out["sdi_game_clock"] = out["game_minutes_elapsed"].clip(lower=0, upper=REGULATION_GAME_MINUTES) / REGULATION_GAME_MINUTES

    shot_type_scores = {
        "dunk": 0.10,
        "layup": 0.20,
        "hook": 0.55,
        "floater": 0.65,
        "2pt_jump": 0.75,
        "3pt_jump": 0.60,
        "other": 0.35,
    }
    out["sdi_shot_type"] = out["shot_type_cat"].map(shot_type_scores).fillna(0.35)

    zone_difficulty = {
        "Restricted Area": 0.10,
        "In The Paint (Non-RA)": 0.40,
        "Mid-Range": 0.75,
        "Left Corner 3": 0.45,
        "Right Corner 3": 0.45,
        "Above the Break 3": 0.60,
        "Backcourt": 0.95,
    }
    out["sdi_zone"] = out.get("SHOT_ZONE_BASIC", pd.Series("", index=out.index)).map(zone_difficulty).fillna(0.50)

    out["sdi_angle"] = out["abs_shot_angle"] / (np.pi / 2)

    active_weight_total = sum(SDI_COMPONENT_WEIGHTS.values())
    weighted_total = sum(out[name] * weight for name, weight in SDI_COMPONENT_WEIGHTS.items())
    out["SDI"] = weighted_total / active_weight_total
    return out


def build_xfg_feature_frame_from_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature frame expected by the trained xFG model."""
    engineered = engineer_nba_features(df)
    return engineered[XFG_FEATURES].copy()


def build_shot_frame_from_dict(shot: dict) -> pd.DataFrame:
    return pd.DataFrame([shot])


def compute_sdi_for_shot_dict(shot: dict) -> float:
    features = compute_sdi_components(build_shot_frame_from_dict(shot))
    if features.empty or "SDI" not in features.columns:
        return 0.0
    value = pd.to_numeric(features["SDI"], errors="coerce").iloc[0]
    return 0.0 if pd.isna(value) else float(value)


def current_usage_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_vars = sorted(set().union(*CURRENT_FEATURE_USAGE.values()))
    for variable in all_vars:
        rows.append(
            {
                "variable_name": variable,
                "used_in_xfg": variable in CURRENT_FEATURE_USAGE["xFG"],
                "used_in_gam": variable in CURRENT_FEATURE_USAGE["GAM"],
                "used_in_sdi": variable in CURRENT_FEATURE_USAGE["SDI"],
                "used_in_app": variable in CURRENT_FEATURE_USAGE["app"],
            }
        )
    return rows


def poster_model_snapshot_rows() -> list[dict[str, object]]:
    """Return a poster-ready summary of the core active GAM/SDI inputs."""
    return [row.copy() for row in POSTER_MODEL_SNAPSHOT]


def poster_model_snapshot_frame() -> pd.DataFrame:
    """Return the poster-ready model snapshot as a DataFrame."""
    return pd.DataFrame(poster_model_snapshot_rows())
