"""Matched NBA/NHL comparison figures for the 2014-2024 story."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from utils import APP_BG, MUTED_TEXT, PANEL_BG, TEXT_COLOR, apply_theme

WINDOW_LABEL = "2014-2024"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"

NBA_SUMMARY_PATH = DATA_DIR / "nba_player_summary_2014_2024.csv"
NHL_SUMMARY_PATH = DATA_DIR / "nhl_player_summary_2014_2024.csv"
NBA_GAM_PATH = DATA_DIR / "nba_gam_distance_2014_2024.csv"
NHL_GAM_PATH = DATA_DIR / "nhl_gam_distance_2014_2024.csv"
NBA_SDI_FIGURE = FIGURES_DIR / "nba_sdi_vs_actual_2014_2024.png"
NHL_SDI_FIGURE = FIGURES_DIR / "nhl_sdi_vs_actual_2014_2024.png"
NBA_GAM_FIGURE = FIGURES_DIR / "nba_gam_distance_2014_2024.png"
NHL_GAM_FIGURE = FIGURES_DIR / "nhl_gam_distance_2014_2024.png"
NBA_POSITION_FIGURE = FIGURES_DIR / "nba_sdi_by_position_2014_2024.png"
NHL_POSITION_FIGURE = FIGURES_DIR / "nhl_sdi_by_position_2014_2024.png"


st.set_page_config(page_title="NBA vs NHL Comparison", layout="wide")
apply_theme()
st.title("🆚 NBA vs NHL Comparison")
st.markdown(
    f'<div style="color: {MUTED_TEXT}; margin-bottom: 24px;">'
    f"Matched cross-sport figures built from the shared {WINDOW_LABEL} comparison pipeline. "
    f"The page uses the same generated outputs as the story R Markdown document."
    f"</div>",
    unsafe_allow_html=True,
)

missing = [p for p in [
    NBA_SUMMARY_PATH,
    NHL_SUMMARY_PATH,
    NBA_GAM_PATH,
    NHL_GAM_PATH,
    NBA_SDI_FIGURE,
    NHL_SDI_FIGURE,
    NBA_GAM_FIGURE,
    NHL_GAM_FIGURE,
    NBA_POSITION_FIGURE,
    NHL_POSITION_FIGURE,
] if not p.exists()]
if missing:
    st.error(
        "Comparison outputs are missing. Run "
        "`python3 analysis/nba/cross_sport_comparison.py` first.\n\n"
        + "\n".join(str(p) for p in missing)
    )
    st.stop()

import pandas as pd
nba_summary = pd.read_csv(NBA_SUMMARY_PATH)
nhl_summary = pd.read_csv(NHL_SUMMARY_PATH)

metric_cols = st.columns(4)
metric_cols[0].metric("NBA Players", f"{len(nba_summary):,}")
metric_cols[1].metric("NHL Players", f"{len(nhl_summary):,}")
metric_cols[2].metric("NBA Attempts", f"{int(nba_summary['attempts'].sum()):,}")
metric_cols[3].metric("NHL Shots", f"{int(nhl_summary['attempts'].sum()):,}")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.image(str(NBA_SDI_FIGURE), use_container_width=True)
    st.caption("Efficiency vs expectation view. Labels mark selected overperformers, underperformers, high-SDI shooters, and high-volume scorers.")
with col2:
    st.image(str(NHL_SDI_FIGURE), use_container_width=True)
    st.caption("Efficiency vs expectation view. The NHL figure mirrors the NBA styling with the same label-selection rules.")

st.divider()

col3, col4 = st.columns(2)
with col3:
    st.image(str(NBA_GAM_FIGURE), use_container_width=True)
    st.caption("The NBA GAM panel isolates how distance shifts scoring odds after controlling for shot context.")
with col4:
    st.image(str(NHL_GAM_FIGURE), use_container_width=True)
    st.caption("The NHL GAM panel is clipped at 100 feet so the displayed distance effect stays focused on the meaningful scoring range.")

st.divider()

col5, col6 = st.columns(2)
with col5:
    st.image(str(NBA_POSITION_FIGURE), use_container_width=True)
    st.caption("NBA position-cluster SDI figure. Color marks Guards, Forwards, and Centers; the soft ellipses show each cluster shape.")
with col6:
    st.image(str(NHL_POSITION_FIGURE), use_container_width=True)
    st.caption("NHL position-cluster SDI figure. Wings are combined from left and right side skaters so the NHL groups read as C/W/D.")

st.divider()
st.subheader("Generated Files")
st.code(
    "\n".join(
        str(path)
        for path in [
            NBA_SUMMARY_PATH,
            NHL_SUMMARY_PATH,
            NBA_GAM_PATH,
            NHL_GAM_PATH,
            NBA_SDI_FIGURE,
            NHL_SDI_FIGURE,
            NBA_GAM_FIGURE,
            NHL_GAM_FIGURE,
            NBA_POSITION_FIGURE,
            NHL_POSITION_FIGURE,
        ]
    ),
    language="text",
)
