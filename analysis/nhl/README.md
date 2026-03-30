# NHL Expected Goals Analysis

## Overview

This folder contains the NHL-side data prep and capstone analysis inputs used by the dashboard and proposal materials. The current workflow relies on MoneyPuck shot data, preserves shot-level `xGoal`, and computes SDI-style difficulty features for downstream figures and app pages.

## Quick Start

```bash
cd analysis/nhl
pip install -r requirements.txt
python prepare_app_data.py
```

## Data Sources

- **MoneyPuck**: NHL shot data (2007-2024)
- Reference the proposal files in `docs/proposal/` for the current write-up and comparison figures

## Outputs

- `data/app_data/nhl_shots_2024.csv.gz`: season-filtered app dataset for Streamlit NHL pages
- `capstonemodels.Rmd`: NHL-focused draft analysis notebook/write-up
- raw MoneyPuck CSVs stored under `data/`

## Notes

- `prepare_app_data.py` expects `data/shots_2007-2024.csv` to exist locally
- the main cross-sport figure generation lives in `analysis/nba/cross_sport_comparison.py`
