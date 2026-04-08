# NHL Analysis

This folder contains the NHL-side data preparation, expected-goal modeling, SDI variants, and figure exports that feed the final cross-sport comparison.

## Workflow Overview

The NHL pipeline does three main jobs:

1. prepares app-ready shot data from the larger MoneyPuck archive
2. fits repo-owned expected-goal and distance-effect outputs
3. exports NHL summaries and figures used by the dashboard, story, and final poster

## Main Files

### Core Scripts

- `prepare_app_data.py`: prepares NHL app datasets from the local MoneyPuck source
- `gam_analysis.py`: fits the NHL GAM and writes the main figure/table outputs
- `modeling.py`: shared NHL modeling helpers
- `sdi_analysis_nba_style.py`: alternate NHL SDI experiment that mirrors the NBA scale more closely

### Documentation

- `SDI_METHODS.md`: explains the original NHL SDI versus the alternate NBA-style NHL SDI
- `capstonemodels.Rmd`: earlier NHL-focused write-up
- `drianna_walker_capstonemodels.Rmd`: alternate draft notebook/write-up

### App Helpers

- `app/nhl_utils.py`: helper functions used by the Streamlit NHL pages

## Important Data Outputs

### App Data

- `data/app_data/nhl_shots_2024.csv.gz`
- `data/app_data/nhl_shots_2014_2024.csv.gz`

### Poster / Comparison Tables

- `data/nhl_player_summary_2014_2024.csv`
- `data/nhl_position_summary_2014_2024.csv`
- `data/nhl_gam_distance_2014_2024.csv`
- `data/nhl_gam_angle_2014_2024.csv`
- `data/nhl_model_metrics_gam.csv`

### Key Figures

- `figures/nhl_sdi_vs_actual_2014_2024.png`
- `figures/nhl_sdi_by_position_2014_2024.png`
- `figures/nhl_gam_distance_2014_2024.png`
- `figures/nhl_gam_angle_2014_2024.png`
- `figures/nhl_gam_distance_empty_net_comparison_2014_2024.png`

## Typical Commands

Install dependencies:

```bash
cd analysis/nhl
pip install -r requirements.txt
```

Refresh app data:

```bash
python prepare_app_data.py
```

Refresh NHL modeling outputs:

```bash
python gam_analysis.py
python sdi_analysis_nba_style.py
```

## Notes

- `prepare_app_data.py` expects `data/shots_2007-2024.csv` to exist locally.
- The final cross-sport comparison is orchestrated from `analysis/nba/cross_sport_comparison.py`, but the NHL model outputs live here.
- The original and alternate NHL SDI methods are intentionally both preserved. Read [`SDI_METHODS.md`](SDI_METHODS.md) before treating those figures as interchangeable.
