# Final Poster Guide

This guide maps the final poster sections to the files in the repository that support them.

## Project Summary

Final poster story:

- compare shot difficulty and expected scoring across the NBA and NHL
- use a matched `2014-2024` historical window so the cross-sport figures share the same frame
- pair player-level SDI views with model-based GAM distance effects

Dashboard note:

- the dashboard is a companion demo
- the final poster should cite the matched `2014-2024` exports, not the NBA-only `2025-26` demo sample

## Poster Section Map

### Abstract / Motivation

Use:

- [`README.md`](../README.md)
- [`docs/proposal/Matched_Comparison_Story.Rmd`](proposal/Matched_Comparison_Story.Rmd)

Key message:

- raw efficiency is incomplete on its own
- shot difficulty changes the baseline in both sports
- the comparison is strongest when both leagues are constrained to the same historical window

### Data Snapshot Table

Use:

- [`analysis/nba/data/poster_model_snapshot.md`](../analysis/nba/data/poster_model_snapshot.md)
- [`analysis/nba/export_poster_snapshot.py`](../analysis/nba/export_poster_snapshot.py)
- [`analysis/nba/feature_spec.py`](../analysis/nba/feature_spec.py)

This is the cleanest source for the "which variables are we actually using?" section of the poster.

### Methods

NBA side:

- [`analysis/nba/expected_points_analysis.py`](../analysis/nba/expected_points_analysis.py)
- [`analysis/nba/gam_analysis.py`](../analysis/nba/gam_analysis.py)
- [`analysis/nba/sdi_analysis.py`](../analysis/nba/sdi_analysis.py)

NHL side:

- [`analysis/nhl/gam_analysis.py`](../analysis/nhl/gam_analysis.py)
- [`analysis/nhl/modeling.py`](../analysis/nhl/modeling.py)
- [`analysis/nhl/prepare_app_data.py`](../analysis/nhl/prepare_app_data.py)
- [`analysis/nhl/SDI_METHODS.md`](../analysis/nhl/SDI_METHODS.md)

Recommended methods language:

- expected scoring is estimated from repo-owned shot models
- SDI is sport-specific in construction but shared in interpretation
- higher SDI means a player takes harder shots on average

### Results: SDI Scatter

Use:

- [`analysis/nba/figures/nba_sdi_vs_actual_2014_2024.png`](../analysis/nba/figures/nba_sdi_vs_actual_2014_2024.png)
- [`analysis/nhl/figures/nhl_sdi_vs_actual_2014_2024.png`](../analysis/nhl/figures/nhl_sdi_vs_actual_2014_2024.png)
- [`analysis/nba/data/nba_player_summary_2014_2024.csv`](../analysis/nba/data/nba_player_summary_2014_2024.csv)
- [`analysis/nhl/data/nhl_player_summary_2014_2024.csv`](../analysis/nhl/data/nhl_player_summary_2014_2024.csv)

Talking points:

- point color shows residual efficiency
- point size shows volume
- high-SDI players are not automatically inefficient; some still outperform expectation

### Results: Distance Effect / GAM

Use:

- [`analysis/nba/figures/nba_gam_distance_2014_2024.png`](../analysis/nba/figures/nba_gam_distance_2014_2024.png)
- [`analysis/nhl/figures/nhl_gam_distance_2014_2024.png`](../analysis/nhl/figures/nhl_gam_distance_2014_2024.png)
- [`analysis/nba/data/nba_gam_distance_2014_2024.csv`](../analysis/nba/data/nba_gam_distance_2014_2024.csv)
- [`analysis/nhl/data/nhl_gam_distance_2014_2024.csv`](../analysis/nhl/data/nhl_gam_distance_2014_2024.csv)

Talking points:

- distance is the cleanest shared difficulty signal across sports
- both curves show scoring probability dropping as distance increases
- the shape of decline differs by sport, which is part of the analytical story

### Results: Position Clusters

Use:

- [`analysis/nba/figures/nba_sdi_by_position_2014_2024.png`](../analysis/nba/figures/nba_sdi_by_position_2014_2024.png)
- [`analysis/nhl/figures/nhl_sdi_by_position_2014_2024.png`](../analysis/nhl/figures/nhl_sdi_by_position_2014_2024.png)
- [`analysis/nba/data/nba_position_summary_2014_2024.csv`](../analysis/nba/data/nba_position_summary_2014_2024.csv)
- [`analysis/nhl/data/nhl_position_summary_2014_2024.csv`](../analysis/nhl/data/nhl_position_summary_2014_2024.csv)

Talking points:

- role structure matters
- shot difficulty is not distributed randomly across player positions
- the two sports show different role clusters, but both exhibit structured shot environments

### Dashboard / Demo Section

Use:

- [`app/streamlit_app.py`](../app/streamlit_app.py)
- [`app/demo_content.py`](../app/demo_content.py)
- [`app/overview_page.py`](../app/overview_page.py)
- [`app/sdi_explorer_page.py`](../app/sdi_explorer_page.py)
- [`app/gam_explorer_page.py`](../app/gam_explorer_page.py)

Good framing:

- the dashboard is an interactive companion to the poster
- it lets viewers inspect shot maps, summaries, and matched comparison figures
- it is especially useful during live presentation or Q&A

## Best Files To Cite In A Presentation

If you only cite a few repo locations on the poster or in speaker notes, use these:

- [`README.md`](../README.md)
- [`docs/proposal/Matched_Comparison_Story.Rmd`](proposal/Matched_Comparison_Story.Rmd)
- [`analysis/nba/data/poster_model_snapshot.md`](../analysis/nba/data/poster_model_snapshot.md)
- [`analysis/nba/figures/nba_gam_distance_2014_2024.png`](../analysis/nba/figures/nba_gam_distance_2014_2024.png)
- [`analysis/nhl/figures/nhl_gam_distance_2014_2024.png`](../analysis/nhl/figures/nhl_gam_distance_2014_2024.png)

## Refresh Commands

```bash
python analysis/nba/export_poster_snapshot.py
python analysis/nba/cross_sport_comparison.py
python analysis/nhl/gam_analysis.py
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```
