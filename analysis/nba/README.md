# NBA Analysis

This folder contains both sides of the NBA capstone work:

- the `2025-26` NBA-only modeling and dashboard workflow
- the NBA contribution to the final matched `2014-2024` NBA/NHL comparison

## What Lives Here

### Dashboard Surface

- `app/streamlit_app.py`: main Streamlit home page for the poster companion app
- `app/overview_page.py`: poster-oriented landing page
- `app/sdi_explorer_page.py`: interactive cross-sport SDI explorer
- `app/gam_explorer_page.py`: curated GAM figure explorer
- `app/demo_content.py`: poster-oriented copy and figure groups used across the app
- `app/utils.py`: plotting, filtering, and model helper functions for the dashboard

### Modeling And Exports

- `expected_points_analysis.py`: trains the NBA expected-field-goal workflow
- `gam_analysis.py`: builds the NBA GAM-based difficulty outputs
- `sdi_analysis.py`: computes NBA shot-difficulty summaries
- `cross_sport_comparison.py`: writes the matched NBA/NHL `2014-2024` outputs used by the story and dashboard
- `export_poster_snapshot.py`: exports the poster-ready variable snapshot table
- `feature_spec.py`: shared feature definitions used across models, app logic, and poster exports

## Main Data And Figure Outputs

### Poster / Final Comparison Outputs

- `data/nba_shots_2014_2024.csv.gz`
- `data/nba_player_summary_2014_2024.csv`
- `data/nba_position_summary_2014_2024.csv`
- `data/nba_gam_distance_2014_2024.csv`
- `data/poster_model_snapshot.csv`
- `data/poster_model_snapshot.md`
- `figures/nba_sdi_vs_actual_2014_2024.png`
- `figures/nba_sdi_by_position_2014_2024.png`
- `figures/nba_gam_distance_2014_2024.png`

### Dashboard / NBA Demo Outputs

- `data/nba_shots_2025-26.csv.gz`
- `data/shots_with_xp_2025-26.parquet`
- `data/player_box_usage_2025-26.csv`
- `data/player_box_traditional_2025-26.csv`
- `data/player_box_advanced_2025-26.csv`
- `models/xp_model_2025-26.joblib`
- `models/gam_model_2025-26.pkl`

## Typical Commands

Install dependencies:

```bash
cd analysis/nba
pip install -r requirements.txt
```

Refresh core NBA outputs:

```bash
python expected_points_analysis.py
python gam_analysis.py
python sdi_analysis.py
```

Refresh poster-specific exports:

```bash
python cross_sport_comparison.py
python export_poster_snapshot.py
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Notes

- The app still uses the `2025-26` NBA sample for interactive exploration.
- The final poster and cross-sport write-up use the matched `2014-2024` outputs.
- If you are preparing presentation material, start with `data/poster_model_snapshot.md`, `cross_sport_comparison.py`, and the figures under `figures/` with `2014_2024` in the name.
