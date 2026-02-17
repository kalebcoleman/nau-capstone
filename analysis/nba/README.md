# NBA Expected Points and Shot Analytics

Standalone NBA analysis package for capstone deliverables.

## Folder Structure

```
analysis/nba/
├── expected_points_analysis.py
├── calibration_diagnostics.py
├── gam_analysis.py
├── advanced_analytics.py
├── player_performance_analysis.py
├── shot_density.py
├── salary_collector.py
├── value_analysis.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── nba_shots_2025-26.csv.gz
│   ├── nba_shots_training.csv.gz
│   ├── player_box_usage_2025-26.csv
│   ├── player_box_traditional_2025-26.csv
│   ├── player_box_advanced_2025-26.csv
│   ├── shots_with_xp_2025-26.parquet
│   └── *.csv outputs
├── figures/
├── models/
├── utils/
└── requirements.txt
```

## Data Files

- `nba_shots_2025-26.csv.gz`: full 2025-26 regular season shot sample (~146K shots, gzip-compressed)
- `nba_shots_training.csv.gz`: six-season training set for model scripts (2020-21 through 2025-26)
- `player_box_usage_2025-26.csv`: usage rates for SDI/role context
- `player_box_traditional_2025-26.csv`: position/minutes context
- `player_box_advanced_2025-26.csv`: assist percentage context
- `shots_with_xp_2025-26.parquet`: precomputed shot-level xP/POE output

## Run Order

1. `pip install -r requirements.txt`
2. `python expected_points_analysis.py`
3. `python calibration_diagnostics.py`
4. `python gam_analysis.py`
5. `python advanced_analytics.py`
6. `python player_performance_analysis.py`
7. `python shot_density.py`
8. `python salary_collector.py` (optional; precomputed salary CSV included)
9. `python value_analysis.py`
10. `streamlit run app/streamlit_app.py`

## Key Metrics

- `xFG`: expected field-goal probability from the logistic model
- `POE`: points over expected at shot and player levels
- `SDI`: shot difficulty index based on distance, angle, clock, and shot type
- `POE/$M`: POE per $1M salary for value context

## Notes

- This package is pre-populated with outputs and figures so it works as an example repository immediately.
- Scripts are configured for standalone CSV/parquet inputs in `data/`.
- Streamlit app is scoped to 2025-26 sample data for capstone sharing.
