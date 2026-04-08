# Data Directory

This folder exposes the two main historical shot archives at the top level of the repository so reviewers can find the core data quickly.

## Included Files

| File | Scope | Source Copy |
| --- | --- | --- |
| [`nba_shots_2014_2024.csv.gz`](nba_shots_2014_2024.csv.gz) | Historical NBA archive used for the matched cross-sport comparison | [`analysis/nba/data/nba_shots_2014_2024.csv.gz`](../analysis/nba/data/nba_shots_2014_2024.csv.gz) |
| [`nhl_shots_2014_2024.csv.gz`](nhl_shots_2014_2024.csv.gz) | Historical NHL archive used for the matched cross-sport comparison | [`analysis/nhl/data/app_data/nhl_shots_2014_2024.csv.gz`](../analysis/nhl/data/app_data/nhl_shots_2014_2024.csv.gz) |

## Why The Files Are Compressed

The raw archives are large enough that storing them as `.csv.gz` keeps the repository manageable while still shipping the complete shot tables used by the project.

## Related Data Locations

- `analysis/nba/data/`: NBA model inputs, summaries, figures, and poster snapshot outputs
- `analysis/nhl/data/`: NHL model outputs and derived comparison tables
- `analysis/nhl/data/app_data/`: NHL app-specific extracts used by the Streamlit pages
