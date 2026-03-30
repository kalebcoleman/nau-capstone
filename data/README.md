# Data Directory

This folder exposes the main shot archives at the top level of the repository so visitors can find them immediately.

## Included Files

| File | Scope | Source Copy |
|------|-------|-------------|
| [`nba_shots_2014_2024.csv.gz`](nba_shots_2014_2024.csv.gz) | NBA shot-level archive for 2014-2024 | [`analysis/nba/data/nba_shots_2014_2024.csv.gz`](../analysis/nba/data/nba_shots_2014_2024.csv.gz) |
| [`nhl_shots_2014_2024.csv.gz`](nhl_shots_2014_2024.csv.gz) | NHL shot-level archive for 2014-2024 | [`analysis/nhl/data/app_data/nhl_shots_2014_2024.csv.gz`](../analysis/nhl/data/app_data/nhl_shots_2014_2024.csv.gz) |

## Why The Files Are Compressed

The raw NBA and NHL shot tables are large enough that we compressed them as `.gz` files to keep them small enough to store in GitHub while still making the full datasets available in the repo.
