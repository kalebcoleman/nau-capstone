# NAU Capstone: Sports Expected Points Analysis

A comprehensive analysis of expected points/goals across NHL and NBA, examining offensive strategy and player performance through quantitative shot quality models.

![Shot Difficulty Index vs Expected FG%](analysis/nba/figures/sdi_vs_xfg_scatter.png)

## Overview

This project analyzes hockey and basketball shot data to identify player performance beyond traditional metrics. Using expected points/goal models, we quantify how players perform relative to expectation.

### Key Metrics

| Metric | Description |
|--------|-------------|
| **xFG/xG** | Expected field-goal (NBA) / expected goal (NHL) probability |
| **POE** | Points/Goals Over Expected at shot and player levels |
| **SDI** | Shot Difficulty Index based on distance, angle, clock, and shot type |
| **POE/$M** | POE per $1M salary for value context |

## Project Structure

```
nau-capstone/
├── docs/
│   └── proposal/
│       ├── Proposal.pdf
│       └── Proposal.Rmd
├── analysis/
│   ├── nba/
│   │   ├── expected_points_analysis.py
│   │   ├── calibration_diagnostics.py
│   │   ├── gam_analysis.py
│   │   ├── advanced_analytics.py
│   │   ├── player_performance_analysis.py
│   │   ├── shot_density.py
│   │   ├── value_analysis.py
│   │   ├── app/
│   │   │   └── streamlit_app.py
│   │   ├── data/
│   │   ├── figures/
│   │   ├── models/
│   │   ├── utils/
│   │   └── requirements.txt
│   └── nhl/
│       └── README.md
└── requirements.txt
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### NBA Analysis

```bash
cd analysis/nba
pip install -r requirements.txt

# Run analysis scripts
python expected_points_analysis.py
python calibration_diagnostics.py
python gam_analysis.py
python advanced_analytics.py
python player_performance_analysis.py
python shot_density.py
python value_analysis.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

### NHL Analysis

```bash
cd analysis/nhl
pip install -r requirements.txt
# See NHL README for details
```

## Data Sources

- **NBA Stats API**: Official league data with optical tracking (25fps)
- **MoneyPuck**: NHL shot data (2007-2024)
- **Player Salaries**: basketball-reference.com

## Models

- **xFG Model (NBA)**: Logistic regression for expected field goal probability
- **xG Model (NHL)**: LASSO regression for expected goals
- **GAM**: Generalized Additive Models for spatial analysis
- **GMM Clustering**: Gaussian Mixture Models for player archetypes

## Requirements

See `requirements.txt` for full dependency list:

- pandas, numpy, pyarrow (data processing)
- matplotlib, plotly, seaborn, streamlit (visualization)
- scikit-learn, joblib, pygam (machine learning)
- requests, beautifulsoup4 (data collection)

## License

NAU Capstone Project - Spring 2026
