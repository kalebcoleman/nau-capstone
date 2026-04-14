# NHL SDI Methods

This note documents the two NHL Shot Difficulty Index implementations currently used in the repository:

- the original NHL SDI already present in the earlier capstone workflow
- the alternate NBA-style NHL SDI added for shape comparison

The point of keeping both is methodological transparency. The alternate version is an experiment, not a silent replacement.

## 1. Original NHL SDI

The original NHL SDI appears in:

- [`capstonemodels.Rmd`](capstonemodels.Rmd)
- [`prepare_app_data.py`](prepare_app_data.py)
- [`../nba/cross_sport_comparison.py`](../nba/cross_sport_comparison.py)

### Formula

The historical NHL SDI is built from larger-scale components:

- `difficulty_distance = (shotDistance / max_dist) * 100`
- `difficulty_angle = abs(shotAngle) / max_angle * 100`
- `difficulty_rebound = 30 if rebound else 0`
- `difficulty_goalie_froze = 20 if goalie_froze else 0`

Then:

`SDI = 0.4 * difficulty_distance + 0.3 * difficulty_angle + 0.2 * difficulty_rebound + 0.1 * difficulty_goalie_froze`

### Interpretation

- Larger values mean harder shots by construction.
- The raw weighted total lands in the `20s` and low `30s`, but the exported SDI used in the app and comparison figures is divided by `100`.
- That final presentation-scale normalization keeps the NHL plot on the same `0-1` axis family as the NBA plot without changing player ordering or the scatter shape.
- This SDI remains a within-NHL ranking device first; the rescaling only improves cross-sport readability.

### Limitation

The rebound term in the original NHL SDI increases difficulty, but the shot data suggests rebounds are usually better scoring opportunities than ordinary shots. That does not make the original chart useless, but it does mean the metric is partly heuristic rather than a pure expected-hardness score.

## 2. Alternate NBA-Style NHL SDI

The alternate NHL SDI lives in:

- [`sdi_analysis_nba_style.py`](sdi_analysis_nba_style.py)

### Goal

This script was added to answer a specific question:

- if NHL SDI is rebuilt to look more like the NBA SDI, does the shape of the player-level scatter change much?

### Formula

The alternate version uses normalized `0-1` components and weights that sum to `1`:

- `sdi_distance = clip(shotDistance, 0, 90) / 90`
- `sdi_angle = clip(abs(shotAngle), 0, 90) / 90`
- `sdi_shot_type = lookup by shotType`
- `sdi_context = 0.55 + 0.25*goalie_froze - 0.20*rebound - 0.10*rush`
- `sdi_period = (clip(period, 1, 4) - 1) / 3`

Then:

`SDI = 0.30 * distance + 0.20 * angle + 0.20 * shot_type + 0.15 * context + 0.15 * period`

### Why These Terms Were Chosen

- `distance`: parallel to the NBA SDI, where longer shots are harder
- `angle`: parallel to the NBA SDI, where wider or more extreme geometry is harder
- `shot_type`: used as the NHL analogue to NBA shot-type difficulty buckets
- `context`: used instead of shot-clock pressure because hockey does not have a shot clock
- `period`: kept as a light temporal/context term so the formula stays structurally similar to the NBA version

### Output Files

- [`data/nhl_player_sdi_nba_style_2014_2024.csv`](data/nhl_player_sdi_nba_style_2014_2024.csv)
- [`figures/nhl_sdi_nba_style_vs_actual_2014_2024.png`](figures/nhl_sdi_nba_style_vs_actual_2014_2024.png)

### What Changed

- The NHL SDI scale now sits near the NBA scale.
- Player average SDI is around `0.42` instead of around `25`.
- Rebounds and rush chances now reduce difficulty instead of increasing it.

## 3. What Did Not Change Much

After rebuilding the NHL SDI in a more NBA-like way, the overall scatter still does not look dramatically different.

That matters because it suggests the visible structure in the NHL player plot is not only a scaling artifact. Part of the shape appears to come from the underlying shot-distribution differences between player roles.

In plain terms:

- defensemen tend to live in a long-distance, low-conversion shot neighborhood
- forwards tend to take more interior and net-front attempts
- changing the SDI formula changes the axis scale, but it does not erase the underlying role structure in the shot data

That is a plausible reason the chart still looks like it has two broad groups and still does not show many extreme green or red points.

## 4. Why There Are Not Many Extreme Green/Red Players

In both NHL versions, point color is based on:

`residual = actual_goal_pct - expected_goal_pct`

If there are not many extreme green or red points, that usually means:

- player-level finishing differences are modest after aggregation
- expected-goal averages are already explaining much of the variation
- the minimum-shot threshold removes noisy small-sample outliers

That is not necessarily a problem. It can simply mean the model-plus-aggregation layer is stable and conservative.

## 5. Recommended Use

Use the original NHL SDI when:

- you want continuity with the existing capstone workflow
- you are reproducing the earlier NHL figures already in the repository

Use the alternate NBA-style NHL SDI when:

- you want a more apples-to-apples scale relative to the NBA SDI
- you want to test whether the plot shape is driven by scaling or by the data itself

Do not claim that the original and alternate SDIs are interchangeable. They answer similar questions, but they are constructed differently.
