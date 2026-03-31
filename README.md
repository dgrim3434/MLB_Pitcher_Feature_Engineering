# Pitcher ERA Prediction — Feature Engineering
 
A feature engineering pipeline for predicting 2025 MLB pitcher ERA using FanGraphs standard, advanced, and Statcast pitching data from the 2023 and 2024 seasons. Built as part of a self-directed ML engineering curriculum focused on real-world data science and AI engineering skills.
 
---
 
## Data Sources
 
All data sourced from FanGraphs (membership required). Three tables were downloaded for the 2023 and 2024 seasons:
 
- **Standard pitching stats** — traditional counting and rate stats (ERA, IP, K, BB, HR, etc.)
- **Advanced (+) stats** — park and league adjusted metrics (ERA-, FIP-, xFIP-, K%+, GB%+, etc.)
- **Statcast stats** — contact quality metrics (EV, Barrel%, HardHit%, xERA)
 
Data is not included in this repository due to FanGraphs licensing. Stats can be downloaded directly from [fangraphs.com/leaders](https://www.fangraphs.com/leaders).
 
---
 
## Population
 
- Pitchers with **60+ IP** in each of the 2023, 2024, and 2025 seasons
- Minimum **1 game started** — pure relievers who reached 60 IP were excluded
- Pitchers appearing in all three seasons only — ensures lag features are available for every row
- **97 pitchers** in the final feature matrix
 
---
 
## Features Engineered
 
### Standard Stats (2024 values + 2023→2024 trends)
- `K_per_9`, `BB_per_9`, `H_per_9`, `HR_per_9` — rate stats normalized per 9 innings
- `K_to_BB` — strikeout to walk ratio, measures command and dominance
- `Quality_Start_percent` — quality starts as a percentage of games started
- `ERA_label` — ordinal ERA bin (0=Elite <3.00, 1=Good 3-3.99, 2=Average 4-4.99, 3=Poor 5+)
- `YoY_ERA`, `YoY_K_per_9`, `YoY_BB_per_9` — year-over-year change from 2023 to 2024
 
### Advanced (+) Stats (2024 values + 2023→2024 trends)
- `ERA-`, `FIP-`, `xFIP-` — park and league adjusted ERA estimators, lower is better
- `K%+`, `BB%+` — adjusted strikeout and walk rates
- `GB%+` — ground ball rate, ground balls suppress home runs
- `BABIP+`, `LOB%+` — luck indicators, high BABIP+ and low LOB%+ suggest ERA will regress
- `YoY_ERA_minus`, `YoY_FIP_minus`, `YoY_K_plus_pct` — year-over-year trends in key adjusted metrics
 
### Statcast (2024 values + 2023→2024 trends)
- `EV` — average exit velocity allowed
- `Barrel%` — percentage of batted balls classified as barrels, strongest contact quality signal
- `HardHit%` — hard hit ball rate
- `xERA` — expected ERA based on quality of contact allowed
- `YoY_Barrel_pct`, `YoY_xERA`, `YoY_HardHit_pct` — year-over-year trends in contact quality
 
---
 
## Target Variable
 
`Target` — the pitcher's 2025 ERA. Shifted onto the 2024 row using `shift(-1)` within each pitcher group. No 2025 features were used as predictors to prevent data leakage.
 
---
 
## Key Engineering Decisions
 
- **Lag features built from a single concatenated dataframe** — all three seasons were stacked into one table with a `Year` column. `groupby('PlayerId').diff()` computed year-over-year changes cleanly without any reshaping.
- **Target leakage prevention** — only 2023 and 2024 data was used for advanced and Statcast tables. 2025 ERA was the sole piece of 2025 data included, used only as the prediction target.
- **Reliever exclusion via Quality Start filter** — pitchers with 0 game starts were dropped after feature engineering, removing relievers who accumulated 60+ IP without starting.
- **PlayerId as join key** — joined all three tables on FanGraphs PlayerId rather than name to avoid mismatches from name encoding differences or players who changed teams.
 
---
 
## Next Steps
 
- Train a baseline model (Ridge Regression, Random Forest) on the 2024 feature matrix to predict 2025 ERA
- Evaluate using cross-validation RMSE and compare against a naive baseline (predict 2024 ERA as 2025 ERA)
- Add additional lag years (2022) to build longer trend windows
- Incorporate pitcher arsenal data (pitch mix, velocity by pitch type) from FanGraphs pitch-level tables
- Expand to a rolling multi-year pipeline that updates automatically each season
 
---
 
## Curriculum Context
 
This project is the Week 11 mini-capstone from a self-directed Python ML engineering curriculum. Each week covers a core data science topic through real-world coding challenges, followed by a baseball project applying that week's concepts to real data.
 
Week 11 topic: **Feature Engineering for Machine Learning**
