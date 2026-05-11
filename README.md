# Grand Slam Predictions — ATP & WTA

A full end-to-end machine learning pipeline to predict ATP and WTA Grand Slam outcomes, starting with the Australian Open and French Open 2026. Built on 25 years of historical match data, surface-specific Elo ratings, rolling player form, and bracket-aware Monte Carlo simulation with bootstrap confidence intervals.

Historical ATP and WTA data sourced from: http://www.tennis-data.co.uk/alldata.php

---

## Supported tournaments

| Key | Tournament | Surface |
|-----|-----------|---------|
| `ao26` | Australian Open 2026 | Hard |
| `fo26` | French Open 2026 | Clay |

Adding a new tournament takes two steps: add an entry to `config/tournaments.py` and drop a draw CSV into `data/draws/<key>/`.

---

## What this project does

- Builds separate ML models per tournament and tour (ATP/WTA)
- Learns from surface-specific Elo ratings, rolling form, betting odds, and rankings
- Filters training data to the correct surface for each tournament
- Simulates the full 128-player bracket round by round
- Outputs title probabilities with 90% bootstrap confidence intervals
- Stress-tests each model with calibration curves, permutation importance, and walk-forward backtesting

---

## Modelling approach

### 1. Raw data ingestion
- ATP & WTA match results (2000–2025)
- Betting odds (Bet365)
- ATP & WTA rankings

### 2. Feature engineering

| Feature | Description |
|---------|-------------|
| `winrate_diff` | Rolling win rate difference (last N matches) |
| `odds_diff` | Rolling average odds difference |
| `matches_diff` | Matches played difference |
| `rank_diff` | Log-scaled ranking difference |
| `surface_wr_diff` | Rolling win rate on this specific surface (last 20 matches) |
| `elo_diff_global` | Global Elo rating difference (all surfaces) |
| `elo_diff_surface` | Surface-specific Elo rating difference |

Elo ratings are computed from scratch across the full 2000–2025 dataset using a K=32 base factor (K=48 for Grand Slams). Separate Elo tracks are maintained for global, clay, hard, and grass.

### 3. Model
- Logistic Regression with StandardScaler
- Time-aware 80/20 train/test split
- Surface-filtered training data (e.g. clay-only matches for French Open)
- Separate models per tournament and tour

### 4. Tournament simulation
- Real draw CSV loaded per tournament
- Simulates R128 → Final match by match
- Monte Carlo (10,000 runs) → title probability per player
- Bootstrap confidence intervals (200 × 1,000 sims) → 90% CI per player

---

## Model performance

### French Open 2026 (clay)

| Tour | Accuracy | Log Loss | Calibration Error | Walk-forward Accuracy |
|------|----------|----------|-------------------|----------------------|
| ATP  | 64.2%    | 0.632    | 0.039 ✓           | 66.3% ± 2.5%         |
| WTA  | 66.8%    | 0.603    | 0.015 ✓           | 65.1% ± 1.6%         |

### Most important features (permutation importance)

**ATP clay:** `elo_diff_surface` > `elo_diff_global` > `rank_diff`
Surface Elo dominates — historical clay dominance is the strongest signal.

**WTA clay:** `elo_diff_global` > `elo_diff_surface` > `rank_diff`
Global dominance carries more weight — top WTA players (Swiatek, Sabalenka) are strong across all surfaces.

### FO26 title probabilities (placeholder draw)

**ATP**

| Player | Probability | 90% CI |
|--------|------------|--------|
| Alcaraz C. | 28.1% | [25.8% – 30.2%] |
| Sinner J. | 23.6% | [21.4% – 26.0%] |
| Djokovic N. | 11.9% | [10.3% – 13.8%] |
| Zverev A. | 5.0% | [4.0% – 6.3%] |
| Ruud C. | 2.9% | [2.1% – 3.6%] |

**WTA**

| Player | Probability | 90% CI |
|--------|------------|--------|
| Swiatek I. | 31.8% | [29.2% – 34.6%] |
| Gauff C. | 12.0% | [10.3% – 13.7%] |
| Sabalenka A. | 11.8% | [10.1% – 13.6%] |
| Anisimova A. | 7.0% | [5.8% – 8.5%] |
| Rybakina E. | 6.6% | [5.4% – 8.0%] |

*Note: probabilities above use a placeholder draw (AO26 draw). Final probabilities will be updated when the real FO26 draw is released (~22 May 2026).*

---

## Project structure

```
grand-slam-predictions/
├── config/
│   ├── tournaments.py        # Tournament configs (surface, paths, draw size)
│   └── surfaces.py           # Surface maps and training filters
├── data/
│   ├── raw/                  # ATP/WTA rankings + tennis_data match files
│   ├── draws/
│   │   ├── ao26/             # atp_draw.csv, wta_draw.csv
│   │   └── fo26/             # atp_draw.csv, wta_draw.csv
│   └── processed/
│       ├── atp/              # Intermediate CSVs for ATP pipeline
│       └── wta/              # Intermediate CSVs for WTA pipeline
├── models/
│   ├── ao26/                 # atp_model.pkl, wta_model.pkl
│   └── fo26/                 # atp_model.pkl, wta_model.pkl
├── results/
│   ├── ao26/                 # Title probabilities + stress test charts
│   └── fo26/                 # Title probabilities + stress test charts
└── scripts/
    ├── 00_load_all_tour_data.py
    ├── 01_preprocess_raw_data.py
    ├── 02_build_player_history.py
    ├── 02b_build_player_history_all_surfaces.py
    ├── 03_build_rolling_features.py
    ├── 03b_build_rolling_features_all_surfaces.py
    ├── 04_build_ml_dataset.py
    ├── 05_train_model.py
    ├── 06_simulate_tournament.py
    ├── 07_build_surface_features.py
    ├── 08_build_elo_ratings.py
    ├── 09_stress_test.py
    ├── name_utils.py
    └── predict_match.py
```

---

## How to run

### First-time setup (run once)

```bash
# Build all-surface player history and rolling stats
python3 scripts/02b_build_player_history_all_surfaces.py
python3 scripts/03b_build_rolling_features_all_surfaces.py

# Build Elo ratings (global + clay/hard/grass)
python3 scripts/08_build_elo_ratings.py
```

### Run for a specific tournament

Replace `fo26` with `ao26` for the Australian Open.

```bash
# 1. Build surface-specific rolling features
python3 scripts/07_build_surface_features.py --tour atp --tournament fo26
python3 scripts/07_build_surface_features.py --tour wta --tournament fo26

# 2. Build ML dataset
python3 scripts/04_build_ml_dataset.py --tournament fo26 --tour atp
python3 scripts/04_build_ml_dataset.py --tournament fo26 --tour wta

# 3. Train models
python3 scripts/05_train_model.py --tournament fo26 --tour atp
python3 scripts/05_train_model.py --tournament fo26 --tour wta

# 4. Simulate bracket (requires draw CSV in data/draws/fo26/)
python3 scripts/06_simulate_tournament.py --tournament fo26 --tour atp
python3 scripts/06_simulate_tournament.py --tournament fo26 --tour wta

# 5. Stress test
python3 scripts/09_stress_test.py --tournament fo26 --tour atp
python3 scripts/09_stress_test.py --tournament fo26 --tour wta
```

### Adding a new tournament (e.g. Wimbledon 2026)

1. Add an entry to `config/tournaments.py` with surface `"grass"`
2. Add `"wimbledon26"` to `config/surfaces.py` TOURNAMENT_SURFACE dict
3. Drop `atp_draw.csv` and `wta_draw.csv` into `data/draws/wimbledon26/`
4. Run the pipeline above with `--tournament wimbledon26`

---

## Draw CSV format

Draw files go in `data/draws/<tournament>/atp_draw.csv` and `wta_draw.csv`.
One row per first-round match, two columns:

```
player_A,player_B
Alcaraz C.,Zandschulp B.
Sinner J.,Cazaux A.
...
```

---

## Dependencies

```
pandas
numpy
scikit-learn
joblib
matplotlib
openpyxl
```

Install with:
```bash
pip3 install pandas numpy scikit-learn joblib matplotlib openpyxl
```
