# Wildfire Burn Severity Prediction

Predict the post-fire burn severity of wildfires in the Pacific Northwest
from pre-fire conditions — fire location, fuel type, vegetation,
topography, and a 30-day weather lookback — using gradient-boosted trees
and a multilayer perceptron.

## Problem

Given a wildfire's ignition date and location, predict its post-fire
**burn severity class** as defined by the MTBS / BARC scale:

| Code | Description                                 |
|------|---------------------------------------------|
| 1    | Unburned / Underburned to Low Burn Severity |
| 2    | Low Burn Severity                           |
| 3    | Moderate Burn Severity                      |
| 4    | High Burn Severity                          |

Classes 5 ("Increased Greenness") and 6 ("Background/No Data") appear in
the source data but are not real severity outcomes (combined < 1.2% of
samples) and are dropped during modeling.

## Data

The cleaned dataset is built by two scripts in [src/data/](src/data/) from
three free public sources:

1. **Wildfire records** — [data/raw/ML_Project_Data.csv](data/raw/ML_Project_Data.csv).
   One row per fire with start date, lat/lon, acres burned, severity
   label, fuel model (FBFM40), and biophysical setting (BPS).
2. **ERA5 weather reanalysis** — `data/raw/ML_weather_data/*.nc`.
   Hourly 2-m temperature, dewpoint, 10-m wind components, and total
   precipitation, on a 0.25° grid covering Idaho.
3. **SRTM 30m elevation** — fetched on demand from the public
   [OpenTopoData API](https://www.opentopodata.org/datasets/srtm/).

[src/data/clean_data.py](src/data/clean_data.py) merges the wildfire
records with ERA5 weather. For each fire it pulls a **30-day weather
window ending on the ignition date** at the nearest grid point, derives
hourly relative humidity, wind speed, and vapor pressure deficit, then
aggregates over 3-day, 7-day, and 30-day windows to produce the weather
features below.

[src/data/add_topography.py](src/data/add_topography.py) augments the
result with elevation, slope, and aspect derived from a 3×3 SRTM kernel
sampled around each fire.

The final cleaned table is written to
[data/processed/cleaned_data.csv](data/processed/cleaned_data.csv) — 26
columns, one row per fire.

### Final feature set

| Feature                              | Source                  | Notes                                    |
|--------------------------------------|-------------------------|------------------------------------------|
| `ACRES_BURNED`                       | Fire record             | Log-transformed during modeling          |
| `LATITUDE`, `LONGITUDE`              | Fire record             | Fire location                            |
| `FBFM40_CODE`                        | Fire record             | Surface fuel model (categorical)         |
| `BPS_CODE`                           | Fire record             | Biophysical setting (categorical)        |
| `TEMPERATURE`                        | ERA5, 3-day mean        | °C                                       |
| `PRECIPITATION`                      | ERA5, 3-day sum         | meters                                   |
| `WIND_SPEED`                         | ERA5, 3-day mean        | m/s; computed hourly then averaged       |
| `HUMIDITY`                           | ERA5, 3-day mean        | %; from temp + dewpoint                  |
| `TEMPERATURE_7D_MAX`                 | ERA5, 7-day max         | Peak temperature in week leading up      |
| `HUMIDITY_7D_MIN`                    | ERA5, 7-day min         | Driest hour in week leading up           |
| `WIND_SPEED_7D_MAX`                  | ERA5, 7-day max         | Peak gust in week leading up             |
| `PRECIPITATION_7D_SUM`               | ERA5, 7-day sum         | Recent rainfall                          |
| `VPD_7D_MEAN`, `VPD_7D_MAX`          | ERA5, derived           | Vapor pressure deficit (kPa)             |
| `PRECIPITATION_30D_SUM`              | ERA5, 30-day sum        | Drought-state proxy                      |
| `DAYS_SINCE_RAIN`                    | ERA5, derived           | Days since last day with ≥ 5 mm rain     |
| `ELEVATION`                          | SRTM 30m                | meters                                   |
| `SLOPE`                              | SRTM 30m, derived       | degrees (Horn's method, 100 m kernel)    |
| `ASPECT_SIN`, `ASPECT_COS`           | SRTM 30m, derived       | Cyclical encoding of compass aspect      |
| `MONTH`, `DAY_OF_YEAR`               | Derived                 | With sin/cos cyclical encoding           |

## Models

All four files live in [src/models/](src/models/):

- **[xgboost_model.py](src/models/xgboost_model.py)** — gradient-boosted
  trees with class-stratified train/val/test splits and early stopping.
- **[mlp_model.py](src/models/mlp_model.py)** — scikit-learn
  `MLPClassifier` (64 → 32 ReLU) with standardized numerics, one-hot
  categoricals, and early stopping.
- **[compare_models.py](src/models/compare_models.py)** — head-to-head
  comparison of XGBoost vs MLP across 5 train/test splits, reporting
  accuracy ± std, macro F1 ± std, and per-class F1.
- **[ensemble_model.py](src/models/ensemble_model.py)** — combines a
  seed-ensemble MLP (5 seeds, predicted probabilities averaged) with
  XGBoost via soft-vote probability averaging.

The majority-class baseline (always predict "Low") is **49.5%**, so any
model needs to clear that bar to be useful.

### Results

Single 80/20 stratified split with `random_state=42`:

| Model                                       | Accuracy   |
|---------------------------------------------|------------|
| Majority-class baseline                     | 49.5%      |
| XGBoost                                     | 51.92%     |
| MLP (single seed)                           | 46.69% (high variance — see below) |
| MLP ensemble (5 seeds, prob. average)       | 52.96%     |
| **XGBoost + MLP-ensemble (soft vote)**      | **54.01%** |

Across 5 random splits the MLP varies a lot (std 2.66%), which the
ensemble collapses by averaging probabilities; XGBoost is much more
stable (std 0.84%).

## Running

The fastest way to use the project is the menu in [src/main.py](src/main.py):

```bash
# From the project root: /Semester Project/

# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. (Only needed once, or after raw data changes)
#    Build the cleaned dataset. Slow on first run because it loads
#    ~1000 ERA5 .nc files; cached daily aggregates make it fast after.
python -m src.data.clean_data
python -m src.data.add_topography

# 3. Launch the interactive menu
python -m src.main
```

The menu offers:

```
1. Train and evaluate XGBoost     — single-model run with confusion-matrix
                                    heatmap + feature-importance plot
2. Train and evaluate MLP         — single-model run with confusion-matrix +
                                    per-class F1 plot
3. Compare XGBoost vs MLP         — head-to-head text table + side-by-side
                                    confusion matrices + per-class F1 bars
q. Quit
```

Each option prints a clean text summary first, then opens matplotlib
windows. Closing the windows returns you to the menu.

### Running components individually

If you want to skip the menu, every model file works as a standalone
script:

```bash
python -m src.models.xgboost_model      # train + evaluate XGBoost
python -m src.models.mlp_model          # train + evaluate MLP
python -m src.models.compare_models     # 5-split head-to-head comparison
python -m src.models.ensemble_model     # XGBoost + seed-ensemble MLP
```

## Project layout

```
Semester Project/
├── data/
│   ├── raw/                             # source CSV + ERA5 .nc files (gitignored)
│   └── processed/cleaned_data.csv       # merged dataset (26 cols)
├── src/
│   ├── main.py                          # menu-driven entry point
│   ├── data/
│   │   ├── clean_data.py                # builds cleaned_data.csv from CSV + ERA5
│   │   └── add_topography.py            # augments CSV with elevation/slope/aspect
│   ├── models/
│   │   ├── xgboost_model.py
│   │   ├── mlp_model.py
│   │   ├── compare_models.py            # XGBoost vs MLP head-to-head
│   │   └── ensemble_model.py            # XGB + MLP-ensemble soft vote
│   └── visualization/
│       └── plots.py                     # confusion-matrix / F1 / importance plots
├── requirements.txt
└── README.md
```
