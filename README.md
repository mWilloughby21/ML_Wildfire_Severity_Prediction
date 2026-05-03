# Wildfire Burn Severity Prediction

Predict the post-fire burn severity of wildfires in **Idaho** from
pre-fire conditions — fire location, surface fuel model, biophysical
setting, topography, and a 30-day weather lookback — using
gradient-boosted trees and a multilayer perceptron.

## Problem

Given a wildfire's ignition date and location, predict its post-fire
**burn severity class** as defined by the MTBS / BARC scale:

| Code | Description                                 |
|------|---------------------------------------------|
| 1    | Unburned / Underburned to Low Burn Severity |
| 2    | Low Burn Severity                           |
| 3    | Moderate Burn Severity                      |
| 4    | High Burn Severity                          |

Classes 5 ("Increased Greenness") and 6 ("Background / No Data") appear
in the source data but are not real severity outcomes (combined ~1.1%
of samples) and are dropped during modeling, along with rows that are
missing a severity code. This leaves **1,432** of 1,508 fires for
training and evaluation.

## Data

The dataset spans **1,508 wildfires in Idaho from 1984–2024**, built
by combining four free public sources:

| Source         | Provider                                      | What it contributes                                            |
|----------------|-----------------------------------------------|----------------------------------------------------------------|
| **MTBS**       | USGS / USFS — [mtbs.gov](https://www.mtbs.gov/) | Fire perimeters, ignition dates, acres burned, BARC severity class |
| **LANDFIRE**   | USGS / USFS — [landfire.gov](https://landfire.gov/) | Surface fuel model (FBFM40) and biophysical setting (BPS) sampled at each fire's ignition point |
| **ERA5**       | ECMWF / Copernicus Climate Data Store ([cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/)) | Hourly 2-m temperature & dewpoint, 10-m wind components, total precipitation on a 0.25° grid |
| **SRTM 30m**   | NASA, served by [OpenTopoData](https://www.opentopodata.org/datasets/srtm/) | 30-meter elevation grid (used for elevation, slope, aspect)    |

The MTBS and LANDFIRE pulls are pre-merged into a single CSV at
[data/raw/ML_Project_Data.csv](data/raw/ML_Project_Data.csv) — one row
per fire with start date, lat/lon, acres, severity label, FBFM40 code,
and BPS code (23 unique fuel models, 66 unique biophysical settings).

ERA5 reanalysis data was downloaded from the Copernicus Climate Data
Store (CDS) for the Idaho bounding box, one month per request, and
stored as 492 monthly subfolders under
`data/raw/ML_weather_data/era5_idaho_YYYY_MM/` (each containing two
NetCDF files — one for instantaneous fields, one for accumulated
precipitation, ~984 `.nc` files total).

### Building the cleaned dataset

Two scripts in [src/data/](src/data/) turn the raw inputs into the
modeling table:

[src/data/clean_data.py](src/data/clean_data.py) merges the wildfire
records with ERA5 weather. For each fire it pulls a **30-day weather
window ending on the ignition date** at the nearest ERA5 grid point,
derives hourly relative humidity (Magnus formula), wind speed
(`sqrt(u² + v²)`), and vapor pressure deficit (Tetens), then
aggregates over 3-day, 7-day, and 30-day windows.

[src/data/add_topography.py](src/data/add_topography.py) augments the
result with elevation, slope, and aspect. For each fire it queries the
public OpenTopoData SRTM endpoint for a 3×3 elevation kernel spaced
~100 m apart around the ignition point, computes slope and aspect via
Horn's method, and writes them back into the same CSV. Calls are
batched (100 points per request) and rate-limited to one request per
second.

The final cleaned table is at
[data/processed/cleaned_data.csv](data/processed/cleaned_data.csv) —
1,508 rows × 26 columns. This file is checked into the repo so the
models can be run without re-fetching the ~984 ERA5 NetCDFs or
re-querying OpenTopoData.

### Final feature set (used by the models)

| Feature                              | Source                  | Notes                                       |
|--------------------------------------|-------------------------|---------------------------------------------|
| `LOG_ACRES`                          | Derived from `ACRES_BURNED` | `log1p(acres)`; replaces raw acres      |
| `LATITUDE`, `LONGITUDE`              | MTBS                    | Fire ignition point                         |
| `FBFM40_CODE`                        | LANDFIRE                | Surface fuel model (one-hot in MLP)         |
| `BPS_CODE`                           | LANDFIRE                | Biophysical setting (one-hot in MLP)        |
| `TEMPERATURE`                        | ERA5, 3-day mean        | °C                                          |
| `PRECIPITATION`                      | ERA5, 3-day sum         | meters                                      |
| `WIND_SPEED`                         | ERA5, 3-day mean        | m/s; computed hourly then averaged          |
| `HUMIDITY`                           | ERA5, 3-day mean        | %; from temperature + dewpoint              |
| `TEMPERATURE_7D_MAX`                 | ERA5, 7-day max         | Peak temperature in week leading up         |
| `HUMIDITY_7D_MIN`                    | ERA5, 7-day min         | Driest hour in week leading up              |
| `WIND_SPEED_7D_MAX`                  | ERA5, 7-day max         | Peak gust in week leading up                |
| `PRECIPITATION_7D_SUM`               | ERA5, 7-day sum         | Recent rainfall                             |
| `VPD_7D_MEAN`, `VPD_7D_MAX`          | ERA5, derived           | Vapor pressure deficit (kPa)                |
| `PRECIPITATION_30D_SUM`              | ERA5, 30-day sum        | Drought-state proxy                         |
| `DAYS_SINCE_RAIN`                    | ERA5, derived           | Days since last day with ≥ 5 mm rain        |
| `ELEVATION`                          | SRTM 30m                | meters                                      |
| `SLOPE`                              | SRTM 30m, derived       | degrees (Horn's method, 100 m kernel)       |
| `ASPECT_SIN`, `ASPECT_COS`           | SRTM 30m, derived       | Cyclical encoding of compass aspect         |
| `MONTH`, `DAY_OF_YEAR`               | Derived from `START_DATE` | Raw integer date features (XGBoost only)  |
| `MONTH_SIN`, `MONTH_COS`             | Derived from `START_DATE` | `sin/cos(2π · month / 12)`                |
| `DOY_SIN`, `DOY_COS`                 | Derived from `START_DATE` | `sin/cos(2π · day_of_year / 365)`         |

XGBoost handles the categorical fuel/BPS codes via one-hot expansion in
`load_features` (pandas `get_dummies`); the MLP pipeline does the same
inside a scikit-learn `ColumnTransformer` with `OneHotEncoder` and
applies a `StandardScaler` to all numeric columns.

## Models

All four files live in [src/models/](src/models/):

- **[xgboost_model.py](src/models/xgboost_model.py)** — gradient-boosted
  trees (`XGBClassifier`, `multi:softprob`, 1000 estimators with early
  stopping at 30 rounds, `learning_rate=0.03`, `max_depth=3`,
  `min_child_weight=5`, `subsample=0.7`, `colsample_bytree=0.7`,
  `reg_lambda=1.0`, `tree_method='hist'`). Class-stratified 80/20
  train/test split, with 20% of the training portion held out for
  early-stopping validation.
- **[mlp_model.py](src/models/mlp_model.py)** — scikit-learn
  `MLPClassifier` (hidden layers 64 → 32, ReLU, Adam,
  `alpha=1e-3`, `batch_size=64`, `learning_rate_init=1e-3`, max 500
  iterations, early stopping with 15% internal validation and
  `n_iter_no_change=20`). Numerics are standardized; categoricals
  are one-hot encoded via `ColumnTransformer`.
- **[compare_models.py](src/models/compare_models.py)** — standalone
  script (not wired into the menu) that retrains both models across 5
  stratified 80/20 splits (seeds `[42, 0, 1, 7, 13]`) and prints
  accuracy ± std, macro F1 ± std, per-class F1, training time, and
  confusion matrices for the seed=42 run.
- **[ensemble_model.py](src/models/ensemble_model.py)** — combines a
  seed-ensemble MLP (5 seeds: `[0, 1, 2, 3, 4]`, predicted
  probabilities averaged) with XGBoost via soft-vote probability
  averaging (`(xgb_proba + mlp_proba) / 2`, then `argmax`).

The majority-class baseline (always predict "Low") on the 1,432-fire
filtered dataset is **50.1%**, so any model needs to clear that bar
to be useful.

### Results — single 80/20 stratified split (seed=42)

| Model                                       | Accuracy   |
|---------------------------------------------|------------|
| Majority-class baseline                     | 50.1%      |
| XGBoost                                     | 51.92%     |
| MLP (single seed)                           | 46.69%     |
| MLP ensemble (5 seeds, prob. average)       | 52.96%     |
| **XGBoost + MLP-ensemble (soft vote)**      | **54.01%** |

The soft-vote ensemble beats either component on its own — the XGBoost
+ MLP-ensemble combination outperforms XGBoost alone by ~2.1 points
and the MLP-ensemble alone by ~1.1 points on the seed=42 holdout.

### Results — 5-split stability check (`compare_models.py`)

| Metric              | XGBoost              | MLP                  |
|---------------------|----------------------|----------------------|
| Accuracy            | 51.29% ± 1.71%       | 49.76% ± 2.66%       |
| Macro F1            | 0.299 ± 0.018        | 0.319 ± 0.035        |
| Per-class F1 — Unburned/Low | 0.094 ± 0.030 | 0.118 ± 0.051       |
| Per-class F1 — Low          | 0.682 ± 0.010 | 0.671 ± 0.020       |
| Per-class F1 — Moderate     | 0.211 ± 0.038 | 0.277 ± 0.064       |
| Per-class F1 — High         | 0.209 ± 0.038 | 0.210 ± 0.118       |
| Mean train time     | ~0.3 s               | ~0.1 s               |

XGBoost is the more stable model on accuracy (std 1.71% vs 2.66%) and
the much more stable model on per-class F1 (especially for the
"High" class, where MLP swings ~5× as widely). The MLP edges out
XGBoost on macro F1 because it's more willing to predict the minority
classes — which is exactly why the soft-vote ensemble helps: XGBoost
contributes calibration on the dominant "Low" class, the MLP ensemble
contributes coverage on the rarer ones.

### Feature importance (XGBoost, seed=42)

Only XGBoost reports feature importance here — tree-based models
expose it natively (each split is attributable to one feature, so the
total gain per feature is just bookkeeping). The MLP doesn't, because
a neural net distributes signal across thousands of weights and
nonlinear activations; getting a comparable ranking would require
permutation importance (`sklearn.inspection.permutation_importance`),
which isn't currently wired in.

The categorical features `BPS_CODE` and `FBFM40_CODE` are one-hot
encoded for training, so XGBoost reports an importance score for each
of the 23 fuel-model levels and 66 BPS levels separately. The top-15
plot in [src/visualization/plots.py](src/visualization/plots.py)
**sums** the dummies of each categorical back into a single feature
before ranking — averaging would dilute their importance with all the
inactive levels and unfairly penalize variables with many categories.

Top features after that aggregation:

| Rank | Feature                 | Importance |
|------|-------------------------|------------|
| 1    | `BPS_CODE`              | 0.239      |
| 2    | `FBFM40_CODE`           | 0.173      |
| 3    | `ELEVATION`             | 0.041      |
| 4    | `SLOPE`                 | 0.037      |
| 5    | `LATITUDE`              | 0.030      |
| 6    | `MONTH_SIN`             | 0.026      |
| 7    | `WIND_SPEED_7D_MAX`     | 0.026      |
| 8    | `MONTH_COS`             | 0.025      |
| 9    | `WIND_SPEED`            | 0.024      |
| 10   | `MONTH`                 | 0.024      |
| 11   | `ASPECT_COS`            | 0.024      |
| 12   | `TEMPERATURE_7D_MAX`    | 0.023      |
| 13   | `ASPECT_SIN`            | 0.023      |
| 14   | `VPD_7D_MEAN`           | 0.022      |
| 15   | `PRECIPITATION_30D_SUM` | 0.022      |

The big takeaway: **vegetation context dominates**. The two LANDFIRE
variables together account for ~41% of total feature importance — what
*can* burn matters more than the weather conditions in the days
leading up to ignition. Topography (elevation, slope, aspect) is the
next strongest signal, followed by seasonal timing (`MONTH*`) and
fire-weather extremes (peak wind, peak temperature, mean VPD).

## Running

The fastest way to use the project is the menu in
[src/main.py](src/main.py):

```bash
# From the project root: /Semester Project/

# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. (Only needed once, or after raw data changes)
#    Build the cleaned dataset. Slow on first run because it loads
#    ~984 ERA5 .nc files; cached daily aggregates make it fast after.
#    The pre-built cleaned_data.csv is already committed, so you can
#    skip this step unless you want to rebuild from scratch.
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
4. Train and evaluate Ensemble    — XGBoost + 5-seed MLP ensemble combined by
                                    soft-vote probability averaging
q. Quit
```

Each option prints a clean text summary first, then opens matplotlib
windows. Closing the windows returns you to the menu.

> **Note:** menu option 3 retrains both models once on a single
> `random_state=42` split for a side-by-side comparison.
> `src.models.compare_models` is a separate, more thorough sweep
> (5 seeds, mean ± std) — run it as its own command (see below).

### Running components individually

Every model file works as a standalone script:

```bash
python -m src.models.xgboost_model      # train + evaluate XGBoost
python -m src.models.mlp_model          # train + evaluate MLP
python -m src.models.compare_models     # 5-split head-to-head sweep
python -m src.models.ensemble_model     # XGBoost + seed-ensemble MLP
```

## Project layout

```
Semester Project/
├── data/
│   ├── raw/                                # source MTBS+LANDFIRE CSV + ERA5 .nc (gitignored)
│   │   ├── ML_Project_Data.csv             # 1,508 fires, MTBS + LANDFIRE merged
│   │   └── ML_weather_data/                # 492 monthly ERA5 subfolders
│   │       └── era5_idaho_YYYY_MM/
│   │           ├── data_stream-oper_stepType-instant.nc
│   │           └── data_stream-oper_stepType-accum.nc
│   └── processed/
│       └── cleaned_data.csv                # 1,508 × 26, committed for convenience
├── src/
│   ├── main.py                             # menu-driven entry point
│   ├── data/
│   │   ├── clean_data.py                   # builds cleaned_data.csv from CSV + ERA5
│   │   └── add_topography.py               # augments CSV with elevation/slope/aspect
│   ├── models/
│   │   ├── xgboost_model.py                # gradient-boosted trees
│   │   ├── mlp_model.py                    # 64→32 ReLU MLP with sklearn pipeline
│   │   ├── compare_models.py               # 5-seed XGB vs MLP sweep (CLI only)
│   │   └── ensemble_model.py               # XGB + MLP-ensemble soft-vote
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                        # confusion matrix / F1 / importance plots
├── models/                                 # reserved (gitignored) for saved models
├── outputs/                                # reserved (gitignored) for saved figures
├── requirements.txt
└── README.md
```
