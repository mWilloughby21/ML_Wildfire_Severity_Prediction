# clean_data.py

# python -m src.data.clean_data

from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

CLEANED_DATA_PATH = Path("data/processed/cleaned_data.csv")
BASE_TABLE_PATH = Path("data/raw/ML_Project_Data.csv")
WEATHER_FOLDER_PATH = Path("data/raw/ML_weather_data")

# ============================================================
# NEW: thresholds + lookback windows for derived weather features.
# ERA5 precipitation is in meters per timestep, so 0.005 == 5 mm.
# ============================================================
RAIN_DAY_THRESHOLD_M = 0.005
LOOKBACK_DAYS = 30


## Main function for data cleaning
def clean_data():
    df, ds = load_data()
    ds_daily = transform_weather_data(ds)
    # ============================================================
    # NEW: materialize the daily dataset into memory once upfront.
    # Without this, every per-fire .compute() re-opens many .nc
    # files. The daily aggregates fit easily in RAM (~800 MB) and
    # turn the per-fire loop into pure in-memory slicing.
    # ============================================================
    print('Materializing daily weather dataset into memory...')
    ds_daily = ds_daily.load()
    print('Done.')
    df = merge_data(df, ds_daily)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print_data_info(df)


## Merge weather data with base table on date
def merge_data(df: pd.DataFrame, ds_daily: xr.Dataset) -> pd.DataFrame:
    features = []

    for i, fire in df.iterrows():
        print(f"Processing fire {i + 1}/{len(df)}")
        date = fire['START_DATE']

        # ============================================================
        # CHANGED: pull a single 30-day window per fire and derive all
        # aggregations from it (was a 3-day window only).
        # ============================================================
        window_30d = ds_daily.sel(
            valid_time=slice(date - pd.Timedelta(days=LOOKBACK_DAYS - 1), date)
        ).sel(
            latitude=fire['LATITUDE'],
            longitude=fire['LONGITUDE'],
            method='nearest',
        )

        # 3-day window (existing features, kept for backward compatibility)
        w3 = window_30d.sel(valid_time=slice(date - pd.Timedelta(days=3), date))
        # --- NEW: 7-day window for fire-weather extremes ---
        w7 = window_30d.sel(valid_time=slice(date - pd.Timedelta(days=6), date))

        # --- NEW: days since last meaningful rain (>= 5mm) within 30 days ---
        precip_30d = window_30d['PRECIPITATION'].values
        times_30d = window_30d['valid_time'].values
        rain_idx = np.where(precip_30d >= RAIN_DAY_THRESHOLD_M)[0]
        if len(rain_idx) > 0:
            last_rain = times_30d[rain_idx[-1]]
            days_since_rain = int(
                (np.datetime64(date) - last_rain) / np.timedelta64(1, 'D')
            )
        else:
            days_since_rain = LOOKBACK_DAYS

        features.append({
            # Existing 3-day features
            'TEMPERATURE': float(w3['TEMPERATURE_MEAN'].mean()),
            'PRECIPITATION': float(w3['PRECIPITATION'].sum()),
            'WIND_SPEED': float(w3['WIND_SPEED_MEAN'].mean()),
            'HUMIDITY': float(w3['HUMIDITY_MEAN'].mean()),
            # --- NEW: 7-day fire-weather extremes ---
            'TEMPERATURE_7D_MAX': float(w7['TEMPERATURE_MAX'].max()),
            'HUMIDITY_7D_MIN': float(w7['HUMIDITY_MIN'].min()),
            'WIND_SPEED_7D_MAX': float(w7['WIND_SPEED_MAX'].max()),
            'PRECIPITATION_7D_SUM': float(w7['PRECIPITATION'].sum()),
            # --- NEW: vapor pressure deficit (kPa) ---
            'VPD_7D_MEAN': float(w7['VPD_MEAN'].mean()),
            'VPD_7D_MAX': float(w7['VPD_MAX'].max()),
            # --- NEW: 30-day drought indicators ---
            'PRECIPITATION_30D_SUM': float(window_30d['PRECIPITATION'].sum()),
            'DAYS_SINCE_RAIN': days_since_rain,
        })

    weather_df = pd.DataFrame(features)
    df = pd.concat([df.reset_index(drop=True), weather_df], axis=1)
    return df


## Transform weather dataset to correct features and daily resolution
# ============================================================
# CHANGED: now derives wind speed, humidity, and VPD at HOURLY
# resolution before resampling. The original computed daily means
# of u10/v10 first, then took the magnitude — which under-estimates
# wind speed when direction varies. Same issue for humidity, which
# is non-linear in temp/dewpoint. Daily MAX/MIN aggregations are
# also new (used for 7-day extremes).
# ============================================================
def transform_weather_data(ds):
    # Hourly derived variables
    temp_c = ds['t2m'] - 273.15
    dew_c = ds['d2m'] - 273.15
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)

    humidity = 100 * np.exp(
        (17.625 * dew_c) / (243.04 + dew_c)
        - (17.625 * temp_c) / (243.04 + temp_c)
    )

    # --- NEW: vapor pressure deficit (Tetens formula, kPa) ---
    es_t = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    es_td = 0.6108 * np.exp(17.27 * dew_c / (dew_c + 237.3))
    vpd = es_t - es_td

    daily = lambda da, how: getattr(da.resample(valid_time='1D'), how)()

    ds_daily = xr.Dataset({
        'TEMPERATURE_MEAN': daily(temp_c, 'mean'),
        'TEMPERATURE_MAX': daily(temp_c, 'max'),       # NEW
        'HUMIDITY_MEAN': daily(humidity, 'mean'),
        'HUMIDITY_MIN': daily(humidity, 'min'),        # NEW
        'WIND_SPEED_MEAN': daily(wind_speed, 'mean'),
        'WIND_SPEED_MAX': daily(wind_speed, 'max'),    # NEW
        'VPD_MEAN': daily(vpd, 'mean'),                # NEW
        'VPD_MAX': daily(vpd, 'max'),                  # NEW
        'PRECIPITATION': daily(ds['tp'], 'sum'),
    })
    return ds_daily


## Load base table and weather data
def load_data():
    # Load base table into DataFrame and set date format
    df = pd.read_csv(BASE_TABLE_PATH)
    df['START_DATE'] = pd.to_datetime(df['START_DATE']).dt.date

    # Load weather data into DataSet
    ds = xr.open_mfdataset(
        f"{WEATHER_FOLDER_PATH}/*/*.nc",
        combine='by_coords',
        engine='netcdf4',
        chunks='auto',
        compat='no_conflicts',
    )

    return df, ds


## Print basic info about the cleaned data
def print_data_info(df):
    print(df.head())
    print(df.info())
    print(df.describe())


if __name__ == "__main__":
    clean_data()
