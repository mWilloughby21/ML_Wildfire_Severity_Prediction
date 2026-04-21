# clean_data.py

# python -m src.data.clean_data

import time
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

CLEANED_DATA_PATH = Path('data/processed/cleaned_data.csv')
BASE_TABLE_PATH = Path("data/raw/ML_Project_Data.csv")
WEATHER_FOLDER_PATH = Path("data/raw/ML_weather_data")

## Main function for data cleaning
def clean_data():
    df, ds = load_data()
    ds_daily = transform_weather_data(ds)
    df = merge_data(df, ds_daily)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print_data_info(df)

## Merge weather data with base table on date
def merge_data(df: pd.DataFrame, ds_daily: xr.Dataset) -> pd.DataFrame:
    features = []
    
    # Extract weather features for each fire based on date and location
    for i, fire in df.iterrows():
        print(f"Processing fire {i + 1}/{len(df)}")
        date = fire['START_DATE']
        
        # 3-day window of weather data leading up to fire date at fire location
        window = ds_daily.sel(
            valid_time=slice(date - pd.Timedelta(days=3), date)
        )
        window = window.sel(
            latitude=fire['LATITUDE'],
            longitude=fire['LONGITUDE'],
            method='nearest'
        )
        
        # Avereage features over 3-day window
        window = window.compute()
        
        features.append({
            'TEMPERATURE': window['TEMPERATURE'].mean().item(),
            'PRECIPITATION': window['PRECIPITATION'].sum().item(),
            'WIND_SPEED': window['WIND_SPEED'].mean().item(),
            'HUMIDITY': window['HUMIDITY'].mean().item(),
        })
    
    # Create temp weather features DataFrame
    weather_df = pd.DataFrame(features)
    
    # Concatenate with base table
    df = pd.concat([df.reset_index(drop=True), weather_df], axis=1)
    
    return df

## Transform weather dataset to coorect features and daily resolution
def transform_weather_data(ds):
    # Get daily averages
    ds_daily = xr.Dataset({
        't2m': ds['t2m'].resample(valid_time='1D').mean(), # Temperature at 2m
        'd2m': ds['d2m'].resample(valid_time='1D').mean(), # Dewpoint at 2m
        'u10': ds['u10'].resample(valid_time='1D').mean(), # E-W wind at 10m
        'v10': ds['v10'].resample(valid_time='1D').mean(), # N-S wind at 10m
        'tp': ds['tp'].resample(valid_time='1D').sum(), # Total precipitation
    })
    
    # Convert wind vectors to single magnitude
    ds_daily['WIND_SPEED'] = np.sqrt(ds_daily['u10']**2 + ds_daily['v10']**2)
    
    # Rename precipitation variable
    ds_daily = ds_daily.rename({'tp': 'PRECIPITATION'})
    
    # Convert temperature from Kelvin to Celsius
    ds_daily['TEMPERATURE'] = ds_daily['t2m'] - 273.15
    ds_daily['DEWPOINT'] = ds_daily['d2m'] - 273.15
    
    # Calculate relative humidity from temperature and dewpoint
    ds_daily['HUMIDITY'] = 100 * np.exp(
        (17.625*ds_daily['DEWPOINT']) / (243.04 + ds_daily['DEWPOINT'])
        - (17.625*ds_daily['TEMPERATURE']) / (243.04 + ds_daily['TEMPERATURE'])
    )
    
    # Drop original componets
    ds_daily = ds_daily.drop_vars(['u10', 'v10', 'd2m', 't2m', 'DEWPOINT'])
    
    return ds_daily

## Load base table and weather data
def load_data():
    # Load base table into DataFrame and set date format
    df = pd.read_csv(BASE_TABLE_PATH)
    df['START_DATE'] = pd.to_datetime(df['START_DATE']).dt.date
    
    # Load weather data into DataSet
    ds = xr.open_mfdataset(
        f'{WEATHER_FOLDER_PATH}/*/*.nc',
        combine='by_coords',
        engine='netcdf4',
        chunks='auto',
        compat='no_conflicts'
    )
    
    return df, ds

## Print basic info about the cleaned data
def print_data_info(df):
    print(df.head())
    print(df.info())
    print(df.describe())

def temp():
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(df.head())

if __name__ == "__main__":
    # clean_data()
    temp()