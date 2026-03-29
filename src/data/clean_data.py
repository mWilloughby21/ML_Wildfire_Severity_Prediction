# clean_data.py

# python -m src.data.clean_data

import pandas as pd
from pathlib import Path

CSV_PATH = Path('data/raw/wildfires.csv')

## Force datatypes to str type
DTYPE_MAP = {
    "CN": "str",
    "COMPLEXNAME": "str",
    "SOFIRENUM": "str",
    "LOCALFIRENUM": "str",
    "SECURITYID": "str",
    "COMMENTS": "str",
    "DATASOURCE": "str"
}


def clean_data():
    ## Read csv into dataframe
    df = pd.read_csv(CSV_PATH, dtype=DTYPE_MAP, low_memory=False)
    
    # Cleaning data
    df = fix_invalid_values(df)
    df = date_time_features(df)
    df = replace_blank_with_na(df)
    df = drop_columns(df)
    df = rename_columns(df)
    
    # Drop null values
    df = df.dropna()
    
    # Temp print dataframe info
    print(df.shape)
    print(df.head(5))
    print(df.info())
    print(df.describe())
    
    return df

def fix_invalid_values(df):
    
    # Valid latitude and longitude values
    df = df[
        df['LATDD83'].between(31, 50) &
        df['LONGDD83'].between(-125, -102)
    ]
    
    # Years between 1960 and 2026
    df = df[df['FIREYEAR'].between(1960, 2026)]
    
    # Total acres greater than 0
    df = df[(df['TOTALACRES'] > 0)]
    
    # Wildfire categorry only
    df = df[df['FIRETYPECATEGORY'].isin(['WF'])]
    
    return df

def date_time_features(df):
    # Create DURATION column in days
    df['DURATION'] = (
        pd.to_datetime(df['FIREOUTDATETIME'], errors='coerce') - 
        pd.to_datetime(df['DISCOVERYDATETIME'], errors='coerce')
    ).dt.total_seconds() / (86400)
    
    # Try to fill remaining NA with median of TOTALACRES and FIRETYPECATEGORY
    df['DURATION'] = df.groupby(['TOTALACRES', 'FIRETYPECATEGORY'])['DURATION'].transform(lambda x: x.fillna(x.median()))
    
    # Try to fill remaining NA with median of TOTALACRES
    df['DURATION'] = df.groupby(['TOTALACRES'])['DURATION'].transform(lambda x: x.fillna(x.median()))
    
    # Final Fallback
    df['DURATION'] = df['DURATION'].fillna(df["DURATION"].median())
    
    return df

def replace_blank_with_na(df):
    ## Replace blank values with NA
    df['FIRETYPECATEGORY'] = df['FIRETYPECATEGORY'].apply(lambda x: pd.NA if isinstance(x, str) and x.strip() == "" else x)
    
    return df

def drop_columns(df):
    df = df.drop(columns=[
        'X', 'Y', 'OBJECTID', 'GLOBALID', 'FIREOCCURID', 'FIREOCCURID', 'CN', 'REVDATE', 'FIRENAME', 'COMPLEXNAME', 'FIREYEAR', 'UNIQFIREID', 
        'SOFIRENUM', 'LOCALFIRENUM', 'SECURITYID', 'SIZECLASS', 'COMMENTS', 'DATASOURCE', 'OWNERAGENCY', 'UNITIDOWNER', 'PROTECTIONAGENCY', 'UNITIDPROTECT', 
        'FIRETYPECATEGORY','POINTTYPE', 'PERIMEXISTS', 'FIRERPTQC', 'DBSOURCEID', 'DBSOURCEDATE', 'ACCURACY', 'FIREOUTDATETIME', 'DISCOVERYDATETIME'
    ])
    
    return df

def rename_columns(df):
    df.rename(columns={
        'TOTALACRES': 'TOTAL_ACRES',
        'STATCAUSE': 'STAT_CAUSE',
        'LATDD83': 'LAT',
        'LONGDD83': 'LONG',
    }, inplace=True)
    
    return df

if __name__ == "__main__":
    clean_data()