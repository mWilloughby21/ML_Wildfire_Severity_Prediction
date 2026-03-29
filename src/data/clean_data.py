# clean_data.py

import pandas as pd
from pathlib import Path

CSV_PATH = Path('data/raw/wildfires.csv')

## Force datatypes to str type
dtype_map = {
    "CN": "str",
    "COMPLEXNAME": "str",
    "SOFIRENUM": "str",
    "LOCALFIRENUM": "str",
    "SECURITYID": "str",
    "COMMENTS": "str",
    "DATASOURCE": "str"
}


## Read csv into dataframe
df = pd.read_csv(CSV_PATH, dtype=dtype_map, low_memory=False)


## Fix invalid values
df = df[
    df['LATDD83'].between(-90, 90) &
    df['LONGDD83'].between(-180, 180)
]
df = df[(df['TOTALACRES'] > 0)]


## Date features
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


## Drop useless columns
df = df.drop(columns=[
    'OBJECTID', 'GLOBALID', 'FIREOCCURID', 'FIREOCCURID', 'CN', 'REVDATE', 'FIRENAME', 'COMPLEXNAME', 'FIREYEAR', 'UNIQFIREID', 
    'SOFIRENUM', 'LOCALFIRENUM', 'SECURITYID', 'SIZECLASS', 'STATCAUSE', 'COMMENTS', 'DATASOURCE', 'OWNERAGENCY', 'UNITIDOWNER', 'PROTECTIONAGENCY', 
    'UNITIDPROTECT', 'POINTTYPE', 'PERIMEXISTS', 'FIRERPTQC', 'DBSOURCEID', 'DBSOURCEDATE', 'ACCURACY', 'FIREOUTDATETIME', 'DISCOVERYDATETIME'
])


## Replace blank values with NA
df['FIRETYPECATEGORY'] = df['FIRETYPECATEGORY'].apply(lambda x: pd.NA if isinstance(x, str) and x.strip() == "" else x)


## Drop null values
df = df.dropna()

## Rename columns
df.rename(columns={
    'TOTALACRES': 'TOTAL_ACRES',
    'LATDD83': 'LAT',
    'LONGDD83': 'LONG',
    'FIRETYPECATEGORY': 'FIRE_TYPE_CATEGORY'
}, inplace=True)

print(df.shape)
print(df.head(5))
print(df.info())
print(df.describe())