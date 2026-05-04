# mlp_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/cleaned_data.csv")
VALID_SEVERITY_CODES = [1, 2, 3, 4]
NUMERIC_FEATURES = [
    'LATITUDE', 'LONGITUDE',
    # 3-Day Means
    'TEMPERATURE', 'PRECIPITATION', 'WIND_SPEED', 'HUMIDITY',
    # 7-Day Fire-Weather Extremes
    'TEMPERATURE_7D_MAX', 'HUMIDITY_7D_MIN', 'WIND_SPEED_7D_MAX',
    'PRECIPITATION_7D_SUM', 'VPD_7D_MEAN', 'VPD_7D_MAX',
    # 30-Day drought indicators
    'PRECIPITATION_30D_SUM', 'DAYS_SINCE_RAIN',
    # Topography
    'ELEVATION', 'SLOPE', 'ASPECT_SIN', 'ASPECT_COS',
    # Date / Size Features
    'LOG_ACRES', 'MONTH_SIN', 'MONTH_COS', 'DOY_SIN', 'DOY_COS',
]
CATEGORICAL_FEATURES = ['FBFM40_CODE', 'BPS_CODE']


def load_features():
    data = pd.read_csv(DATA_PATH)
    data = data[data['BURN_SEVERITY_CODE'].isin(VALID_SEVERITY_CODES)].copy()
    start = pd.to_datetime(data['START_DATE'])
    month = start.dt.month
    doy = start.dt.dayofyear
    data['MONTH_SIN'] = np.sin(2 * np.pi * month / 12)
    data['MONTH_COS'] = np.cos(2 * np.pi * month / 12)
    data['DOY_SIN'] = np.sin(2 * np.pi * doy / 365)
    data['DOY_COS'] = np.cos(2 * np.pi * doy / 365)
    data['LOG_ACRES'] = np.log1p(data['ACRES_BURNED'])
    
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data['BURN_SEVERITY_CODE'].astype(int) - 1  # 0-based labels
    return X, y

def build_pipeline():
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ])
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-3,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )

    return Pipeline([('prep', preprocessor), ('mlp', mlp)])

def train_and_evaluate(seed=42):
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    
    pipeline = build_pipeline()
    pipeline.named_steps['mlp'].random_state = seed
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    mlp = pipeline.named_steps['mlp']

    return {
        'name': 'MLP',
        'model': pipeline,
        'y_true': y_test.to_numpy(),
        'y_pred': y_pred,
        'y_proba': y_proba,
        'class_names': ['Unburned/Low', 'Low', 'Moderate', 'High'],
        'n_iter': mlp.n_iter_,
        'best_val_score': mlp.best_validation_score_,
    }

def train_model():
    result = train_and_evaluate()
    print_evaluation(result['y_true'], result['y_pred'])
    print(f"\nTrained for {result['n_iter']} iterations; "
          f"best internal val accuracy: {result['best_val_score']:.4f}")

def print_evaluation(y_test, y_pred):
    target_names = ['Unburned/Low', 'Low', 'Moderate', 'High']
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=target_names, zero_division=0)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")