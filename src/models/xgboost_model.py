# xgboost_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_PATH = Path("data/processed/cleaned_data.csv")
VALID_SEVERITY_CODES = [1, 2, 3, 4]


def load_features():
    data = pd.read_csv(DATA_PATH)
    data = data[data['BURN_SEVERITY_CODE'].isin(VALID_SEVERITY_CODES)].copy()
    start = pd.to_datetime(data['START_DATE'])
    data['MONTH'] = start.dt.month
    data['DAY_OF_YEAR'] = start.dt.dayofyear
    data['MONTH_SIN'] = np.sin(2 * np.pi * data['MONTH'] / 12)
    data['MONTH_COS'] = np.cos(2 * np.pi * data['MONTH'] / 12)
    data['DOY_SIN'] = np.sin(2 * np.pi * data['DAY_OF_YEAR'] / 365)
    data['DOY_COS'] = np.cos(2 * np.pi * data['DAY_OF_YEAR'] / 365)
    data['LOG_ACRES'] = np.log1p(data['ACRES_BURNED'])
    data = pd.get_dummies(data, columns=['FBFM40_CODE', 'BPS_CODE'])
    drop_cols = [
        'ACRES_BURNED', 'START_DATE',
        'BURN_SEVERITY_CODE', 'SEVERITY_DESC',
        'FBFM40_TITLE', 'BPS_NAME',
    ]
    X = data.drop(columns=drop_cols)
    y = data['BURN_SEVERITY_CODE'].astype(int) - 1
    return X, y

def train_and_evaluate(seed=42):
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train,
    )
    
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        random_state=42,
        tree_method='hist',
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    
    return {
        'name': 'XGBoost',
        'model': model,
        'y_true': y_test.to_numpy(),
        'y_pred': y_pred,
        'y_proba': y_proba,
        'feature_importances': importances,
        'class_names': ['Unburned/Low', 'Low', 'Moderate', 'High'],
    }

def train_model():
    result = train_and_evaluate()
    print_evaluation(result['y_true'], result['y_pred'])
    print_top_features(result['feature_importances'], n=15)

def print_evaluation(y_test, y_pred):
    target_names = ['Unburned/Low', 'Low', 'Moderate', 'High']
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=target_names, zero_division=0)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

def print_top_features(importances, n=15):
    print(f"\nTop {n} features by importance:")
    print(importances.sort_values(ascending=False).head(n).to_string())