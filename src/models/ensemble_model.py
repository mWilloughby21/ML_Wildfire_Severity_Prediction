# ensemble_model.py

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.models.mlp_model import build_pipeline as build_mlp_pipeline
from src.models.mlp_model import load_features as mlp_load_features
from src.models.xgboost_model import load_features as xgb_load_features


ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 42
TARGET_NAMES = ['Unburned/Low', 'Low', 'Moderate', 'High']


def train_xgb(X_train, y_train, X_val, y_val):
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def train_mlp_ensemble(X_train, y_train, seeds, verbose=True):
    models = []
    for s in seeds:
        pipeline = build_mlp_pipeline()
        pipeline.named_steps['mlp'].random_state = s
        pipeline.fit(X_train, y_train)
        models.append(pipeline)
        if verbose:
            print(f"  trained MLP with seed={s}")
    return models

def train_and_evaluate(seed=42):
    X_xgb, y = xgb_load_features()
    X_mlp, _ = mlp_load_features()
    
    Xxgb_tr, Xxgb_te, y_tr, y_te = train_test_split(
        X_xgb, y, test_size=0.2, random_state=seed, stratify=y,
    )
    Xmlp_tr, Xmlp_te, _, _ = train_test_split(
        X_mlp, y, test_size=0.2, random_state=seed, stratify=y,
    )
    Xxgb_tr2, Xxgb_val, y_tr2, y_val = train_test_split(
        Xxgb_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr,
    )
    
    xgb = train_xgb(Xxgb_tr2, y_tr2, Xxgb_val, y_val)
    mlps = train_mlp_ensemble(Xmlp_tr, y_tr, ENSEMBLE_SEEDS, verbose=False)
    
    xgb_proba = xgb.predict_proba(Xxgb_te)
    mlp_proba = average_proba(mlps, Xmlp_te)
    combined_proba = (xgb_proba + mlp_proba) / 2.0
    y_pred = combined_proba.argmax(axis=1)
    
    return {
        'name': f'Ensemble (XGB + MLP×{len(ENSEMBLE_SEEDS)} soft vote)',
        'y_true': np.asarray(y_te),
        'y_pred': y_pred,
        'y_proba': combined_proba,
        'class_names': TARGET_NAMES,
        'xgb_pred': xgb_proba.argmax(axis=1),
        'mlp_pred': mlp_proba.argmax(axis=1),
    }

def average_proba(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)

def report(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def main():
    X_xgb, y = xgb_load_features()
    X_mlp, y_mlp = mlp_load_features()
    assert (y.values == y_mlp.values).all()
    
    Xxgb_tr, Xxgb_te, y_tr, y_te = train_test_split(
        X_xgb, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y,
    )
    Xmlp_tr, Xmlp_te, _, _ = train_test_split(
        X_mlp, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y,
    )
    Xxgb_tr2, Xxgb_val, y_tr2, y_val = train_test_split(
        Xxgb_tr, y_tr, test_size=0.2, random_state=SPLIT_SEED, stratify=y_tr,
    )
    
    print("Training XGBoost...")
    xgb = train_xgb(Xxgb_tr2, y_tr2, Xxgb_val, y_val)
    
    print(f"Training MLP ensemble ({len(ENSEMBLE_SEEDS)} seeds)...")
    mlps = train_mlp_ensemble(Xmlp_tr, y_tr, ENSEMBLE_SEEDS)
    
    per_seed_accs = [accuracy_score(y_te, m.predict(Xmlp_te)) for m in mlps]
    print("\nIndividual MLP seed accuracies:")
    for s, a in zip(ENSEMBLE_SEEDS, per_seed_accs):
        print(f"  seed {s}: {a:.4f}")
    print(f"  mean={np.mean(per_seed_accs):.4f}, std={np.std(per_seed_accs):.4f}")
    
    xgb_proba = xgb.predict_proba(Xxgb_te)
    mlp_proba = average_proba(mlps, Xmlp_te)
    combined_proba = (xgb_proba + mlp_proba) / 2.0
    
    xgb_pred = xgb_proba.argmax(axis=1)
    mlp_pred = mlp_proba.argmax(axis=1)
    combined_pred = combined_proba.argmax(axis=1)
    
    report("XGBoost (single)", y_te, xgb_pred)
    report(f"MLP ensemble ({len(ENSEMBLE_SEEDS)} seeds, probability average)", y_te, mlp_pred)
    report("Combined: XGBoost + MLP-ensemble", y_te, combined_pred)
    
    print("\n=== Summary ===")
    print(f"XGBoost (single):                 {accuracy_score(y_te, xgb_pred):.4f}")
    print(f"MLP ensemble:                     {accuracy_score(y_te, mlp_pred):.4f}")
    print(f"Combined XGB + MLP-ensemble:      {accuracy_score(y_te, combined_pred):.4f}")