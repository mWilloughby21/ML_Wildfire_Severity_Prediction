# compare_models.py

import time
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.models.mlp_model import build_pipeline as build_mlp_pipeline
from src.models.mlp_model import load_features as mlp_load_features
from src.models.xgboost_model import load_features as xgb_load_features

# Multiple seeds give a robust mean ± std (esp. important for the MLP).
SPLIT_SEEDS = [42, 0, 1, 7, 13]
TARGET_NAMES = ['Unburned/Low', 'Low', 'Moderate', 'High']


def train_xgb(X_train, y_train):
    Xtr, Xv, ytr, yv = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
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
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    return model


def train_mlp(X_train, y_train, seed):
    pipeline = build_mlp_pipeline()
    pipeline.named_steps['mlp'].random_state = seed
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # quiet sklearn convergence chatter
        pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(y_true, y_pred):
    """Return dict of accuracy, macro F1, and per-class F1."""
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(TARGET_NAMES)), zero_division=0,
    )
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'per_class_f1': f1,
    }


def run_for_seed(X_xgb, X_mlp, y, seed):
    Xxgb_tr, Xxgb_te, y_tr, y_te = train_test_split(
        X_xgb, y, test_size=0.2, random_state=seed, stratify=y,
    )
    Xmlp_tr, Xmlp_te, _, _ = train_test_split(
        X_mlp, y, test_size=0.2, random_state=seed, stratify=y,
    )

    t0 = time.time()
    xgb = train_xgb(Xxgb_tr, y_tr)
    xgb_time = time.time() - t0
    xgb_pred = xgb.predict(Xxgb_te)

    t0 = time.time()
    mlp = train_mlp(Xmlp_tr, y_tr, seed=seed)
    mlp_time = time.time() - t0
    mlp_pred = mlp.predict(Xmlp_te)

    return {
        'y_true': np.asarray(y_te),
        'xgb': {**evaluate(y_te, xgb_pred), 'time_s': xgb_time, 'pred': xgb_pred},
        'mlp': {**evaluate(y_te, mlp_pred), 'time_s': mlp_time, 'pred': mlp_pred},
    }


def aggregate(metric_lists):
    """Stack a list of arrays/scalars and return (mean, std)."""
    arr = np.stack([np.asarray(m) for m in metric_lists])
    return arr.mean(axis=0), arr.std(axis=0)


def fmt(mean, std, pct=False):
    if pct:
        return f"{mean*100:5.2f}% ± {std*100:.2f}%"
    return f"{mean:.3f} ± {std:.3f}"


def main():
    X_xgb, y = xgb_load_features()
    X_mlp, _ = mlp_load_features()

    print(f"Running {len(SPLIT_SEEDS)} train/test splits per model...\n")
    runs = []
    for seed in SPLIT_SEEDS:
        print(f"  seed={seed}", end='', flush=True)
        runs.append(run_for_seed(X_xgb, X_mlp, y, seed))
        print(
            f"   XGB acc={runs[-1]['xgb']['accuracy']:.4f}"
            f"   MLP acc={runs[-1]['mlp']['accuracy']:.4f}"
        )

    xgb_acc_mean, xgb_acc_std = aggregate([r['xgb']['accuracy'] for r in runs])
    mlp_acc_mean, mlp_acc_std = aggregate([r['mlp']['accuracy'] for r in runs])
    xgb_mf1_mean, xgb_mf1_std = aggregate([r['xgb']['macro_f1'] for r in runs])
    mlp_mf1_mean, mlp_mf1_std = aggregate([r['mlp']['macro_f1'] for r in runs])
    xgb_pcf1_mean, xgb_pcf1_std = aggregate([r['xgb']['per_class_f1'] for r in runs])
    mlp_pcf1_mean, mlp_pcf1_std = aggregate([r['mlp']['per_class_f1'] for r in runs])
    xgb_time_mean, _ = aggregate([r['xgb']['time_s'] for r in runs])
    mlp_time_mean, _ = aggregate([r['mlp']['time_s'] for r in runs])

    print("\n" + "=" * 64)
    print(f"  Comparison across {len(SPLIT_SEEDS)} stratified 80/20 splits")
    print("=" * 64)
    print(f"  {'Metric':<20} {'XGBoost':<22} {'MLP':<22}")
    print(f"  {'-'*20} {'-'*22} {'-'*22}")
    print(f"  {'Accuracy':<20} {fmt(xgb_acc_mean, xgb_acc_std, pct=True):<22} "
          f"{fmt(mlp_acc_mean, mlp_acc_std, pct=True):<22}")
    print(f"  {'Macro F1':<20} {fmt(xgb_mf1_mean, xgb_mf1_std):<22} "
          f"{fmt(mlp_mf1_mean, mlp_mf1_std):<22}")
    print(f"  {'Train time (s)':<20} {xgb_time_mean:<22.2f} {mlp_time_mean:<22.2f}")

    print("\n  Per-class F1 (mean across seeds):")
    print(f"  {'Class':<16} {'XGBoost':<20} {'MLP':<20}")
    print(f"  {'-'*16} {'-'*20} {'-'*20}")
    for i, name in enumerate(TARGET_NAMES):
        print(f"  {name:<16} {fmt(xgb_pcf1_mean[i], xgb_pcf1_std[i]):<20} "
              f"{fmt(mlp_pcf1_mean[i], mlp_pcf1_std[i]):<20}")

    print("\n  Confusion matrices (seed=42, rows=true, cols=predicted):")
    seed42 = next(r for r, s in zip(runs, SPLIT_SEEDS) if s == 42)
    print("    XGBoost:")
    for row in confusion_matrix(seed42['y_true'], seed42['xgb']['pred']):
        print("      " + "  ".join(f"{v:4d}" for v in row))
    print("    MLP:")
    for row in confusion_matrix(seed42['y_true'], seed42['mlp']['pred']):
        print("      " + "  ".join(f"{v:4d}" for v in row))


if __name__ == "__main__":
    main()
