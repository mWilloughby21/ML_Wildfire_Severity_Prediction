# plots.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sns.set_theme(style='whitegrid', context='talk')


ONE_HOT_PREFIXES = ('FBFM40_CODE_', 'BPS_CODE_')


def _per_class_f1(y_true, y_pred, n_classes):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0,
    )
    return f1

def plot_single_model_summary(result):
    classes = result['class_names']
    cm = confusion_matrix(result['y_true'], result['y_pred'], labels=range(len(classes)))
    f1 = _per_class_f1(result['y_true'], result['y_pred'], len(classes))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{result['name']} — wildfire severity prediction", fontsize=18, y=1.02)
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=classes, yticklabels=classes, ax=axes[0],
    )
    axes[0].set_title('Confusion matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_xticklabels(classes, rotation=30, ha='right')
    axes[0].set_yticklabels(classes, rotation=0)

    bars = axes[1].bar(classes, f1, color=sns.color_palette('Blues_d', len(classes)))
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('F1 score')
    axes[1].set_title('Per-class F1')
    axes[1].set_xticklabels(classes, rotation=30, ha='right')
    for bar, v in zip(bars, f1):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=11)
    
    fig.tight_layout()
    return fig

def _collapse_one_hot(importances):
    keep = importances[~importances.index.str.startswith(ONE_HOT_PREFIXES)].copy()
    for prefix in ONE_HOT_PREFIXES:
        mask = importances.index.str.startswith(prefix)
        if mask.any():
            keep[prefix.rstrip('_')] = importances[mask].sum()
    return keep

def plot_feature_importance(importances, top_n=15, title='Top features'):
    importances = _collapse_one_hot(importances)
    top = importances.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index, top.values, color=sns.color_palette('Blues_d', len(top)))
    ax.set_title(title)
    ax.set_xlabel('Importance')
    fig.tight_layout()
    return fig

def plot_comparison(result_a, result_b):
    """Side-by-side confusion matrices + overlaid per-class F1 bars."""
    classes = result_a['class_names']
    n = len(classes)
    cm_a = confusion_matrix(result_a['y_true'], result_a['y_pred'], labels=range(n))
    cm_b = confusion_matrix(result_b['y_true'], result_b['y_pred'], labels=range(n))
    f1_a = _per_class_f1(result_a['y_true'], result_a['y_pred'], n)
    f1_b = _per_class_f1(result_b['y_true'], result_b['y_pred'], n)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_f1 = fig.add_subplot(gs[1, :])
    
    fig.suptitle(f"{result_a['name']} vs {result_b['name']}", fontsize=18, y=1.00)
    
    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax_a)
    ax_a.set_title(result_a['name'])
    ax_a.set_xlabel('Predicted'); ax_a.set_ylabel('True')
    ax_a.set_xticklabels(classes, rotation=30, ha='right')
    ax_a.set_yticklabels(classes, rotation=0)

    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax_b)
    ax_b.set_title(result_b['name'])
    ax_b.set_xlabel('Predicted'); ax_b.set_ylabel('True')
    ax_b.set_xticklabels(classes, rotation=30, ha='right')
    ax_b.set_yticklabels(classes, rotation=0)

    width = 0.38
    x = np.arange(n)
    ax_f1.bar(x - width / 2, f1_a, width, label=result_a['name'], color=sns.color_palette('Blues_d')[2])
    ax_f1.bar(x + width / 2, f1_b, width, label=result_b['name'], color=sns.color_palette('Greens_d')[2])
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(classes, rotation=20, ha='right')
    ax_f1.set_ylim(0, 1)
    ax_f1.set_ylabel('F1 score')
    ax_f1.set_title('Per-class F1 (higher is better)')
    ax_f1.legend(loc='upper right')
    for i, (a, b) in enumerate(zip(f1_a, f1_b)):
        ax_f1.text(i - width / 2, a + 0.02, f"{a:.2f}", ha='center', fontsize=10)
        ax_f1.text(i + width / 2, b + 0.02, f"{b:.2f}", ha='center', fontsize=10)
    
    fig.tight_layout()
    return fig