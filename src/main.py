# main.py

# python -m src.main

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from sklearn.metrics import accuracy_score

from src.models.ensemble_model import train_and_evaluate as run_ensemble
from src.models.mlp_model import train_and_evaluate as run_mlp
from src.models.xgboost_model import train_and_evaluate as run_xgb
from src.visualization.plots import (
    plot_comparison,
    plot_feature_importance,
    plot_single_model_summary,
)

MENU = """
================================================================
    Wildfire Burn-Severity Predictor
================================================================
    1. Train and evaluate XGBoost
    2. Train and evaluate MLP
    3. Compare XGBoost vs MLP (side-by-side)
    4. Train and evaluate Ensemble (XGB + MLP soft vote)
    q. Quit
"""

## Text Rendering Helpers
def banner(title):
    line = '=' * 64
    print(f"\n{line}\n  {title}\n{line}")

def headline_metrics(result):
    y_true, y_pred = result['y_true'], result['y_pred']
    classes = result['class_names']
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, macro_f1

def render_metrics_block(result):
    y_true, y_pred = result['y_true'], result['y_pred']
    classes = result['class_names']
    acc, macro_f1 = headline_metrics(result)

    print(f"  Test set: {len(y_true)} fires (20% stratified holdout)")
    print()
    print(f"  Accuracy   :  {acc * 100:6.2f}%")
    print(f"  Macro F1   :  {macro_f1:6.3f}")
    print()

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0,
    )
    print("  Per-class breakdown:")
    print(f"    {'Class':<16} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>8}")
    print("    " + "-" * 52)
    for cls, pi, ri, fi, si in zip(classes, p, r, f1, support):
        print(f"    {cls:<16} {pi:>10.3f} {ri:>8.3f} {fi:>6.3f} {si:>8d}")

def render_comparison_block(result_a, result_b):
    classes = result_a['class_names']
    n = len(classes)
    
    acc_a, mf1_a = headline_metrics(result_a)
    acc_b, mf1_b = headline_metrics(result_b)
    p_a, r_a, f1_a, _ = precision_recall_fscore_support(
        result_a['y_true'], result_a['y_pred'], labels=range(n), zero_division=0,
    )
    p_b, r_b, f1_b, _ = precision_recall_fscore_support(
        result_b['y_true'], result_b['y_pred'], labels=range(n), zero_division=0,
    )
    
    print(f"  Test set: {len(result_a['y_true'])} fires (20% stratified holdout)")
    print()
    name_a, name_b = result_a['name'], result_b['name']
    print(f"  {'Metric':<14} {name_a:>14} {name_b:>14} {'Winner':>10}")
    print("  " + "-" * 56)
    
    def line(metric, a, b, fmt='{:>14.3f}', higher_is_better=True):
        winner = name_a if (a > b if higher_is_better else a < b) else name_b
        if a == b:
            winner = '—'
        print(f"  {metric:<14} {fmt.format(a):>14} {fmt.format(b):>14} {winner:>10}")
        
    line('Accuracy', acc_a, acc_b, fmt='{:>13.2%}')
    line('Macro F1', mf1_a, mf1_b)
    print()

    print("  Per-class F1:")
    print(f"    {'Class':<16} {name_a:>10} {name_b:>10} {'Winner':>10}")
    print("    " + "-" * 50)
    for cls, fa, fb in zip(classes, f1_a, f1_b):
        winner = name_a if fa > fb else (name_b if fb > fa else '—')
        print(f"    {cls:<16} {fa:>10.3f} {fb:>10.3f} {winner:>10}")


## Menu Actions
def action_xgboost():
    banner('XGBoost — train and evaluate')
    print('  Training (this takes a few seconds)...')
    result = run_xgb()
    
    print()
    render_metrics_block(result)
    print()
    print('  Showing visualizations — close the windows to return to the menu.')
    fig1 = plot_single_model_summary(result)
    fig2 = plot_feature_importance(
        result['feature_importances'], top_n=15,
        title='XGBoost — top 15 features by importance',
    )
    
    plt.show()
    plt.close(fig1)
    plt.close(fig2)

def action_mlp():
    banner('MLP — train and evaluate')
    print('  Training (this takes a few seconds)...')
    result = run_mlp()
    
    print()
    render_metrics_block(result)
    print()
    print(f"  Trained for {result['n_iter']} iterations "
          f"(internal val peak: {result['best_val_score']:.3f})")
    print()
    print('  Showing visualization — close the window to return to the menu.')
    fig = plot_single_model_summary(result)
    
    plt.show()
    plt.close(fig)


def action_compare():
    banner('Comparison — XGBoost vs MLP')
    print('  Training XGBoost...')
    result_xgb = run_xgb()
    print('  Training MLP...')
    result_mlp = run_mlp()
    
    print()
    render_comparison_block(result_xgb, result_mlp)
    print()
    print('  Showing comparison visualization — close the window to return to the menu.')
    fig = plot_comparison(result_xgb, result_mlp)
    
    plt.show()
    plt.close(fig)

def action_ensemble():
    banner('Ensemble — XGBoost + MLP (soft vote)')
    print('  Training XGBoost and MLP seeds, then averaging probabilities...')
    result = run_ensemble()

    print()
    render_metrics_block(result)
    print()
    ens_acc = accuracy_score(result['y_true'], result['y_pred'])
    xgb_acc = accuracy_score(result['y_true'], result['xgb_pred'])
    mlp_acc = accuracy_score(result['y_true'], result['mlp_pred'])
    print('  Component accuracies on the same test set:')
    print(f"    XGBoost alone        :  {xgb_acc * 100:6.2f}%")
    print(f"    MLP ensemble alone   :  {mlp_acc * 100:6.2f}%")
    print(f"    Soft-vote ensemble   :  {ens_acc * 100:6.2f}%")
    print()
    print('  Showing visualization — close the window to return to the menu.')
    fig = plot_single_model_summary(result)

    plt.show()
    plt.close(fig)


## Main
def main():
    while True:
        print(MENU)
        choice = input('  Choose an option: ').strip().lower()
        if choice == '1':
            action_xgboost()
        elif choice == '2':
            action_mlp()
        elif choice == '3':
            action_compare()
        elif choice == '4':
            action_ensemble()
        elif choice in ('q', 'quit', 'exit', '5', ''):
            print('  Goodbye.')
            break
        else:
            print(f"  Unknown option: {choice!r}")


if __name__ == '__main__':
    main()
