"""
Find optimal distance threshold for plagiarism detection.
Performs Ablation Study (Cosine vs Euclidean) and identifies the best metric.
Evaluates CLEWS, WEALY, and the pre-calculated FUSION model.
"""

import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

# Resolve repository root and load logging_util without relying on src package path
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent

logging_util_path = repo_root / "src" / "utils" / "logging_util.py"
spec = importlib.util.spec_from_file_location("logging_util", str(logging_util_path))
if spec is None or spec.loader is None:
    raise FileNotFoundError(f"Could not load logging_util from {logging_util_path}")
logging_util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_util)
setup_logging = logging_util.setup_logging

# Initialize logging for this script (logs/optimal_threshold.txt)
setup_logging(__file__)


def load_distances(csv_path):
    """Load distance metrics from CSV and assign labels correctly."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Distance file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df['is_plagiarised'] = (~df['final_mod_type'].str.startswith('Negative')).astype(int)

    unique_labels = df['final_mod_type'].unique().tolist()
    class_counts = df['is_plagiarised'].value_counts().to_dict()

    if set(df['is_plagiarised'].unique()) != {0, 1}:
        raise ValueError(
            f"Labeling error: both classes not present. final_mod_type unique values: {unique_labels}. "
            f"Computed is_plagiarised counts: {class_counts}."
        )

    min_class_count = min(class_counts.values()) if class_counts else 0
    if min_class_count < 5:
        raise ValueError(
            f"Insufficient class representation: each class must have at least 5 samples. "
            f"Computed is_plagiarised counts: {class_counts}."
        )

    if len(df) < 50:
        raise ValueError(f"Insufficient data: need at least 50 samples, found {len(df)}")

    for col in ['cosine_distance', 'euclidean_distance']:
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"NaN values found in distance column '{col}'.")

    return df


def _compute_optimal_threshold(y_train, y_train_scores, invert_logic):
    fpr, tpr, thresholds = roc_curve(y_train, y_train_scores)
    j_scores = tpr - fpr
    optimal_index = np.nanargmax(j_scores)

    if invert_logic:
        optimal_threshold = -thresholds[optimal_index]
    else:
        optimal_threshold = thresholds[optimal_index]

    return optimal_threshold


def _compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def evaluate_metric(df, distance_col, n_splits=5):
    """
    Evaluate a metric via 5-Fold Stratified Cross-Validation and return aggregate results.
    Includes Strict Nested CV for AUC and Dynamic Logic Detection.
    """
    if distance_col not in df.columns:
        raise ValueError(f"Missing distance column {distance_col} in DataFrame.")

    if df[distance_col].isna().any():
        raise ValueError(f"NaN values found in distance column '{distance_col}'.")

    y = df['is_plagiarised'].values
    distances = df[distance_col].values

    col_name_lower = distance_col.lower()
    if 'distance' in col_name_lower or 'divergence' in col_name_lower:
        invert_logic = True
    elif 'similarity' in col_name_lower or 'confidence' in col_name_lower:
        invert_logic = False
    else:
        raise ValueError(f"CRITICAL ERROR: Could not determine logic for column '{distance_col}'. Please ensure it contains"
                         f" 'distance' or 'similarity' in the name.")

    scores = -distances if invert_logic else distances
    fpr_full, tpr_full, _ = roc_curve(y, scores)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    fold_thresholds = []
    cv_aucs = []  

    for fold_index, (train_idx, test_idx) in enumerate(skf.split(distances, y), start=1):
        y_train, y_test = y[train_idx], y[test_idx]
        dist_train, dist_test = distances[train_idx], distances[test_idx]

        y_train_scores = -dist_train if invert_logic else dist_train

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Fold {fold_index} has only one class in training labels; cannot compute ROC.")

        opt_threshold = _compute_optimal_threshold(y_train, y_train_scores, invert_logic)
        fold_thresholds.append(opt_threshold)

        if invert_logic:
            y_test_pred = (dist_test <= opt_threshold).astype(int)
        else:
            y_test_pred = (dist_test >= opt_threshold).astype(int)

        metrics = _compute_metrics(y_test, y_test_pred)
        fold_metrics.append(metrics)

        y_test_scores = -dist_test if invert_logic else dist_test
        fpr_fold, tpr_fold, _ = roc_curve(y_test, y_test_scores)
        fold_auc = auc(fpr_fold, tpr_fold)
        cv_aucs.append(fold_auc)

        print(
            f"[{distance_col}] Fold {fold_index}/{n_splits}: optimal_threshold={opt_threshold:.6f}, "
            f"AUC={fold_auc:.4f}, accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
        )

    agg_metrics = {
        'optimal_threshold': float(np.mean(fold_thresholds)),
        'accuracy': float(np.mean([m['accuracy'] for m in fold_metrics])),
        'precision': float(np.mean([m['precision'] for m in fold_metrics])),
        'recall': float(np.mean([m['recall'] for m in fold_metrics])),
        'f1': float(np.mean([m['f1'] for m in fold_metrics])),
        'roc_auc': float(np.mean(cv_aucs)),
        'tp': int(np.sum([m['tp'] for m in fold_metrics])),
        'tn': int(np.sum([m['tn'] for m in fold_metrics])),
        'fp': int(np.sum([m['fp'] for m in fold_metrics])),
        'fn': int(np.sum([m['fn'] for m in fold_metrics])),
        'cv_optimal_thresholds': fold_thresholds,
        'cv_f1_mean_label': '5-Fold CV Mean F1-Score',
        'cv_auc_label': '5-Fold CV Mean AUC'
    }

    return agg_metrics, fpr_full, tpr_full


def plot_ablation_roc(fpr_cos, tpr_cos, auc_cos, fpr_euc, tpr_euc, auc_euc, output_path, model_name):
    """Plot ROC curves for all metrics on the same graph for direct comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(fpr_cos, tpr_cos, color='blue', lw=2.5,
            label=f'Cosine Distance (AUC = {auc_cos:.3f})')

    if fpr_euc is not None:
        ax.plot(fpr_euc, tpr_euc, color='red', lw=2.5, linestyle=':',
                label=f'Euclidean Distance (AUC = {auc_euc:.3f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Chance')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Metric Comparison - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {output_path}")


def plot_distance_distributions(df, optimal_threshold, distance_col, output_path):
    """Plot the overlapping distributions of plagiarised vs baseline pairs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plagiarised = df[df['is_plagiarised'] == 1][distance_col]
    baseline = df[df['is_plagiarised'] == 0][distance_col]

    sns.kdeplot(data=plagiarised, fill=True, color='red', label='Plagiarism (Positives)', ax=ax, alpha=0.5)
    sns.kdeplot(data=baseline, fill=True, color='blue', label='Baseline (Negatives)', ax=ax, alpha=0.5)

    ax.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2,
               label=f'Optimal Threshold ({optimal_threshold:.3f})')

    ax.set_title(f'{distance_col.replace("_", " ").title()} Distributions', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{distance_col.title()} (Lower = More Similar)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_model_analysis(csv_path, model_name):
    """Runs the full ablation study for a given model."""
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping {model_name} analysis.")
        return None

    print(f"\n[{model_name}] Loading distances...")
    df = load_distances(csv_path)

    print(f"[{model_name}] Evaluating Cosine Distance (5-Fold CV)...")
    res_cos, fpr_cos, tpr_cos = evaluate_metric(df, 'cosine_distance')

    # Αλλαγή: Δυναμικός έλεγχος αν υπάρχει στήλη Ευκλείδειας Απόστασης
    if 'euclidean_distance' in df.columns:
        print(f"[{model_name}] Evaluating Euclidean Distance (5-Fold CV)...")
        res_euc, fpr_euc, tpr_euc = evaluate_metric(df, 'euclidean_distance')
        
        print(f"\n=== ABLATION STUDY RESULTS ({model_name}) ===")
        print(f"{'Metric':<20} | {'AUC':<7} | {'5-Fold CV Mean F1-Score':<25} | {'Opt. Threshold'}")
        print("-" * 90)
        print(f"{'Cosine':<20} | {res_cos['roc_auc']:.4f}  | {res_cos['f1']:.4f}                 | {res_cos['optimal_threshold']:.4f}")
        print(f"{'Euclidean':<20} | {res_euc['roc_auc']:.4f}  | {res_euc['f1']:.4f}                 | {res_euc['optimal_threshold']:.4f}")

        if res_euc['roc_auc'] > res_cos['roc_auc']:
            winner_metric = 'Euclidean'
            winning_results = res_euc
        else:
            winner_metric = 'Cosine'
            winning_results = res_cos
            
        plot_ablation_roc(
            fpr_cos, tpr_cos, res_cos['roc_auc'],
            fpr_euc, tpr_euc, res_euc['roc_auc'],
            f'plots/threshold/{model_name.lower()}_ablation_roc.png', model_name
        )
        plot_distance_distributions(df, res_euc['optimal_threshold'], 'euclidean_distance',
                                    f'plots/threshold/{model_name.lower()}_euclidean_distributions.png')
    else:
        print(f"[{model_name}] Euclidean Distance not found. Skipping Euclidean comparison.")
        winner_metric = 'Cosine'
        winning_results = res_cos
        
        # Plot with only Cosine
        plot_ablation_roc(
            fpr_cos, tpr_cos, res_cos['roc_auc'],
            None, None, None,
            f'plots/threshold/{model_name.lower()}_roc.png', model_name
        )

    print(f"\n=> WINNING METRIC: {winner_metric} Distance")

    plot_distance_distributions(df, res_cos['optimal_threshold'], 'cosine_distance',
                                f'plots/threshold/{model_name.lower()}_cosine_distributions.png')

    winning_results['metric'] = winner_metric
    return winning_results


def process_and_save_thresholds(summary_data, summary_path="results/threshold/threshold_analysis_summary.csv"):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_path, index=False)
    print(f"\nSummary of winning metrics saved to: {summary_path}")
    return df


def main():
    print("=" * 70)
    print("ABLATION STUDY & THRESHOLD ANALYSIS FOR PLAGIARISM DETECTION")
    print("=" * 70)

    # Single Models
    results_clews = run_model_analysis("results/distances/clews_distances.csv", "CLEWS")
    results_wealy = run_model_analysis("results/distances/wealy_distances.csv", "WEALY")
    
    # FUSION
    results_fusion = run_model_analysis("results/fusion/optimal_fused_distances.csv", "FUSION")

    summary_data = []
    if results_clews:
        summary_data.append({
            'model': 'CLEWS',
            'metric': results_clews.get('metric', 'Unknown'),
            **{k: v for k, v in results_clews.items() if isinstance(v, (int, float))}
        })
    if results_wealy:
        summary_data.append({
            'model': 'WEALY',
            'metric': results_wealy.get('metric', 'Unknown'),
            **{k: v for k, v in results_wealy.items() if isinstance(v, (int, float))}
        })
    if results_fusion:
        summary_data.append({
            'model': 'FUSION',
            'metric': results_fusion.get('metric', 'Unknown'),
            **{k: v for k, v in results_fusion.items() if isinstance(v, (int, float))}
        })

    if summary_data:
        process_and_save_thresholds(summary_data, "results/threshold/threshold_analysis_summary.csv")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()