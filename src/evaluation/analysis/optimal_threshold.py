"""
Find optimal distance threshold for plagiarism detection.
Evaluates Cosine, Euclidean, Manhattan, and Pearson distances.
Optimizes threshold based on maximum F0.5-Score via Precision-Recall Curve.
Selects the winning metric based strictly on the highest Mean F0.5-Score.

- Follows standard practice in audio/music fingerprinting: threshold sweep on
  validation set maximizing F0.5-Score per metric.
- Uses Stratified K-Fold CV for robust, unbiased estimation.
- Reports PR-AUC (preferred over ROC-AUC for imbalanced detection tasks).

Vocal-aware policy:
- CLEWS  : threshold computed on ALL pairs (acoustic, always valid).
- WEALY  : threshold computed on vocal-valid pairs ONLY (speech model, domain constraint).
- FUSION : threshold computed on ALL pairs using adaptive fused distances.
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, precision_recall_curve, auc, fbeta_score
)
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import logging

# Resolve repository root and load logging_util
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

setup_logging(__file__)
logger = logging.getLogger(__name__)

# Import centralized utilities
sys.path.insert(0, str(repo_root / "src"))
from utils.constants import (
    MODEL_PATHS, OUTPUT_DIRS, SUMMARY_FILES,
    DISTANCE_METRICS, NUM_K_FOLDS, RANDOM_STATE,
    PLOT_LINE_STYLES, PLOT_DPI,
    VOCAL_RATIOS_CSV, BETA
)
from utils.categorization import get_ground_truth_label, fbeta_score_curve
from utils.vocal_metadata import attach_vocal_metadata, filter_vocal_valid


# Models that require vocal-valid filtering before threshold optimization
VOCAL_FILTERED_MODELS = {"WEALY"}


# DATA LOADING
def load_distances(csv_path: str, model_name: str) -> pd.DataFrame:
    """
    Load distance CSV and apply ground truth labels.

    For WEALY: attaches vocal metadata and filters to vocal-valid pairs only,
    because WEALY is a speech model and produces unreliable embeddings on
    segments without vocal content.

    For CLEWS and FUSION: no filtering — all pairs are valid.
    """
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"Distance file not found: {csv_path_obj}")

    print(f"Loading distances from {csv_path_obj}...")
    df = pd.read_csv(csv_path_obj, low_memory=False)
    df['is_plagiarised'] = df['final_mod_type'].apply(get_ground_truth_label)

    # Validate distance columns
    for col in DISTANCE_METRICS:
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"NaN values found in distance column '{col}'.")

    # Filter WEALY to vocal-valid pairs only ---
    if model_name in VOCAL_FILTERED_MODELS:
        vocal_csv_exists = Path(VOCAL_RATIOS_CSV).exists()
        if vocal_csv_exists:
            n_before = len(df)
            df = attach_vocal_metadata(df)
            df = filter_vocal_valid(df)
            n_after = len(df)
            print(
                f"[{model_name}] Vocal filter applied: "
                f"{n_before} → {n_after} pairs "
                f"({n_before - n_after} non-vocal pairs excluded from threshold optimization)"
            )
        else:
            logger.warning(
                f"[{model_name}] Vocal metadata not found at {VOCAL_RATIOS_CSV}. "
                f"Proceeding without vocal filtering — thresholds may be unreliable."
            )

    print(f"Dataset loaded: {len(df)} pairs, {df['is_plagiarised'].sum()} positives.")
    return df


# THRESHOLD OPTIMIZATION HELPERS
def _compute_optimal_threshold(
    y_train:        np.ndarray,
    y_train_scores: np.ndarray,
    invert_logic:   bool,
    beta:           float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute the threshold that maximises F-beta on the training fold.

    Returns
    -------
    threshold_distance : float  - optimal threshold in original distance space
    best_fbeta         : float  - F-beta at that threshold
    best_f1            : float  - F1 at the same threshold (for reporting)
    """
    y_train        = np.asarray(y_train)
    y_train_scores = np.asarray(y_train_scores)

    if len(np.unique(y_train)) < 2:
        logger.warning("Fold contains only one class. Returning default threshold.")
        fallback = 0.0 if invert_logic else 1.0
        return fallback, 0.0, 0.0

    precision, recall, thresholds = precision_recall_curve(y_train, y_train_scores)
    fbeta_scores = fbeta_score_curve(precision, recall, beta)
    f1_scores    = fbeta_score_curve(precision, recall, beta=1.0)

    if len(fbeta_scores) > 1:
        optimal_index = int(np.argmax(fbeta_scores[:-1]))
    else:
        logger.warning("PR curve has insufficient points. Using fallback threshold.")
        fallback = 0.0 if invert_logic else 1.0
        return fallback, 0.0, 0.0

    score_threshold    = thresholds[optimal_index]
    best_fbeta         = float(fbeta_scores[optimal_index])
    best_f1            = float(f1_scores[optimal_index])
    threshold_distance = -score_threshold if invert_logic else score_threshold
    return threshold_distance, best_fbeta, best_f1


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta:   float = 0.5,
) -> Dict:
    """
    Compute classification metrics for a single fold prediction.
    Uses sklearn's fbeta_score consistently.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'fbeta':     fbeta,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# CROSS-VALIDATED EVALUATION
def evaluate_metric(
    df:            pd.DataFrame,
    distance_col:  str,
    n_splits:      Optional[int] = None,
    beta:          float = 0.5,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate a single distance metric with Stratified K-Fold CV.
    Threshold per fold is chosen by maximising F-beta (default beta=0.5).
    """
    if n_splits is None:
        n_splits = NUM_K_FOLDS

    y         = np.asarray(df['is_plagiarised'].to_numpy(), dtype=np.float64)
    distances = np.asarray(df[distance_col].to_numpy(),     dtype=np.float64)

    col_lower    = distance_col.lower()
    invert_logic = 'distance' in col_lower or 'divergence' in col_lower
    scores       = -distances if invert_logic else distances

    prec_full, rec_full, _ = precision_recall_curve(y, scores)

    skf             = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics    = []
    fold_thresholds = []
    cv_aucs         = []

    for _, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros_like(distances).reshape(-1, 1), y), start=1
    ):
        y_train,    y_test    = y[train_idx],         y[test_idx]
        dist_train, dist_test = distances[train_idx], distances[test_idx]

        scores_train        = -dist_train if invert_logic else dist_train
        opt_threshold, _, _ = _compute_optimal_threshold(
            y_train, scores_train, invert_logic, beta=beta
        )
        fold_thresholds.append(opt_threshold)

        y_test_pred = (
            (dist_test <= opt_threshold) if invert_logic else (dist_test >= opt_threshold)
        ).astype(int)
        fold_metrics.append(_compute_metrics(y_test, y_test_pred, beta=beta))

        scores_test            = -dist_test if invert_logic else dist_test
        prec_fold, rec_fold, _ = precision_recall_curve(y_test, scores_test)
        cv_aucs.append(auc(rec_fold, prec_fold))

    agg_metrics = {
        'optimal_threshold': float(np.mean(fold_thresholds)),
        'beta':              beta,
        'fbeta':             float(np.mean([m['fbeta'] for m in fold_metrics])),
        'f1':                float(np.mean([m['f1']    for m in fold_metrics])),
        'accuracy':          float(np.mean([m['accuracy']  for m in fold_metrics])),
        'precision':         float(np.mean([m['precision'] for m in fold_metrics])),
        'recall':            float(np.mean([m['recall']    for m in fold_metrics])),
        'pr_auc':            float(np.mean(cv_aucs)),
    }
    return agg_metrics, prec_full, rec_full


# PLOTTING
def plot_ablation_pr(
    plot_data:   Dict,
    output_path: str,
    model_name:  str,
):
    """PR curves for all distance metrics on one model."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange']

    for idx, (metric_name, data) in enumerate(plot_data.items()):
        prec, rec, pr_auc = data['prec'], data['rec'], data['auc']
        ax.plot(
            rec, prec,
            color=colors[idx % len(colors)],
            linewidth=2.5,
            linestyle=PLOT_LINE_STYLES[idx % len(PLOT_LINE_STYLES)],
            label=f'{metric_name} (PR AUC = {pr_auc:.3f})',
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(
        f'Precision-Recall Curve Comparison — {model_name}',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PR curve plot -> {output_path}")


def plot_distance_distributions(
    df:                pd.DataFrame,
    optimal_threshold: float,
    distance_col:      str,
    output_path:       str,
    beta:              float = 0.5,
):
    """KDE plot of positive vs negative distance distributions with threshold line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plagiarised = df[df['is_plagiarised'] == 1][distance_col]
    baseline    = df[df['is_plagiarised'] == 0][distance_col]

    # Clip to 99th percentile to remove outliers
    if not plagiarised.empty:
        p99 = np.percentile(plagiarised.dropna(), 99)
        plagiarised = plagiarised[plagiarised <= p99]
    if not baseline.empty:
        p99 = np.percentile(baseline.dropna(), 99)
        baseline = baseline[baseline <= p99]

    if not plagiarised.empty:
        sns.kdeplot(
            data=plagiarised, fill=True, color='red',
            label='Plagiarism (Positives)', ax=ax, alpha=0.5
        )
    if not baseline.empty:
        sns.kdeplot(
            data=baseline, fill=True, color='blue',
            label='Baseline (Negatives)', ax=ax, alpha=0.5
        )

    ax.axvline(
        x=optimal_threshold, color='black', linestyle='--', linewidth=2,
        label=f'Opt. F{beta} Threshold ({optimal_threshold:.3f})'
    )
    ax.set_title(
        f'{distance_col.replace("_", " ").title()} Distributions',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel(f'{distance_col.title()} (Lower = More Similar)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved distance distribution plot -> {output_path}")


# PER-MODEL PIPELINE
def run_model_analysis(
    csv_path:   str,
    model_name: str,
    beta:       float = 0.5,
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Full threshold-optimisation pipeline for one model.
    Winner selected by highest mean F-beta on held-out test folds.
    """
    if not Path(csv_path).exists():
        logger.warning(f"{csv_path} not found. Skipping {model_name}.")
        return None, []

    df = load_distances(csv_path, model_name)
    metrics_to_test = [
        col for col in df.columns
        if col.endswith('_distance') or '+' in col
    ]

    best_fbeta      = -1.0
    winner_metric   = None
    winning_results: Dict       = {}
    plot_data:       Dict       = {}
    all_metrics:     List[Dict] = []

    beta_label = f'F{beta}'
    print(f"\n=== ABLATION STUDY ({model_name}) — optimising {beta_label} ===")
    print(
        f"{'Metric':<40} | {'PR-AUC':<7} | "
        f"{beta_label+' (test)':<12} | {'F1 (test)':<10} | {'Opt. Threshold':<14}"
    )
    print("-" * 95)

    for metric in metrics_to_test:
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not in dataframe. Skipping.")
            continue

        res, prec, rec = evaluate_metric(df, metric, beta=beta)
        print_name     = metric.split('_')[0].capitalize()

        print(
            f"{print_name:<15} | {res['pr_auc']:<7.4f} | "
            f"{res['fbeta']:<12.4f} | {res['f1']:<10.4f} | "
            f"{res['optimal_threshold']:<14.4f}"
        )

        res_to_store = {**res, 'model': model_name, 'metric': metric}
        all_metrics.append(res_to_store)
        plot_data[print_name] = {'prec': prec, 'rec': rec, 'auc': res['pr_auc']}

        if res['fbeta'] > best_fbeta:
            best_fbeta      = res['fbeta']
            winner_metric   = metric
            winning_results = res

    if winner_metric:
        print_winner = winner_metric.split('_')[0].capitalize()
        print(
            f"=> WINNING METRIC: {print_winner} Distance "
            f"(Best {beta_label}: {best_fbeta:.4f}, "
            f"F1 at same threshold: {winning_results['f1']:.4f})"
        )

        plot_ablation_pr(
            plot_data,
            f'{OUTPUT_DIRS["threshold_plots"]}/{model_name.lower()}_ablation_pr.pdf',
            model_name,
        )
        plot_distance_distributions(
            df,
            winning_results['optimal_threshold'],
            winner_metric,
            f'{OUTPUT_DIRS["threshold_plots"]}'
            f'/{model_name.lower()}_{winner_metric}_distributions.pdf',
            beta=beta,
        )
        winning_results['metric'] = winner_metric
        return winning_results, all_metrics

    logger.warning(f"No valid metrics found for {model_name}.")
    return None, []


# SAVE
def process_and_save_thresholds(
    summary_data: List[Dict],
    summary_path: str,
) -> pd.DataFrame:
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_path, index=False)
    print(f"Summary saved -> {summary_path}")
    return df


def main():

    print("=" * 70)
    print(f"ABLATION STUDY & THRESHOLD ANALYSIS  (optimising F{BETA})")
    print("=" * 70)
    print()
    print("Vocal-aware policy:")
    print("  CLEWS  : all pairs")
    print("  WEALY  : vocal-valid pairs only")
    print("  FUSION : all pairs (adaptive fused distances)")
    print()

    models = {
        "CLEWS":  MODEL_PATHS["CLEWS"],
        "WEALY":  MODEL_PATHS["WEALY"],
        "FUSION": MODEL_PATHS["FUSION"],
    }

    summary_data = []
    all_detailed = []

    for model_name, csv_path in models.items():
        results, all_metrics = run_model_analysis(csv_path, model_name, beta=BETA)
        all_detailed.extend(all_metrics)
        if results:
            summary_data.append({
                'model':  model_name,
                'metric': results.get('metric', 'Unknown'),
                **results,
            })

    if summary_data:
        process_and_save_thresholds(
            summary_data, SUMMARY_FILES['threshold_analysis']
        )

    if all_detailed:
        detailed_df = pd.DataFrame(all_detailed)
        cols = [
            'model', 'metric', 'beta', 'optimal_threshold',
            'fbeta', 'f1', 'pr_auc', 'precision', 'recall', 'accuracy'
        ]
        cols = [c for c in cols if c in detailed_df.columns]
        detailed_df[cols].to_csv(SUMMARY_FILES['all_metrics_detailed'], index=False)
        print(f"Detailed report saved -> {SUMMARY_FILES['all_metrics_detailed']}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()