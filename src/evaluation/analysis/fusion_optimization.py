"""
Exhaustive Fusion Optimization for Plagiarism Detection.
Performs grid search over 16 metric combinations × 21 alpha weights (336 trials).
Optimizes strictly for F0.5-Score via Precision-Recall curve thresholding.

Design Principles:
1. Data Alignment: Strict inner merge on identical pairs. Mismatches are dropped.
2. Normalization: Min-Max scaling to [0, 1].
3. Exhaustive Search: 4 CLEWS metrics × 4 WEALY metrics × 21 alpha weights = 336 trials.
4. Winner Selection: Absolute max F0.5-Score wins. Outputs full grid CSV + winning fused distances.
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Union, Optional
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
import logging

# Logging Setup
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
    DISTANCE_METRICS as METRICS,
    MERGE_KEYS,
    ALPHA_VALUES,
    MODEL_PATHS,
    OUTPUT_DIRS,
    SUMMARY_FILES,
    TEST_SIZE,
    RANDOM_STATE
)
from utils.categorization import get_ground_truth_label


def load_and_align_data(clews_csv: str, wealy_csv: str) -> pd.DataFrame:
    """
    Load & strictly align datasets.
    Only pairs present in BOTH models with identical metadata are kept.
    """
    logger.info(f"Loading CLEWS from {clews_csv}")
    df_c = pd.read_csv(clews_csv)
    logger.info(f"Loading WEALY from {wealy_csv}")
    df_w = pd.read_csv(wealy_csv)

    # Keep only merge keys + all possible metric columns
    cols_c = MERGE_KEYS + METRICS
    cols_w = MERGE_KEYS + METRICS
    df_c = df_c[[c for c in cols_c if c in df_c.columns]]
    df_w = df_w[[c for c in cols_w if c in df_w.columns]]

    # Strict inner join: only pairs present in BOTH models with identical keys
    df_merged = pd.merge(df_c, df_w, on=MERGE_KEYS, how='inner', suffixes=('_clews', '_wealy'))
    
    # Create binary label
    df_merged['is_plagiarised'] = df_merged['final_mod_type'].apply(get_ground_truth_label)
    
    initial_total = len(df_c) + len(df_w)
    aligned_len = len(df_merged)
    logger.info(f"Alignment: {initial_total} total rows → {aligned_len} perfectly matched pairs.")
    
    if aligned_len == 0:
        raise ValueError("No matching pairs found between CLEWS and WEALY datasets.")
    return df_merged


def normalize_with_train_stats(df_train: pd.DataFrame, df_val: pd.DataFrame, 
                                cols: list) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Min-Max normalization using TRAIN statistics only to prevent data leakage.
    """
    df_train = df_train.copy()
    df_val = df_val.copy()
    norm_stats = {}
    
    for col in cols:
        min_v = df_train[col].min()
        max_v = df_train[col].max()
        rng = max_v - min_v
        if rng < 1e-9:  # Avoid division by zero for constant columns
            rng = 1e-9
        
        norm_stats[col] = {'min': min_v, 'max': max_v, 'range': rng}
        
        # Apply normalization
        df_train[f'{col}_norm'] = (df_train[col] - min_v) / rng
        df_val[f'{col}_norm'] = (df_val[col] - min_v) / rng
        
    return df_train, df_val, norm_stats


def _fbeta_score_curve(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute F-beta scores along a PR curve.

    F_beta = (1 + beta²) * P * R / (beta² * P + R)

    beta < 1  → weights Precision more  (e.g. beta=0.5 → Precision 4x recall)
    beta = 1  → harmonic mean of P and R (standard F1)
    beta > 1  → weights Recall more
    """
    beta_sq = beta ** 2
    numerator   = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    return np.divide(numerator, denominator,
                     out=np.zeros_like(numerator), where=denominator != 0)


def find_internal_fbeta_threshold(distances: np.ndarray, y_true, beta: float = 0.5) -> float:
    """
    INTERNAL USE ONLY: Finds the optimal threshold for the validation split 
    to score the current Alpha value during grid search.
    Optimizes for maximum F-beta score (default beta=0.5 for precision-weighted).
    """
    y_true = np.asarray(y_true)
    
    if len(np.unique(y_true)) < 2:
        return float(np.median(distances))
        
    # PR curve expects: higher score → more likely positive. Distances are inverted.
    scores = -distances
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # F-beta calculation with zero-division protection
    fbeta_scores = _fbeta_score_curve(precision, recall, beta)
    
    if len(fbeta_scores) > 1:
        optimal_idx = np.argmax(fbeta_scores[:-1])
    else:
        return float(np.median(distances))
        
    return -thresholds[optimal_idx]


def run_exhaustive_grid_search(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    """
    Exhaustive search over 16 metric combos × 21 alphas (336 trials).
    """
    results = []
    best_f05 = -1.0
    best_config = None
    
    # Generate all 16 metric combinations: 4 CLEWS × 4 WEALY
    metric_combos = list(itertools.product(METRICS, METRICS))
    total_trials = len(metric_combos) * len(ALPHA_VALUES)  # 16 × 21 = 336
    
    logger.info(f"Starting exhaustive grid search: {total_trials} trials...")
    
    y_train = df_train['is_plagiarised'].values
    y_val = df_val['is_plagiarised'].values
    
    for trial_idx, (m_c, m_w) in enumerate(metric_combos, 1):
        col_c_train = f'{m_c}_clews_norm'
        col_w_train = f'{m_w}_wealy_norm'
        col_c_val = f'{m_c}_clews_norm'
        col_w_val = f'{m_w}_wealy_norm'
        
        dist_c_train = df_train[col_c_train].values
        dist_w_train = df_train[col_w_train].values
        dist_c_val = df_val[col_c_val].values
        dist_w_val = df_val[col_w_val].values
        
        for alpha in ALPHA_VALUES:
            beta = 1.0 - alpha
            
            # Fuse training distances
            train_fused = alpha * dist_c_train + beta * dist_w_train
            
            # Internal scoring mechanism to evaluate this alpha
            opt_th = find_internal_fbeta_threshold(train_fused, y_train, beta=0.5)
            
            val_fused = alpha * dist_c_val + beta * dist_w_val
            y_val_pred = (val_fused <= opt_th).astype(int)
            val_f05 = fbeta_score(np.asarray(y_val), y_val_pred, beta=0.5, zero_division=0)
            
            trial_res = {
                'clews_metric': m_c,
                'wealy_metric': m_w,
                'alpha': round(float(alpha), 2),
                'beta': round(float(beta), 2),
                'internal_val_f05_score': round(float(val_f05), 4)
            }
            results.append(trial_res)
            
            if val_f05 > best_f05:
                best_f05 = val_f05
                best_config = trial_res.copy()
                
        if trial_idx % 4 == 0:
            logger.info(f"Progress: {trial_idx * len(ALPHA_VALUES)}/{total_trials} trials. Current best F0.5: {best_f05:.4f}")
            
    logger.info(f"Grid search complete. Best internal validation F0.5: {best_f05:.4f}")
    return pd.DataFrame(results), best_config


def generate_final_fused_distances(df_full: pd.DataFrame, best_config: Optional[dict], norm_stats: dict) -> pd.DataFrame:
    """
    Applies the winning configuration to the full dataset and exports it cleanly.
    Output column is strictly named 'fused_distance'.
    """
    if best_config is None:
        raise ValueError("best_config cannot be None. Grid search must produce a valid configuration.")
    
    m_c = best_config['clews_metric']
    m_w = best_config['wealy_metric']
    alpha = best_config['alpha']
    beta = best_config['beta']
    
    df_out = df_full.copy()
    
    for suffix, metric in [('clews', m_c), ('wealy', m_w)]:
        col = f'{metric}_{suffix}'
        stats = norm_stats[col]
        df_out[f'{col}_norm'] = (df_out[col] - stats['min']) / stats['range']
        
    # Dynamic column name for the fused distance
    fusion_col_name = f"{m_c}+{m_w}"
    
    df_out[fusion_col_name] = alpha * df_out[f'{m_c}_clews_norm'] + beta * df_out[f'{m_w}_wealy_norm']
    keep_cols = MERGE_KEYS + ['is_plagiarised', fusion_col_name]
    
    return df_out[keep_cols]


def save_results(all_results: pd.DataFrame, best_config: Optional[dict], df_final: pd.DataFrame, output_dir: str, output_plots: str):
    if best_config is None:
        raise ValueError("best_config cannot be None.")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_plots).mkdir(parents=True, exist_ok=True)
    
    grid_path = Path(output_dir) / 'fusion_grid_search_results.csv'
    all_results.to_csv(grid_path, index=False)
    
    config_path = Path(output_dir) / 'optimal_fusion_config.csv'
    config_df = pd.DataFrame([best_config])
    config_df.to_csv(config_path, index=False)
    
    dist_path = Path(output_dir) / 'optimal_fused_distances.csv'
    df_final.to_csv(dist_path, index=False)
    logger.info(f"Saved cleanly named fused distances to {dist_path}")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = all_results.pivot_table(
        index='clews_metric', 
        columns='wealy_metric', 
        values='internal_val_f05_score', 
        aggfunc='max'
    )
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('WEALY Metric')
    ax.set_ylabel('CLEWS Metric')
    ax.set_title('Max Validation F0.5-Score per Metric Pair')
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color='white', fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Internal F0.5-Score')
    fig.tight_layout()
    plot_path = Path(output_plots) / 'fusion_heatmap_max_fbeta_score.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Alpha vs F-beta-Score Curve
    best_m_c = best_config['clews_metric']
    best_m_w = best_config['wealy_metric']
    
    df_best_pair = all_results[
        (all_results['clews_metric'] == best_m_c) & 
        (all_results['wealy_metric'] == best_m_w)
    ]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(df_best_pair['alpha'], df_best_pair['internal_val_f05_score'], marker='o', linestyle='-', color='purple', lw=2)
    ax2.axvline(x=best_config['alpha'], color='red', linestyle='--', label=f"Optimal α = {best_config['alpha']:.2f}")
    
    ax2.set_xlabel('Weight CLEWS (α)', fontsize=12)
    ax2.set_ylabel('Internal Validation F0.5-Score', fontsize=12)
    ax2.set_title(f'F0.5-Score vs Alpha\n({best_m_c} & {best_m_w})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    plot_path2 = Path(output_plots) / 'fusion_fbeta_score_vs_alpha.png'
    fig2.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)


def main():
    logger.info("=" * 70)
    logger.info("EXHAUSTIVE FUSION OPTIMIZATION")
    logger.info("=" * 70)
    
    clews_csv = MODEL_PATHS["CLEWS"]
    wealy_csv = MODEL_PATHS["WEALY"]
    output_dir = OUTPUT_DIRS["fusion"]
    output_plots = OUTPUT_DIRS["fusion_plots"]
    
    if not Path(clews_csv).exists() or not Path(wealy_csv).exists():
        logger.error("Input distance CSVs not found. Run metrics.py first.")
        return
        
    logger.info("Aligning datasets...")
    df_aligned = load_and_align_data(clews_csv, wealy_csv)
    
    logger.info("Splitting & normalizing...")
    df_train, df_val = train_test_split(
        df_aligned, 
        test_size=TEST_SIZE, 
        stratify=df_aligned['is_plagiarised'], 
        random_state=RANDOM_STATE
    )
    
    norm_cols = [f'{m}_{s}' for m in METRICS for s in ['clews', 'wealy']]
    df_train, df_val, norm_stats = normalize_with_train_stats(df_train, df_val, norm_cols)
    
    logger.info("Running exhaustive grid search (336 trials)...")
    all_results, best_config = run_exhaustive_grid_search(df_train, df_val)
    
    if best_config is None:
        logger.error("No valid configuration found.")
        return
    
    logger.info("Generating final fused distances...")
    df_final = generate_final_fused_distances(df_aligned, best_config, norm_stats)
    save_results(all_results, best_config, df_final, output_dir, output_plots)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"WINNER: {best_config['clews_metric']} + {best_config['wealy_metric']}")
    logger.info(f"WEIGHTS: α={best_config['alpha']:.2f}, β={best_config['beta']:.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()