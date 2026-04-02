"""
Find optimal (alpha, beta) weights for late fusion.
Uses grid search to maximize F1-score for plagiarism detection.
Includes MRR and Failure Recovery Analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import importlib.util
from pathlib import Path

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

# Initialize logging
setup_logging(__file__)

def load_threshold_summary(summary_csv='results/threshold/threshold_analysis_summary.csv'):
    """Load the threshold summary with winning metrics for CLEWS and WEALY."""
    if not os.path.exists(summary_csv):
        return {}
    df_summary = pd.read_csv(summary_csv)
    mapping = {}
    for _, row in df_summary.iterrows():
        model = str(row.get('model', '')).strip().upper()
        metric = row.get('metric', 'cosine_distance')
        mapping[model] = metric
    return mapping

def resolve_metric_column(metric_name):
    if not isinstance(metric_name, str):
        return 'cosine_distance'
    name = metric_name.lower().replace(' ', '_')
    if name in ['cosine', 'cosine_distance']: return 'cosine_distance'
    if name in ['euclidean', 'euclidean_distance']: return 'euclidean_distance'
    if name in ['symmetric_kl', 'symmetric_kl_distance']: return 'symmetric_kl_distance'
    return metric_name

def load_fusion_data(clews_csv, wealy_csv, threshold_summary='results/threshold/threshold_analysis_summary.csv'):
    df_clews = pd.read_csv(clews_csv)
    df_wealy = pd.read_csv(wealy_csv)
    metric_map = load_threshold_summary(threshold_summary)
    
    clews_metric = resolve_metric_column(metric_map.get('CLEWS', 'cosine_distance'))
    wealy_metric = resolve_metric_column(metric_map.get('WEALY', 'cosine_distance'))

    df_clews = df_clews.rename(columns={clews_metric: 'dist_clews'})
    df_wealy = df_wealy.rename(columns={wealy_metric: 'dist_wealy'})

    merge_cols = ['pair_id', 'time', 'final_mod_type', 'filename_mod', 'filename_ori']
    df_merged = pd.merge(df_clews[merge_cols + ['dist_clews']], 
                         df_wealy[merge_cols + ['dist_wealy']], on=merge_cols, how='inner')
    df_merged['is_plagiarised'] = (~df_merged['final_mod_type'].str.startswith('Negative')).astype(int)
    return df_merged

def compute_fused_distance(df, alpha, beta=None):
    if beta is None: beta = 1.0 - alpha
    for col in ['dist_clews', 'dist_wealy']:
        min_v, max_v = df[col].min(), df[col].max()
        df[f'{col}_norm'] = (df[col] - min_v) / (max_v - min_v + 1e-8)
    df['fused_distance'] = (alpha * df['dist_clews_norm'] + beta * df['dist_wealy_norm'])
    return df

def normalize_with_train_minmax(df_train, df_test):
    """Normalize test data using min/max from train data."""
    min_clews = df_train['dist_clews'].min()
    max_clews = df_train['dist_clews'].max()
    min_wealy = df_train['dist_wealy'].min()
    max_wealy = df_train['dist_wealy'].max()

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train['dist_clews_norm'] = (df_train['dist_clews'] - min_clews) / (max_clews - min_clews + 1e-8)
    df_train['dist_wealy_norm'] = (df_train['dist_wealy'] - min_wealy) / (max_wealy - min_wealy + 1e-8)

    df_test['dist_clews_norm'] = (df_test['dist_clews'] - min_clews) / (max_clews - min_clews + 1e-8)
    df_test['dist_wealy_norm'] = (df_test['dist_wealy'] - min_wealy) / (max_wealy - min_wealy + 1e-8)

    return df_train, df_test

def evaluate_alpha_oof(df, alpha, n_splits=5):
    """Evaluate alpha using Out-Of-Fold predictions to get unbiased F1-score."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = []
    oof_indices = []

    for train_idx, test_idx in skf.split(df, df['is_plagiarised']):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        # Normalize using train min/max
        df_train_norm, df_test_norm = normalize_with_train_minmax(df_train, df_test)

        # Fuse
        beta = 1.0 - alpha
        df_train_norm['fused_distance'] = alpha * df_train_norm['dist_clews_norm'] + beta * df_train_norm['dist_wealy_norm']
        df_test_norm['fused_distance'] = alpha * df_test_norm['dist_clews_norm'] + beta * df_test_norm['dist_wealy_norm']

        # Find optimal threshold on train
        y_train = df_train_norm['is_plagiarised'].values
        y_scores_train = -df_train_norm['fused_distance'].values
        fpr, tpr, thresholds = roc_curve(y_train, y_scores_train)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = -thresholds[optimal_idx]

        # Apply threshold to test
        y_test_pred = (df_test_norm['fused_distance'] <= optimal_threshold).astype(int)
        oof_predictions.extend(y_test_pred)
        oof_indices.extend(test_idx)

    # Sort predictions back to original order
    oof_predictions = [x for _, x in sorted(zip(oof_indices, oof_predictions))]

    # Compute OOF F1
    y_true_oof = df['is_plagiarised'].values
    oof_f1 = f1_score(y_true_oof, oof_predictions)

    return oof_f1

def find_optimal_threshold_for_fusion(df, distance_col='fused_distance'):
    y_true = df['is_plagiarised'].values
    y_scores = -df[distance_col].values
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return -thresholds[optimal_idx]

def compute_failure_rate(df, distance_col, threshold):
    failures = (df[distance_col] > threshold).sum()
    return 100.0 * failures / len(df)

def compute_failure_rate_by_modification(df, distance_col, threshold):
    rows = []
    for mod_name, group in df.groupby('final_mod_type'):
        rate = compute_failure_rate(group, distance_col, threshold)
        rows.append({'final_mod_type': mod_name, 'failure_rate': rate, 'n_samples': len(group)})
    return pd.DataFrame(rows)

def compute_retrieval_metrics(df, distance_col, k=1):
    """Compute Recall@k and MRR (Mean Reciprocal Rank)."""
    total, hits, rr_sum = 0, 0, 0.0
    for _, g in df.groupby('pair_id'):
        if len(g) < 2: continue
        g = g.sort_values(distance_col)
        positives = g[g['is_plagiarised'] == 1]
        for _, row in positives.iterrows():
            total += 1
            rank = (g[distance_col] < row[distance_col]).sum() + 1
            if rank <= k: hits += 1
            rr_sum += 1.0 / rank
    return (0.0, 0.0) if total == 0 else (hits / total, rr_sum / total)

def compute_neighborhood_similarity(df, distance_col, k=10):
    """Computes similarity using a strict Non-Parametric Empirical CDF (eCDF)."""
    df = df.copy()
    plag_df = df[~df['final_mod_type'].str.startswith('Negative')].copy()
    baseline = df[df['final_mod_type'].str.startswith('Negative')][distance_col].values
    
    if len(baseline) == 0:
        plag_df['relative_distance'] = 0.0
        plag_df['neighborhood_confidence'] = 0.0
        return plag_df
        
    mean_baseline = np.mean(baseline)
    
    # Sort the baseline distances for efficient ranking
    sorted_baseline = np.sort(baseline)
    n_baseline = len(sorted_baseline)
    
    # Find the rank of each plagiarized sample's distance within the baseline distribution
    indices = np.searchsorted(sorted_baseline, plag_df[distance_col].values)
    
    # Estimate neighborhood confidence as the proportion of baseline samples that are farther than the plagiarized sample
    plag_df['neighborhood_confidence'] = 1.0 - (indices / n_baseline)
    
    # Relative distance can be used as an additional feature (lower means more similar to original)
    plag_df['relative_distance'] = plag_df[distance_col] / (mean_baseline + 1e-9)
    
    return plag_df

def evaluate_fusion(df, alpha, beta=None):
    df_copy = compute_fused_distance(df.copy(), alpha, beta)
    th = find_optimal_threshold_for_fusion(df_copy)
    y_pred = (df_copy['fused_distance'] <= th).astype(int)
    f1 = f1_score(df_copy['is_plagiarised'], y_pred)
    fail_rate = compute_failure_rate(df_copy, 'fused_distance', th)
    return f1, th, fail_rate, df_copy

def grid_search_optimal_weights(df, alpha_range=np.arange(0.0, 1.01, 0.05)):
    results = []
    for alpha in alpha_range:
        oof_f1 = evaluate_alpha_oof(df, alpha)
        results.append({'alpha': alpha, 'beta': 1.0-alpha, 'oof_f1_score': oof_f1})
    res_df = pd.DataFrame(results)
    best_row = res_df.loc[res_df['oof_f1_score'].idxmax()]
    return res_df, best_row

def plot_fusion_performance(results_df, best_result, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['alpha'], results_df['f1_score'], 'b-o', linewidth=2, label='F1-Score')
    ax.scatter(best_result['alpha'], best_result['f1_score'], color='red', s=150, zorder=5, 
               label=f'Optimal: α={best_result["alpha"]:.2f}, F1={best_result["f1_score"]:.4f}')
    ax.set_xlabel('Alpha (CLEWS Weight)')
    ax.set_ylabel('F1-Score')
    ax.set_title('Fusion Performance vs CLEWS Weight (α)')
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close()

def main():
    clews_csv, wealy_csv = "results/distances/clews_distances.csv", "results/distances/wealy_distances.csv"
    if not os.path.exists(clews_csv) or not os.path.exists(wealy_csv): return
    
    os.makedirs('results/fusion', exist_ok=True)
    os.makedirs('plots/fusion', exist_ok=True)

    df_merged = load_fusion_data(clews_csv, wealy_csv)
    results_df, best_res = grid_search_optimal_weights(df_merged)
    
    print(f"Best alpha selected via OOF F1: {best_res['alpha']:.2f} (OOF F1: {best_res['oof_f1_score']:.4f})")
    
    # Now apply the best alpha to the entire dataset for final artifacts
    f1_f, th_f, fail_f, df_f = evaluate_fusion(df_merged, best_res['alpha'], best_res['beta'])
    
    # Evaluate Individual Models for Baseline
    f1_c, th_c, _, df_c = evaluate_fusion(df_merged, 1.0, 0.0)
    f1_w, th_w, _, df_w = evaluate_fusion(df_merged, 0.0, 1.0)
    
    # Retrieval Metrics (Recall@1 & MRR)
    rec1_c, mrr_c = compute_retrieval_metrics(df_c, 'fused_distance', k=1)
    rec1_w, mrr_w = compute_retrieval_metrics(df_w, 'fused_distance', k=1)
    rec1_f, mrr_f = compute_retrieval_metrics(df_f, 'fused_distance', k=1)

    print("-" * 50)
    print(f"BEST WEIGHTS | Alpha (CLEWS): {best_res['alpha']:.2f} | Beta (WEALY): {best_res['beta']:.2f} | OOF F1: {best_res['oof_f1_score']:.4f}")
    print(f"CLEWS  | F1: {f1_c:.4f} | R@1: {rec1_c:.4f} | MRR: {mrr_c:.4f}")
    print(f"WEALY  | F1: {f1_w:.4f} | R@1: {rec1_w:.4f} | MRR: {mrr_w:.4f}")
    print(f"Fusion | F1: {f1_f:.4f} | R@1: {rec1_f:.4f} | MRR: {mrr_f:.4f}")
    print("-" * 50)
    
    # Failure Recovery Comparison
    fail_clews = compute_failure_rate_by_modification(df_c, 'fused_distance', th_c)
    fail_fusion = compute_failure_rate_by_modification(df_f, 'fused_distance', th_f)
    
    recovery_df = pd.merge(fail_clews, fail_fusion, on='final_mod_type', suffixes=('_clews', '_fusion'))
    recovery_df['recovery_delta'] = recovery_df['failure_rate_clews'] - recovery_df['failure_rate_fusion']
    recovery_df.to_csv('results/fusion/fusion_failure_recovery_detailed.csv', index=False)
    
    # Save Outputs
    compute_neighborhood_similarity(df_f, 'fused_distance').to_csv('results/fusion/fusion_neighborhood_similarity.csv', index=False)
    df_f.rename(columns={'fused_distance': 'cosine_distance'}).to_csv("results/fusion/optimal_fused_distances.csv", index=False)
    
    # Prepare data for plotting
    plot_results_df = results_df.rename(columns={'oof_f1_score': 'f1_score'})
    plot_best_res = best_res.copy()
    plot_best_res['f1_score'] = plot_best_res['oof_f1_score']
    
    plot_fusion_performance(plot_results_df, plot_best_res, 'plots/fusion/fusion_performance.png')

    config_df = pd.DataFrame([{
        'model': 'Fusion',
        'alpha': best_res['alpha'],
        'beta': best_res['beta'],
        'optimal_threshold': th_f,
        'f1_score': best_res['oof_f1_score']
    }])
    config_df.to_csv('results/fusion/optimal_fusion_config.csv', index=False)
    print("Fusion configuration saved to: results/fusion/optimal_fusion_config.csv")
    
    print("Optimization Complete. Results saved to 'results/fusion/' and 'plots/fusion/' directories.")

if __name__ == "__main__":
    main()