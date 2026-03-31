#!/usr/bin/env python3
"""
Find optimal (alpha, beta) weights for late fusion.
Uses grid search to maximize F1-score for plagiarism detection.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

def load_fusion_data(clews_csv, wealy_csv):
    """Load and merge CLEWS and WEALY distances."""
    df_clews = pd.read_csv(clews_csv)
    df_wealy = pd.read_csv(wealy_csv)
    
    # Rename distance columns
    df_clews = df_clews.rename(columns={'cosine_distance': 'dist_clews'})
    df_wealy = df_wealy.rename(columns={'cosine_distance': 'dist_wealy'})
    
    # Merge on common columns
    merge_cols = ['pair_id', 'time', 'final_mod_type', 'filename_mod', 'filename_ori']
    
    # Inner join to ensure both models have predictions
    df_merged = pd.merge(
        df_clews[merge_cols + ['dist_clews']],
        df_wealy[merge_cols + ['dist_wealy']],
        on=merge_cols,
        how='inner'
    )
    
    # Create binary label
    df_merged['is_plagiarised'] = (~df_merged['final_mod_type'].str.startswith('Negative')).astype(int)
    
    return df_merged

def compute_fused_distance(df, alpha, beta=None):
    """Compute fused distance with given weights."""
    if beta is None:
        beta = 1.0 - alpha
    
    # Min-Max normalization for consistent scale
    for col in ['dist_clews', 'dist_wealy']:
        min_val = df[col].min()
        max_val = df[col].max()
        df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val + 1e-8)
    
    df['fused_distance'] = (alpha * df['dist_clews_norm'] + 
                           beta * df['dist_wealy_norm'])
    
    return df

def find_optimal_threshold_for_fusion(df, distance_col='fused_distance'):
    """Find optimal threshold for fused distances."""
    y_true = df['is_plagiarised'].values
    y_scores = -df[distance_col].values
    
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = -thresholds[optimal_idx]
    
    return optimal_threshold

def evaluate_fusion(df, alpha, beta=None):
    """Evaluate fusion performance with given weights."""
    df_copy = df.copy()
    df_copy = compute_fused_distance(df_copy, alpha, beta)
    
    optimal_threshold = find_optimal_threshold_for_fusion(df_copy)
    
    y_true = df_copy['is_plagiarised'].values
    y_pred = (df_copy['fused_distance'] <= optimal_threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred)
    
    return f1, optimal_threshold

def grid_search_optimal_weights(df, alpha_range=np.arange(0.0, 1.01, 0.05)):
    """Grid search for optimal alpha weight."""
    results = []
    
    for alpha in alpha_range:
        beta = 1.0 - alpha
        f1, threshold = evaluate_fusion(df, alpha, beta)
        
        results.append({
            'alpha': alpha,
            'beta': beta,
            'f1_score': f1,
            'optimal_threshold': threshold
        })
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1_score'].idxmax()
    best_result = results_df.loc[best_idx]
    
    return results_df, best_result

def plot_fusion_performance(results_df, best_result, output_path):
    """Plot F1-score vs alpha weight."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results_df['alpha'], results_df['f1_score'], 
           'b-o', linewidth=2, markersize=6, label='F1-Score')
    
    # Mark optimal point
    ax.scatter(best_result['alpha'], best_result['f1_score'],
              color='red', s=150, zorder=5, 
              label=f'Optimal: α={best_result["alpha"]:.2f}, β={best_result["beta"]:.2f}, F1={best_result["f1_score"]:.4f}')
    
    ax.set_xlabel('Alpha (CLEWS Weight)', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Fusion Performance vs CLEWS Weight (α)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fusion performance plot saved to: {output_path}")

def main():
    print("=" * 70)
    print("FUSION WEIGHT OPTIMIZATION")
    print("=" * 70)
    
    clews_csv = "data/clews_distances.csv"
    wealy_csv = "data/wealy_distances.csv"
    
    if not os.path.exists(clews_csv) or not os.path.exists(wealy_csv):
        print("Error: Distance files not found. Run metrics.py first.")
        return
    
    print("\nLoading and merging distance data...")
    df_merged = load_fusion_data(clews_csv, wealy_csv)
    print(f"Merged dataset: {len(df_merged)} pairs")
    
    print("\nRunning grid search for optimal weights...")
    results_df, best_result = grid_search_optimal_weights(df_merged)
    
    print("\n" + "=" * 70)
    print("OPTIMAL FUSION WEIGHTS")
    print("=" * 70)
    print(f"Alpha (CLEWS):  {best_result['alpha']:.2f}")
    print(f"Beta (WEALY):   {best_result['beta']:.2f}")
    print(f"Optimal Threshold: {best_result['optimal_threshold']:.4f}")
    print(f"F1-Score:       {best_result['f1_score']:.4f}")
    print("=" * 70)
    
    # Compare with single models
    print("\nComparison with Single Models:")
    f1_clews, _ = evaluate_fusion(df_merged, alpha=1.0, beta=0.0)
    f1_wealy, _ = evaluate_fusion(df_merged, alpha=0.0, beta=1.0)
    
    print(f"  CLEWS-only:  F1 = {f1_clews:.4f}")
    print(f"  WEALY-only:  F1 = {f1_wealy:.4f}")
    print(f"  Fusion:      F1 = {best_result['f1_score']:.4f}")
    print(f"  Improvement: +{best_result['f1_score'] - max(f1_clews, f1_wealy):.4f} over best single model")

    print("\nSaving optimal fused dataset for final comparison...")
    df_optimal = df_merged.copy()
    df_optimal = compute_fused_distance(df_optimal, best_result['alpha'], best_result['beta'])
    # Το μετονομάζουμε σε cosine_distance για να "κουμπώσει" τέλεια με το final_comparison.py
    df_optimal = df_optimal.rename(columns={'fused_distance': 'cosine_distance'})
    df_optimal.to_csv("data/optimal_fused_distances.csv", index=False)
    print("Saved to: data/optimal_fused_distances.csv")
    
    # Save results
    results_df.to_csv("results/fusion_grid_search_results.csv", index=False)
    
    # Save best configuration
    best_config = {
        'alpha': best_result['alpha'],
        'beta': best_result['beta'],
        'optimal_threshold': best_result['optimal_threshold'],
        'f1_score': best_result['f1_score'],
        'f1_clews_only': f1_clews,
        'f1_wealy_only': f1_wealy,
        'improvement': best_result['f1_score'] - max(f1_clews, f1_wealy)
    }
    pd.DataFrame([best_config]).to_csv("results/optimal_fusion_config.csv", index=False)
    
    # Plot
    plot_fusion_performance(results_df, best_result, 'plots/fusion_performance.png')
    
    print("\n" + "=" * 70)
    print("FUSION OPTIMIZATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()