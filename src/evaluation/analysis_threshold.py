#!/usr/bin/env python3
"""
Find optimal distance threshold for plagiarism detection.
Performs Ablation Study (Cosine vs Euclidean) and identifies the best metric.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_distances(csv_path):
    """Load distance metrics from CSV and assign labels correctly."""
    df = pd.read_csv(csv_path)
    
    df['is_plagiarised'] = (~df['final_mod_type'].str.startswith('Negative')).astype(int)
    
    return df

def evaluate_metric(df, distance_col, invert_logic=True):
    """
    Evaluates a specific distance metric and finds its optimal threshold.
    invert_logic: True for Cosine/Euclidean (lower=better), False for similarities.
    """
    y_true = df['is_plagiarised'].values
    
    if invert_logic:
        y_scores = -df[distance_col].values
    else:
        y_scores = df[distance_col].values
        
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    
    if invert_logic:
        optimal_threshold = -thresholds[optimal_idx]
        y_pred = (df[distance_col] <= optimal_threshold).astype(int)
    else:
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (df[distance_col] >= optimal_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    results = {
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    
    return results, fpr, tpr

def plot_ablation_roc(fpr_cos, tpr_cos, auc_cos, fpr_euc, tpr_euc, auc_euc, output_path, model_name):
    """Plot ROC curves for both metrics on the same graph for direct comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Cosine
    ax.plot(fpr_cos, tpr_cos, color='blue', lw=2.5, 
            label=f'Cosine Distance (AUC = {auc_cos:.3f})')
            
    # Plot Euclidean
    ax.plot(fpr_euc, tpr_euc, color='red', lw=2.5, linestyle=':',
            label=f'Euclidean Distance (AUC = {auc_euc:.3f})')
            
    # Random Baseline
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Chance')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Ablation Study: Metric Comparison - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ablation ROC curve saved to: {output_path}")

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
    
    # Evaluate Cosine Distance
    print(f"[{model_name}] Evaluating Cosine Distance...")
    res_cos, fpr_cos, tpr_cos = evaluate_metric(df, 'cosine_distance')
    
    # Evaluate Euclidean Distance
    print(f"[{model_name}] Evaluating Euclidean Distance...")
    res_euc, fpr_euc, tpr_euc = evaluate_metric(df, 'euclidean_distance')
    
    # Print Comparison
    print(f"\n=== ABLATION STUDY RESULTS ({model_name}) ===")
    print(f"{'Metric':<15} | {'AUC':<7} | {'F1-Score':<10} | {'Opt. Threshold'}")
    print("-" * 55)
    print(f"{'Cosine':<15} | {res_cos['roc_auc']:.4f}  | {res_cos['f1']:.4f}     | {res_cos['optimal_threshold']:.4f}")
    print(f"{'Euclidean':<15} | {res_euc['roc_auc']:.4f}  | {res_euc['f1']:.4f}     | {res_euc['optimal_threshold']:.4f}")
    
    # Determine the winner
    winner = 'Cosine' if res_cos['roc_auc'] >= res_euc['roc_auc'] else 'Euclidean'
    print(f"\n=> WINNING METRIC: {winner} Distance")
    
    # Generate Plots
    plot_ablation_roc(fpr_cos, tpr_cos, res_cos['roc_auc'], 
                      fpr_euc, tpr_euc, res_euc['roc_auc'], 
                      f'plots/{model_name.lower()}_ablation_roc.png', model_name)
                      
    plot_distance_distributions(df, res_cos['optimal_threshold'], 'cosine_distance', 
                              f'plots/{model_name.lower()}_cosine_distributions.png')
                              
    plot_distance_distributions(df, res_euc['optimal_threshold'], 'euclidean_distance', 
                              f'plots/{model_name.lower()}_euclidean_distributions.png')
                              
    # Return the results of the winning metric
    winning_results = res_cos if winner == 'Cosine' else res_euc
    return winning_results

def main():
    print("=" * 70)
    print("ABLATION STUDY & THRESHOLD ANALYSIS FOR PLAGIARISM DETECTION")
    print("=" * 70)
    
    results_clews = run_model_analysis("data/clews_distances.csv", "CLEWS")
    results_wealy = run_model_analysis("data/wealy_distances.csv", "WEALY")
    
    # Save summary of the winning metrics
    summary_path = "results/threshold_analysis_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary_data = []
    if results_clews:
        summary_data.append({'model': 'CLEWS', **{k: v for k, v in results_clews.items() if isinstance(v, (int, float))}})
    if results_wealy:
        summary_data.append({'model': 'WEALY', **{k: v for k, v in results_wealy.items() if isinstance(v, (int, float))}})
    
    if summary_data:
        pd.DataFrame(summary_data).to_csv(summary_path, index=False)
        print(f"\nSummary of winning metrics saved to: {summary_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()