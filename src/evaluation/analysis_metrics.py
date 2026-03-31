#!/usr/bin/env python3
"""
Compute per-category metrics: Accuracy, Precision, Recall, F1-Score.
Analyzes performance for each modification type separately against all negatives.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def load_and_prepare(csv_path, optimal_threshold):
    """Load distances, clean labels, and apply threshold for binary classification."""
    df = pd.read_csv(csv_path)
    
    # Human Plagiarism
    smp_conditions = df['final_mod_type'].isin(['SMP_plag', 'SMP_plag_doubt', 'SMP_remake'])
    df.loc[smp_conditions, 'final_mod_type'] = 'Human Plagiarism (SMP)'
    
    # Create binary labels
    df['predicted_plagiarised'] = (df['cosine_distance'] <= optimal_threshold).astype(int)
    df['actual_plagiarised'] = (~df['final_mod_type'].str.startswith('Negative')).astype(int)
    
    return df

def compute_metrics_per_category(df):
    """Compute metrics for each modification type evaluated against all Hard Negatives."""
    results = []
    
    df_negatives = df[df['actual_plagiarised'] == 0]
    
    # Get all unique modification types that are not negatives
    mod_types = [m for m in df['final_mod_type'].unique() if not str(m).startswith('Negative')]
    
    for mod_type in mod_types:
        df_mod = df[df['final_mod_type'] == mod_type]
        
        if len(df_mod) < 5:  # Skip categories with too few samples
            continue
            
        # Combine current modification type with all negatives for evaluation
        subset = pd.concat([df_mod, df_negatives])
        
        y_true = subset['actual_plagiarised'].values
        y_pred = subset['predicted_plagiarised'].values
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Categorize into macro categories (Human Plagiarism, AI Base, AI Pitch, AI Tempo, AI Both)
        macro_cat = 'Unknown'
        if 'Human Plagiarism' in mod_type: macro_cat = 'Human Plagiarism'
        elif mod_type == 'musicgen': macro_cat = 'AI (Base)'
        elif 'pitch' in mod_type and 'tempo' in mod_type: macro_cat = 'AI (Pitch + Tempo)'
        elif 'pitch' in mod_type: macro_cat = 'AI (Pitch)'
        elif 'tempo' in mod_type: macro_cat = 'AI (Tempo)'
        
        results.append({
            'macro_category': macro_cat,
            'mod_type': mod_type,
            'n_positives': len(df_mod),
            'n_negatives': len(df_negatives),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_negatives': fn,
            'false_positives': fp,
            'mean_distance': df_mod['cosine_distance'].mean(),
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=['macro_category', 'f1_score'], ascending=[True, False])
    return df_results

def print_summary_table(df_results, title):
    """Print formatted summary table."""
    print(f"\n{'=' * 110}")
    print(f" {title}")
    print(f"{'=' * 110}")
    print(f"{'Macro Category':<20} | {'Specific Modification':<30} | {'N(+)':>5} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Mean Dist':>9}")
    print(f"{'-' * 110}")
    
    for _, row in df_results.iterrows():
        print(f"{row['macro_category']:<20} | {row['mod_type']:<30} | {row['n_positives']:>5} | "
              f"{row['accuracy']:>6.3f} | {row['precision']:>6.3f} | "
              f"{row['recall']:>6.3f} | {row['f1_score']:>6.3f} | "
              f"{row['mean_distance']:>9.4f}")
    
    print(f"{'=' * 110}")

def main():
    print("=" * 70)
    print("PER-CATEGORY METRICS ANALYSIS (vs Negative Baselines)")
    print("=" * 70)
    
    threshold_summary = "results/threshold_analysis_summary.csv"
    
    thresholds = {}
    if os.path.exists(threshold_summary):
        df_thresh = pd.read_csv(threshold_summary)
        for _, row in df_thresh.iterrows():
            thresholds[row['model']] = row['optimal_threshold']
    else:
        thresholds = {'CLEWS': 0.8440, 'WEALY': 0.7810} # Fallback thresholds
        print("Warning: threshold_analysis_summary.csv not found. Using previously found thresholds.")
    
    # Analyze CLEWS
    clews_path = "data/clews_distances.csv"
    if os.path.exists(clews_path):
        print("\n[1/2] CLEWS Per-Category Metrics")
        df_clews = load_and_prepare(clews_path, thresholds.get('CLEWS', 0.8440))
        results_clews = compute_metrics_per_category(df_clews)
        print_summary_table(results_clews, f"CLEWS PERFORMANCE (Threshold: {thresholds.get('CLEWS', 0.8440):.4f})")
        results_clews['model'] = 'CLEWS'
        
        os.makedirs("results", exist_ok=True)
        results_clews.to_csv("results/clews_per_category_metrics.csv", index=False)
    
    # Analyze WEALY
    wealy_path = "data/wealy_distances.csv"
    if os.path.exists(wealy_path):
        print("\n[2/2] WEALY Per-Category Metrics")
        df_wealy = load_and_prepare(wealy_path, thresholds.get('WEALY', 0.7810))
        results_wealy = compute_metrics_per_category(df_wealy)
        print_summary_table(results_wealy, f"WEALY PERFORMANCE (Threshold: {thresholds.get('WEALY', 0.7810):.4f})")
        results_wealy['model'] = 'WEALY'
        results_wealy.to_csv("results/wealy_per_category_metrics.csv", index=False)
    
    # Combined comparison
    if 'results_clews' in locals() and 'results_wealy' in locals():
        combined = pd.concat([results_clews, results_wealy], ignore_index=True)
        combined.to_csv("results/combined_per_category_metrics.csv", index=False)
        print(f"\nCombined results saved to: results/combined_per_category_metrics.csv")

if __name__ == "__main__":
    main()