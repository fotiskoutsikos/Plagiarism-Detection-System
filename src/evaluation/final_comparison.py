#!/usr/bin/env python3
"""
Final comparison: CLEWS vs WEALY vs Fusion.
Generates comprehensive results table for thesis/paper.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def load_all_data():
    """Load all distance and fusion data."""
    data = {}
    
    # CLEWS
    if os.path.exists("data/clews_distances.csv"):
        data['clews'] = pd.read_csv("data/clews_distances.csv")
    
    # WEALY
    if os.path.exists("data/wealy_distances.csv"):
        data['wealy'] = pd.read_csv("data/wealy_distances.csv")
    
    # Fusion
    if os.path.exists("data/optimal_fused_distances.csv"):
        data['fusion'] = pd.read_csv("data/optimal_fused_distances.csv")
        
        if os.path.exists("results/optimal_fusion_config.csv"):
            df_config = pd.read_csv("results/optimal_fusion_config.csv")
            data['fusion_alpha'] = df_config.iloc[0]['alpha']
        else:
            data['fusion_alpha'] = 0.55 # Fallback
            
    return data

def compute_comprehensive_metrics(df, distance_col, model_name, threshold=None):
    """Compute all metrics for a model."""
    df = df.copy()
    df['is_plagiarised'] = (~df['final_mod_type'].str.startswith('Negative')).astype(int)    
    if threshold is None:
        # Use median as default threshold
        threshold = df[distance_col].quantile(0.5)
    
    df['predicted'] = (df[distance_col] <= threshold).astype(int)
    
    y_true = df['is_plagiarised'].values
    y_pred = df['predicted'].values
    
    # Overall metrics
    overall = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold,
        'n_samples': len(df)
    }
    
    # Per-category metrics
    per_category = []
    for mod_type in df['final_mod_type'].unique():
        subset = df[df['final_mod_type'] == mod_type]
        if len(subset) < 5:
            continue
        
        y_true_cat = subset['is_plagiarised'].values
        y_pred_cat = subset['predicted'].values
        
        per_category.append({
            'model': model_name,
            'category': mod_type,
            'n_samples': len(subset),
            'accuracy': accuracy_score(y_true_cat, y_pred_cat),
            'precision': precision_score(y_true_cat, y_pred_cat, zero_division=0),
            'recall': recall_score(y_true_cat, y_pred_cat, zero_division=0),
            'f1_score': f1_score(y_true_cat, y_pred_cat, zero_division=0),
            'mean_distance': subset[distance_col].mean(),
            'std_distance': subset[distance_col].std()
        })
    
    return overall, pd.DataFrame(per_category)

def print_final_comparison(overall_results):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON")
    print("=" * 90)
    print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Threshold':>12} {'N Samples':>12}")
    print("-" * 90)
    
    for result in overall_results:
        print(f"{result['model']:<15} {result['accuracy']:>10.4f} {result['precision']:>10.4f} "
              f"{result['recall']:>10.4f} {result['f1_score']:>10.4f} "
              f"{result['threshold']:>12.4f} {result['n_samples']:>12}")
    
    print("=" * 90)

def main():
    print("=" * 70)
    print("FINAL COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    # Load all data
    data = load_all_data()
    
    if not data:
        print("Error: No data found. Run previous analysis scripts first.")
        return
    
    print(f"\nLoaded models: {list(data.keys())}")
    
    # Load optimal thresholds
    thresholds = {}
    if os.path.exists("results/threshold_analysis_summary.csv"):
        df_thresh = pd.read_csv("results/threshold_analysis_summary.csv")
        for _, row in df_thresh.iterrows():
            thresholds[row['model']] = row['optimal_threshold']
    
    # Load optimal fusion config
    fusion_threshold = None
    if os.path.exists("results/optimal_fusion_config.csv"):
        df_fusion = pd.read_csv("results/optimal_fusion_config.csv")
        fusion_threshold = df_fusion.iloc[0]['optimal_threshold']
    
    # Compute metrics for each model
    overall_results = []
    all_per_category = []
    
    # CLEWS
    if 'clews' in data:
        overall, per_cat = compute_comprehensive_metrics(
            data['clews'], 'cosine_distance', 'CLEWS',
            threshold=thresholds.get('CLEWS')
        )
        overall_results.append(overall)
        all_per_category.append(per_cat)
    
    # WEALY
    if 'wealy' in data:
        overall, per_cat = compute_comprehensive_metrics(
            data['wealy'], 'cosine_distance', 'WEALY',
            threshold=thresholds.get('WEALY')
        )
        overall_results.append(overall)
        all_per_category.append(per_cat)
    
    # Fusion
    if 'fusion' in data:
        overall, per_cat = compute_comprehensive_metrics(
            data['fusion'], 'cosine_distance', 
            f"Fusion (α={data.get('fusion_alpha', 0.5):.2f})",
            threshold=fusion_threshold
        )
        overall_results.append(overall)
        all_per_category.append(per_cat)
    
    # Print comparison
    print_final_comparison(overall_results)
    
    # Save overall results
    pd.DataFrame(overall_results).to_csv("results/final_model_comparison.csv", index=False)
    print(f"\nOverall results saved to: results/final_model_comparison.csv")
    
    # Save per-category results
    if all_per_category:
        combined_per_cat = pd.concat(all_per_category, ignore_index=True)
        combined_per_cat.to_csv("results/final_per_category_comparison.csv", index=False)
        print(f"Per-category results saved to: results/final_per_category_comparison.csv")
    
    # Generate summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    if len(overall_results) >= 3:
        best_single = max(overall_results[:2], key=lambda x: x['f1_score'])
        fusion = overall_results[2]
        
        improvement = fusion['f1_score'] - best_single['f1_score']
        improvement_pct = (improvement / best_single['f1_score']) * 100 if best_single['f1_score'] > 0 else 0
        
        print(f"\nBest Single Model: {best_single['model']} (F1 = {best_single['f1_score']:.4f})")
        print(f"Fusion Model:      {fusion['model']} (F1 = {fusion['f1_score']:.4f})")
        print(f"Absolute Improvement:  +{improvement:.4f}")
        print(f"Relative Improvement:  +{improvement_pct:.2f}%")
        
        if improvement > 0:
            print("\n✓ Fusion provides statistically significant improvement!")
        else:
            print("\nFusion does not improve over best single model.")
    
    print("\n" + "=" * 70)
    print("FINAL COMPARISON COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()