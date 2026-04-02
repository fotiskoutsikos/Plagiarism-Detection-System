"""
Binary Classification Analysis for Plagiarism Detection (Pairwise).
Evaluates the model using FIXED thresholds imported from 'threshold_analysis_summary.csv'.
"""

import os
import sys
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score

# Resolve repository root and load logging_util.py
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


def run_binary_classification_evaluation(df_all, model_name, fixed_threshold):
    """Evaluate pairwise similarity using a predefined distance threshold."""
    df_eval = df_all.copy()
    
    # Ground Truth (0 = Negative Pairs, 1 = Plagiarism/Modified)
    df_eval['is_positive'] = ~df_eval['final_mod_type'].str.startswith('Negative', na=False)
    df_eval['y_true'] = df_eval['is_positive'].astype(int)
    
    # Grouping categories for analysis
    df_eval['category_grouped'] = df_eval['final_mod_type']
    smp_conditions = df_eval['category_grouped'].isin(['SMP_plag', 'SMP_plag_doubt', 'SMP_remake'])
    df_eval.loc[smp_conditions, 'category_grouped'] = 'Human Plagiarism (SMP)'
    df_eval.loc[df_eval['category_grouped'].str.contains('MusicGen|AI', case=False, na=False), 'category_grouped'] = 'AI Plagiarism (MusicGen)'
    df_eval.loc[df_eval['y_true'] == 0, 'category_grouped'] = 'Negative Pairs'

    # Check if cosine_distance column exists
    if 'cosine_distance' not in df_eval.columns:
        raise ValueError("Column 'cosine_distance' not found in the dataset.")
        
    # Apply the fixed threshold to get binary predictions
    df_eval['y_pred'] = (df_eval['cosine_distance'] <= fixed_threshold).astype(int)
    
    
    metrics_list = []
    
    # Analysis per Category
    positive_categories = df_eval[df_eval['y_true'] == 1]['category_grouped'].unique()
    
    for cat in sorted(positive_categories):
        mask = (df_eval['category_grouped'] == cat) | (df_eval['y_true'] == 0)
        df_sub = df_eval[mask]
        
        tn, fp, fn, tp = confusion_matrix(df_sub['y_true'], df_sub['y_pred'], labels=[0, 1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics_list.append({
            'Category': cat,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Total_Queries': tp + fn
        })
        
    # Final Overall Metrics (All Positives vs All Negatives)
    tn, fp, fn, tp = confusion_matrix(df_eval['y_true'], df_eval['y_pred'], labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics_list.append({
        'Category': 'OVERALL',
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Total_Queries': tp + fn
    })
    
    df_metrics = pd.DataFrame(metrics_list)
    
    # Print results in a formatted table
    print(f"\n{'=' * 95}")
    print(f" BINARY CLASSIFICATION PERFORMANCE: {model_name.upper()}")
    print(f" Applied FIXED Threshold: <= {fixed_threshold:.4f}")
    print(f"{'=' * 95}")
    print(f"{'Modification Category':<25} | {'Precision':>9} | {'Recall':>8} | {'F1-Score':>8} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>6}")
    print(f"{'-' * 95}")

    for _, row in df_metrics.iterrows():
        is_overall = row['Category'] == 'OVERALL'
        prefix = "► " if is_overall else "  "
        print(f"{prefix}{row['Category']:<23} | {row['Precision']:>8.1%} | {row['Recall']:>7.1%} | {row['F1-Score']:>7.1%} | {row['TP']:>5} | {row['FP']:>5} | {row['FN']:>5} | {row['TN']:>6}")

    print(f"{'=' * 95}\n")
    
    # Save results to CSV
    os.makedirs('results/binary_classification', exist_ok=True)
    model_path = f'results/binary_classification/{model_name.lower()}_binary_metrics.csv'
    df_metrics.to_csv(model_path, index=False)
    
    summary_path = 'results/binary_classification/binary_summary.csv'
    df_summary = df_metrics.copy()
    df_summary.insert(0, 'Model', model_name)
    df_summary['Applied_Threshold'] = fixed_threshold
    
    df_summary.to_csv(summary_path, index=False, mode='a', header=not os.path.exists(summary_path))

    return df_metrics


def main():
    print("=" * 80)
    print("STARTING BINARY CLASSIFICATION WITH FIXED THRESHOLDS")
    print("=" * 80)
    
    # Remove old summary if exists
    if os.path.exists('results/binary_classification/binary_summary.csv'): 
        os.remove('results/binary_classification/binary_summary.csv')

    # Load Thresholds
    threshold_file = "results/threshold/threshold_analysis_summary.csv"  
    thresholds_dict = {}
    
    if os.path.exists(threshold_file):
        df_thresh = pd.read_csv(threshold_file)
        print(f"Loaded {len(df_thresh)} thresholds from {threshold_file}:")
        for _, row in df_thresh.iterrows():
            model_key = str(row['model']).upper()
            opt_thresh = float(row['optimal_threshold'])
            thresholds_dict[model_key] = opt_thresh
            print(f"  -> {model_key}: {opt_thresh:.4f}")
    else:
        print(f"\n[ERROR] Required threshold file '{threshold_file}' not found.")
        return

    # Load distance CSVs for each model (if they exist) and run evaluation
    FILES = {
        "CLEWS": "results/distances/clews_distances.csv",
        "WEALY": "results/distances/wealy_distances.csv",
        "FUSION": "results/fusion/optimal_fused_distances.csv"
    }

    processed_any = False
    for model, csv_path in FILES.items():
        if os.path.exists(csv_path):
            if model in thresholds_dict:
                fixed_thresh = thresholds_dict[model]
                df_all = pd.read_csv(csv_path)
                run_binary_classification_evaluation(df_all, model, fixed_thresh)
                processed_any = True
            else:
                print(f"Warning: No threshold mapped for {model} in {threshold_file}. Skipping.")
        else:
            print(f"Warning: Distance file not found -> {csv_path}")
            
    if processed_any:
        print("\n[SUCCESS] Binary classification analysis saved to: results/binary_classification/")
    else:
        print("\n[FAILED] No distance CSV files found to process.")

if __name__ == "__main__":
    main()