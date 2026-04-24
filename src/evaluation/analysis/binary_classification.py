"""
Binary Classification Analysis for Plagiarism Detection (Pairwise).
Evaluates the model using FIXED thresholds and WINNING METRICS imported from 'threshold_analysis_summary.csv'.
Categorizes results precisely into Human, Original+DSP, AI(Base), and AI+DSP.
Now also calculates F0.5-score and exports a False Positive Tier Breakdown.
"""

import os
import sys
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
from sklearn.metrics import confusion_matrix, fbeta_score

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

# Import centralized utilities
sys.path.insert(0, str(repo_root / "src"))
from utils.constants import MODEL_PATHS, OUTPUT_DIRS, SUMMARY_FILES, MERGE_KEYS
from utils.categorization import get_ground_truth_label, clean_mod_type, categorize_modification

# Initialize logging
setup_logging(__file__)


def run_binary_classification_evaluation(df_all, model_name, fixed_threshold, target_metric):
    """Evaluate pairwise similarity using the predefined distance threshold AND metric."""
    df_eval = df_all.copy()
    
    # Ground Truth (0 = Negative Pairs, 1 = Plagiarism/Modified)
    df_eval['y_true'] = df_eval['final_mod_type'].apply(get_ground_truth_label)
    
    # Clean the 'final_mod_type' from the 'Negative_' prefix to categorize correctly
    df_eval['clean_mod_type'] = df_eval['final_mod_type'].apply(clean_mod_type)
    
    # Apply Granular Categorization
    df_eval['category_grouped'] = df_eval['clean_mod_type'].apply(categorize_modification)
    
    # Override category for Negatives to group them together in the final printout
    df_eval.loc[df_eval['y_true'] == 0, 'category_grouped'] = 'Negative Pairs'

    # Validate that the winning metric exists in this dataset
    if target_metric not in df_eval.columns:
        raise ValueError(f"Winning metric '{target_metric}' not found in the {model_name} dataset.")
        
    # Apply the fixed threshold (Lower distance = more similar -> Predict 1)
    df_eval['y_pred'] = (df_eval[target_metric] <= fixed_threshold).astype(int)
    
    metrics_list = []
    
    # Analysis per Category (Positives Only)
    positive_categories = df_eval[df_eval['y_true'] == 1]['category_grouped'].unique()
    
    for cat in sorted(positive_categories):
        mask = (df_eval['category_grouped'] == cat) | (df_eval['y_true'] == 0)
        df_sub = df_eval[mask]
        
        tn, fp, fn, tp = confusion_matrix(df_sub['y_true'], df_sub['y_pred'], labels=[0, 1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate F0.5 (beta=0.5) to emphasize Precision
        f0_5 = fbeta_score(df_sub['y_true'], df_sub['y_pred'], beta=0.5, zero_division=0)
        
        metrics_list.append({
            'Category': cat,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'F0.5-Score': f0_5,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Total_Queries': tp + fn
        })
        
    # DETAILED ANALYSIS (Metrics per Mod-Type)
    detailed_metrics = []
    unique_mods = df_eval[df_eval['y_true'] == 1]['clean_mod_type'].unique()
    
    for mod in sorted(unique_mods):
        mask = (df_eval['clean_mod_type'] == mod) | (df_eval['y_true'] == 0)
        df_sub = df_eval[mask]
        
        if len(df_sub['y_true'].unique()) < 2:
            continue
            
        tn_s, fp_s, fn_s, tp_s = confusion_matrix(df_sub['y_true'], df_sub['y_pred'], labels=[0, 1]).ravel()
        
        prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
        rec_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        f1_s = 2 * (prec_s * rec_s) / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0.0
        f0_5_s = fbeta_score(df_sub['y_true'], df_sub['y_pred'], beta=0.5, zero_division=0)
        
        detailed_metrics.append({
            'Modification_Type': mod,
            'Precision': prec_s,
            'Recall': rec_s,
            'F1-Score': f1_s,
            'F0.5-Score': f0_5_s,
            'TP': tp_s, 'FP': fp_s, 'FN': fn_s, 'TN': tn_s
        })
        
    df_detailed = pd.DataFrame(detailed_metrics)
    
    # Save detailed metrics to CSV
    detailed_path = f"{OUTPUT_DIRS['binary_classification']}/{model_name.lower()}_detailed_metrics.csv"
    df_detailed.to_csv(detailed_path, index=False)

    # Final Overall Metrics (All Positives vs All Negatives)
    tn, fp, fn, tp = confusion_matrix(df_eval['y_true'], df_eval['y_pred'], labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f0_5 = fbeta_score(df_eval['y_true'], df_eval['y_pred'], beta=0.5, zero_division=0)
    
    metrics_list.append({
        'Category': 'OVERALL',
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'F0.5-Score': f0_5,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Total_Queries': tp + fn
    })
    
    df_metrics = pd.DataFrame(metrics_list)
    
    # Print results in a formatted table
    print(f"\n{'=' * 115}")
    print(f" BINARY CLASSIFICATION PERFORMANCE: {model_name.upper()}")
    print(f" Metric: {target_metric} | Threshold: <= {fixed_threshold:.4f}")
    print(f"{'=' * 115}")
    print(f"{'Modification Category':<30} | {'Precision':>9} | {'Recall':>8} | {'F1-Score':>8} | {'F0.5-Score':>9} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>6}")
    print(f"{'-' * 115}")

    for _, row in df_metrics.iterrows():
        is_overall = row['Category'] == 'OVERALL'
        prefix = "► " if is_overall else "  "
        print(f"{prefix}{row['Category']:<28} | {row['Precision']:>8.1%} | {row['Recall']:>7.1%} | {row['F1-Score']:>7.1%} | {row['F0.5-Score']:>8.1%} | {row['TP']:>5} | {row['FP']:>5} | {row['FN']:>5} | {row['TN']:>6}")

    print(f"{'=' * 115}\n")
    
    # Save results to CSV
    os.makedirs(OUTPUT_DIRS['binary_classification'], exist_ok=True)
    model_path = os.path.join(OUTPUT_DIRS['binary_classification'], f'{model_name.lower()}_binary_metrics.csv')
    df_metrics.to_csv(model_path, index=False)
    
    summary_path = SUMMARY_FILES['binary_summary']
    df_summary = df_metrics.copy()
    df_summary.insert(0, 'Model', model_name)
    df_summary.insert(1, 'Metric', target_metric)
    df_summary['Applied_Threshold'] = fixed_threshold
    
    df_summary.to_csv(summary_path, index=False, mode='a', header=not os.path.exists(summary_path))

    # False Positive Tier Breakdown 
    # Check if 'negative_tier' column exists (from metrics.py)
    if 'negative_tier' in df_eval.columns:
        false_positives_df = df_eval[(df_eval['y_true'] == 0) & (df_eval['y_pred'] == 1)]
        if not false_positives_df.empty:
            # Group by broad category and tier
            fp_breakdown = false_positives_df.groupby(['category_grouped', 'negative_tier']).size().reset_index(name='FP_Count')
            fp_breakdown_path = os.path.join(OUTPUT_DIRS['binary_classification'], f"{model_name.lower()}_{target_metric}_fp_tier_breakdown.csv")
            fp_breakdown.to_csv(fp_breakdown_path, index=False)
            print(f"False Positive Tier Breakdown saved to: {fp_breakdown_path}")
        else:
            print("Error: No False Positives found – perfect separation!")
    else:
        print("Error: 'negative_tier' column not present; skipping FP tier analysis.")
    # =============================================================

    return df_metrics


def main():
    print("=" * 80)
    print("STARTING BINARY CLASSIFICATION WITH OPTIMAL THRESHOLDS")
    print("=" * 80)
    
    # Remove old summary if exists
    summary_path = SUMMARY_FILES['binary_summary']
    if os.path.exists(summary_path): 
        os.remove(summary_path)

    # Load Thresholds AND Metrics
    threshold_file = SUMMARY_FILES['threshold_analysis']
    thresholds_dict = {}
    
    if os.path.exists(threshold_file):
        df_thresh = pd.read_csv(threshold_file)
        print(f"Loaded {len(df_thresh)} optimal configurations from {threshold_file}:")
        for _, row in df_thresh.iterrows():
            model_key = str(row['model']).upper()
            opt_metric = str(row['metric'])
            opt_thresh = float(row['optimal_threshold'])
            thresholds_dict[model_key] = {'metric': opt_metric, 'threshold': opt_thresh}
            print(f"  -> {model_key}: {opt_metric} (Threshold: {opt_thresh:.4f})")
    else:
        print(f"\n[ERROR] Required threshold file '{threshold_file}' not found. Run optimal_threshold.py first.")
        return

    # Load distance CSVs for each model and run evaluation
    processed_any = False
    for model, csv_path in MODEL_PATHS.items():
        if os.path.exists(csv_path):
            if model in thresholds_dict:
                fixed_thresh = thresholds_dict[model]['threshold']
                target_metric = thresholds_dict[model]['metric']
                
                df_all = pd.read_csv(csv_path)
                try:
                    run_binary_classification_evaluation(df_all, model, fixed_thresh, target_metric)
                    processed_any = True
                except ValueError as e:
                    print(f"\n[ERROR] Skipping {model}: {e}")
            else:
                print(f"Warning: No threshold mapped for {model} in {threshold_file}. Skipping.")
        else:
            print(f"Warning: Distance file not found -> {csv_path}")
            
    if processed_any:
        print(f"\n[SUCCESS] Binary classification analysis saved to: {OUTPUT_DIRS['binary_classification']}/")
    else:
        print("\n[FAILED] No distance CSV files found to process.")

if __name__ == "__main__":
    main()