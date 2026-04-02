"""
Final comparison: CLEWS vs WEALY vs Fusion.
Aggregates results from Binary Classification and Attribution (when ready)
to generate a comprehensive results table.
"""

import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

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

# Initialize logging for this script
setup_logging(__file__)


# Set to True when the attribution script has been successfully run 
# and you want to merge Binary Classification with Retrieval Metrics.
ENABLE_ATTRIBUTION_MERGE = False 

def load_summaries():
    """Load the pre-computed summary files from previous steps."""
    data = {}
    
    # 1. Load Binary Classification Summary
    binary_path = "results/binary_classification/binary_summary.csv"
    if os.path.exists(binary_path):
        data['binary'] = pd.read_csv(binary_path)
    else:
        print(f"[ERROR] Could not find {binary_path}. Run binary_classification.py first.")
        
    # 2. Load Attribution Summary (If enabled and exists)
    attr_path = "results/attribution/attribution_summary.csv"
    if ENABLE_ATTRIBUTION_MERGE:
        if os.path.exists(attr_path):
            data['attribution'] = pd.read_csv(attr_path)
        else:
            print(f"[WARNING] Attribution merge is enabled, but {attr_path} was not found. Skipping attribution.")
            
    return data

def print_binary_comparison(df_binary):
    """Print the overall binary classification results nicely."""
    # Filter to OVERALL category only for the top-level table
    df_overall = df_binary[df_binary['Category'] == 'OVERALL'].copy()
    
    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON (BINARY CLASSIFICATION)")
    print("=" * 90)
    print(f"{'Model':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Threshold':>12} {'ROC-AUC':>10}")
    print("-" * 90)
    
    for _, row in df_overall.iterrows():
        model = str(row['Model']).upper()
        # Fallback names in case old keys exist
        thresh = row.get('Applied_Threshold', row.get('Optimal_Threshold', 0.0))
        roc = row.get('ROC_AUC', 0.0)
        
        print(f"{model:<15} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
              f"{row['F1-Score']:>10.4f} {thresh:>12.4f} {roc:>10.4f}")
    
    print("=" * 90)
    return df_overall

def print_summary_statistics(df_overall):
    """Calculate and print the improvement brought by the Fusion model."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS & FUSION IMPACT")
    print("=" * 70)
    
    if len(df_overall) >= 2:
        df_overall['Model_UPPER'] = df_overall['Model'].str.upper()
        
        base_models = df_overall[df_overall['Model_UPPER'].isin(['CLEWS', 'WEALY'])]
        fusion_model = df_overall[df_overall['Model_UPPER'] == 'FUSION']
        
        if not base_models.empty and not fusion_model.empty:
            best_single = base_models.loc[base_models['F1-Score'].idxmax()]
            fusion = fusion_model.iloc[0]
            
            improvement = fusion['F1-Score'] - best_single['F1-Score']
            improvement_pct = (improvement / best_single['F1-Score']) * 100 if best_single['F1-Score'] > 0 else 0
            
            print(f"\nBest Single Model: {best_single['Model_UPPER']} (F1 = {best_single['F1-Score']:.4f})")
            print(f"Fusion Model:      {fusion['Model_UPPER']} (F1 = {fusion['F1-Score']:.4f})")
            print(f"Absolute Improvement:  +{improvement:.4f}")
            print(f"Relative Improvement:  +{improvement_pct:.2f}%")
            
            if improvement > 0.005:
                print("\n[✓] Fusion provides a significant improvement over the best single model!")
            elif improvement > 0:
                print("\n[~] Fusion provides a marginal improvement.")
            else:
                print("\n[!] Fusion does not improve over the best single model.")
        else:
            print("Not enough models (Need CLEWS/WEALY and FUSION) to compute improvement.")

def main():
    print("=" * 70)
    print("FINAL COMPREHENSIVE AGGREGATION")
    print("=" * 70)
    
    # Load Data
    data = load_summaries()
    if 'binary' not in data:
        return
        
    df_binary = data['binary']
    
    # Print Binary Stats
    df_overall = print_binary_comparison(df_binary)
    print_summary_statistics(df_overall)
    
    # Merge Logic (Attribution)
    os.makedirs('results', exist_ok=True)
    
    if 'attribution' in data:
        df_attr = data['attribution']
        # Standardize 'Model' column naming for safety before merging
        if 'model' in df_attr.columns:
            df_attr = df_attr.rename(columns={'model': 'Model'})
            
        # Convert Model names to uppercase to ensure they match (CLEWS == clews)
        df_binary['Merge_Key'] = df_binary['Model'].str.upper()
        df_attr['Merge_Key'] = df_attr['Model'].str.upper()
        
        # Merge on Model and Category
        merged = pd.merge(df_binary, df_attr, on=['Merge_Key', 'Category'], how='outer', suffixes=('_bin', '_attr'))
        merged = merged.drop(columns=['Merge_Key'])
        
        output_path = "results/final_merged_comparison.csv"
        merged.to_csv(output_path, index=False)
        print(f"\n[SUCCESS] Merged Master CSV (Binary + Attribution) saved to: {output_path}")
        
    else:
        output_path = "results/final_binary_comparison.csv"
        df_binary.to_csv(output_path, index=False)
        print(f"\n[SUCCESS] Final comparison saved to: {output_path}")
        if not ENABLE_ATTRIBUTION_MERGE:
            print("\n(Note: Attribution integration is currently disabled via 'ENABLE_ATTRIBUTION_MERGE' flag.)")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()