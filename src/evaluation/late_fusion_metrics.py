import pandas as pd
import numpy as np
import os

def calculate_late_fusion(clews_csv, wealy_csv, output_csv, alpha=0.5, beta=0.5):
    """
    Apply Late Fusion to CLEWS and WEALY distances.
    Formula: Distance = alpha * d_clews + beta * d_wealy
    Args:
        clews_csv (str): Path to CLEWS distances CSV.
        wealy_csv (str): Path to WEALY distances CSV.
        output_csv (str): Path to save the fused results CSV.
        alpha (float): Weight for CLEWS distance in fusion.
        beta (float): Weight for WEALY distance in fusion.
    """
    print(f"\nStarting Fusion with alpha = {alpha}, beta = {beta}...")
    
    # Load both CSVs
    df_clews = pd.read_csv(clews_csv)
    df_wealy = pd.read_csv(wealy_csv)

    # Keep only cosine distance and rename
    df_clews = df_clews.rename(columns={'cosine_distance': 'dist_clews'})
    df_wealy = df_wealy.rename(columns={'cosine_distance': 'dist_wealy'})

    # Safe merge
    # Merge on reference columns
    merge_cols = ['pair_id', 'time', 'final_mod_type', 'filename_mod', 'filename_ori']
    
    # Bring only 'dist_wealy' column to avoid duplication
    df_merged = pd.merge(
        df_clews, 
        df_wealy[merge_cols + ['dist_wealy']], 
        on=merge_cols, 
        how='outer' # Use outer join to avoid selection bias when one model fails
    )

    # Impute missing values from one model with the mean of that model (or by another strategy)
    df_merged['dist_clews'] = df_merged['dist_clews'].fillna(df_merged['dist_clews'].mean())
    df_merged['dist_wealy'] = df_merged['dist_wealy'].fillna(df_merged['dist_wealy'].mean())

    # Min-Max normalization for consistent scale before late fusion
    for col in ['dist_clews', 'dist_wealy']:
        min_val = df_merged[col].min()
        max_val = df_merged[col].max()
        df_merged[col] = (df_merged[col] - min_val) / (max_val - min_val + 1e-8)

    # Fusion with independent weights
    df_merged['fused_distance'] = (alpha * df_merged['dist_clews']) + (beta * df_merged['dist_wealy'])

    # Normalize fused distance to [0,1] for comparability
    min_val = df_merged['fused_distance'].min()
    max_val = df_merged['fused_distance'].max()
    df_merged['fused_distance'] = (df_merged['fused_distance'] - min_val) / (max_val - min_val + 1e-8)

    # Save results
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Keep useful columns
    final_cols = merge_cols + ['dist_clews', 'dist_wealy', 'fused_distance']
    df_final = df_merged[final_cols]
    
    df_final.to_csv(output_csv, index=False)
    print(f"Success! Results saved to: {output_csv}")
    print(f"Total pairs analyzed: {len(df_final)}")


if __name__ == "__main__":
    # Files created by metrics.py
    CLEWS_CSV_PATH = "data/clews_distances.csv"
    WEALY_CSV_PATH = "data/wealy_distances.csv"

    # Grid search for independent alpha/beta weights
    alphas = [0.2, 0.5, 0.8]
    betas = [0.2, 0.5, 0.8]

    for a in alphas:
        for b in betas:
            OUT_PATH = f"data/fusion_results_a{a}_b{b}.csv"
            calculate_late_fusion(CLEWS_CSV_PATH, WEALY_CSV_PATH, OUT_PATH, alpha=a, beta=b)

    print("Grid search completed. Fused outputs are saved under data/fusion_results_a*_b*.csv")