import pandas as pd
import numpy as np
import os

def calculate_late_fusion(clews_csv, wealy_csv, output_csv, alpha=0.5):
    """
    Apply Late Fusion to CLEWS and WEALY distances.
    Formula: Distance = alpha * CLEWS_dist + (1 - alpha) * WEALY_dist
    Args:
        clews_csv (str): Path to CLEWS distances CSV.
        wealy_csv (str): Path to WEALY distances CSV.
        output_csv (str): Path to save the fused results CSV.
        alpha (float): Weight for CLEWS distance in fusion (0 <= alpha <= 1).
    """
    print(f"\n--- Starting Fusion with Alpha = {alpha} ---")
    
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
        how='inner' # Inner join ensures only songs that passed both models
    )

    # Fusion
    df_merged['fused_distance'] = (alpha * df_merged['dist_clews']) + ((1.0 - alpha) * df_merged['dist_wealy'])

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
    
    # Loop to test multiple alphas
    alphas_to_test = [0.2, 0.5, 0.8]
    
    for a in alphas_to_test:
        OUT_PATH = f"data/fusion_results_alpha_{a}.csv"
        calculate_late_fusion(CLEWS_CSV_PATH, WEALY_CSV_PATH, OUT_PATH, alpha=a)