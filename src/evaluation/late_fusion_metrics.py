import pandas as pd
import numpy as np
import os

def calculate_late_fusion(clews_csv, wealy_csv, output_csv, alpha=0.5):
    """
    Apply Late Fusion using Convex Combination: alpha * CLEWS + (1 - alpha) * WEALY.
    """
    beta = round(1.0 - alpha, 2)
    # print(f"Starting Fusion with CLEWS weight (alpha) = {alpha}, WEALY weight (beta) = {beta}...")
    
    # Load both CSVs
    df_clews = pd.read_csv(clews_csv)
    df_wealy = pd.read_csv(wealy_csv)

    # Keep only cosine distance and rename
    df_clews = df_clews.rename(columns={'cosine_distance': 'dist_clews'})
    df_wealy = df_wealy.rename(columns={'cosine_distance': 'dist_wealy'})

    # Merge exactly on the same files (Inner Join for fair comparison)
    merge_cols = ['pair_id', 'time', 'final_mod_type', 'filename_mod', 'filename_ori']
    
    df_merged = pd.merge(
        df_clews, 
        df_wealy[merge_cols + ['dist_wealy']], 
        on=merge_cols, 
        how='inner' # Strict comparison
    )

    # Min-Max normalization for consistent scale before fusion
    for col in ['dist_clews', 'dist_wealy']:
        min_val = df_merged[col].min()
        max_val = df_merged[col].max()
        df_merged[col] = (df_merged[col] - min_val) / (max_val - min_val + 1e-8)

    df_merged['cosine_distance'] = (alpha * df_merged['dist_clews']) + (beta * df_merged['dist_wealy'])

    # Save results
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    final_cols = merge_cols + ['dist_clews', 'dist_wealy', 'cosine_distance']
    df_final = df_merged[final_cols]
    
    df_final.to_csv(output_csv, index=False)
    print(f"Success! Results saved to: {output_csv}")

if __name__ == "__main__":
    CLEWS_CSV_PATH = "data/clews_distances.csv"
    WEALY_CSV_PATH = "data/wealy_distances.csv"

    # Linear Grid Search from 0.0 to 1.0 with 0.1 step
    alphas = np.round(np.arange(0.0, 1.1, 0.1), 1)

    print("Starting Grid Search for Convex Late Fusion...")
    for a in alphas:
        b = round(1.0 - a, 1)
        OUT_PATH = f"data/fusion_results/fusion_results_alpha_{a}_beta_{b}.csv"
        calculate_late_fusion(CLEWS_CSV_PATH, WEALY_CSV_PATH, OUT_PATH, alpha=a)

    print(f"Completed! Generated {len(alphas)} fused output files in 'data/fusion_results/'.")