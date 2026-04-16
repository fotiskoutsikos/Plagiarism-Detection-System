import argparse
import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# Resolve repository root and load logging_util without relying on src package path
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
from utils.dataset_builder import build_positive_pairs

# Initialize logging for this script (logs/metrics.txt)
setup_logging(__file__)


def compute_distances(parquet_path, smp_metadata_path, output_csv_path):
    """Compute distances separating plagiarism positive cases and 1-to-K hard negative pairs."""

    # Build positive pairs using centralized utility
    df_positives = build_positive_pairs(parquet_path, smp_metadata_path)

    # 1-to-K Hard Negatives (Same Modification Type)
    print(f"\n--- Generating 1-to-K Hard Negative pairs ({len(df_positives)} Positives) ---")
    K_NEGATIVES = 5  # Αναλογία 1 θετικό προς 5 αρνητικά
    negatives = []
    
    for mod_type, group in df_positives.groupby('final_mod_type'):
        n_samples = len(group)
        if n_samples < 2: 
            continue 
        
        group = group.sort_values(by=['pair_id', 'time']).reset_index(drop=True)
        indices = np.arange(n_samples)
        
        actual_K = min(K_NEGATIVES, n_samples - 1)
        
        for k in range(1, actual_K + 1):
            shifted_indices = (indices + k) % n_samples
            
            group_neg = group.copy()
            group_neg['filename_mod'] = group['filename_mod'].iloc[shifted_indices].values
            group_neg['embedding_mod'] = group['embedding_mod'].iloc[shifted_indices].values
            group_neg['pair_id_mod'] = group['pair_id'].iloc[shifted_indices].values
            
            group_neg = group_neg[group_neg['pair_id'] != group_neg['pair_id_mod']].copy()
            
            group_neg['final_mod_type'] = 'Negative_' + str(mod_type)
            group_neg = group_neg.drop(columns=['pair_id_mod'])
            negatives.append(group_neg)

    df_negatives = pd.concat(negatives, ignore_index=True) if negatives else pd.DataFrame()
    df_all_pairs = pd.concat([df_positives, df_negatives], ignore_index=True)

    df_all_pairs = df_all_pairs.drop_duplicates(subset=['filename_ori', 'filename_mod', 'final_mod_type']).reset_index(drop=True)

    # Compute distances for ALL cases
    print(f"--- Computing Distances Matrix ({len(df_positives)} Positives + {len(df_negatives)} Negatives) ---")
    results = []
    for _, row in df_all_pairs.iterrows():
        emb_mod_np = np.array(row['embedding_mod'].tolist(), dtype=np.float32)
        emb_ori_np = np.array(row['embedding_ori'].tolist(), dtype=np.float32)

        emb_mod = torch.tensor(emb_mod_np)
        emb_ori = torch.tensor(emb_ori_np)

        if emb_mod.ndim == 1: emb_mod = emb_mod.unsqueeze(0)
        if emb_ori.ndim == 1: emb_ori = emb_ori.unsqueeze(0)

        # Min-length truncation for safe execution
        min_t = min(emb_ori.shape[0], emb_mod.shape[0])
        emb_ori = emb_ori[:min_t]
        emb_mod = emb_mod[:min_t]

        eps = 1e-6
        ori_norm = emb_ori / (torch.norm(emb_ori, dim=-1, keepdim=True) + eps)
        mod_norm = emb_mod / (torch.norm(emb_mod, dim=-1, keepdim=True) + eps)

        # -- Cosine Distance --
        sim_matrix = torch.matmul(ori_norm, mod_norm.T)
        dist_matrix = 1.0 - sim_matrix
        final_dist = dist_matrix.mean().item()


        # -- Manhattan / L1 Distance --
        manhattan_dist = torch.dist(emb_ori, emb_mod, p=1).item()

        # -- Pearson Correlation Distance --
        ori_centered = emb_ori - emb_ori.mean(dim=-1, keepdim=True)
        mod_centered = emb_mod - emb_mod.mean(dim=-1, keepdim=True)
        
        ori_c_norm = ori_centered / (torch.norm(ori_centered, dim=-1, keepdim=True) + eps)
        mod_c_norm = mod_centered / (torch.norm(mod_centered, dim=-1, keepdim=True) + eps)
        
        pearson_sim_matrix = torch.matmul(ori_c_norm, mod_c_norm.T)
        pearson_dist = 1.0 - pearson_sim_matrix.mean().item()

        results.append({
            'pair_id': row['pair_id'],
            'time': row['time'],
            'final_mod_type': row['final_mod_type'],
            'filename_mod': row['filename_mod'],
            'filename_ori': row['filename_ori'],
            'euclidean_distance': torch.dist(emb_ori, emb_mod).item(),
            'cosine_distance': final_dist,
            'manhattan_distance': manhattan_dist,
            'pearson_distance': pearson_dist,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=['final_mod_type', 'pair_id'])

    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved Pairwise Distances to: {output_csv_path}")

    # Export full summary without truncation
    summary = df_results.groupby('final_mod_type')[[
        'euclidean_distance', 'cosine_distance', 'manhattan_distance', 'pearson_distance'
    ]].mean()
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("\nSummary of average distances by modification type:")
        print(summary)
        
    summary_csv_path = output_csv_path.replace('.csv', '_summary.csv')
    summary.to_csv(summary_csv_path)
    print(f"Saved full summary to: {summary_csv_path}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Compute distances (Metric Learning) for CLEWS/WEALY.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=['clews', 'wealy', 'all'], 
        default='all',
        help="Choose model (clews, wealy, or all)"
    )
    args = parser.parse_args()

    # Paths relative to project root
    SMP_CSV = "data/Final_dataset_pairs.csv"

    if args.model in ['clews', 'all']:
        CLEWS_PARQUET = "data/clews_embeddings.parquet"
        CLEWS_RESULTS = "results/distances/clews_distances.csv" 
        
        print("\n=== Calculating distances for CLEWS ===")
        if os.path.exists(CLEWS_PARQUET):
            compute_distances(CLEWS_PARQUET, SMP_CSV, CLEWS_RESULTS)
        else:
            print(f"Error: File {CLEWS_PARQUET} not found.")

    if args.model in ['wealy', 'all']:
        WEALY_PARQUET = "data/wealy_embeddings.parquet"
        WEALY_RESULTS = "results/distances/wealy_distances.csv"
        
        print("\n=== Calculating distances for WEALY ===")
        if os.path.exists(WEALY_PARQUET):
            compute_distances(WEALY_PARQUET, SMP_CSV, WEALY_RESULTS)
        else:
            print(f"Error: File {WEALY_PARQUET} not found.")