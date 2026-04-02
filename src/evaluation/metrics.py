import argparse
import os
import sys
import ast
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine
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

# Initialize logging for this script (logs/metrics.txt)
setup_logging(__file__)

def compute_distances(parquet_path, smp_metadata_path, output_csv_path):
    """Compute distances using metadata mapping for human SMP and AI/DSP modifications."""

    # Load data
    df = pd.read_parquet(parquet_path)
    df = df.drop_duplicates(subset=['filename'], keep='last').reset_index(drop=True)
    df_meta = pd.read_csv(smp_metadata_path)

    # Build human mapping from metadata
    mapping_records = []
    for _, row in df_meta.iterrows():
        pair_id = int(row['pair_number'])
        relation = str(row['relation'])

        ori_times = ast.literal_eval(row['ori_times']) if pd.notnull(row['ori_times']) else []
        comp_times = ast.literal_eval(row['comp_times']) if pd.notnull(row['comp_times']) else []

        # Nested loop to create a record for each combination of ori_time and comp_time
        for o_time in ori_times:
            for c_time in comp_times:
                mapping_records.append({
                    'pair_id': pair_id,
                    'ori_time': int(o_time),
                    'comp_time': int(c_time),
                    'relation': relation,
                })

    df_human_mapping = pd.DataFrame(mapping_records)

    # Parse filename schema
    def parse_filename(filename):
        try:
            clean = filename.replace('.wav', '')
            parts = clean.split('_')

            if len(parts) < 4:
                raise ValueError(f"Unexpected filename format: {filename}")

            pair_id = int(parts[1])
            ori_comp = parts[2]
            time = int(parts[3].replace('s', ''))

            if len(parts) >= 5:
                mod_type = "_".join(parts[4:])
            else:
                mod_type = 'none'

            return pd.Series([pair_id, ori_comp, time, mod_type])
        except Exception as e:
            print(f"Warning: Could not parse filename '{filename}': {e}")
            return pd.Series([None, None, None, None])

    # Apply parse_filename to extract metadata columns correctly
    parsed_meta = df['filename'].apply(parse_filename)
    parsed_meta.columns = ['pair_id', 'ori_comp', 'time', 'mod_type']
    df = pd.concat([df, parsed_meta], axis=1)

    # Split data sets
    df_pure_ori = df[(df['ori_comp'] == 'ori') & (df['mod_type'] == 'none')].copy()
    df_smp_comp = df[(df['ori_comp'] == 'comp') & (df['mod_type'] == 'none')].copy()
    df_ai_mod = df[df['mod_type'] != 'none'].copy()
    
    # All base segments (both ori and comp) without modifications for AI/DSP comparison
    df_all_base = df[df['mod_type'] == 'none'].copy()

    # Merge 1 - Human Plagiarism via mapping
    df_human = pd.merge(
        df_human_mapping,
        df_pure_ori,
        left_on=['pair_id', 'ori_time'],
        right_on=['pair_id', 'time'],
        how='inner',
    )

    df_human = pd.merge(
        df_human,
        df_smp_comp,
        left_on=['pair_id', 'comp_time'],
        right_on=['pair_id', 'time'],
        how='inner',
        suffixes=('_ori', '_mod'),
    )

    df_human['final_mod_type'] = 'SMP_' + df_human['relation'].astype(str)
    
    # Keep original time for human pairs
    df_human['time'] = df_human['time_ori']

    df_human = df_human[[
        'pair_id', 'time', 'final_mod_type',
        'filename_mod', 'filename_ori',
        'embedding_mod', 'embedding_ori',
    ]]

    # Merge 2 - AI/DSP modifications
    df_ai = pd.merge(
        df_ai_mod,
        df_all_base,
        on=['pair_id', 'time', 'ori_comp'],
        suffixes=('_mod', '_ori'),
        how='inner',
    )
    
    df_ai['final_mod_type'] = df_ai['mod_type_mod']
    df_ai = df_ai[[
        'pair_id', 'time', 'final_mod_type',
        'filename_mod', 'filename_ori',
        'embedding_mod', 'embedding_ori',
    ]]

    # Merge 3 - Hard Negative Pairs per modification type (modality-matched)
    df_positives = pd.concat([df_human, df_ai], axis=0, ignore_index=True, sort=False)

    hard_negatives = []
    for mod_type, df_mod in df_positives.groupby('final_mod_type'):
        df_mod = df_mod.reset_index(drop=True)
        if len(df_mod) < 2:
            continue

        df_mod_shifted = df_mod.shift(-1)
        df_mod_shifted.loc[len(df_mod) - 1] = df_mod.loc[0]

        df_hard = pd.DataFrame({
            'pair_id': df_mod['pair_id'],
            'time': df_mod['time'],
            'final_mod_type': 'Negative_' + str(mod_type),
            'filename_mod': df_mod_shifted['filename_mod'],
            'filename_ori': df_mod['filename_ori'],
            'embedding_mod': df_mod_shifted['embedding_mod'],
            'embedding_ori': df_mod['embedding_ori'],
        })

        hard_negatives.append(df_hard)

    df_hard_negatives = pd.concat(hard_negatives, axis=0, ignore_index=True, sort=False) if hard_negatives else pd.DataFrame(columns=df_positives.columns)

    # Combine positives with new hard negatives baseline
    df_final = pd.concat([df_positives, df_hard_negatives], axis=0, ignore_index=True, sort=False)



    results = []
    for _, row in df_final.iterrows():
        emb_mod_np = np.array(row['embedding_mod'].tolist(), dtype=np.float32)
        emb_ori_np = np.array(row['embedding_ori'].tolist(), dtype=np.float32)

        emb_mod = torch.tensor(emb_mod_np)
        emb_ori = torch.tensor(emb_ori_np)

        if emb_mod.ndim == 1: emb_mod = emb_mod.unsqueeze(0)
        if emb_ori.ndim == 1: emb_ori = emb_ori.unsqueeze(0)

        eps = 1e-6
        ori_norm = emb_ori / (torch.norm(emb_ori, dim=-1, keepdim=True) + eps)
        mod_norm = emb_mod / (torch.norm(emb_mod, dim=-1, keepdim=True) + eps)

        sim_matrix = torch.matmul(ori_norm, mod_norm.T)
        dist_matrix = 1.0 - sim_matrix

        final_dist = dist_matrix.mean().item()

        results.append({
            'pair_id': row['pair_id'],
            'time': row['time'],
            'final_mod_type': row['final_mod_type'],
            'filename_mod': row['filename_mod'],
            'filename_ori': row['filename_ori'],
            'euclidean_distance': torch.dist(emb_ori, emb_mod).item(),
            'cosine_distance': final_dist,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=['final_mod_type', 'pair_id'])

    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(output_csv_path, index=False)

    summary = df_results.groupby('final_mod_type')[['euclidean_distance', 'cosine_distance']].mean()
    print(f"Saved distances to: {output_csv_path}")
    print("\nSummary of average distances by final_mod_type:")
    print(summary)


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
        print("Calculating distances for CLEWS...")
        if os.path.exists(CLEWS_PARQUET):
            compute_distances(CLEWS_PARQUET, SMP_CSV, CLEWS_RESULTS)
        else:
            print(f"Error: File {CLEWS_PARQUET} not found.")

    if args.model in ['wealy', 'all']:
        WEALY_PARQUET = "data/wealy_embeddings.parquet"
        WEALY_RESULTS = "results/distances/wealy_distances.csv"
        print("\nCalculating distances for WEALY...")
        if os.path.exists(WEALY_PARQUET):
            compute_distances(WEALY_PARQUET, SMP_CSV, WEALY_RESULTS)
        else:
            print(f"Error: File {WEALY_PARQUET} not found.")