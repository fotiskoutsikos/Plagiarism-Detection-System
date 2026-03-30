import argparse
import os
import ast
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import torch

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

    # Merge 3 - Negative Pairs (Baseline)
    df_negative = df_pure_ori.copy().reset_index(drop=True)

    # Find all pair_ids
    unique_pairs = sorted(df_negative['pair_id'].unique())

    # Mapping: pair_id -> next pair_id (circular shift)
    target_pairs = unique_pairs[1:] + [unique_pairs[0]]
    pair_shift_map = dict(zip(unique_pairs, target_pairs))

    # Get the first segment of each target pair_id to use as baseline comparison
    df_first_segments = df_negative.groupby('pair_id').first().reset_index()

    # Map each negative pair to its target pair_id
    df_negative['target_pair_id'] = df_negative['pair_id'].map(pair_shift_map)

    # Merge original embeddings
    df_baseline_merged = pd.merge(
        df_negative,
        df_first_segments,
        left_on='target_pair_id',
        right_on='pair_id',
        suffixes=('_ori', '_mod') # ori: the original segment, mod: the baseline segment from another pair
    )

    df_baseline = pd.DataFrame({
        'pair_id': df_baseline_merged['pair_id_ori'],
        'time': df_baseline_merged['time_ori'],
        'final_mod_type': 'Negative_Baseline',
        'filename_mod': df_baseline_merged['filename_mod'],  # The baseline segment from another pair
        'filename_ori': df_baseline_merged['filename_ori'],  # Original segment
        'embedding_mod': df_baseline_merged['embedding_mod'],
        'embedding_ori': df_baseline_merged['embedding_ori']
    })

    # Combine
    df_final = pd.concat([df_human, df_ai, df_baseline], axis=0, ignore_index=True, sort=False)



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
        CLEWS_RESULTS = "data/clews_distances.csv"
        print("Calculating distances for CLEWS...")
        if os.path.exists(CLEWS_PARQUET):
            compute_distances(CLEWS_PARQUET, SMP_CSV, CLEWS_RESULTS)
        else:
            print(f"Error: File {CLEWS_PARQUET} not found.")

    if args.model in ['wealy', 'all']:
        WEALY_PARQUET = "data/wealy_embeddings.parquet"
        WEALY_RESULTS = "data/wealy_distances.csv"
        print("\nCalculating distances for WEALY...")
        if os.path.exists(WEALY_PARQUET):
            compute_distances(WEALY_PARQUET, SMP_CSV, WEALY_RESULTS)
        else:
            print(f"Error: File {WEALY_PARQUET} not found.")