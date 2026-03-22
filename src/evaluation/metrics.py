import os
import ast
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine


def compute_distances(parquet_path, smp_metadata_path, output_csv_path):
    """Compute distances using metadata mapping for human SMP and AI/DSP modifications."""

    # Load data
    df = pd.read_parquet(parquet_path)
    df_meta = pd.read_csv(smp_metadata_path)

    # Build human mapping from metadata
    mapping_records = []
    for _, row in df_meta.iterrows():
        pair_id = int(row['pair_number'])
        relation = str(row['relation'])

        ori_times = ast.literal_eval(row['ori_times']) if pd.notnull(row['ori_times']) else []
        comp_times = ast.literal_eval(row['comp_times']) if pd.notnull(row['comp_times']) else []

        if len(ori_times) != len(comp_times):
            raise ValueError(
                f"Mismatch times lengths for pair {pair_id}: ori_times={ori_times}, comp_times={comp_times}"
            )

        for o_time, c_time in zip(ori_times, comp_times):
            mapping_records.append({
                'pair_id': pair_id,
                'ori_time': int(o_time),
                'comp_time': int(c_time),
                'relation': relation,
            })

    df_human_mapping = pd.DataFrame(mapping_records)

    # Parse filename schema
    def parse_filename(filename):
        clean = filename.replace('.wav', '')
        parts = clean.split('_')

        if len(parts) < 4:
            raise ValueError(f"Unexpected filename format: {filename}")

        pair_id = int(parts[1])
        ori_comp = parts[2]
        time = int(parts[3].replace('s', ''))

        if len(parts) == 5:
            mod_type = parts[4]
        else:
            mod_type = 'none'

        return pd.Series([pair_id, ori_comp, time, mod_type])

    df[['pair_id', 'ori_comp', 'time', 'mod_type']] = df['filename'].apply(parse_filename)

    # Split data sets
    df_pure_ori = df[(df['ori_comp'] == 'ori') & (df['mod_type'] == 'none')].copy()
    df_smp_comp = df[(df['ori_comp'] == 'comp') & (df['mod_type'] == 'none')].copy()
    df_ai_mod = df[df['mod_type'] != 'none'].copy()

    # Merge 1 - Human Plagiarism via mapping
    df_human = pd.merge(
        df_human_mapping,
        df_pure_ori,
        left_on=['pair_id', 'ori_time'],
        right_on=['pair_id', 'time'],
        how='left',
    )

    df_human = pd.merge(
        df_human,
        df_smp_comp,
        left_on=['pair_id', 'comp_time'],
        right_on=['pair_id', 'time'],
        how='left',
        suffixes=('_ori', '_mod'),
    )

    df_human['final_mod_type'] = 'SMP_' + df_human['relation'].astype(str)
    df_human = df_human.rename(columns={'ori_time': 'time'})

    df_human = df_human[[
        'pair_id',
        'time',
        'final_mod_type',
        'filename_mod',
        'filename_ori',
        'embedding_mod',
        'embedding_ori',
    ]]

    # Merge 2 - AI/DSP modifications
    df_ai = pd.merge(
        df_ai_mod,
        df_pure_ori,
        on=['pair_id', 'time', 'ori_comp'],
        suffixes=('_mod', '_ori'),
        how='inner',
    )
    df_ai['final_mod_type'] = df_ai['mod_type_mod']
    df_ai = df_ai[[
        'pair_id',
        'time',
        'final_mod_type',
        'filename_mod',
        'filename_ori',
        'embedding_mod',
        'embedding_ori',
    ]]

    # Combine
    df_final = pd.concat([df_human, df_ai], axis=0, ignore_index=True, sort=False)

    results = []
    for _, row in df_final.iterrows():
        emb_mod = np.asarray(row['embedding_mod'])
        emb_ori = np.asarray(row['embedding_ori'])

        eucl_dist = euclidean(emb_ori, emb_mod)
        cos_dist = cosine(emb_ori, emb_mod)

        results.append({
            'pair_id': row['pair_id'],
            'time': row['time'],
            'final_mod_type': row['final_mod_type'],
            'filename_mod': row['filename_mod'],
            'filename_ori': row['filename_ori'],
            'euclidean_distance': eucl_dist,
            'cosine_distance': cos_dist,
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
    SMP_CSV = "../../data/Final_dataset_pairs.csv"
    CLEWS_PARQUET = "../../data/clews_embeddings.parquet"
    CLEWS_RESULTS = "../../data/clews_distances.csv"

    WEALY_PARQUET = "../../data/wealy_embeddings.parquet"
    WEALY_RESULTS = "../../data/wealy_distances.csv"

    compute_distances(CLEWS_PARQUET, SMP_CSV, CLEWS_RESULTS)
    compute_distances(WEALY_PARQUET, SMP_CSV, WEALY_RESULTS)