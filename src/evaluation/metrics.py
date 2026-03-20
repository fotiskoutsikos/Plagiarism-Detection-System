import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine


def compute_distances(parquet_path, output_csv_path):
    """Compute distances between original and modified audio embeddings."""

    # Load the parquet file into a DataFrame
    df = pd.read_parquet(parquet_path)

    def parse_filename(filename):
        name = filename.replace('.wav', '')
        parts = name.split('_')
        
        # parts = ['pair', '1', 'ori', '0s', 'musicgen'] (generated)
        # parts = ['pair', '1', 'ori', '0s'] (original)
        
        pair_id = int(parts[1])
        ori_comp = parts[2]
        time = int(parts[3].replace('s', ''))
        
        if len(parts) == 5:
            mod_type = parts[4] # 'musicgen', 'audioldm', 'mgeldm
        else:
            mod_type = 'original'
            
        return pd.Series([pair_id, ori_comp, time, mod_type])

    # Parse filename to extract pair_id, ori_comp, time, and mod_type
    df[['pair_id', 'ori_comp', 'time', 'mod_type']] = df['filename'].apply(parse_filename)

    # Split into original and modified DataFrames
    df_ori = df[df['mod_type'] == 'original'].copy()
    df_mod = df[df['mod_type'] != 'original'].copy()

    # Merge on pair_id, ori_comp AND time to create exact positive pairs!
    df_merged = pd.merge(
        df_mod, 
        df_ori, 
        on=['pair_id', 'ori_comp', 'time'], 
        suffixes=('_mod', '_ori')
    )

    # Check if merge resulted in any pairs
    if df_merged.empty:
        print(f"Warning: No matching pairs found in {parquet_path}. Skipping.")
        return
        

 # Compute distances for each pair
    results = []
    for _, row in df_merged.iterrows():
        emb_ori = np.array(row['embedding_ori'])
        emb_mod = np.array(row['embedding_mod'])

        eucl_dist = euclidean(emb_ori, emb_mod)
        cos_dist = cosine(emb_ori, emb_mod)

        results.append({
            'pair_id': row['pair_id'],
            'time': row['time'],
            'mod_type': row['mod_type_mod'],
            'filename_mod': row['filename_mod'],
            'filename_ori': row['filename_ori'],
            'euclidean_distance': eucl_dist,
            'cosine_distance': cos_dist,
        })

    # Convert results to DataFrame and sort
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=['mod_type', 'pair_id'])

    # Create output directory if needed
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save to CSV without index
    df_results.to_csv(output_csv_path, index=False)
    print(f"Distances computed and saved to {output_csv_path}")

    # Print summary statistics
    summary = df_results.groupby('mod_type')[['euclidean_distance', 'cosine_distance']].mean()
    print("\nSummary of average distances by modification type:")
    print(summary)


if __name__ == "__main__":
    # Dummy paths for CLEWS and WEALY
    CLEWS_PARQUET = "../../data/clews_embeddings.parquet"
    CLEWS_RESULTS = "../../data/clews_distances.csv"
    WEALY_PARQUET = "../../data/wealy_embeddings.parquet"
    WEALY_RESULTS = "../../data/wealy_distances.csv"

    # Compute distances for CLEWS
    compute_distances(CLEWS_PARQUET, CLEWS_RESULTS)

    # Compute distances for WEALY
    compute_distances(WEALY_PARQUET, WEALY_RESULTS)