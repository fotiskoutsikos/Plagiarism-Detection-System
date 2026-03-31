#!/usr/bin/env python3
"""
Attribution / Retrieval Analysis for Plagiarism Detection.
Evaluates if the model can find the original song from a database 
when queried with a modified (plagiarised/AI) version.
"""

import os
import ast
import pandas as pd
import numpy as np

def clean_embedding(emb):
    """Robust embedding extraction (same as umap_analysis)."""
    if isinstance(emb, str):
        try: emb = ast.literal_eval(emb)
        except: return np.array([], dtype=np.float32)
    def extract_numbers(item):
        if isinstance(item, (list, tuple, np.ndarray)):
            flat = []
            for x in item: flat.extend(extract_numbers(x))
            return flat
        elif item is not None and not pd.isna(item):
            return [float(item)]
        return []
    try: return np.array(extract_numbers(emb), dtype=np.float32)
    except: return np.array([], dtype=np.float32)

def run_retrieval_evaluation(csv_path, parquet_path, model_name):
    print(f"\n[{model_name}] Loading Data for Retrieval Analysis...")
    
    # Ground Truth: Original-Modified pairs with modification types
    df_pairs = pd.read_csv(csv_path)
    
    # Keep `final_mod_type` that are not negatives for retrieval evaluation
    df_pos = df_pairs[~df_pairs['final_mod_type'].str.startswith('Negative')].copy()
    
    # Human Plagiarism Consolidation
    smp_conditions = df_pos['final_mod_type'].isin(['SMP_plag', 'SMP_plag_doubt', 'SMP_remake'])
    df_pos.loc[smp_conditions, 'final_mod_type'] = 'Human Plagiarism (SMP)'
    
    # Load Embeddings
    df_emb = pd.read_parquet(parquet_path)
    df_emb = df_emb.drop_duplicates(subset=['filename'])
    df_emb['embedding'] = df_emb['embedding'].apply(clean_embedding)
    
        # Φιλτράρισμα έγκυρων διαστάσεων
    mode_dim = pd.Series([len(x) for x in df_emb['embedding']]).mode()[0]
    df_emb = df_emb[df_emb['embedding'].apply(len) == mode_dim]
    
    # Create a dictionary for fast lookup
    emb_dict = dict(zip(df_emb['filename'], df_emb['embedding']))
    
    # Create gallery (originals) and queries (modified)
    unique_originals = df_pos['filename_ori'].unique()
    gallery_filenames = [f for f in unique_originals if f in emb_dict]
    gallery_embs = np.array([emb_dict[f] for f in gallery_filenames])
    
    if len(gallery_embs) == 0:
        print(f"Error: No valid original embeddings found for {model_name}.")
        return None
        
    # Normalize gallery embeddings for cosine similarity
    gallery_norms = np.linalg.norm(gallery_embs, axis=1, keepdims=True)
    gallery_embs_norm = gallery_embs / (gallery_norms + 1e-8)
    
    gallery_filenames_np = np.array(gallery_filenames)
    
    # Retrieval Evaluation
    print(f"[{model_name}] Searching database of {len(gallery_filenames)} original songs...")
    query_results = []
    
    for _, row in df_pos.iterrows():
        q_name = row['filename_mod']
        t_name = row['filename_ori']
        mod_type = row['final_mod_type']
        
        # Skip if query or target not in embeddings
        if q_name not in emb_dict or t_name not in gallery_filenames:
            continue
            
        q_emb = emb_dict[q_name]
        q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        
        # Compute cosine similarities
        sims = np.dot(gallery_embs_norm, q_emb_norm)
        
        # Sort gallery by similarity (descending)
        sorted_indices = np.argsort(-sims)
        sorted_filenames = gallery_filenames_np[sorted_indices]
        
        # Find rank of the true original
        true_rank_array = np.where(sorted_filenames == t_name)[0]
        if len(true_rank_array) > 0:
            rank = true_rank_array[0] + 1  
            query_results.append({
                'category': mod_type,
                'rank': rank
            })
            
    # Calculate metrics
    df_results = pd.DataFrame(query_results)
    
    metrics_list = []
    # Calculate metrics for each category
    for cat in df_results['category'].unique():
        cat_ranks = df_results[df_results['category'] == cat]['rank'].values
        metrics_list.append({
            'Category': cat,
            'Recall@1': np.mean(cat_ranks <= 1),
            'Recall@5': np.mean(cat_ranks <= 5),
            'Recall@10': np.mean(cat_ranks <= 10),
            'MRR': np.mean(1.0 / cat_ranks),
            'Queries': len(cat_ranks)
        })
        
    # Overall metrics
    all_ranks = df_results['rank'].values
    metrics_list.append({
        'Category': 'OVERALL',
        'Recall@1': np.mean(all_ranks <= 1),
        'Recall@5': np.mean(all_ranks <= 5),
        'Recall@10': np.mean(all_ranks <= 10),
        'MRR': np.mean(1.0 / all_ranks),
        'Queries': len(all_ranks)
    })
    
    df_metrics = pd.DataFrame(metrics_list)
    # Sort with OVERALL at the end
    df_metrics = df_metrics.sort_values(by='Category', key=lambda col: col == 'OVERALL')
    
    # Print results
    print(f"\n{'=' * 85}")
    print(f" RETRIEVAL PERFORMANCE: {model_name}")
    print(f"{'=' * 85}")
    print(f"{'Modification Category':<25} | {'Recall@1':>9} | {'Recall@5':>9} | {'MRR':>8} | {'Queries':>8}")
    print(f"{'-' * 85}")
    
    for _, row in df_metrics.iterrows():
        is_overall = row['Category'] == 'OVERALL'
        prefix = "► " if is_overall else "  "
        print(f"{prefix}{row['Category']:<23} | {row['Recall@1']:>8.1%} | {row['Recall@5']:>8.1%} | {row['MRR']:>8.3f} | {row['Queries']:>8}")
        if not is_overall and _ == len(df_metrics) - 2:
            print(f"{'-' * 85}")
            
    print(f"{'=' * 85}\n")
    
    os.makedirs("results", exist_ok=True)
    df_metrics.to_csv(f"results/{model_name.lower()}_retrieval_metrics.csv", index=False)
    return df_metrics

def main():
    print("=" * 70)
    print("MUSIC INFORMATION RETRIEVAL (SOURCE IDENTIFICATION) ANALYSIS")
    print("=" * 70)
    
    if os.path.exists("data/clews_distances.csv") and os.path.exists("data/clews_embeddings.parquet"):
        run_retrieval_evaluation("data/clews_distances.csv", "data/clews_embeddings.parquet", "CLEWS")
    else:
        print("Warning: CLEWS data files not found.")
        
    if os.path.exists("data/wealy_distances.csv") and os.path.exists("data/wealy_embeddings.parquet"):
        run_retrieval_evaluation("data/wealy_distances.csv", "data/wealy_embeddings.parquet", "WEALY")
    else:
        print("Warning: WEALY data files not found.")
        
    print("ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()