"""
Attribution / Retrieval Analysis for Plagiarism Detection.
Evaluates if the model can find the original song from a database 
when queried with a modified (plagiarised/AI) version.
"""

import os
import sys
import ast
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path

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

# Initialize logging
setup_logging(__file__)

# Initialize logging for this script (logs/attribution.txt)
setup_logging(__file__)

def process_and_save_metrics(model_name, df_metrics):
    # Table output formatting
    print(f"\n{'=' * 85}")
    print(f" RETRIEVAL PERFORMANCE: {model_name}")
    print(f"{'=' * 85}")
    print(f"{'Modification Category':<25} | {'Recall@1':>9} | {'Recall@5':>9} | {'MRR':>8} | {'ConfGap':>9} | {'Queries':>8}")
    print(f"{'-' * 85}")

    for _, row in df_metrics.iterrows():
        is_overall = row['Category'] == 'OVERALL'
        prefix = "► " if is_overall else "  "
        print(f"{prefix}{row['Category']:<23} | {row['Recall@1']:>8.1%} | {row['Recall@5']:>8.1%} | {row['MRR']:>8.3f} | {row['Avg_Confidence_Gap']:>8.3f} | {row['Queries']:>8}")

    print(f"{'=' * 85}\n")

    os.makedirs('results', exist_ok=True)
    model_path = f'results/attribution/{model_name.lower()}_retrieval_metrics.csv'
    df_metrics.to_csv(model_path, index=False)

    summary_path = 'results/attribution/attribution_summary.csv'
    df_summary = df_metrics.copy()
    df_summary['model'] = model_name
    df_summary = df_summary[['model', 'Category', 'Recall@1', 'Recall@5', 'Recall@10', 'MRR', 'Avg_Confidence_Gap', 'Std_Confidence_Gap', 'P95_Confidence_Gap', 'Queries']]
    df_summary.to_csv(summary_path, index=False, mode='a', header=not os.path.exists(summary_path))

    print(f"Saved retrieval metrics for {model_name} to {model_path}")

def run_retrieval_evaluation_from_pairs(df_pairs, df_all, model_name):
    """Evaluate retrieval from a precomputed pairwise distance table."""
    # Use a similar mapping strategy as run_retrieval_evaluation to get mod_type grouping
    df_pos = df_pairs[~df_pairs['final_mod_type'].str.startswith('Negative')].copy()
    df_pos['category_raw'] = df_pos['final_mod_type']
    df_pos['category_grouped'] = df_pos['category_raw']

    smp_conditions = df_pos['category_grouped'].isin(['SMP_plag', 'SMP_plag_doubt', 'SMP_remake'])
    df_pos.loc[smp_conditions, 'category_grouped'] = 'Human Plagiarism (SMP)'
    df_pos.loc[df_pos['category_grouped'].str.contains('MusicGen|AI', case=False, na=False), 'category_grouped'] = 'AI Plagiarism (MusicGen)'

    # Hash Map
    candidates_dict = dict(tuple(df_all.groupby('filename_mod')))

    # Determine if the metric is a distance (lower is better) or similarity/confidence (higher is better)
    is_distance = True
    if 'similarity' in 'cosine_distance' or 'confidence' in 'cosine_distance':
        is_distance = False

    query_results = []
    for _, row in df_pos.iterrows():
        q_name = row['filename_mod']
        t_name = row['filename_ori']

        if q_name not in candidates_dict:
            query_results.append({
                'category_grouped': row['category_grouped'],
                'category_raw': row['category_raw'],
                'rank': 999999, 
                'confidence_gap': 0.0,
                'pool_size': 0
            })
            continue

        candidates = candidates_dict[q_name]
        pool_size = len(candidates)

        # Sort by metric
        candidates = candidates.sort_values('cosine_distance', ascending=is_distance)
        ranks = candidates.reset_index(drop=True)

        true_rows = ranks[ranks['filename_ori'] == t_name]
        
        if true_rows.empty:
            query_results.append({
                'category_grouped': row['category_grouped'],
                'category_raw': row['category_raw'],
                'rank': pool_size + 1,
                'confidence_gap': 0.0,
                'pool_size': pool_size
            })
            continue

        # Rank is 1-based index of the last true match (worst case if multiple matches exist)
        rank = int(true_rows.index[-1]) + 1 

        top1_dist = float(ranks.loc[0, 'cosine_distance'])
        top2_dist = float(ranks.loc[1, 'cosine_distance']) if len(ranks) > 1 else top1_dist
        
        confidence_gap = abs(top2_dist - top1_dist)

        query_results.append({
            'category_grouped': row['category_grouped'],
            'category_raw': row['category_raw'],
            'rank': rank,
            'confidence_gap': confidence_gap,
            'pool_size': pool_size
        })

    if len(query_results) == 0:
        print(f"No queries evaluated for {model_name} (fused).")
        return None

    df_results = pd.DataFrame(query_results)
    metrics_list = []

    for cat_group in sorted(df_results['category_grouped'].dropna().unique()):
        cat_rows = df_results[df_results['category_grouped'] == cat_group]
        cat_ranks = np.asarray(cat_rows['rank'].values, dtype=float)
        cat_gap = np.asarray(cat_rows['confidence_gap'].values, dtype=float)
        metrics_list.append({
            'Category': cat_group,
            'Recall@1': float(np.mean(cat_ranks <= 1)),
            'Recall@5': float(np.mean(cat_ranks <= 5)),
            'Recall@10': float(np.mean(cat_ranks <= 10)),
            'MRR': float(np.mean(1.0 / cat_ranks)),
            'Avg_Confidence_Gap': float(np.nanmean(cat_gap)),
            'Std_Confidence_Gap': float(np.nanstd(cat_gap)),
            'P95_Confidence_Gap': float(np.nanpercentile(cat_gap, 95)),
            'Queries': int(len(cat_ranks))
        })

        if cat_group == 'AI Plagiarism (MusicGen)':
            for raw_cat in sorted(cat_rows['category_raw'].dropna().unique()):
                raw_rows = cat_rows[cat_rows['category_raw'] == raw_cat]
                raw_ranks = np.asarray(raw_rows['rank'].values, dtype=float)
                raw_gap = np.asarray(raw_rows['confidence_gap'].values, dtype=float)
                metrics_list.append({
                    'Category': raw_cat,
                    'Recall@1': float(np.mean(raw_ranks <= 1)),
                    'Recall@5': float(np.mean(raw_ranks <= 5)),
                    'Recall@10': float(np.mean(raw_ranks <= 10)),
                    'MRR': float(np.mean(1.0 / raw_ranks)),
                    'Avg_Confidence_Gap': float(np.nanmean(raw_gap)),
                    'Std_Confidence_Gap': float(np.nanstd(raw_gap)),
                    'P95_Confidence_Gap': float(np.nanpercentile(raw_gap, 95)),
                    'Queries': int(len(raw_ranks))
                })

    all_ranks = np.asarray(df_results['rank'].values, dtype=float)
    all_gaps = np.asarray(df_results['confidence_gap'].values, dtype=float)
    metrics_list.append({
        'Category': 'OVERALL',
        'Recall@1': float(np.mean(all_ranks <= 1)),
        'Recall@5': float(np.mean(all_ranks <= 5)),
        'Recall@10': float(np.mean(all_ranks <= 10)),
        'MRR': float(np.mean(1.0 / all_ranks)),
        'Avg_Confidence_Gap': float(np.nanmean(all_gaps)),
        'Std_Confidence_Gap': float(np.nanstd(all_gaps)),
        'P95_Confidence_Gap': float(np.nanpercentile(all_gaps, 95)),
        'Queries': int(len(all_ranks))
    })

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics.sort_values(by='Category', key=lambda col: col == 'OVERALL')

    process_and_save_metrics(model_name, df_metrics)

    return df_metrics


def main():
    print("=" * 80)
    print("STARTING ATTRIBUTION ANALYSIS (PAIRWISE ONLY)")
    print("=" * 80)
    
    # Remove old summary if exists to avoid appending to stale data
    if os.path.exists('results/attribution/attribution_summary.csv'): 
        os.remove('results/attribution/attribution_summary.csv')

    os.makedirs('results/attribution', exist_ok=True)

    # Load distances
    FILES = {
        "CLEWS": "results/distances/clews_distances.csv",
        "WEALY": "results/distances/wealy_distances.csv",
        "Fusion": "results/distances/optimal_fused_distances.csv"
    }

    df_base = pd.read_csv("results/distances/clews_distances.csv") if os.path.exists("results/distances/clews_distances.csv") else None

    if df_base is not None:
        for model, csv_path in FILES.items():
            if os.path.exists(csv_path):
                df_all = pd.read_csv(csv_path)
                run_retrieval_evaluation_from_pairs(df_base, df_all, model)
    
    print("\n[SUCCESS] Attribution analysis saved: results/attribution/attribution_summary.csv")

if __name__ == "__main__":
    main()