"""
Creates a single, deterministic evaluation pair list for all models.

Uses CLEWS embeddings to mine:
- Positive pairs (from SMP)
- Random negatives (cyclic-shift within same mod_type)
- Intra-category nearest negatives (hardest within same mod_type)
- Global nearest negatives (hardest across all mod_types)

Output: data/evaluation_master_pairs.csv
Columns:
    pair_id, time, final_mod_type, filename_ori, filename_mod, negative_tier, is_plagiarised
"""

import argparse
import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

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

sys.path.insert(0, str(repo_root / "src"))
from utils.dataset_builder import build_positive_pairs
from utils.constants import SMP_CSV

setup_logging(__file__)


# SEARCH POOL
def _build_search_pool(df_positives: pd.DataFrame):
    """
    Build a normalised embedding matrix and companion metadata arrays
    from all *modified* embeddings in df_positives.

    The pool is built from embedding_mod of each positive pair.
    The query during mining is embedding_ori (see build_master_pairs).

    Returns
-
    pool_emb_norm  : np.ndarray  (N, D) – L2-normalised
    pool_pair_ids  : np.ndarray  (N,)   – pair_id of every candidate
    pool_mod_types : np.ndarray  (N,)   – final_mod_type of every candidate
    """
    eps = 1e-6
    pair_ids  = []
    mod_types = []
    pool_emb  = []

    for _, row in df_positives.iterrows():
        pair_ids.append(row["pair_id"])
        mod_types.append(row["final_mod_type"])

        emb = np.array(row["embedding_mod"].tolist(), dtype=np.float32)
        t = torch.tensor(emb)
        if t.ndim > 1:
            t = t.mean(dim=0)
        pool_emb.append(F.normalize(t, dim=-1, eps=eps).cpu().numpy())

    return (
        np.stack(pool_emb),
        np.array(pair_ids),
        np.array(mod_types),
    )


# NEGATIVE MINING
def _find_hardest_candidate(
    query_norm:    np.ndarray,
    pool_emb_norm: np.ndarray,
    allowed_mask:  np.ndarray,
) -> int | None:
    """
    Return the pool index of the nearest allowed candidate (max cosine sim).
    Returns None if no eligible candidate exists.
    """
    if not allowed_mask.any():
        return None

    sims = pool_emb_norm @ query_norm
    sims[~allowed_mask] = float("-inf")
    return int(sims.argmax())


def _build_neg_row(
    row:          pd.Series,
    pool_idx:     int,
    tier:         str,
    df_positives: pd.DataFrame,
) -> dict:
    """Build one negative pair row — defined outside the loop to avoid closure bugs."""
    return {
        "pair_id":        row["pair_id"],
        "time":           row["time"],
        "final_mod_type": f"Negative_{row['final_mod_type']}",
        "filename_ori":   row["filename_ori"],
        "filename_mod":   df_positives.iloc[pool_idx]["filename_mod"],
        "negative_tier":  tier,
        "is_plagiarised": 0,
    }


# MAIN PIPELINE
def build_master_pairs(
    parquet_path:    str,
    smp_metadata_path: str,
    output_csv_path: str,
):
    """
    Build evaluation pair list using CLEWS embeddings for mining.
    """
    print(f"Loading positive pairs from {parquet_path} …")
    df_positives = build_positive_pairs(parquet_path, smp_metadata_path)
    # Ensure clean sequential index for cyclic shift correctness
    df_positives = df_positives.reset_index(drop=True)
    print(f"Loaded {len(df_positives)} positive pairs.")

    print("Building search pool …")
    pool_emb_norm, pool_pair_ids, pool_mod_types = _build_search_pool(df_positives)
    print(f"Search pool: {len(pool_pair_ids)} candidates  |  dim: {pool_emb_norm.shape[1]}")

    print("\nMining Random / Intra-Category Nearest / Global Nearest negatives …")
    eps = 1e-6
    random_rows    = []
    intra_rows     = []
    global_rows    = []

    for counter, (_, row) in enumerate(df_positives.iterrows()):
        target_pair_id = row["pair_id"]
        target_mod     = row["final_mod_type"]

        # Query = embedding_ori
        emb_ori = np.array(row["embedding_ori"].tolist(), dtype=np.float32)
        t = torch.tensor(emb_ori)
        if t.ndim > 1:
            t = t.mean(dim=0)
        query_norm = F.normalize(t, dim=-1, eps=eps).cpu().numpy()

        diff_pair_mask = (pool_pair_ids != target_pair_id)

        # Random: deterministic cyclic-shift within same mod_type
        same_mod_mask = (pool_mod_types == target_mod) & diff_pair_mask
        same_mod_idxs = np.where(same_mod_mask)[0]
        random_idx    = None
        if len(same_mod_idxs) > 0:
            # Use sequential counter (not pandas idx) for reproducibility
            pos_in_group = np.searchsorted(same_mod_idxs, counter) % len(same_mod_idxs)
            random_idx   = int(same_mod_idxs[pos_in_group])

        # Intra-Category Nearest
        intra_mask = (pool_mod_types == target_mod) & diff_pair_mask
        intra_idx  = _find_hardest_candidate(query_norm, pool_emb_norm, intra_mask)

        # Global Nearest
        global_idx = _find_hardest_candidate(query_norm, pool_emb_norm, diff_pair_mask)

        if random_idx is not None:
            random_rows.append(_build_neg_row(row, random_idx, "random", df_positives))
        if intra_idx is not None:
            intra_rows.append(_build_neg_row(row, intra_idx, "intra_category_nearest", df_positives))
        if global_idx is not None:
            global_rows.append(_build_neg_row(row, global_idx, "global_nearest", df_positives))

    print(f"  random negatives         : {len(random_rows)}")
    print(f"  intra_category_nearest   : {len(intra_rows)}")
    print(f"  global_nearest           : {len(global_rows)}")

    # Positive pairs
    print("\nBuilding positive pair rows …")
    pos_rows = [
        {
            "pair_id":        row["pair_id"],
            "time":           row["time"],
            "final_mod_type": row["final_mod_type"],
            "filename_ori":   row["filename_ori"],
            "filename_mod":   row["filename_mod"],
            "negative_tier":  "N/A",
            "is_plagiarised": 1,
        }
        for _, row in df_positives.iterrows()
    ]

    # Assemble
    df_results = pd.concat(
        [
            pd.DataFrame(pos_rows),
            pd.DataFrame(random_rows),
            pd.DataFrame(intra_rows),
            pd.DataFrame(global_rows),
        ],
        ignore_index=True,
    )

    df_results = (
        df_results
        .drop_duplicates(subset=["filename_ori", "filename_mod", "final_mod_type", "negative_tier"])
        .sort_values(by=["final_mod_type", "negative_tier", "pair_id"])
        .reset_index(drop=True)
    )

    # Save
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    print(f"\nSaved pair list → {output_csv_path}")
    print(f"Total rows: {len(df_results)}")

    # Summary
    total_pos = int((df_results["is_plagiarised"] == 1).sum())
    total_neg = int((df_results["is_plagiarised"] == 0).sum())
    prevalence = total_pos / len(df_results) * 100

    print(f"\n{'=' * 50}")
    print(f"PAIR LIST SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total pairs      : {len(df_results)}")
    print(f"  Positives        : {total_pos}  ({prevalence:.1f}%)")
    print(f"  Negatives        : {total_neg}  ({100 - prevalence:.1f}%)")
    print(f"\n  Breakdown by tier:")

    tier_summary = (
        df_results.groupby(["negative_tier", "is_plagiarised"])
        .size()
        .reset_index(name="count")
    )
    print(tier_summary.to_string(index=False))

    print(f"\n  Breakdown by mod_type:")
    mod_summary = (
        df_results.groupby(["final_mod_type", "negative_tier"])
        .size()
        .reset_index(name="count")
    )
    with pd.option_context("display.max_rows", None):
        print(mod_summary.to_string(index=False))

    summary_path = output_csv_path.replace(".csv", "_summary.csv")
    mod_summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build evaluation pair list.")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/clews_embeddings.parquet",
        help="Path to CLEWS embeddings parquet.",
    )
    parser.add_argument(
        "--smp",
        type=str,
        default=SMP_CSV,
        help="Path to SMP metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/pairs/evaluation_master_pairs.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("Building Pair List")
    print(f"{'=' * 60}")

    if os.path.exists(args.parquet):
        build_master_pairs(args.parquet, args.smp, args.output)
    else:
        print(f"Error: {args.parquet} not found.")