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

# Import centralized utilities
sys.path.insert(0, str(repo_root / "src"))
from utils.dataset_builder import build_positive_pairs
from utils.constants import DISTANCE_METRICS, MODEL_PATHS, SMP_CSV, EMBEDDING_PATHS

# Initialize logging for this script
setup_logging(__file__)


# DISTANCE COMPUTATION
def _compute_all_distances(emb_ori: torch.Tensor, emb_mod: torch.Tensor) -> dict:
    """Return the four distance scalars for a single pair of embedding tensors."""
    eps = 1e-6

    if emb_ori.ndim == 1:
        emb_ori = emb_ori.unsqueeze(0)
    if emb_mod.ndim == 1:
        emb_mod = emb_mod.unsqueeze(0)

    min_t = min(emb_ori.shape[0], emb_mod.shape[0])
    emb_ori = emb_ori[:min_t]
    emb_mod = emb_mod[:min_t]

    # Cosine Distance
    ori_norm = emb_ori / (torch.norm(emb_ori, dim=-1, keepdim=True) + eps)
    mod_norm = emb_mod / (torch.norm(emb_mod, dim=-1, keepdim=True) + eps)
    cosine_dist = (1.0 - torch.matmul(ori_norm, mod_norm.T)).mean().item()

    # Euclidean Distance
    euclidean_dist = torch.dist(emb_ori, emb_mod).item()

    # Manhattan / L1 Distance
    manhattan_dist = torch.dist(emb_ori, emb_mod, p=1).item()

    # Pearson Correlation Distance
    ori_c = emb_ori - emb_ori.mean(dim=-1, keepdim=True)
    mod_c = emb_mod - emb_mod.mean(dim=-1, keepdim=True)
    ori_c_norm = ori_c / (torch.norm(ori_c, dim=-1, keepdim=True) + eps)
    mod_c_norm = mod_c / (torch.norm(mod_c, dim=-1, keepdim=True) + eps)
    pearson_dist = (1.0 - torch.matmul(ori_c_norm, mod_c_norm.T)).mean().item()

    return {
        "euclidean_distance": euclidean_dist,
        "cosine_distance":    cosine_dist,
        "manhattan_distance": manhattan_dist,
        "pearson_distance":   pearson_dist,
    }


# SEARCH POOL
def _build_search_pool(df_positives: pd.DataFrame, device: torch.device):
    """
    Build a normalised embedding matrix and companion metadata arrays from
    all *modified* embeddings in df_positives.

    Returns
    -------
    pool_emb_norm  : torch.Tensor  (N, D)  – L2-normalised, on `device`
    pool_pair_ids  : np.ndarray    (N,)    – pair_id of every candidate
    pool_mod_types : np.ndarray    (N,)    – final_mod_type of every candidate
    pool_raw_embs  : list[np.ndarray]      – raw (unnormalised) embeddings
    """
    eps = 1e-6
    raw_embs  = []
    pair_ids  = []
    mod_types = []

    for _, row in df_positives.iterrows():
        emb = np.array(row["embedding_mod"].tolist(), dtype=np.float32)
        raw_embs.append(emb)
        pair_ids.append(row["pair_id"])
        mod_types.append(row["final_mod_type"])

    pool_pair_ids  = np.array(pair_ids)
    pool_mod_types = np.array(mod_types)

    pool_2d = []
    for emb in raw_embs:
        t = torch.tensor(emb)
        pool_2d.append(t if t.ndim == 1 else t.mean(dim=0))

    pool_matrix    = torch.stack(pool_2d).to(device)
    pool_emb_norm  = F.normalize(pool_matrix, dim=-1, eps=eps)

    return pool_emb_norm, pool_pair_ids, pool_mod_types, raw_embs


# NEGATIVE MINING
def _find_hardest_candidate( query_norm: torch.Tensor, pool_emb_norm: torch.Tensor, allowed_mask:  torch.Tensor) -> int | None:
    """
    Return the pool index of the nearest allowed candidate (max cosine sim).
    Returns None if no eligible candidate exists.
    """
    if not allowed_mask.any():
        return None

    sims = torch.mv(pool_emb_norm, query_norm)
    sims[~allowed_mask] = float("-inf")
    return int(sims.argmax().item())


# MAIN PIPELINE
def compute_distances(parquet_path: str, smp_metadata_path: str, output_csv_path: str):
    """
    Compute pairwise distances for:
      • Positive pairs              (ground-truth plagiarism)
      • Random Negatives            (cyclic-shift within same mod_type)
      • Intra-Category Nearest      (nearest within same mod_type, different pair_id)
      • Global Nearest              (nearest across ALL mod_types, different pair_id)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_positives = build_positive_pairs(parquet_path, smp_metadata_path)
    print(f"\nLoaded {len(df_positives)} positive pairs.")

    print("Building search pool …")
    pool_emb_norm, pool_pair_ids, pool_mod_types, pool_raw_embs = \
        _build_search_pool(df_positives, device)
    print(f"Search pool: {len(pool_pair_ids)} candidates  |  "
          f"embedding dim: {pool_emb_norm.shape[1]}")

    eps = 1e-6

    def _query_norm(raw_emb: np.ndarray) -> torch.Tensor:
        t = torch.tensor(raw_emb, device=device)
        if t.ndim > 1:
            t = t.mean(dim=0)
        return F.normalize(t, dim=-1, eps=eps)

    # Mine negatives 
    print("\nMining Random / Intra-Category Nearest / Global Nearest negatives …")

    random_rows               = []
    intra_category_nearest_rows = []
    global_nearest_rows       = []

    for idx, row in df_positives.iterrows():
        target_pair_id = row["pair_id"]
        target_mod     = row["final_mod_type"]
        emb_ori_raw    = np.array(row["embedding_ori"].tolist(), dtype=np.float32)
        q_norm         = _query_norm(emb_ori_raw)

        diff_pair_mask = (pool_pair_ids != target_pair_id)
        diff_pair_t    = torch.tensor(diff_pair_mask, device=device)

        # Random: deterministic cyclic-shift within same mod_type
        same_mod_mask = (pool_mod_types == target_mod) & diff_pair_mask
        same_mod_idxs = np.where(same_mod_mask)[0]
        random_idx    = None
        if len(same_mod_idxs) > 0:
            pos_in_group = np.searchsorted(same_mod_idxs, idx) % len(same_mod_idxs)
            random_idx   = int(same_mod_idxs[pos_in_group])

        # Intra-Category Nearest
        intra_mask_t = torch.tensor(
            (pool_mod_types == target_mod) & diff_pair_mask, device=device
        )
        intra_idx = _find_hardest_candidate(q_norm, pool_emb_norm, intra_mask_t)

        # Global Nearest
        global_idx = _find_hardest_candidate(q_norm, pool_emb_norm, diff_pair_t)

        def _build_neg_row(pool_idx: int, tier: str) -> dict:
            emb_mod_raw = np.array(
                df_positives.iloc[pool_idx]["embedding_mod"].tolist(), dtype=np.float32
            )
            dists = _compute_all_distances(
                torch.tensor(emb_ori_raw), torch.tensor(emb_mod_raw)
            )
            return {
                "pair_id":        row["pair_id"],
                "time":           row["time"],
                "final_mod_type": f"Negative_{target_mod}",
                "filename_ori":   row["filename_ori"],
                "filename_mod":   df_positives.iloc[pool_idx]["filename_mod"],
                "negative_tier":  tier,
                **dists,
            }

        if random_idx is not None:
            random_rows.append(_build_neg_row(random_idx, "random"))
        if intra_idx is not None:
            intra_category_nearest_rows.append(_build_neg_row(intra_idx, "intra_category_nearest"))
        if global_idx is not None:
            global_nearest_rows.append(_build_neg_row(global_idx, "global_nearest"))

    print(f"  random negatives              : {len(random_rows)}")
    print(f"  intra_category_nearest        : {len(intra_category_nearest_rows)}")
    print(f"  global_nearest negatives      : {len(global_nearest_rows)}")

    # Positive pairs distances 
    print("\nComputing distances for positive pairs …")
    pos_rows = []
    for _, row in df_positives.iterrows():
        emb_ori_raw = np.array(row["embedding_ori"].tolist(), dtype=np.float32)
        emb_mod_raw = np.array(row["embedding_mod"].tolist(), dtype=np.float32)
        dists = _compute_all_distances(
            torch.tensor(emb_ori_raw), torch.tensor(emb_mod_raw)
        )
        pos_rows.append({
            "pair_id":        row["pair_id"],
            "time":           row["time"],
            "final_mod_type": row["final_mod_type"],
            "filename_ori":   row["filename_ori"],
            "filename_mod":   row["filename_mod"],
            "negative_tier":  "N/A",
            **dists,
        })

    # Assemble, deduplicate & save 
    df_results = pd.concat(
        [
            pd.DataFrame(pos_rows),
            pd.DataFrame(random_rows),
            pd.DataFrame(intra_category_nearest_rows),
            pd.DataFrame(global_nearest_rows),
        ],
        ignore_index=True,
    )

    df_results = (
        df_results.drop_duplicates( subset=["filename_ori", "filename_mod", "final_mod_type", "negative_tier"])
        .sort_values(by=["final_mod_type", "negative_tier", "pair_id"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    print(f"\nSaved pairwise distances → {output_csv_path}")
    print(f"Total rows: {len(df_results)}")

    # Summary 
    summary = df_results.groupby(
        ["final_mod_type", "negative_tier"]
    )[DISTANCE_METRICS].mean()

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("\nSummary of average distances by modification type × tier:")
        print(summary)

    summary_csv_path = output_csv_path.replace(".csv", "_summary.csv")
    summary.to_csv(summary_csv_path)
    print(f"Saved full summary → {summary_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Compute distances for CLEWS / WEALY.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["clews", "wealy", "all"],
        default="all",
        help="Choose model (clews, wealy, or all)",
    )
    args = parser.parse_args()

    MODEL_CONFIG = {
        "clews": {
            "parquet": "data/clews_embeddings.parquet",
            "output":  MODEL_PATHS["CLEWS"],
        },
        "wealy": {
            "parquet": "data/wealy_embeddings.parquet",
            "output":  MODEL_PATHS["WEALY"],
        },
    }

    models_to_run = (
        ["clews", "wealy"] if args.model == "all" else [args.model]
    )

    for model_key in models_to_run:
        cfg = MODEL_CONFIG[model_key]
        print(f"\n{'=' * 60}")
        print(f"=== Calculating distances for {model_key.upper()} ===")
        print(f"{'=' * 60}")
        if os.path.exists(cfg["parquet"]):
            compute_distances(cfg["parquet"], SMP_CSV, cfg["output"])
        else:
            print(f"Error: {cfg['parquet']} not found. Skipping.")