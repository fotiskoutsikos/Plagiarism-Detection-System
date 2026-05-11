"""
Scores distances for a given model on a fixed  pair list.
All models (CLEWS, WEALY) use the same evaluation_master_pairs.csv.
"""

import argparse
import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import torch

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
from utils.constants import DISTANCE_METRICS, MODEL_PATHS

setup_logging(__file__)

def _embedding_to_numpy(emb) -> np.ndarray:
    try:
        if isinstance(emb, np.ndarray):
            if emb.dtype != object:
                return emb.astype(np.float32, copy=False)
            emb = emb.tolist()
        elif hasattr(emb, "tolist"):
            emb = emb.tolist()

        arr = np.array(emb, dtype=np.float32)
        return arr

    except Exception as e:
        raise ValueError(
            f"Failed to convert embedding to float32 numpy array. "
            f"Type={type(emb)}, sample={str(emb)[:200]}"
        ) from e

# DISTANCE COMPUTATION
def _compute_all_distances(
    emb_ori: torch.Tensor,
    emb_mod: torch.Tensor,
) -> dict:
    """Return the four distance scalars for a single pair of embedding tensors."""
    eps = 1e-6

    if emb_ori.ndim == 1:
        emb_ori = emb_ori.unsqueeze(0)
    if emb_mod.ndim == 1:
        emb_mod = emb_mod.unsqueeze(0)

    min_t   = min(emb_ori.shape[0], emb_mod.shape[0])
    emb_ori = emb_ori[:min_t]
    emb_mod = emb_mod[:min_t]

    # Cosine Distance
    ori_norm    = emb_ori / (torch.norm(emb_ori, dim=-1, keepdim=True) + eps)
    mod_norm    = emb_mod / (torch.norm(emb_mod, dim=-1, keepdim=True) + eps)
    cosine_dist = (1.0 - torch.matmul(ori_norm, mod_norm.T)).mean().item()

    # Euclidean Distance
    euclidean_dist = torch.dist(emb_ori, emb_mod).item()

    # Manhattan Distance
    manhattan_dist = torch.dist(emb_ori, emb_mod, p=1).item()

    # Pearson Correlation Distance
    ori_c      = emb_ori - emb_ori.mean(dim=-1, keepdim=True)
    mod_c      = emb_mod - emb_mod.mean(dim=-1, keepdim=True)
    ori_c_norm = ori_c / (torch.norm(ori_c, dim=-1, keepdim=True) + eps)
    mod_c_norm = mod_c / (torch.norm(mod_c, dim=-1, keepdim=True) + eps)
    pearson_dist = (1.0 - torch.matmul(ori_c_norm, mod_c_norm.T)).mean().item()

    return {
        "euclidean_distance": euclidean_dist,
        "cosine_distance":    cosine_dist,
        "manhattan_distance": manhattan_dist,
        "pearson_distance":   pearson_dist,
    }


# MAIN PIPELINE
def compute_distances(parquet_path: str, pair_list_csv: str, output_csv_path: str, model_name: str = ""):
    """
    Scores distances for the given model on the pair list.
    Produces the exact same summary style as the old metrics.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pair list
    df_pairs = pd.read_csv(pair_list_csv, keep_default_na=False)
    print(f"Loaded pair list: {len(df_pairs)} rows")

    # Load embeddings
    print("Loading embeddings …")
    df_emb = pd.read_parquet(parquet_path)
    emb_map = {
        filename: _embedding_to_numpy(emb)
        for filename, emb in zip(df_emb["filename"], df_emb["embedding"])
    }
    print(f"Loaded {len(emb_map)} embeddings.")

    # Score each pair
    rows = []
    missing_rows = []

    for _, pair in df_pairs.iterrows():
        ori = emb_map.get(pair["filename_ori"])
        mod = emb_map.get(pair["filename_mod"])

        if ori is None or mod is None:
            missing_rows.append({
                "pair_id": pair["pair_id"],
                "time": pair["time"],
                "filename_ori": pair["filename_ori"],
                "filename_mod": pair["filename_mod"],
                "negative_tier": pair["negative_tier"],
            })
            continue

        dists = _compute_all_distances(
            torch.tensor(ori, dtype=torch.float32),
            torch.tensor(mod, dtype=torch.float32)
        )
        rows.append({
            "pair_id":        pair["pair_id"],
            "time":           pair["time"],
            "final_mod_type": pair["final_mod_type"],
            "filename_ori":   pair["filename_ori"],
            "filename_mod":   pair["filename_mod"],
            "negative_tier":  pair["negative_tier"],
            "is_plagiarised": pair["is_plagiarised"],
            **dists,
        })

    if missing_rows:
        missing_df = pd.DataFrame(missing_rows)
        missing_path = output_csv_path.replace(".csv", "_missing_pairs.csv")
        missing_df.to_csv(missing_path, index=False)
        print(
            f"[{model_name}] Found {len(missing_rows)} pairs with missing embeddings. "
            f"Report saved to: {missing_path}"
        )

    df_results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved distances → {output_csv_path}")
    print(f"Total rows: {len(df_results)}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY OF AVERAGE DISTANCES — {model_name.upper() if model_name else 'MODEL'}")
    print(f"{'=' * 60}")

    summary = df_results.groupby(
        ["final_mod_type", "negative_tier"]
    )[DISTANCE_METRICS].mean().round(4)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(summary)

    # Save summary
    summary_path = output_csv_path.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nSaved full summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score distances for a model on the pair list."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["clews", "wealy", "all"],
        default="all",
        help="Model to score (clews, wealy, or all).",
    )
    parser.add_argument(
        "--pair-list",
        type=str,
        default="results/pairs/evaluation_master_pairs.csv",
        help="Path to pair list CSV.",
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

    if not Path(args.pair_list).exists():
        raise FileNotFoundError(
            f"Pair list not found: {args.pair_list}\n"
            f"Run build_pairs.py first."
        )

    for model_key in models_to_run:
        cfg = MODEL_CONFIG[model_key]
        print(f"\n{'=' * 60}")
        print(f"Scoring {model_key.upper()}")
        print(f"{'=' * 60}")
        if Path(cfg["parquet"]).exists():
            compute_distances(
                cfg["parquet"],
                args.pair_list,
                cfg["output"],
                model_name=model_key,
            )
        else:
            print(f"Error: {cfg['parquet']} not found. Skipping.")