"""
Classifier Feature Builder Utility.

Assembles a unified feature matrix for all evaluation pairs (positives + negatives)
by merging:
    - CLEWS distance metrics
    - WEALY distance metrics
    - CLEWS embedding delta summary features (11 features)
    - WEALY embedding delta summary features (11 features)
    - Vocal metadata (3 features)

The delta summary features are computed using the same mathematical logic
as explainability.py, with the stable/volatile dimension reference frame
derived exclusively from positive pairs (no leakage from negatives).

This module does NOT perform any training or evaluation.
It produces a single parquet file consumed by classifier.py.

Usage:
    python -m src.utils.classifier_features

Output:
    results/classification/classifier_features.parquet
"""

import sys
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

#  Resolve repository root & logging 
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
logging_util.setup_logging(__file__)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(repo_root / "src"))

from utils.constants import (
    MERGE_KEYS, MODEL_PATHS, EMBEDDING_PATHS,
    DISTANCE_METRICS, VOCAL_RATIOS_CSV,
)
from utils.dataset_builder import clean_embedding
from utils.vocal_metadata import attach_vocal_metadata

# Output path 
OUTPUT_DIR  = Path("results/classification")
OUTPUT_FILE = OUTPUT_DIR / "classifier_features.parquet"


def _load_csv_for_merge(path: str) -> pd.DataFrame:
    """
    Load a CSV intended for merge operations while preserving literal strings
    like 'N/A' and normalizing merge-key dtypes.
    """
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)

    # Normalize merge key dtypes
    if "pair_id" in df.columns:
        df["pair_id"] = pd.to_numeric(df["pair_id"], errors="coerce").astype("Int64")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")

    for col in ["filename_ori", "filename_mod", "final_mod_type", "negative_tier"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df

# Delta summary features 
def compute_delta_summary_features(
    delta_matrix:  np.ndarray,
    stable_dims:   np.ndarray,
    volatile_dims: np.ndarray,
    global_q75:    float,
) -> pd.DataFrame:
    """
    Compute pair-level summary features from absolute delta vectors.

    This is the same mathematical logic used in explainability.py's
    export_pairwise_delta_features(), extracted as a pure function
    for reuse across positives and negatives.

    Args:
        delta_matrix:  (N, D) array of |emb_mod - emb_ori| per pair.
        stable_dims:   Indices of the most stable dimensions (from positives).
        volatile_dims: Indices of the most volatile dimensions (from positives).
        global_q75:    75th-percentile threshold for "active" dimensions (from positives).

    Returns:
        DataFrame with N rows and 11 feature columns:
            delta_mean, delta_std, delta_median, delta_max,
            delta_l2, delta_p90, delta_p95,
            active_dims_q75_global,
            stable_core_mean_delta, volatile_shell_mean_delta,
            stable_to_volatile_ratio
    """
    delta_mean   = np.mean(delta_matrix, axis=1)
    delta_std    = np.std(delta_matrix, axis=1)
    delta_median = np.median(delta_matrix, axis=1)
    delta_max    = np.max(delta_matrix, axis=1)
    delta_l2     = np.linalg.norm(delta_matrix, axis=1)
    delta_p90    = np.percentile(delta_matrix, 90, axis=1)
    delta_p95    = np.percentile(delta_matrix, 95, axis=1)

    active_dims_q75_global = np.sum(delta_matrix > global_q75, axis=1)

    stable_core_mean_delta    = np.mean(delta_matrix[:, stable_dims], axis=1)
    volatile_shell_mean_delta = np.mean(delta_matrix[:, volatile_dims], axis=1)

    stable_to_volatile_ratio = np.divide(
        stable_core_mean_delta,
        volatile_shell_mean_delta,
        out=np.zeros_like(stable_core_mean_delta),
        where=volatile_shell_mean_delta != 0,
    )

    return pd.DataFrame({
        "delta_mean":                delta_mean,
        "delta_std":                 delta_std,
        "delta_median":              delta_median,
        "delta_max":                 delta_max,
        "delta_l2":                  delta_l2,
        "delta_p90":                 delta_p90,
        "delta_p95":                 delta_p95,
        "active_dims_q75_global":    active_dims_q75_global.astype(int),
        "stable_core_mean_delta":    stable_core_mean_delta,
        "volatile_shell_mean_delta": volatile_shell_mean_delta,
        "stable_to_volatile_ratio":  stable_to_volatile_ratio,
    })


# Helpers 
def _build_embedding_map(parquet_path: str) -> dict:
    """
    Load embeddings parquet and return {filename: 1D float32 array} dict.
    Uses clean_embedding() from dataset_builder for robust parsing.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding parquet not found: {path}")

    df = pd.read_parquet(path)
    emb_map = {}

    for filename, emb in zip(df["filename"], df["embedding"]):
        cleaned = clean_embedding(emb)
        if len(cleaned) > 0:
            emb_map[filename] = cleaned

    print(f"  Loaded {len(emb_map)} embeddings from {path.name}")
    return emb_map


def _compute_delta_matrix_for_pairs(
    df: pd.DataFrame,
    emb_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each pair row, compute |emb_mod - emb_ori|.

    Returns:
        delta_matrix: (N, D) array — absolute deltas for valid pairs.
        valid_mask:    (N_total,) bool array — which rows had both embeddings.
    """
    n_total    = len(df)
    valid_mask = np.zeros(n_total, dtype=bool)
    deltas     = []

    for i, (_, row) in enumerate(df.iterrows()):
        emb_ori = emb_map.get(row["filename_ori"])
        emb_mod = emb_map.get(row["filename_mod"])

        if emb_ori is not None and emb_mod is not None:
            min_dim = min(len(emb_ori), len(emb_mod))
            if min_dim > 0:
                delta = np.abs(emb_mod[:min_dim] - emb_ori[:min_dim])
                deltas.append(delta)
                valid_mask[i] = True

    if not deltas:
        return np.array([]), valid_mask

    # Ensure all deltas have same dimension
    target_dim = max(len(d) for d in deltas)
    uniform    = []
    valid_indices = np.where(valid_mask)[0]
    final_valid   = np.zeros(n_total, dtype=bool)

    for idx, d in zip(valid_indices, deltas):
        if len(d) == target_dim:
            uniform.append(d)
            final_valid[idx] = True

    if not uniform:
        return np.array([]), np.zeros(n_total, dtype=bool)

    return np.stack(uniform), final_valid


def _compute_reference_from_positives(
    delta_matrix: np.ndarray,
    n_core: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the stable/volatile reference frame from positive pairs only.

    Same logic as explainability.py's plot_stable_core():
        - global_q75: 75th percentile of mean delta per dimension
        - stable_dims: n_core dimensions with lowest mean delta
        - volatile_dims: n_core dimensions with highest mean delta

    Args:
        delta_matrix: (N_pos, D) array of positive-pair deltas.
        n_core: Number of stable/volatile dims. Default: min(100, D // 5).

    Returns:
        (global_q75, stable_dims, volatile_dims)
    """
    ndim = delta_matrix.shape[1]

    if n_core is None:
        n_core = min(100, ndim // 5)

    global_mean       = np.mean(delta_matrix, axis=0)
    global_q75        = float(np.percentile(global_mean, 75))
    stability_ranking = np.argsort(global_mean)
    stable_dims       = stability_ranking[:n_core]
    volatile_dims     = stability_ranking[-n_core:]

    return global_q75, stable_dims, volatile_dims


# Main assembly 
def build_classifier_feature_table(
    pair_list_csv:  str = "results/pairs/evaluation_master_pairs.csv",
    clews_dist_csv: str = MODEL_PATHS["CLEWS"],
    wealy_dist_csv: str = MODEL_PATHS["WEALY"],
    clews_parquet:  str = EMBEDDING_PATHS["CLEWS"],
    wealy_parquet:  str = EMBEDDING_PATHS["WEALY"],
    output_path:    str = str(OUTPUT_FILE),
) -> pd.DataFrame:
    """
    Build the full classifier feature table for all pairs.

    Steps:
        1. Load evaluation_master_pairs.csv (pair definitions + labels)
        2. Merge CLEWS distances (4 features)
        3. Merge WEALY distances (4 features)
        4. Load CLEWS embeddings → compute deltas → reference from positives
           → delta summary features for ALL pairs (11 features)
        5. Same for WEALY (11 features)
        6. Attach vocal metadata (3 features)
        7. Save unified feature table

    Returns:
        The assembled DataFrame (also saved to disk).
    """
    print("=" * 60)
    print("BUILDING CLASSIFIER FEATURE TABLE")
    print("=" * 60)

    # 1. Load pair list
    print("\n[1/7] Loading pair list…")
    df = _load_csv_for_merge(pair_list_csv)
    print(f"  Pairs: {len(df)} ({df['is_plagiarised'].astype(int).sum()} positives)")

    # 2. Merge CLEWS distances
    print("\n[2/7] Merging CLEWS distances…")
    df_clews_dist = _load_csv_for_merge(clews_dist_csv)
    clews_dist_cols = {m: f"clews_{m}" for m in DISTANCE_METRICS if m in df_clews_dist.columns}

    df_clews_dist = df_clews_dist[MERGE_KEYS + list(clews_dist_cols.keys())].copy()
    df_clews_dist = df_clews_dist.rename(columns=clews_dist_cols)

    df = df.merge(df_clews_dist, on=MERGE_KEYS, how="left")
    print(f"  Added {len(clews_dist_cols)} CLEWS distance features")

    # 3. Merge WEALY distances
    print("\n[3/7] Merging WEALY distances…")
    df_wealy_dist = _load_csv_for_merge(wealy_dist_csv)
    wealy_dist_cols = {m: f"wealy_{m}" for m in DISTANCE_METRICS if m in df_wealy_dist.columns}

    df_wealy_dist = df_wealy_dist[MERGE_KEYS + list(wealy_dist_cols.keys())].copy()
    df_wealy_dist = df_wealy_dist.rename(columns=wealy_dist_cols)

    df = df.merge(df_wealy_dist, on=MERGE_KEYS, how="left")
    print(f"  Added {len(wealy_dist_cols)} WEALY distance features")

    # 4. CLEWS delta summary features
    print("\n[4/7] Computing CLEWS delta summary features…")
    df = _add_delta_features_for_model(df, clews_parquet, prefix="clews")

    # 5. WEALY delta summary features
    print("\n[5/7] Computing WEALY delta summary features…")
    df = _add_delta_features_for_model(df, wealy_parquet, prefix="wealy")

    # 6. Vocal metadata
    print("\n[6/7] Attaching vocal metadata…")
    if Path(VOCAL_RATIOS_CSV).exists():
        df = attach_vocal_metadata(df)
        print("  Vocal features attached.")
    else:
        logger.warning(
            f"  Vocal metadata not found at {VOCAL_RATIOS_CSV}. "
            f"Skipping vocal features."
        )

    # 7. Save
    print("\n[7/7] Saving feature table…")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    df.to_parquet(output_path, index=False)
    print(f"  Saved → {output_path}")

    # Summary
    feature_cols = [
        c for c in df.columns
        if c.startswith(("clews_", "wealy_"))
        and c not in MERGE_KEYS
        and "filename" not in c
        and "source_key" not in c
    ]
    vocal_cols = [c for c in df.columns if "vocal" in c.lower()]

    print(f"\n{'=' * 60}")
    print(f"FEATURE TABLE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total rows:        {len(df)}")
    print(f"  Positives:         {df['is_plagiarised'].astype(int).sum()}")
    print(f"  Negatives:         {(~df['is_plagiarised'].astype(bool)).sum()}")
    print(f"  Distance features: {sum(1 for c in feature_cols if 'distance' in c)}")
    print(f"  Delta features:    {sum(1 for c in feature_cols if 'delta' in c or 'stable' in c or 'volatile' in c or 'active' in c)}")
    print(f"  Vocal features:    {len(vocal_cols)}")
    print(f"  Total features:    {len(feature_cols) + len(vocal_cols)}")
    print(f"{'=' * 60}")

    return df


def _add_delta_features_for_model(
    df: pd.DataFrame,
    parquet_path: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Compute delta summary features for one embedding model and merge into df.

    Steps:
        1. Load embedding map
        2. Compute delta matrix for ALL pairs
        3. Compute reference (stable/volatile) from POSITIVES ONLY
        4. Compute summary features for ALL pairs using positive reference
        5. Merge with prefix (e.g. clews_delta_mean)

    Args:
        df: The master pair DataFrame.
        parquet_path: Path to the embedding parquet.
        prefix: Column prefix (e.g. "clews" or "wealy").

    Returns:
        df with 11 new prefixed delta feature columns.
    """
    # 1. Load embeddings
    emb_map = _build_embedding_map(parquet_path)

    # 2. Compute deltas for ALL pairs
    delta_matrix_all, valid_mask_all = _compute_delta_matrix_for_pairs(df, emb_map)

    if delta_matrix_all.size == 0:
        logger.warning(f"  [{prefix}] No valid deltas computed. Filling with NaN.")
        delta_cols = [
            "delta_mean", "delta_std", "delta_median", "delta_max",
            "delta_l2", "delta_p90", "delta_p95",
            "active_dims_q75_global",
            "stable_core_mean_delta", "volatile_shell_mean_delta",
            "stable_to_volatile_ratio",
        ]
        for col in delta_cols:
            df[f"{prefix}_{col}"] = np.nan
        return df

    n_valid = int(valid_mask_all.sum())
    n_miss  = int((~valid_mask_all).sum())
    print(f"  [{prefix}] Delta computed: {n_valid} valid, {n_miss} missing embeddings")

    # 3. Reference from positives only
    is_positive = df["is_plagiarised"].astype(int).values
    pos_in_valid = is_positive[valid_mask_all].astype(bool)

    if pos_in_valid.sum() == 0:
        logger.warning(f"  [{prefix}] No positive deltas for reference. Using all deltas.")
        pos_delta_matrix = delta_matrix_all
    else:
        pos_delta_matrix = delta_matrix_all[pos_in_valid]

    global_q75, stable_dims, volatile_dims = _compute_reference_from_positives(
        pos_delta_matrix
    )
    print(
        f"  [{prefix}] Reference: q75={global_q75:.6f}, "
        f"n_stable={len(stable_dims)}, n_volatile={len(volatile_dims)} "
        f"(from {len(pos_delta_matrix)} positive pairs)"
    )

    # 4. Compute summary features for ALL valid pairs
    df_features = compute_delta_summary_features(
        delta_matrix_all, stable_dims, volatile_dims, global_q75
    )

    # 5. Merge back — align with valid_mask
    df_features = df_features.rename(columns={c: f"{prefix}_{c}" for c in df_features.columns})
    df_features.index = np.where(valid_mask_all)[0]

    for col in df_features.columns:
        df[col] = np.nan
        df.loc[df_features.index, col] = df_features[col].values

    return df


# CLI entry point 
if __name__ == "__main__":
    build_classifier_feature_table()