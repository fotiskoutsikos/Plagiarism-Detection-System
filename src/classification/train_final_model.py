"""
Final Plagiarism Detection Model Training.

Trains the final production classifier artifact using the manually selected
supervised configuration.

Important
---------
This script does NOT perform model selection, ablation, or evaluation.
Its sole purpose is to:

    1. Rebuild the exact selected feature configuration on the full dataset.
    2. Compute and store all training-derived reference statistics needed
       for exact feature reconstruction at inference time.
    3. Calibrate a probability threshold on a stratified hold-out split.
    4. Retrain the classifier on 100% of the available data.
    5. Save a deployment-ready .pkl artifact containing everything needed
       by predict_pair.py to produce identical feature vectors for new pairs.

Current final configuration:
    SELECTED_CONFIG = "hybrid_top512"

This means:
    - Engineered features (no vocal metadata)
    - + Top-512 CLEWS raw delta dimensions

Feature computation consistency:
    - Distances are computed by metrics.py::_compute_all_distances
    - Delta summaries are computed by classifier_features.py::compute_delta_summary_features
    - Reference statistics (stable_dims, volatile_dims, global_q75) are derived
      from positive pairs only via classifier_features.py::_compute_reference_from_positives
    - Top-K CLEWS ranking uses mean absolute shift on positive pairs
    - All of these are stored in the artifact for exact inference reproduction

Output:
    models/final_plagiarism_detector.pkl
"""

import sys
import importlib.util
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve repository root & logging 
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
    EMBEDDING_PATHS,
    RANDOM_STATE,
    BETA,
    CLASSIFIER_FEATURE_TABLE,
    CALIBRATION_SIZE,
    FINAL_MODEL_OUTPUT,
    SELECTED_CONFIG,
)

# Centralized feature computation utilities
from utils.classifier_features import (
    _build_embedding_map,
    _compute_delta_matrix_for_pairs,
    _compute_reference_from_positives,
)

# Centralized classifier engine
from classifier import (
    build_classifier,
    find_optimal_probability_threshold,
)

FEATURE_TABLE = Path(CLASSIFIER_FEATURE_TABLE)
MODEL_OUTPUT  = Path(FINAL_MODEL_OUTPUT)
CAL_SIZE      = CALIBRATION_SIZE


# Helpers 
def _select_columns(df: pd.DataFrame, pattern: str) -> list[str]:
    return sorted([c for c in df.columns if pattern in c])


def _parse_hybrid_k(config_name: str) -> int | None:
    if config_name.startswith("hybrid_top"):
        suffix = config_name.replace("hybrid_top", "")
        try:
            return int(suffix)
        except ValueError:
            raise ValueError(f"Invalid hybrid configuration name: {config_name}")
    return None


def _compute_reference_stats_for_model(
    df: pd.DataFrame,
    y: np.ndarray,
    parquet_path: str,
    model_name: str,
) -> dict:
    """
    Compute training-derived reference statistics for one embedding model.

    Uses the SAME centralized functions as classifier_features.py:
        - _build_embedding_map
        - _compute_delta_matrix_for_pairs
        - _compute_reference_from_positives

    These statistics are essential for exact feature reconstruction
    at inference time and must be stored in the artifact.

    Args:
        df: Full training DataFrame with filename_ori/filename_mod columns.
        y: Binary labels.
        parquet_path: Path to embedding parquet.
        model_name: "CLEWS" or "WEALY" (for logging).

    Returns:
        Dictionary with global_q75, stable_dims, volatile_dims.
    """
    print(f"  Computing {model_name} reference statistics...")

    emb_map = _build_embedding_map(parquet_path)
    delta_valid, valid_mask = _compute_delta_matrix_for_pairs(df, emb_map)
    del emb_map

    if delta_valid.size == 0:
        logger.warning("[%s] No valid deltas. Reference stats will be empty.", model_name)
        return {
            "global_q75": 0.0,
            "stable_dims": [],
            "volatile_dims": [],
        }

    # Extract positive-only deltas (same logic as classifier_features.py)
    pos_in_valid = y[valid_mask].astype(bool)

    if pos_in_valid.sum() == 0:
        logger.warning("[%s] No positive deltas for reference. Using all.", model_name)
        pos_delta_matrix = delta_valid
    else:
        pos_delta_matrix = delta_valid[pos_in_valid]

    # Use the SAME centralized function
    global_q75, stable_dims, volatile_dims = _compute_reference_from_positives(
        pos_delta_matrix
    )

    print(
        f"    [{model_name}] q75={global_q75:.6f}, "
        f"n_stable={len(stable_dims)}, n_volatile={len(volatile_dims)} "
        f"(from {len(pos_delta_matrix)} positive pairs)"
    )

    return {
        "global_q75": float(global_q75),
        "stable_dims": stable_dims.astype(int).tolist(),
        "volatile_dims": volatile_dims.astype(int).tolist(),
    }


def build_features_for_selected_config(
    df: pd.DataFrame,
    y: np.ndarray,
    config_name: str,
) -> tuple[np.ndarray, dict]:
    """
    Build the exact feature matrix for the selected final configuration.

    All feature computation uses centralized utilities from:
        - classifier_features.py (delta summaries, reference stats)
        - metrics.py (distances — already precomputed in feature table)

    Returns:
        X_final : np.ndarray
        metadata: dict containing feature schema, top-K info, and
                  training reference statistics for inference consistency.
    """
    # Identify feature columns 
    clews_dist_cols = [c for c in _select_columns(df, "clews_") if "distance" in c]
    wealy_dist_cols = [c for c in _select_columns(df, "wealy_") if "distance" in c]

    clews_delta_cols = [
        c for c in _select_columns(df, "clews_")
        if any(x in c for x in ["delta_", "stable_", "volatile_", "active_"])
        and "xai_dim" not in c
    ]
    wealy_delta_cols = [
        c for c in _select_columns(df, "wealy_")
        if any(x in c for x in ["delta_", "stable_", "volatile_", "active_"])
        and "xai_dim" not in c
    ]

    vocal_cols = [
        c for c in df.columns
        if c in {
            "pair_vocal_valid",
            "vocal_ratio_ori",
            "vocal_ratio_mod",
            "vocal_valid_ori",
            "vocal_valid_mod",
        }
        and not df[c].isna().all()
    ]

    engineered_no_vocals = (
        clews_dist_cols + wealy_dist_cols
        + clews_delta_cols + wealy_delta_cols
    )
    engineered_with_vocals = engineered_no_vocals + vocal_cols

    # Compute training reference statistics 
    # These are needed by predict_pair.py for exact inference reproduction
    print("\n  Computing training reference statistics for inference consistency...")

    clews_ref = _compute_reference_stats_for_model(
        df, y, EMBEDDING_PATHS["CLEWS"], "CLEWS"
    )
    wealy_ref = _compute_reference_stats_for_model(
        df, y, EMBEDDING_PATHS["WEALY"], "WEALY"
    )

    # Case 1: Engineered (No Vocals) 
    if config_name == "engineered_no_vocals":
        X = df[engineered_no_vocals].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        metadata = {
            "selected_config": config_name,
            "engineered_feature_columns": engineered_no_vocals,
            "full_feature_names": engineered_no_vocals,
            "clews_top_k_indices": [],
            "clews_top_k": 0,
            "uses_vocals": False,
            "uses_clews_topk": False,
            # Training reference statistics
            "clews_reference": clews_ref,
            "wealy_reference": wealy_ref,
        }
        return X, metadata

    # Case 2: Engineered (With Vocals) 
    if config_name == "engineered_with_vocals":
        X = df[engineered_with_vocals].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        metadata = {
            "selected_config": config_name,
            "engineered_feature_columns": engineered_with_vocals,
            "full_feature_names": engineered_with_vocals,
            "clews_top_k_indices": [],
            "clews_top_k": 0,
            "uses_vocals": True,
            "uses_clews_topk": False,
            # Training reference statistics
            "clews_reference": clews_ref,
            "wealy_reference": wealy_ref,
        }
        return X, metadata

    # Case 3: Hybrid (Engineered No Vocals + Top-K CLEWS) 
    k = _parse_hybrid_k(config_name)
    if k is not None:
        logger.info("Building hybrid feature matrix with CLEWS Top-%d.", k)

        # Base engineered features (already in feature table)
        X_eng = df[engineered_no_vocals].values.astype(np.float32)
        X_eng = np.nan_to_num(X_eng, nan=0.0, posinf=0.0, neginf=0.0)

        # Build CLEWS full delta matrix (using centralized utility)
        emb_map = _build_embedding_map(EMBEDDING_PATHS["CLEWS"])
        delta_valid, valid_mask = _compute_delta_matrix_for_pairs(df, emb_map)
        del emb_map

        if delta_valid.size == 0:
            raise RuntimeError(
                "No valid CLEWS deltas could be computed for hybrid training."
            )

        ndim = delta_valid.shape[1]
        delta_matrix = np.zeros((len(df), ndim), dtype=np.float32)
        delta_matrix[valid_mask] = delta_valid.astype(np.float32)
        del delta_valid, valid_mask

        # Rank dimensions (same logic as hybrid_experiments.py)
        pos_mask    = y == 1
        mean_shifts = np.mean(np.abs(delta_matrix[pos_mask]), axis=0)
        ranked_idx  = np.argsort(mean_shifts)[::-1]

        top_k_indices = ranked_idx[:k]
        X_topk        = delta_matrix[:, top_k_indices].astype(np.float32)

        # Final hybrid matrix
        X = np.ascontiguousarray(
            np.hstack([X_eng, X_topk]), dtype=np.float32
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        top_k_feature_names = [
            f"clews_topdim_{int(i)}" for i in top_k_indices
        ]

        metadata = {
            "selected_config": config_name,
            "engineered_feature_columns": engineered_no_vocals,
            "full_feature_names": engineered_no_vocals + top_k_feature_names,
            "clews_top_k_indices": top_k_indices.astype(int).tolist(),
            "clews_top_k": int(k),
            "uses_vocals": False,
            "uses_clews_topk": True,
            # Training reference statistics
            "clews_reference": clews_ref,
            "wealy_reference": wealy_ref,
        }
        return X, metadata

    raise ValueError(
        f"Unknown SELECTED_CONFIG: {config_name}. "
        f"Supported: engineered_no_vocals, engineered_with_vocals, "
        f"hybrid_top256, hybrid_top512, hybrid_top1024."
    )


# Main 
def main() -> None:
    print("=" * 70)
    print("TRAINING FINAL PLAGIARISM DETECTOR (PRODUCTION ARTIFACT)")
    print("=" * 70)
    print(f"Selected configuration: {SELECTED_CONFIG}")

    # [1/4] Load feature table
    print("\n[1/4] Loading feature table...")
    if not FEATURE_TABLE.exists():
        sys.exit(f"[ERROR] Feature table not found: {FEATURE_TABLE}")

    df = pd.read_parquet(FEATURE_TABLE)
    y  = df["is_plagiarised"].astype(int).values

    print(f"  Loaded {len(df):,} pairs")
    print(f"  Positives: {int(y.sum()):,}")
    print(f"  Negatives: {int((y == 0).sum()):,}")

    # [2/4] Build selected feature matrix + reference stats
    print("\n[2/4] Building selected feature matrix...")
    X_final, feature_meta = build_features_for_selected_config(
        df, y, SELECTED_CONFIG
    )

    print(f"\n  Final feature matrix: {X_final.shape}")
    print(f"  Uses vocals: {feature_meta['uses_vocals']}")
    print(f"  Uses CLEWS Top-K: {feature_meta['uses_clews_topk']}")
    if feature_meta["uses_clews_topk"]:
        print(f"  CLEWS Top-K: {feature_meta['clews_top_k']}")
    print(f"  CLEWS ref q75: {feature_meta['clews_reference']['global_q75']:.6f}")
    print(f"  WEALY ref q75: {feature_meta['wealy_reference']['global_q75']:.6f}")

    # [3/4] Calibrate threshold on stratified hold-out
    print(f"\n[3/4] Calibrating threshold on {int(CAL_SIZE * 100)}% hold-out...")

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_final,
        y,
        test_size=CAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    clf_cal = build_classifier(y_tr, random_state=RANDOM_STATE)
    clf_cal.fit(X_tr, y_tr)

    prob_cal = clf_cal.predict_proba(X_cal)[:, 1]
    optimal_threshold = find_optimal_probability_threshold(
        y_cal, prob_cal, beta=BETA
    )

    print(f"  Optimal F{BETA} threshold (calibration set): {optimal_threshold:.4f}")

    del clf_cal, X_tr, X_cal, y_tr, y_cal

    # [4/4] Retrain on 100% and save artifact
    print("\n[4/4] Retraining on 100% of data and saving artifact...")

    clf_final = build_classifier(y, random_state=RANDOM_STATE)
    clf_final.fit(X_final, y)

    artifact = {
        # Model 
        "classifier": clf_final,
        "optimal_threshold": float(optimal_threshold),

        # Configuration 
        "selected_config": feature_meta["selected_config"],
        "classifier_type": "XGBoost",
        "n_features": int(X_final.shape[1]),

        # Feature schema 
        "engineered_feature_columns": feature_meta["engineered_feature_columns"],
        "full_feature_names": feature_meta["full_feature_names"],
        "clews_top_k_indices": feature_meta["clews_top_k_indices"],
        "clews_top_k": int(feature_meta["clews_top_k"]),
        "uses_vocals": bool(feature_meta["uses_vocals"]),
        "uses_clews_topk": bool(feature_meta["uses_clews_topk"]),

        # Training reference statistics (for inference consistency) 
        # These are computed from positive pairs only via
        # classifier_features.py::_compute_reference_from_positives
        # and must be used by predict_pair.py to reconstruct exact
        # delta summary features for new unseen pairs.
        "clews_reference": feature_meta["clews_reference"],
        "wealy_reference": feature_meta["wealy_reference"],

        # Reproducibility / metadata 
        "random_state": RANDOM_STATE,
        "beta": BETA,
        "calibration_size": CAL_SIZE,
        "training_pairs": int(len(df)),
        "positive_pairs": int(y.sum()),
        "negative_pairs": int((y == 0).sum()),
        "trained_on": datetime.now().isoformat(timespec="seconds"),
    }

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT)

    print(f"  Artifact saved → {MODEL_OUTPUT}")

    # Summary 
    print("\n" + "=" * 70)
    print("Production artifact is ready.")
    print(f"  Model        → {MODEL_OUTPUT}")
    print(f"  Config       → {feature_meta['selected_config']}")
    print(f"  Features     → {X_final.shape[1]}")
    print(f"  Threshold    → {optimal_threshold:.4f}")
    print(f"  CLEWS ref q75 → {feature_meta['clews_reference']['global_q75']:.6f}")
    print(f"  WEALY ref q75 → {feature_meta['wealy_reference']['global_q75']:.6f}")
    print(f"  CLEWS stable  → {len(feature_meta['clews_reference']['stable_dims'])} dims")
    print(f"  CLEWS volatile→ {len(feature_meta['clews_reference']['volatile_dims'])} dims")
    print(f"  WEALY stable  → {len(feature_meta['wealy_reference']['stable_dims'])} dims")
    print(f"  WEALY volatile→ {len(feature_meta['wealy_reference']['volatile_dims'])} dims")
    if feature_meta["uses_clews_topk"]:
        print(f"  CLEWS Top-K   → {feature_meta['clews_top_k']} dims")
    print("=" * 70)


if __name__ == "__main__":
    main()