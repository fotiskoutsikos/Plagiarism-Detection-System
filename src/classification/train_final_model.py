"""
Final Plagiarism Detection Model Training.

Trains the final production-ready supervised classifier using the best
feature configuration identified in the ablation study:
    Feature Set D: All Engineered (No Vocals) + CLEWS XAI Top-256

This script:
    1. Extracts the exact features for the whole dataset.
    2. Computes XAI Top-256 indices from the full positive set (global ranking).
    3. Trains on 90% of data, calibrates threshold on held-out 10%.
    4. Retrains on 100% with the calibrated threshold.
    5. Packages everything into a .pkl artifact.
    6. Runs a quick inference demo on calibration samples.

Output:
    results/classification/final_plagiarism_detector.pkl
"""

import sys
import importlib.util
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
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
    EMBEDDING_PATHS, RANDOM_STATE, BETA,
    CLASSIFIER_FEATURE_TABLE, CLASSIFICATION_RESULTS_DIR, XAI_TOP_K, CALIBRATION_SIZE,
    FINAL_MODEL_OUTPUT,
)
from utils.categorization import fbeta_score_curve
from utils.classifier_features import (
    _build_embedding_map,
    _compute_delta_matrix_for_pairs,
)

FEATURE_TABLE = Path(CLASSIFIER_FEATURE_TABLE)
OUTPUT_DIR    = Path(CLASSIFICATION_RESULTS_DIR)
MODEL_OUTPUT  = Path(FINAL_MODEL_OUTPUT)
CAL_SIZE      = CALIBRATION_SIZE   # fraction held out for threshold calibration


# Helpers 

def _select_columns(df: pd.DataFrame, pattern: str) -> list[str]:
    return sorted([c for c in df.columns if pattern in c])


def _find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Returns the probability threshold that maximises F-beta on y_true/y_prob."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    scores = fbeta_score_curve(precision, recall, BETA)
    if len(scores) <= 1:
        return 0.5
    return float(thresholds[int(np.argmax(scores[:-1]))])


def _build_clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.1,
        min_samples_leaf=20,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=0,
    )


# Main 

def main() -> None:
    print("=" * 70)
    print("TRAINING FINAL PLAGIARISM DETECTOR (PRODUCTION MODEL)")
    print("=" * 70)

    # Load feature table 
    print("\n[1/5] Loading feature table...")
    if not FEATURE_TABLE.exists():
        sys.exit(f"[ERROR] Feature table not found: {FEATURE_TABLE}")

    df = pd.read_parquet(FEATURE_TABLE)
    y  = df["is_plagiarised"].astype(int).values
    print(f"  Loaded {len(df):,} pairs  ({y.sum():,} pos / {(1-y).sum():,} neg)")

    # Assemble features 
    print("\n[2/5] Assembling Feature Set D "
          "(Engineered No Vocals + CLEWS XAI Top-256)...")

    # 2a. Engineered columns (no vocal metadata)
    clews_dist_cols  = [c for c in _select_columns(df, "clews_")
                        if "distance" in c]
    wealy_dist_cols  = [c for c in _select_columns(df, "wealy_")
                        if "distance" in c]
    clews_delta_cols = [c for c in _select_columns(df, "clews_")
                        if any(x in c for x in
                               ["delta_", "stable_", "volatile_", "active_"])
                        and "xai_dim" not in c]
    wealy_delta_cols = [c for c in _select_columns(df, "wealy_")
                        if any(x in c for x in
                               ["delta_", "stable_", "volatile_", "active_"])
                        and "xai_dim" not in c]

    engineered_cols = (clews_dist_cols + wealy_dist_cols
                       + clews_delta_cols + wealy_delta_cols)
    X_eng = df[engineered_cols].values.astype(np.float32)

    # 2b. Global XAI Top-256 indices (ranked by positive-pair mean delta)
    #     NOTE: computed on the full dataset; acceptable for a production
    #     artefact because the ablation study validated the feature set
    #     under cross-validation.
    emb_map = _build_embedding_map(EMBEDDING_PATHS["CLEWS"])
    delta_valid, valid_mask = _compute_delta_matrix_for_pairs(df, emb_map)
    del emb_map

    ndim         = delta_valid.shape[1]
    delta_matrix = np.zeros((len(df), ndim), dtype=np.float32)
    delta_matrix[valid_mask] = delta_valid.astype(np.float32)
    del delta_valid

    mean_shifts      = np.mean(delta_matrix[y == 1], axis=0)
    top_256_indices  = np.argsort(mean_shifts)[::-1][:XAI_TOP_K]
    X_xai            = delta_matrix[:, top_256_indices]
    del delta_matrix, mean_shifts

    # 2c. Combine
    X_final = np.ascontiguousarray(
        np.hstack([X_eng, X_xai]), dtype=np.float32
    )
    X_final = np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Feature matrix: {X_final.shape}  "
          f"({len(engineered_cols)} engineered + {XAI_TOP_K} XAI)")

    # Calibrate threshold on a stratified hold-out 
    print(f"\n[3/5] Calibrating threshold on {int(CAL_SIZE*100)}% hold-out...")
    X_tr, X_cal, y_tr, y_cal, idx_tr, idx_cal = train_test_split(
        X_final, y, np.arange(len(df)),
        test_size=CAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    clf_cal = _build_clf()
    clf_cal.fit(X_tr, y_tr)
    prob_cal          = clf_cal.predict_proba(X_cal)[:, 1]
    optimal_threshold = _find_optimal_threshold(y_cal, prob_cal)
    print(f"  Optimal F{BETA} threshold (calibration set): {optimal_threshold:.4f}")
    del clf_cal, X_tr, y_tr   # free memory before full retrain

    # Retrain on 100% and save artifact 
    print("\n[4/5] Retraining on 100% of data and saving artifact...")
    clf_final = _build_clf()
    clf_final.fit(X_final, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "classifier"               : clf_final,
        "optimal_threshold"        : optimal_threshold,
        "engineered_feature_columns": engineered_cols,
        "clews_top_256_indices"    : top_256_indices.tolist(),
    }
    joblib.dump(artifact, MODEL_OUTPUT)
    print(f"  Artifact saved → {MODEL_OUTPUT}")

    # Inference demo on calibration samples 
    print("\n" + "─" * 70)
    print("INFERENCE DEMO  (held-out calibration samples)")
    print("─" * 70)

    rng      = np.random.default_rng(RANDOM_STATE)
    pos_pool = np.where(y_cal == 1)[0]
    neg_pool = np.where(y_cal == 0)[0]
    demo_local = np.concatenate([
        rng.choice(pos_pool, min(2, len(pos_pool)), replace=False),
        rng.choice(neg_pool, min(2, len(neg_pool)), replace=False),
    ])

    # Use the calibration probabilities (out-of-sample)
    for local_i in demo_local:
        global_i   = idx_cal[local_i]
        real_label = "PLAGIARISM"     if y_cal[local_i] == 1 else "NOT PLAGIARISM"
        prob       = prob_cal[local_i]
        pred       = "PLAGIARISM"     if prob >= optimal_threshold else "NOT PLAGIARISM"
        match      = "✓" if real_label == pred else "✗"

        row = df.iloc[global_i]
        print(f"  Pair: {row['pair_id']}  |  Type: {row['final_mod_type']:<22}")
        print(f"    Truth: {real_label:<16}  Pred: {pred:<16} {match}")
        print(f"    Score: {prob*100:.1f}%  (threshold {optimal_threshold*100:.1f}%)\n")

    print("=" * 70)
    print("Production artifact is ready.")
    print(f"  Model   → {MODEL_OUTPUT}")
    print(f"  Features: {X_final.shape[1]}  "
          f"({len(engineered_cols)} eng + {XAI_TOP_K} XAI)")
    print(f"  Threshold (F{BETA}): {optimal_threshold:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()