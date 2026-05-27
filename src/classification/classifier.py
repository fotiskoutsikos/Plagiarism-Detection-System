"""
Modular Binary Plagiarism Classification Engine.

Serves as the reusable execution core for all supervised classification
experiments (ablation study and final evaluation).

Enforces a rigorous evaluation protocol:
    1. Single Non-Linear Classifier: XGBClassifier.
    2. Strict CV: StratifiedGroupKFold grouped by 'filename_ori' (no leakage).
    3. Train-only F0.5 Thresholding: Probability decision threshold optimized
       on train fold predictions, applied without modification to test fold.
    4. Out-of-Fold Collection: Full OOF prediction array returned for
       downstream triple-tier error analysis.

Design notes:
    - No StandardScaler: tree-based models are scale-invariant.
    - No PCA: dimensionality reduction is handled explicitly in ablation.py
      via ranked top-K actual dimensions, not via projection.
    - NaN handling via np.nan_to_num upstream (no SimpleImputer overhead).
    - float32 throughout to halve memory usage vs float64.

This module is designed to be imported and orchestrated by ablation.py
and final_classifier.py. If run standalone it executes a quick
baseline validation run.
"""

import sys
import importlib.util
import logging
from pathlib import Path
from typing import Any

from matplotlib.pyplot import clf
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold

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
    SUMMARY_FILES,
    NUM_K_FOLDS,
    RANDOM_STATE,
    BETA,
)
from utils.categorization import fbeta_score_curve


# Classifier Factory 
def build_classifier(y_train: np.ndarray, random_state: int = RANDOM_STATE) -> XGBClassifier:
    """
    Construct an XGBoost classifier with fixed hyperparameters.
    
    XGBoost is employed as the canonical gradient boosting framework
    (Chen & Guestrin, 2016), widely recognized for strong performance
    on tabular classification tasks.
    
    Class imbalance is addressed via scale_pos_weight, computed dynamically
    from the training fold to avoid leakage.
    
    Hyperparameters are intentionally conservative to avoid overfitting
    on moderate-sized datasets typical of music plagiarism benchmarks.
    
    Args:
        y_train: Training labels for computing scale_pos_weight.
        random_state: Seed for reproducibility.
    
    Returns:
        Configured XGBClassifier instance.
    """
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
    )

# Threshold Optimization 
def find_optimal_probability_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = BETA,
) -> float:
    """
    Find the probability threshold that maximises F-beta score.
    Always optimised strictly on training-fold data to prevent leakage.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    scores = fbeta_score_curve(precision, recall, beta)

    if len(scores) <= 1:
        return 0.5

    optimal_idx = int(np.argmax(scores[:-1]))
    return float(thresholds[optimal_idx])


# Core Experiment Execution 
def run_classifier_experiment(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    experiment_name: str,
    df_meta: pd.DataFrame | None = None,
    n_splits: int = NUM_K_FOLDS,
    random_state: int = RANDOM_STATE,
    beta: float = BETA,
) -> dict[str, Any]:
    """
    Execute a rigorous cross-validated classification experiment.

    Protocol
    --------
    1. StratifiedGroupKFold ensures no song appears in both train and test,
       preventing the model from exploiting song-level identity cues.
    2. The probability threshold is optimised exclusively on train-fold
       predictions (F-beta maximisation) and applied to the test fold
       without further adjustment.
    3. Out-of-fold predictions are collected across all folds, producing
       a complete OOF prediction array aligned with the input DataFrame.
       This enables downstream triple-tier error analysis without any
       additional inference pass.

    Args:
        X            : Feature matrix (N, F). DataFrame or numpy array.
        y            : Binary ground-truth labels (N,).
        groups       : Group keys (e.g. filename_ori) for StratifiedGroupKFold.
        experiment_name: Human-readable label for logging and output tables.
        df_meta      : Optional DataFrame carrying 'final_mod_type',
                       'category_grouped', and 'negative_tier' columns.
                       When provided, OOF metadata is collected alongside
                       predictions for triple-tier analysis.
        n_splits     : Number of CV folds (default from constants).
        random_state : Seed for reproducibility.
        beta         : Beta for F-beta score optimisation and reporting.

    Returns:
        Dictionary containing:
            - Aggregated CV metrics (f05, f1, precision, recall, accuracy).
            - fold_results  : per-fold metric DataFrame.
            - oof_df        : DataFrame with OOF predictions + metadata
                             (only when df_meta is not None).
    """
    # Input normalisation 
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    X_arr = np.ascontiguousarray(X_arr, dtype=np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_features = X_arr.shape

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics: list[dict] = []

    # OOF collection buffers
    oof_y_true  = np.empty(n_samples, dtype=np.int32)
    oof_y_pred  = np.empty(n_samples, dtype=np.int32)
    oof_y_prob  = np.empty(n_samples, dtype=np.float32)
    oof_indices = np.empty(n_samples, dtype=np.int64)

    ptr = 0  # write pointer into OOF buffers

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(X_arr, y, groups), start=1
    ):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = build_classifier(y_train, random_state=random_state)  # Τώρα παίρνει y_train
        clf.fit(X_train, y_train)

        prob_train = clf.predict_proba(X_train)[:, 1]
        prob_test  = clf.predict_proba(X_test)[:, 1]

        # Threshold calibrated on train fold only
        opt_thresh = find_optimal_probability_threshold(y_train, prob_train, beta=beta)
        y_pred     = (prob_test >= opt_thresh).astype(np.int32)

        fold_metrics.append({
            "fold":      fold_idx,
            "threshold": opt_thresh,
            "f05":       fbeta_score(y_test, y_pred, beta=beta, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "accuracy":  accuracy_score(y_test, y_pred),
        })

        # Accumulate OOF
        n_test = len(test_idx)
        oof_y_true [ptr:ptr + n_test] = y_test.astype(np.int32)
        oof_y_pred [ptr:ptr + n_test] = y_pred
        oof_y_prob [ptr:ptr + n_test] = prob_test.astype(np.float32)
        oof_indices[ptr:ptr + n_test] = test_idx
        ptr += n_test

        logger.debug(
            "Fold %d: F0.5=%.4f  P=%.4f  R=%.4f  thresh=%.4f",
            fold_idx,
            fold_metrics[-1]["f05"],
            fold_metrics[-1]["precision"],
            fold_metrics[-1]["recall"],
            opt_thresh,
        )

    # Aggregate metrics 
    df_folds     = pd.DataFrame(fold_metrics)
    mean_metrics = df_folds.mean(axis=0).to_dict()

    # Build OOF DataFrame 
    sort_order  = np.argsort(oof_indices[:ptr])
    oof_indices = oof_indices[:ptr][sort_order]
    oof_y_true  = oof_y_true [:ptr][sort_order]
    oof_y_pred  = oof_y_pred [:ptr][sort_order]
    oof_y_prob  = oof_y_prob [:ptr][sort_order]

    oof_df = pd.DataFrame({
        "original_index": oof_indices,
        "y_true":         oof_y_true,
        "y_pred":         oof_y_pred,
        "y_prob":         oof_y_prob,
    })

    if df_meta is not None:
        meta_cols = [
            c for c in ("final_mod_type", "category_grouped", "negative_tier")
            if c in df_meta.columns
        ]
        oof_df = oof_df.merge(
            df_meta[meta_cols].reset_index(drop=True).iloc[oof_indices].reset_index(drop=True),
            left_index=True,
            right_index=True,
        )

    # Summary dict 
    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "classifier": "XGBoost",
        "n_samples":       n_samples,
        "n_features":      n_features,
        "f05":             mean_metrics["f05"],
        "f1":              mean_metrics["f1"],
        "precision":       mean_metrics["precision"],
        "recall":          mean_metrics["recall"],
        "accuracy":        mean_metrics["accuracy"],
        "mean_threshold":  mean_metrics["threshold"],
        "fold_results":    df_folds,
        "oof_df":          oof_df,
    }

    return summary


# Baseline Loading 
def load_threshold_baselines() -> pd.DataFrame:
    """
    Load pre-computed threshold baselines from threshold_analysis_summary.csv.
    Provides the static-threshold reference point for comparison tables.
    """
    rows = []
    thresh_path = Path(SUMMARY_FILES["threshold_analysis"])

    if thresh_path.exists():
        df_thresh = pd.read_csv(thresh_path)
        for _, r in df_thresh.iterrows():
            model = str(r.get("model", "")).upper()
            rows.append({
                "experiment_name": f"{model} Threshold Baseline",
                "classifier":      "Distance Threshold",
                "f05":       round(float(r.get("fbeta",     0)), 4),
                "f1":        round(float(r.get("f1",        0)), 4),
                "precision": round(float(r.get("precision", 0)), 4),
                "recall":    round(float(r.get("recall",    0)), 4),
                "accuracy":  round(float(r.get("accuracy",  0)), 4),
            })

    if not rows:
        logger.warning("No threshold baselines found for comparison.")

    return pd.DataFrame(rows)


# Console Reporting 
def print_experiment_summary(summary: dict[str, Any]) -> None:
    """Print a standardised, formatted summary of a classification experiment."""
    name  = summary["experiment_name"]
    feats = summary["n_features"]
    print(f"\n  ► {name:<44} | Features: {feats:<5}")
    print(
        f"    F0.5: {summary['f05']:.4f}  |  "
        f"Prec: {summary['precision']:.4f}  |  "
        f"Rec: {summary['recall']:.4f}  |  "
        f"F1: {summary['f1']:.4f}  |  "
        f"Acc: {summary['accuracy']:.4f}"
    )


# Standalone Validation 
if __name__ == "__main__":
    print("=" * 70)
    print("CLASSIFICATION ENGINE — BASELINE VALIDATION RUN")
    print("=" * 70)

    feat_table = Path("data/classifier_features.parquet")
    if not feat_table.exists():
        print(f"[ERROR] Feature table not found: {feat_table}")
        sys.exit(1)

    df       = pd.read_parquet(feat_table)
    y        = df["is_plagiarised"].astype(int).values
    groups   = df["filename_ori"].values
    dist_cols = [c for c in df.columns if "distance" in c]

    if not dist_cols:
        print("[ERROR] No distance features found in parquet.")
        sys.exit(1)

    print(f"Loaded {len(df)} pairs. Running validation on Distances Only...")
    res = run_classifier_experiment(
        X=df[dist_cols], y=y, groups=groups,
        experiment_name="Validation Run (Distances)",
    )
    print_experiment_summary(res)
    print("\nClassification engine is fully operational. Ready for ablation.py.")