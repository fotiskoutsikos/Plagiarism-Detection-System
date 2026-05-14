"""
Modular Binary Plagiarism Classification Engine.

Serves as the reusable execution core for all supervised classification
experiments (e.g., feature ablation, dimensionality reduction).

Enforces a rigorous evaluation protocol:
    1. Single Interpretable Classifier: Logistic Regression (balanced).
    2. Strict CV: StratifiedGroupKFold grouped by 'filename_ori' (no leakage).
    3. Train-only Scaling: Pipeline fitted strictly per train fold.
    4. Train-only F0.5 Thresholding: Probability decision threshold optimized
       on train fold and evaluated on test fold.
    5. Dynamic Dimensionality Reduction: Optional fold-wise PCA integration.

Memory-safe design:
    - Uses float32 throughout to halve memory usage vs float64.
    - NaN handling via np.nan_to_num (no SimpleImputer copy overhead).
    - No median sort allocation (the root cause of the previous OOM crash).

This module is designed to be imported and orchestrated by ablation.py.
If run standalone, it executes a quick baseline validation run.
"""


import sys
import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


# Threshold Optimization 
def _find_optimal_probability_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = BETA,
) -> float:
    """
    Find the probability decision threshold that maximizes F-beta score.
    Optimized strictly on training-fold data to prevent leakage.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    scores = fbeta_score_curve(precision, recall, beta)

    if len(scores) <= 1:
        return 0.5

    optimal_idx = int(np.argmax(scores[:-1]))
    return float(thresholds[optimal_idx])


# Pipeline Factory 
def build_classifier_pipeline(
    use_pca: bool = False,
    n_pca_components: int = 30,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with standard scaling,
    optional PCA reduction, and a balanced Logistic Regression classifier.

    No SimpleImputer — NaN handling is done upstream via np.nan_to_num
    to avoid the massive memory allocation that median imputation causes
    on high-dimensional matrices (the root cause of the OOM crash).
    """
    steps: list[tuple[str, Any]] = [
        ("scaler", StandardScaler()),
    ]

    if use_pca:
        steps.append((
            "pca",
            PCA(n_components=n_pca_components, random_state=random_state),
        ))

    steps.append((
        "clf",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
    ))

    return Pipeline(steps)


# Core Experiment Execution 
def run_classifier_experiment(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    experiment_name: str,
    use_pca: bool = False,
    n_pca_components: int = 30,
    n_splits: int = NUM_K_FOLDS,
    random_state: int = RANDOM_STATE,
    beta: float = BETA,
) -> dict[str, Any]:
    """
    Execute a rigorous cross-validated classification experiment.

    Memory-safe: converts to float32 and replaces NaN with 0.0 upfront,
    avoiding the OOM crash caused by SimpleImputer's median sort on float64.

    Args:
        X: Feature matrix (N_samples, N_features). Can be DataFrame or numpy.
        y: Binary ground truth labels (N_samples,).
        groups: Grouping keys (e.g., filename_ori) for StratifiedGroupKFold.
        experiment_name: Identifier for the experiment.
        use_pca: If True, integrates PCA reduction into the fold pipeline.
        n_pca_components: Number of PCA components to retain.
        n_splits: Number of cross-validation folds.

    Returns:
        Dictionary containing aggregated evaluation metrics and fold summaries.
    """
    # Memory-safe input normalization
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    X_arr = np.ascontiguousarray(X_arr, dtype=np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_features = X_arr.shape

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    pipeline = build_classifier_pipeline(
        use_pca=use_pca,
        n_pca_components=n_pca_components,
        random_state=random_state,
    )

    fold_metrics = []
    pca_variances = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X_arr, y, groups), start=1):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)

        # Track PCA Explained Variance if applicable
        if use_pca:
            explained_var = float(pipeline.named_steps["pca"].explained_variance_ratio_.sum())
            pca_variances.append(explained_var)

        # Predict Probabilities
        prob_train = pipeline.predict_proba(X_train)[:, 1]
        prob_test  = pipeline.predict_proba(X_test)[:, 1]

        # Optimize Decision Threshold on TRAIN
        opt_thresh = _find_optimal_probability_threshold(y_train, prob_train, beta=beta)

        # Apply Threshold on TEST
        y_pred = (prob_test >= opt_thresh).astype(int)

        fold_metrics.append({
            "fold": fold_idx,
            "threshold": opt_thresh,
            "f05": fbeta_score(y_test, y_pred, beta=beta, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred),
        })

    # Aggregate Metrics
    df_folds = pd.DataFrame(fold_metrics)
    mean_metrics = df_folds.mean(axis=0).to_dict()

    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "classifier": "LogisticRegression",
        "n_samples": n_samples,
        "n_features_in": n_features,
        "n_features_model": n_pca_components if use_pca else n_features,
        "use_pca": use_pca,
        "f05": mean_metrics["f05"],
        "f1": mean_metrics["f1"],
        "precision": mean_metrics["precision"],
        "recall": mean_metrics["recall"],
        "accuracy": mean_metrics["accuracy"],
        "mean_threshold": mean_metrics["threshold"],
        "pca_explained_variance": np.mean(pca_variances) if use_pca else None,
        "fold_results": df_folds,
    }

    return summary


# Baseline Loading 
def load_threshold_baselines() -> pd.DataFrame:
    """
    Load pre-computed threshold baselines from threshold_analysis_summary.csv.
    Useful for comparison against learned classifiers.
    """
    rows = []
    thresh_path = Path(SUMMARY_FILES["threshold_analysis"])

    if thresh_path.exists():
        df_thresh = pd.read_csv(thresh_path)
        for _, r in df_thresh.iterrows():
            model = str(r.get("model", "")).upper()
            rows.append({
                "experiment_name": f"{model} Threshold Baseline",
                "classifier": "Distance Threshold",
                "f05": round(float(r.get("fbeta", 0)), 4),
                "f1": round(float(r.get("f1", 0)), 4),
                "precision": round(float(r.get("precision", 0)), 4),
                "recall": round(float(r.get("recall", 0)), 4),
                "accuracy": round(float(r.get("accuracy", 0)), 4),
            })

    if not rows:
        logger.warning("No threshold baselines found for comparison.")

    return pd.DataFrame(rows)


# Console Reporting Utility 
def print_experiment_summary(summary: dict[str, Any]) -> None:
    """Print a standardized, formatted summary of a classification experiment."""
    name = summary["experiment_name"]
    feats = summary["n_features_model"]
    is_pca = " (PCA)" if summary["use_pca"] else ""
    var_str = ""
    if summary["use_pca"] and summary["pca_explained_variance"] is not None:
        var_str = f" [Variance Retained: {summary['pca_explained_variance'] * 100:.1f}%]"

    print(f"\n  ► {name:<32} | Features: {feats:<4}{is_pca}{var_str}")
    print(
        f"    F0.5: {summary['f05']:.4f}  |  "
        f"Prec: {summary['precision']:.4f}  |  "
        f"Rec: {summary['recall']:.4f}  |  "
        f"F1: {summary['f1']:.4f}  |  "
        f"Acc: {summary['accuracy']:.4f}"
    )


# Standalone Validation Run 
if __name__ == "__main__":
    print("=" * 70)
    print("CLASSIFICATION ENGINE - BASELINE VALIDATION RUN")
    print("=" * 70)

    feat_table = Path("data/classifier_features.parquet")
    if not feat_table.exists():
        print(f"[ERROR] Feature table not found: {feat_table}")
        sys.exit(1)

    df = pd.read_parquet(feat_table)
    y = df["is_plagiarised"].astype(int).values
    groups = df["filename_ori"].values

    # Select engineered distance columns as a quick test
    dist_cols = [c for c in df.columns if "distance" in c]
    if not dist_cols:
        print("[ERROR] No distance features found in parquet.")
        sys.exit(1)

    print(f"Loaded {len(df)} pairs. Running validation on Distances Only...")
    res = run_classifier_experiment(
        X=df[dist_cols],
        y=y,
        groups=groups,
        experiment_name="Validation Run (Distances)",
    )

    print_experiment_summary(res)
    print("\nClassification engine is fully operational. Ready for ablation.py.")