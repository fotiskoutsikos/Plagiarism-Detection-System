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
    5. Multi-Seed CI: Optional repetition over N independent random seeds to
       produce mean ± std and 95% confidence intervals for all metrics.
    6. Timing: Per-fold training time and per-sample inference latency
       measured and reported for efficiency analysis.

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
import time
import importlib.util
import logging
from pathlib import Path
from typing import Any

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


# Single-Seed CV Pass (internal)
def _run_single_seed(
    X_arr: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    df_meta: "pd.DataFrame | None",
    n_splits: int,
    beta: float,
) -> dict[str, Any]:
    """
    Execute one complete cross-validation pass with a fixed random seed.

    Timing is measured per fold:
      - train_time_sec   : wall-clock seconds for clf.fit(X_train, y_train)
      - infer_time_ms    : milliseconds per sample for clf.predict_proba(X_test)

    OOF predictions are collected and sorted by original sample index.
    df_meta columns are merged onto the OOF DataFrame when provided.

    Returns a dict of per-fold averages and the OOF DataFrame.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    n_samples = len(X_arr)
    fold_metrics: list[dict] = []

    oof_y_true  = np.empty(n_samples, dtype=np.int32)
    oof_y_pred  = np.empty(n_samples, dtype=np.int32)
    oof_y_prob  = np.empty(n_samples, dtype=np.float32)
    oof_indices = np.empty(n_samples, dtype=np.int64)
    ptr = 0

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(X_arr, y, groups), start=1
    ):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = build_classifier(y_train, random_state=seed)

        # --- Training time ---
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time_sec = time.perf_counter() - t0

        prob_train = clf.predict_proba(X_train)[:, 1]

        # --- Inference time ---
        t0 = time.perf_counter()
        prob_test = clf.predict_proba(X_test)[:, 1]
        infer_time_ms = (time.perf_counter() - t0) / len(X_test) * 1000.0

        opt_thresh = find_optimal_probability_threshold(y_train, prob_train, beta=beta)
        y_pred     = (prob_test >= opt_thresh).astype(np.int32)

        fold_metrics.append({
            "fold":           fold_idx,
            "threshold":      opt_thresh,
            "f05":            fbeta_score(y_test, y_pred, beta=beta, zero_division=0),
            "f1":             f1_score(y_test, y_pred, zero_division=0),
            "precision":      precision_score(y_test, y_pred, zero_division=0),
            "recall":         recall_score(y_test, y_pred, zero_division=0),
            "accuracy":       accuracy_score(y_test, y_pred),
            "train_time_sec": train_time_sec,
            "infer_time_ms":  infer_time_ms,
        })

        n_test = len(test_idx)
        oof_y_true [ptr:ptr + n_test] = y_test.astype(np.int32)
        oof_y_pred [ptr:ptr + n_test] = y_pred
        oof_y_prob [ptr:ptr + n_test] = prob_test.astype(np.float32)
        oof_indices[ptr:ptr + n_test] = test_idx
        ptr += n_test

        logger.debug(
            "Seed %d  Fold %d: F0.5=%.4f  P=%.4f  R=%.4f  "
            "thresh=%.4f  train=%.2fs  infer=%.4fms/sample",
            seed, fold_idx,
            fold_metrics[-1]["f05"],
            fold_metrics[-1]["precision"],
            fold_metrics[-1]["recall"],
            opt_thresh, train_time_sec, infer_time_ms,
        )

    # Sort OOF by original index
    sort_order      = np.argsort(oof_indices[:ptr])
    oof_indices_s   = oof_indices[:ptr][sort_order]
    oof_y_true_s    = oof_y_true [:ptr][sort_order]
    oof_y_pred_s    = oof_y_pred [:ptr][sort_order]
    oof_y_prob_s    = oof_y_prob [:ptr][sort_order]

    oof_df = pd.DataFrame({
        "original_index": oof_indices_s,
        "y_true":         oof_y_true_s,
        "y_pred":         oof_y_pred_s,
        "y_prob":         oof_y_prob_s,
    })

    if df_meta is not None:
        meta_cols = [
            c for c in ("final_mod_type", "category_grouped", "negative_tier")
            if c in df_meta.columns
        ]
        oof_df = oof_df.merge(
            df_meta[meta_cols].reset_index(drop=True).iloc[oof_indices_s].reset_index(drop=True),
            left_index=True,
            right_index=True,
        )

    df_folds     = pd.DataFrame(fold_metrics)
    mean_metrics = df_folds.mean(axis=0).to_dict()

    return {
        "fold_results":       df_folds,
        "oof_df":             oof_df,
        "f05":                mean_metrics["f05"],
        "f1":                 mean_metrics["f1"],
        "precision":          mean_metrics["precision"],
        "recall":             mean_metrics["recall"],
        "accuracy":           mean_metrics["accuracy"],
        "mean_threshold":     mean_metrics["threshold"],
        "mean_train_time_sec": mean_metrics["train_time_sec"],
        "mean_infer_time_ms": mean_metrics["infer_time_ms"],
    }


# Core Experiment Execution
def run_classifier_experiment(
    X: "np.ndarray | pd.DataFrame",
    y: np.ndarray,
    groups: np.ndarray,
    experiment_name: str,
    df_meta: "pd.DataFrame | None" = None,
    n_splits: int = NUM_K_FOLDS,
    random_state: int = RANDOM_STATE,
    beta: float = BETA,
    n_seeds: int = 1,
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
    3. Out-of-Fold predictions are collected across all folds, producing
       a complete OOF prediction array aligned with the input DataFrame.
       This enables downstream triple-tier error analysis without any
       additional inference pass.

    Statistical Rigor
    -----------------
    When n_seeds > 1, the full CV protocol is repeated n_seeds times with
    independent random seeds (random_state, random_state+1, ...).
    All per-fold metric keys gain _std and _ci95 variants in the returned
    summary, enabling confidence interval reporting in tables and plots.
    The OOF DataFrame returned always corresponds to seed 0 (the first run),
    which is sufficient for downstream qualitative error analysis.

    Timing
    ------
    Per-fold training time (seconds) and per-sample inference latency
    (milliseconds) are measured and averaged. These populate
    'mean_train_time_sec' and 'mean_infer_time_ms' in the summary.

    Args:
        X              : Feature matrix (N, F). DataFrame or numpy array.
        y              : Binary ground-truth labels (N,).
        groups         : Group keys (filename_ori) for StratifiedGroupKFold.
        experiment_name: Human-readable label for logging and output tables.
        df_meta        : Optional DataFrame carrying 'final_mod_type',
                         'category_grouped', and 'negative_tier' columns.
        n_splits       : Number of CV folds (default from constants).
        random_state   : Base seed; multi-seed runs use [random_state, ...,
                         random_state + n_seeds - 1].
        beta           : Beta for F-beta score optimisation and reporting.
        n_seeds        : Number of independent CV repetitions for CI
                         estimation. Use 1 (default) for a single run,
                         10 for publication-grade confidence intervals.

    Returns:
        Summary dictionary with:
            experiment_name, classifier, n_samples, n_features, n_seeds
            f05, f1, precision, recall, accuracy, mean_threshold   (means)
            f05_std, f05_ci95                                       (CI)
            precision_std, precision_ci95
            recall_std,    recall_ci95
            f1_std,        f1_ci95
            mean_train_time_sec, mean_infer_time_ms                 (timing)
            fold_results  : per-fold DataFrame (from seed 0)
            oof_df        : OOF predictions + optional metadata (seed 0)
    """
    # Input normalisation
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    X_arr = np.ascontiguousarray(X_arr, dtype=np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_features = X_arr.shape

    seeds = list(range(random_state, random_state + n_seeds))
    seed_results: list[dict] = []

    for i, seed in enumerate(seeds):
        if n_seeds > 1:
            logger.debug(
                "[%s] Seed %d/%d (seed=%d)...",
                experiment_name, i + 1, n_seeds, seed,
            )
        result = _run_single_seed(X_arr, y, groups, seed, df_meta, n_splits, beta)
        seed_results.append(result)

    # Aggregate metrics across seeds
    # Keys that exist in every seed result (excludes fold_results / oof_df)
    metric_keys = [
        "f05", "f1", "precision", "recall", "accuracy",
        "mean_threshold", "mean_train_time_sec", "mean_infer_time_ms",
    ]

    agg: dict[str, float] = {}
    for key in metric_keys:
        vals = np.array([r[key] for r in seed_results], dtype=np.float64)
        agg[key]               = float(np.mean(vals))
        agg[f"{key}_std"]      = float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0
        agg[f"{key}_ci95"]     = (
            1.96 * agg[f"{key}_std"] / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
        )

    # Summary dict — backward-compatible (f05, precision, … are means)
    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "classifier":      "XGBoost",
        "n_samples":       n_samples,
        "n_features":      n_features,
        "n_seeds":         n_seeds,
        # Means (backward compat)
        "f05":             agg["f05"],
        "f1":              agg["f1"],
        "precision":       agg["precision"],
        "recall":          agg["recall"],
        "accuracy":        agg["accuracy"],
        "mean_threshold":  agg["mean_threshold"],
        # Confidence intervals
        "f05_std":         agg["f05_std"],
        "f05_ci95":        agg["f05_ci95"],
        "f1_std":          agg["f1_std"],
        "f1_ci95":         agg["f1_ci95"],
        "precision_std":   agg["precision_std"],
        "precision_ci95":  agg["precision_ci95"],
        "recall_std":      agg["recall_std"],
        "recall_ci95":     agg["recall_ci95"],
        # Timing (averaged over folds and seeds)
        "mean_train_time_sec": agg["mean_train_time_sec"],
        "mean_infer_time_ms":  agg["mean_infer_time_ms"],
        # Detailed artefacts from seed 0 only (sufficient for OOF analysis)
        "fold_results": seed_results[0]["fold_results"],
        "oof_df":       seed_results[0]["oof_df"],
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
                # Baselines have no CI (single deterministic run)
                "f05_ci95":       0.0,
                "precision_ci95": 0.0,
                "recall_ci95":    0.0,
            })

    if not rows:
        logger.warning("No threshold baselines found for comparison.")

    return pd.DataFrame(rows)


# Console Reporting
def print_experiment_summary(summary: dict[str, Any]) -> None:
    """Print a standardised, formatted summary of a classification experiment."""
    name    = summary["experiment_name"]
    feats   = summary["n_features"]
    n_seeds = summary.get("n_seeds", 1)

    ci_str = ""
    if n_seeds > 1 and summary.get("f05_ci95", 0.0) > 0.0:
        ci_str = f" [±{summary['f05_ci95']:.4f}]"

    print(f"\n  ► {name:<44} | Features: {feats:<5} | Seeds: {n_seeds}")
    print(
        f"    F0.5: {summary['f05']:.4f}{ci_str}  |  "
        f"Prec: {summary['precision']:.4f}  |  "
        f"Rec: {summary['recall']:.4f}  |  "
        f"F1: {summary['f1']:.4f}  |  "
        f"Acc: {summary['accuracy']:.4f}"
    )
    if n_seeds > 1:
        print(
            f"    Train: {summary.get('mean_train_time_sec', 0):.2f}s/fold  |  "
            f"Infer: {summary.get('mean_infer_time_ms', 0):.4f}ms/sample"
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

    df        = pd.read_parquet(feat_table)
    y         = df["is_plagiarised"].astype(int).values
    groups    = df["filename_ori"].values
    dist_cols = [c for c in df.columns if "distance" in c]

    if not dist_cols:
        print("[ERROR] No distance features found in parquet.")
        sys.exit(1)

    print(f"Loaded {len(df)} pairs. Running validation on Distances Only (1 seed)...")
    res = run_classifier_experiment(
        X=df[dist_cols], y=y, groups=groups,
        experiment_name="Validation Run (Distances)",
        n_seeds=1,
    )
    print_experiment_summary(res)
    print("\nClassification engine is fully operational. Ready for ablation.py.")
