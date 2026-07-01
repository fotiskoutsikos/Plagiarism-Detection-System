"""
Supervised Classification Results

Produces category-level binary classification results with statistical
rigor (mean ± 95% CI across N_SEEDS independent CV runs).

Protocol
--------
For each seed:
    1. Run 5-Fold StratifiedGroupKFold CV with per-fold threshold calibration
    2. Collect full OOF predictions for that seed
    3. Compute broad-category metrics and FP tier counts

After all seeds:
    4. Aggregate per-seed metrics → mean ± std ± 95% CI per category
    5. Aggregate FP tier counts → mean ± std across seeds
    6. Print and save all results

This produces results that are directly comparable to:
    - ablation.py / hybrid_experiments.py (same multi-seed protocol)
    - binary_classification.py (same triple-tier analysis format)

Outputs
-------
    - results/binary_supervised_classification/supervised_broad_metrics.csv
    - results/binary_supervised_classification/supervised_broad_metrics_ci.csv
    - results/binary_supervised_classification/supervised_fp_tier_breakdown.csv
"""

import sys
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
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
    EMBEDDING_PATHS,
    NUM_K_FOLDS,
    RANDOM_STATE,
    BETA,
    CLASSIFIER_FEATURE_TABLE,
)
from utils.categorization import (
    categorize_modification,
    clean_mod_type,
)
from utils.classifier_features import (
    _build_embedding_map,
    _compute_delta_matrix_for_pairs,
)
from classifier import (
    build_classifier,
    find_optimal_probability_threshold,
)

# Configuration 
SELECTED_CONFIG = "hybrid_top512"
N_SEEDS         = 10

FEATURE_TABLE = Path(CLASSIFIER_FEATURE_TABLE)
OUTPUT_DIR    = Path("results/binary_supervised_classification")


# Helpers 
def _select_columns(df: pd.DataFrame, pattern: str) -> list[str]:
    return sorted([c for c in df.columns if pattern in c])


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Precision":  precision_score(y_true, y_pred, zero_division=0),
        "Recall":     recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":   f1_score(y_true, y_pred, zero_division=0),
        "F0.5-Score": fbeta_score(y_true, y_pred, beta=BETA, zero_division=0),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def _ci95(values: np.ndarray) -> float:
    """95% confidence interval half-width using t-distribution."""
    n = len(values)
    if n < 2:
        return 0.0
    se = np.std(values, ddof=1) / np.sqrt(n)
    return float(stats.t.ppf(0.975, df=n - 1) * se)


# Feature Builder 
def build_features(
    config_name: str,
    df: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    clews_dist_cols = [c for c in _select_columns(df, "clews_") if "distance" in c]
    wealy_dist_cols = [c for c in _select_columns(df, "wealy_") if "distance" in c]
    clews_delta_cols = [
        c for c in _select_columns(df, "clews_")
        if any(x in c for x in ["delta_", "stable_", "volatile_", "active_"])
    ]
    wealy_delta_cols = [
        c for c in _select_columns(df, "wealy_")
        if any(x in c for x in ["delta_", "stable_", "volatile_", "active_"])
    ]
    vocal_cols = [
        c for c in df.columns
        if c in {"pair_vocal_valid", "vocal_ratio_ori", "vocal_ratio_mod",
                 "vocal_valid_ori", "vocal_valid_mod"}
        and not df[c].isna().all()
    ]

    base_no_vocal = clews_dist_cols + wealy_dist_cols + clews_delta_cols + wealy_delta_cols

    if config_name == "engineered_no_vocals":
        return df[base_no_vocal].values, base_no_vocal
    if config_name == "engineered_with_vocals":
        cols = base_no_vocal + vocal_cols
        return df[cols].values, cols
    if config_name.startswith("hybrid_top"):
        k = int(config_name.replace("hybrid_top", ""))
        print(f"    Building CLEWS delta matrix for Top-{k}...")
        emb_map = _build_embedding_map(EMBEDDING_PATHS["CLEWS"])
        delta_valid, valid_mask = _compute_delta_matrix_for_pairs(df, emb_map)
        del emb_map
        ndim = delta_valid.shape[1]
        delta_matrix = np.zeros((len(df), ndim), dtype=np.float32)
        delta_matrix[valid_mask] = delta_valid.astype(np.float32)
        del delta_valid, valid_mask
        mean_shifts = np.mean(np.abs(delta_matrix[y == 1]), axis=0)
        ranked_idx = np.argsort(mean_shifts)[::-1]
        base_X = df[base_no_vocal].values.astype(np.float32)
        top_k = delta_matrix[:, ranked_idx[:k]].astype(np.float32)
        X_hybrid = np.hstack([base_X, top_k])
        names = base_no_vocal + [f"clews_topdim_{int(i)}" for i in ranked_idx[:k]]
        return X_hybrid, names
    raise ValueError(f"Unknown config: {config_name}")


# Single Seed OOF 
def run_single_seed_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Run one complete StratifiedGroupKFold CV with per-fold threshold
    calibration. Returns full OOF binary predictions for all samples.
    """
    X = np.ascontiguousarray(
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float32,
    )
    n_samples = len(y)
    y_pred = np.full(n_samples, -1, dtype=np.int32)

    sgkf = StratifiedGroupKFold(
        n_splits=NUM_K_FOLDS, shuffle=True, random_state=seed
    )

    for train_idx, test_idx in sgkf.split(X, y, groups):
        clf = build_classifier(y[train_idx], random_state=RANDOM_STATE)
        clf.fit(X[train_idx], y[train_idx])

        # Calibrate threshold on training fold
        prob_tr = clf.predict_proba(X[train_idx])[:, 1]
        threshold = find_optimal_probability_threshold(
            y[train_idx], prob_tr, beta=BETA
        )

        # Predict on test fold using calibrated threshold
        prob_test = clf.predict_proba(X[test_idx])[:, 1]
        y_pred[test_idx] = (prob_test >= threshold).astype(int)

    assert (y_pred >= 0).all(), "Some samples were not predicted!"
    return y_pred


# Per-Seed Category Metrics 
def compute_category_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categories: np.ndarray,
    negative_tiers: np.ndarray | None,
) -> tuple[dict, dict]:
    """
    Compute broad-category metrics and FP tier counts for one seed.

    Returns:
        broad_metrics: {category: {metric: value}}
        fp_tiers: {tier_name: count}
    """
    unique_cats = sorted(set(categories[y_true == 1]))

    broad_metrics = {}
    for cat in unique_cats:
        mask = (categories == cat) | (y_true == 0)
        m = _compute_binary_metrics(y_true[mask], y_pred[mask])
        broad_metrics[cat] = m

    # Overall
    m_overall = _compute_binary_metrics(y_true, y_pred)
    broad_metrics["OVERALL"] = m_overall

    # FP tier breakdown
    fp_tiers = {}
    if negative_tiers is not None:
        fp_mask = (y_true == 0) & (y_pred == 1)
        for tier in ["global_nearest", "intra_category_nearest", "random"]:
            fp_tiers[tier] = int(((negative_tiers == tier) & fp_mask).sum())
        fp_tiers["total"] = int(fp_mask.sum())

    return broad_metrics, fp_tiers


# Main 
def main() -> None:
    print("=" * 80)
    print("SUPERVISED CLASSIFICATION — PER-SEED CATEGORY ANALYSIS")
    print(f"Configuration: {SELECTED_CONFIG} | Seeds: {N_SEEDS}")
    print("=" * 80)

    # Load data
    if not FEATURE_TABLE.exists():
        print(f"[ERROR] Feature table not found: {FEATURE_TABLE}")
        return

    print(f"\nLoading feature table from {FEATURE_TABLE}...")
    df = pd.read_parquet(FEATURE_TABLE)
    y = df["is_plagiarised"].astype(int).values
    groups = df["filename_ori"].values
    print(f"  Loaded {len(df):,} pairs  ({y.sum():,} pos / {(y == 0).sum():,} neg)")

    # Build features
    print(f"\nBuilding features for: {SELECTED_CONFIG}...")
    X, feature_names = build_features(SELECTED_CONFIG, df, y)
    print(f"  Feature matrix: {X.shape}  ({len(feature_names)} features)")

    # Build metadata
    clean_mods = df["final_mod_type"].apply(clean_mod_type).values
    categories = np.array([categorize_modification(m) for m in clean_mods])
    categories[y == 0] = "Negative Pairs"
    negative_tiers = df["negative_tier"].values if "negative_tier" in df.columns else None

    # Run per-seed evaluation
    print(f"\nRunning {N_SEEDS}-seed per-seed category analysis...")

    all_broad: list[dict] = []
    all_fp_tiers: list[dict] = []

    for seed_idx in range(N_SEEDS):
        y_pred = run_single_seed_oof(X, y, groups, seed=seed_idx)
        broad, fp_tiers = compute_category_metrics(y_true=y, y_pred=y_pred,
                                                    categories=categories,
                                                    negative_tiers=negative_tiers)
        all_broad.append(broad)
        all_fp_tiers.append(fp_tiers)
        print(f"  Seed {seed_idx + 1}/{N_SEEDS}: "
              f"Overall F0.5={broad['OVERALL']['F0.5-Score']:.4f}  "
              f"Prec={broad['OVERALL']['Precision']:.4f}  "
              f"Rec={broad['OVERALL']['Recall']:.4f}  "
              f"FP={broad['OVERALL']['FP']}")

    # Aggregate across seeds
    print(f"\n{'=' * 120}")
    print(f" SUPERVISED CLASSIFICATION: {SELECTED_CONFIG.upper()}")
    print(f" Model: XGBoost | {N_SEEDS} seeds × {NUM_K_FOLDS}-Fold | mean ± 95% CI")
    print(f"{'=' * 120}")

    # Get all categories
    all_cats = sorted(set(k for b in all_broad for k in b.keys() if k != "OVERALL"))
    all_cats.append("OVERALL")

    metrics_to_report = ["Precision", "Recall", "F0.5-Score", "F1-Score", "TP", "FP"]

    # Header
    print(f"  {'Category':<32} | {'Precision':>16} | {'Recall':>16} | "
          f"{'F0.5':>16} | {'F1':>16} | {'TP':>10} | {'FP':>10}")
    print(f"  {'-' * 116}")

    rows_for_csv = []

    for cat in all_cats:
        seed_values = {m: [] for m in metrics_to_report}
        for broad in all_broad:
            if cat in broad:
                for m in metrics_to_report:
                    seed_values[m].append(broad[cat][m])

        row = {"Category": cat}
        display_parts = []
        for m in metrics_to_report:
            vals = np.array(seed_values[m])
            mean_val = np.mean(vals)
            ci = _ci95(vals)
            row[f"{m}_mean"] = round(mean_val, 4)
            row[f"{m}_ci95"] = round(ci, 4)
            row[f"{m}_std"] = round(np.std(vals, ddof=1), 4)

            if m in ("TP", "FP"):
                display_parts.append(f"{mean_val:>7.0f}±{ci:>4.0f}")
            else:
                display_parts.append(f"{mean_val:>6.1%}±{ci:.1%}")

        prefix = "► " if cat == "OVERALL" else "  "
        print(f"  {prefix}{cat:<30} | " + " | ".join(display_parts))
        rows_for_csv.append(row)

    print(f"  {'=' * 116}\n")

    # FP tier aggregation
    if all_fp_tiers and all_fp_tiers[0]:
        print("  False Positive Tier Breakdown (mean ± 95% CI across seeds):")
        tier_names = ["global_nearest", "intra_category_nearest", "random", "total"]
        for tier in tier_names:
            vals = np.array([fp[tier] for fp in all_fp_tiers if tier in fp])
            mean_val = np.mean(vals)
            ci = _ci95(vals)
            pct = ""
            if tier != "total":
                total_vals = np.array([fp["total"] for fp in all_fp_tiers])
                mean_total = np.mean(total_vals)
                if mean_total > 0:
                    pct = f"  ({mean_val / mean_total * 100:.1f}%)"
            print(f"    {tier:<30}: {mean_val:>8.1f} ± {ci:>6.1f}{pct}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(rows_for_csv)
    ci_path = OUTPUT_DIR / "supervised_broad_metrics_ci.csv"
    df_results.to_csv(ci_path, index=False)
    print(f"\n  Results with CI saved → {ci_path}")

    # Save FP tiers
    if all_fp_tiers and all_fp_tiers[0]:
        fp_rows = []
        for tier in ["global_nearest", "intra_category_nearest", "random"]:
            vals = np.array([fp[tier] for fp in all_fp_tiers])
            fp_rows.append({
                "negative_tier": tier,
                "FP_mean": round(np.mean(vals), 1),
                "FP_std": round(np.std(vals, ddof=1), 1),
                "FP_ci95": round(_ci95(vals), 1),
            })
        df_fp = pd.DataFrame(fp_rows)
        fp_path = OUTPUT_DIR / "supervised_fp_tier_breakdown.csv"
        df_fp.to_csv(fp_path, index=False)
        print(f"  FP tier breakdown saved → {fp_path}")

    print(f"\n{'=' * 80}")
    print(f"COMPLETE — Results saved → {OUTPUT_DIR}/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()