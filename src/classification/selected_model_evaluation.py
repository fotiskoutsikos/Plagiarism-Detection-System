"""
Selected Model Evaluation: Deep Diagnostic Analysis.

This script performs comprehensive error analysis and interpretability
study on manually-selected supervised model configurations.

Configuration flags at the top of the script control which models
to evaluate. Enable/disable by setting to True/False.

Protocol
--------
For each enabled configuration:
1. Build feature matrix for the specified configuration.
2. Run grouped cross-validation with XGBoost.
3. Collect out-of-fold predictions with metadata.
4. Apply triple-tier error analysis (same logic as binary_classification.py):
   - Broad category metrics
   - Granular mod-type metrics
   - False Positive tier breakdown
5. Compute permutation feature importance on a held-out grouped fold.
6. Export all CSVs and plots with configuration-specific naming.

Usage
-----
    Edit the CONFIGS_TO_EVALUATE dictionary below, then:
    python selected_model_evaluation.py

Configuration Options
---------------------
    - "engineered_no_vocals": All distances + delta summaries, no vocal metadata
    - "engineered_with_vocals": All distances + delta summaries + vocal metadata
    - "hybrid_top256": Engineered (no vocals) + Top-256 CLEWS dims
    - "hybrid_top512": Engineered (no vocals) + Top-512 CLEWS dims
    - "hybrid_top1024": Engineered (no vocals) + Top-1024 CLEWS dims

Inputs
------
    - results/classification/classifier_features.parquet
    - data/clews_embeddings.parquet (for hybrid configs)

Outputs (prefixed with config name, e.g., "eval_no_vocals_...")
-------
    - results/classification/{config}_broad_metrics.csv
    - results/classification/{config}_detailed_metrics.csv
    - results/classification/{config}_fp_tier_breakdown.csv
    - results/classification/{config}_feature_importance.csv
    - results/classification/{config}_feature_importance_family.csv
    - plots/classification/{config}_feature_importance.pdf
    - plots/classification/{config}_feature_importance_family.pdf
"""

import sys
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

# CONFIGURATION: Enable/disable models to evaluate
CONFIGS_TO_EVALUATE = {
    "engineered_no_vocals":   True,   # Primary candidate
    "engineered_with_vocals": True,   # Diagnostic comparison
    "hybrid_top256":          False,  # Hybrid enrichment experiments
    "hybrid_top512":          True,
    "hybrid_top1024":         False,
}

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
    PLOT_DPI,
    PLOT_STYLE_PARAMS,
    CLASSIFIER_FEATURE_TABLE,
    CLASSIFICATION_RESULTS_DIR,
    CLASSIFICATION_PLOTS_DIR,
)
from utils.categorization import get_ground_truth_label, categorize_modification, clean_mod_type
from utils.classifier_features import _build_embedding_map, _compute_delta_matrix_for_pairs
from classifier import run_classifier_experiment

plt.rcParams.update(PLOT_STYLE_PARAMS)

FEATURE_TABLE = Path(CLASSIFIER_FEATURE_TABLE)
OUTPUT_DIR    = Path(CLASSIFICATION_RESULTS_DIR)
PLOTS_DIR     = Path(CLASSIFICATION_PLOTS_DIR)


# Helpers 
def _select_columns(df: pd.DataFrame, pattern: str) -> list[str]:
    return sorted([c for c in df.columns if pattern in c])


def _safe_name(text: str) -> str:
    """Convert config name to filesystem-safe identifier."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _get_feature_family(feature: str) -> str:
    """Categorize a feature name into its semantic family."""
    if "vocal" in feature:
        return "Vocal Metadata"
    if feature.startswith("clews_") and "distance" in feature:
        return "CLEWS Distances"
    if feature.startswith("wealy_") and "distance" in feature:
        return "WEALY Distances"
    if feature.startswith("clews_") and any(
        x in feature for x in ["delta_", "stable_", "volatile_", "active_"]
    ):
        return "CLEWS Delta Summaries"
    if feature.startswith("wealy_") and any(
        x in feature for x in ["delta_", "stable_", "volatile_", "active_"]
    ):
        return "WEALY Delta Summaries"
    if feature.startswith("clews_topdim_"):
        return "CLEWS Raw Top-K Dims"
    return "Other"


# Triple-Tier Error Analysis 
def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Standard binary classification metrics (same as binary_classification.py)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Precision":  precision_score(y_true, y_pred, zero_division=0),
        "Recall":     recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":   f1_score(y_true, y_pred, zero_division=0),
        "F0.5-Score": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def run_triple_tier_analysis(
    oof_df:     pd.DataFrame,
    config_name: str,
    output_dir: Path,
) -> None:
    """
    Perform the same three-level error analysis as binary_classification.py.

    Level 1 — Broad category metrics
    Level 2 — Granular mod-type metrics
    Level 3 — False Positive tier breakdown
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(config_name)

    # Level 1: Broad category metrics 
    positive_categories = sorted(
        oof_df.loc[oof_df["y_true"] == 1, "category_grouped"].unique()
    )
    metrics_list: list[dict] = []

    for cat in positive_categories:
        mask   = (oof_df["category_grouped"] == cat) | (oof_df["y_true"] == 0)
        df_sub = oof_df[mask]
        m      = _compute_binary_metrics(df_sub["y_true"].values, df_sub["y_pred"].values)
        metrics_list.append({
            "Category":      cat,
            "Total_Queries": m["TP"] + m["FN"],
            **m,
        })

    m_overall = _compute_binary_metrics(oof_df["y_true"].values, oof_df["y_pred"].values)
    metrics_list.append({
        "Category":      "OVERALL",
        "Total_Queries": m_overall["TP"] + m_overall["FN"],
        **m_overall,
    })

    df_broad = pd.DataFrame(metrics_list)

    # Print broad table
    print(f"\n{'=' * 115}")
    print(f"  TRIPLE-TIER ANALYSIS — {config_name.upper()} (Supervised / OOF)")
    print(f"{'=' * 115}")
    print(
        f"  {'Category':<30} | {'Precision':>9} | {'Recall':>8} | "
        f"{'F1':>7} | {'F0.5':>8} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>6}"
    )
    print(f"  {'-' * 110}")
    for _, row in df_broad.iterrows():
        prefix = "► " if row["Category"] == "OVERALL" else "  "
        print(
            f"  {prefix}{row['Category']:<28} | {row['Precision']:>8.1%} | "
            f"{row['Recall']:>7.1%} | {row['F1-Score']:>6.1%} | "
            f"{row['F0.5-Score']:>7.1%} | {row['TP']:>5} | "
            f"{row['FP']:>5} | {row['FN']:>5} | {row['TN']:>6}"
        )
    print(f"  {'=' * 110}\n")

    broad_path = output_dir / f"{safe}_broad_metrics.csv"
    df_broad.to_csv(broad_path, index=False)
    print(f"  Broad metrics saved → {broad_path}")

    # Level 2: Granular mod-type metrics 
    mod_col = "final_mod_type" if "final_mod_type" in oof_df.columns else None
    if mod_col:
        detailed_list: list[dict] = []
        for mod in sorted(oof_df.loc[oof_df["y_true"] == 1, mod_col].unique()):
            mask   = (oof_df[mod_col] == mod) | (oof_df["y_true"] == 0)
            df_sub = oof_df[mask]
            if len(df_sub["y_true"].unique()) < 2:
                continue
            m = _compute_binary_metrics(df_sub["y_true"].values, df_sub["y_pred"].values)
            detailed_list.append({"Modification_Type": mod, **m})

        df_detailed = pd.DataFrame(detailed_list)
        detail_path = output_dir / f"{safe}_detailed_metrics.csv"
        df_detailed.to_csv(detail_path, index=False)
        print(f"  Detailed metrics saved → {detail_path}")

    # Level 3: False Positive tier breakdown 
    if "negative_tier" in oof_df.columns:
        fp_df = oof_df[(oof_df["y_true"] == 0) & (oof_df["y_pred"] == 1)]
        if not fp_df.empty:
            fp_breakdown = (
                fp_df
                .groupby(["category_grouped", "negative_tier"])
                .size()
                .reset_index(name="FP_Count")
            )
            fp_path = output_dir / f"{safe}_fp_tier_breakdown.csv"
            fp_breakdown.to_csv(fp_path, index=False)
            print(f"  FP tier breakdown saved → {fp_path}")
        else:
            print("  No False Positives in OOF predictions — perfect separation!")
    else:
        print("  'negative_tier' not present in OOF data; skipping FP tier analysis.")


# Feature Importance 
def compute_feature_importance(
    X:              np.ndarray,
    y:              np.ndarray,
    groups:         np.ndarray,
    feature_names:  list[str],
    config_name:    str,
    output_dir:     Path,
    plots_dir:      Path,
    n_repeats:      int = 5,
    max_eval_samples: int = 20_000,
) -> None:
    """
    Permutation feature importance on a grouped held-out fold.
    """
    print(f"\n  Computing permutation importance: {config_name}")

    X = np.ascontiguousarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    sgkf = StratifiedGroupKFold(n_splits=NUM_K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    if len(test_idx) > max_eval_samples:
        rng      = np.random.default_rng(RANDOM_STATE)
        test_idx = rng.choice(test_idx, size=max_eval_samples, replace=False)

    from classifier import build_classifier
    clf = build_classifier(y[train_idx], random_state=RANDOM_STATE)
    clf.fit(X[train_idx], y[train_idx])

    scorer = make_scorer(fbeta_score, beta=BETA, zero_division=0)
    perm   = permutation_importance(
        clf, X[test_idx], y[test_idx],
        scoring=scorer, n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    df_imp = pd.DataFrame({
        "feature":          feature_names,
        "importance_mean":  perm.importances_mean,
        "importance_std":   perm.importances_std,
    })
    df_imp["family"] = df_imp["feature"].apply(_get_feature_family)
    df_imp = df_imp.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    safe = _safe_name(config_name)

    imp_csv = output_dir / f"{safe}_feature_importance.csv"
    df_imp.to_csv(imp_csv, index=False)
    print(f"    CSV → {imp_csv}")

    # Family-level aggregation
    df_family = (
        df_imp.groupby("family", as_index=False)
        .agg(
            mean_importance=("importance_mean", "mean"),
            total_importance=("importance_mean", "sum"),
            n_features=("feature", "count"),
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    fam_csv = output_dir / f"{safe}_feature_importance_family.csv"
    df_family.to_csv(fam_csv, index=False)
    print(f"    Family CSV → {fam_csv}")

    # Top-20 feature plot
    df_top = df_imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(df_top))))
    ax.barh(
        df_top["feature"], df_top["importance_mean"],
        xerr=df_top["importance_std"],
        color="#4CAF50", edgecolor="black", alpha=0.85, capsize=3,
    )
    ax.set_xlabel(f"Permutation Importance (Δ F{BETA})", fontsize=10)
    ax.set_title(f"{config_name} — Top-20 Feature Importances", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    feat_plot = plots_dir / f"{safe}_feature_importance.pdf"
    fig.savefig(feat_plot, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot → {feat_plot}")

    # Family importance plot
    df_fp = df_family.sort_values("mean_importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        df_fp["family"], df_fp["mean_importance"],
        color="#2196F3", edgecolor="black", alpha=0.85,
    )
    ax.set_xlabel(f"Mean Permutation Importance (Δ F{BETA})", fontsize=10)
    ax.set_title(f"{config_name} — Feature Family Importance", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fam_plot = plots_dir / f"{safe}_feature_importance_family.pdf"
    fig.savefig(fam_plot, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot → {fam_plot}")


# Configuration Builder 
def build_features_for_config(
    config_name: str,
    df: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Build feature matrix and feature names for a given configuration.

    Args:
        config_name: One of the predefined configuration identifiers.
        df: The full feature table DataFrame.
        y: Binary labels (used for CLEWS ranking in hybrid configs).

    Returns:
        (X, feature_names)
    """
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
                 "vocal_valid_ori",  "vocal_valid_mod"}
        and not df[c].isna().all()
    ]

    all_engineered_no_vocal = (
        clews_dist_cols + wealy_dist_cols
        + clews_delta_cols + wealy_delta_cols
    )
    all_engineered = all_engineered_no_vocal + vocal_cols

    if config_name == "engineered_no_vocals":
        return df[all_engineered_no_vocal].values, all_engineered_no_vocal

    elif config_name == "engineered_with_vocals":
        return df[all_engineered].values, all_engineered

    elif config_name.startswith("hybrid_top"):
        # Extract K from config name (e.g., "hybrid_top512" → K=512)
        k_str = config_name.replace("hybrid_top", "")
        try:
            k = int(k_str)
        except ValueError:
            raise ValueError(f"Invalid hybrid config name: {config_name}")

        # Build CLEWS delta matrix and rank
        print(f"    Building CLEWS delta matrix for Top-{k}...")
        emb_map = _build_embedding_map(EMBEDDING_PATHS["CLEWS"])
        delta_valid, valid_mask = _compute_delta_matrix_for_pairs(df, emb_map)
        del emb_map

        ndim         = delta_valid.shape[1]
        delta_matrix = np.zeros((len(df), ndim), dtype=np.float32)
        delta_matrix[valid_mask] = delta_valid.astype(np.float32)
        del delta_valid, valid_mask

        pos_mask    = y == 1
        mean_shifts = np.mean(np.abs(delta_matrix[pos_mask]), axis=0)
        ranked_idx  = np.argsort(mean_shifts)[::-1]

        # Combine base + top-K
        base_X     = df[all_engineered_no_vocal].values
        top_k_dims = delta_matrix[:, ranked_idx[:k]]
        X_hybrid   = np.hstack([base_X, top_k_dims])

        feature_names = all_engineered_no_vocal + [
            f"clews_topdim_{int(i)}" for i in ranked_idx[:k]
        ]

        return X_hybrid, feature_names

    else:
        raise ValueError(
            f"Unknown configuration: {config_name}. "
            f"Valid options: engineered_no_vocals, engineered_with_vocals, "
            f"hybrid_top256, hybrid_top512, hybrid_top1024"
        )


# Per-Config Evaluation 
def evaluate_config(
    config_name: str,
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    df_meta: pd.DataFrame,
) -> None:
    """
    Run full evaluation pipeline for a single configuration.
    """
    print("\n" + "=" * 70)
    print(f"EVALUATING CONFIGURATION: {config_name.upper()}")
    print("=" * 70)

    # Build features
    print(f"\n[1/4] Building features for: {config_name}...")
    X, feature_names = build_features_for_config(config_name, df, y)
    print(f"  Feature matrix: {X.shape}  ({len(feature_names)} features)")

    # Run CV and collect OOF
    print(f"\n[2/4] Running cross-validation with XGBoost...")
    res = run_classifier_experiment(
        X=X, y=y, groups=groups,
        experiment_name=config_name,
        df_meta=df_meta,
    )
    print(
        f"\n  ► Overall Performance: "
        f"F0.5={res['f05']:.4f}  Prec={res['precision']:.4f}  "
        f"Rec={res['recall']:.4f}  F1={res['f1']:.4f}"
    )

    oof_df = res["oof_df"]

    # Triple-tier analysis
    print(f"\n[3/4] Running triple-tier error analysis...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_triple_tier_analysis(oof_df, config_name, OUTPUT_DIR)

    # Feature importance
    print(f"\n[4/4] Computing permutation feature importance...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    compute_feature_importance(
        X=X, y=y, groups=groups,
        feature_names=feature_names,
        config_name=config_name,
        output_dir=OUTPUT_DIR,
        plots_dir=PLOTS_DIR,
    )

    print(f"\n{'=' * 70}")
    print(f"Evaluation complete for: {config_name}")
    print(f"{'=' * 70}\n")


# Main 
def main() -> None:
    print("=" * 70)
    print("SELECTED MODEL EVALUATION — DEEP DIAGNOSTIC ANALYSIS")
    print("=" * 70)

    # Check which configs are enabled
    enabled_configs = [name for name, enabled in CONFIGS_TO_EVALUATE.items() if enabled]
    
    if not enabled_configs:
        print("\n[WARNING] No configurations enabled in CONFIGS_TO_EVALUATE.")
        print("Edit the script to enable at least one configuration.")
        return

    print(f"\nConfigurations to evaluate: {', '.join(enabled_configs)}")

    # Load feature table (once)
    if not FEATURE_TABLE.exists():
        print(f"\n[ERROR] Feature table not found: {FEATURE_TABLE}")
        return

    print(f"\nLoading feature table from {FEATURE_TABLE}...")
    df     = pd.read_parquet(FEATURE_TABLE)
    y      = df["is_plagiarised"].astype(int).values
    groups = df["filename_ori"].values
    print(f"  Loaded {len(df):,} pairs  ({y.sum():,} pos / {(y == 0).sum():,} neg)")

    # Prepare metadata (once)
    df_meta = df[["final_mod_type"]].copy()
    df_meta["y_true"]           = df_meta["final_mod_type"].apply(get_ground_truth_label)
    df_meta["clean_mod_type"]   = df_meta["final_mod_type"].apply(clean_mod_type)
    df_meta["category_grouped"] = df_meta["clean_mod_type"].apply(categorize_modification)
    df_meta.loc[df_meta["y_true"] == 0, "category_grouped"] = "Negative Pairs"
    if "negative_tier" in df.columns:
        df_meta["negative_tier"] = df["negative_tier"].values

    # Evaluate each enabled config
    for config_name in enabled_configs:
        evaluate_config(config_name, df, y, groups, df_meta)

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS COMPLETE")
    print(f"  Results → {OUTPUT_DIR}/")
    print(f"  Plots   → {PLOTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()