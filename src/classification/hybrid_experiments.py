"""
Hybrid Feature Experiments: Engineered + Top-K CLEWS Dimensions.

This script performs a targeted follow-up to the ablation study by
combining the best-performing engineered feature set with ranked
top-K actual CLEWS embedding dimensions.

Research Question
-----------------
Do engineered summary features (distances + delta statistics) and
raw informative CLEWS dimensions provide complementary information
when used together?

Rationale
---------
Engineered features offer condensed, interpretable, semantic-level
signals. Raw CLEWS dimensions (especially the top-K most informative)
may carry granular detail lost during summarization. This experiment
tests whether their fusion yields performance gains beyond either
approach alone.

Protocol
--------
1. Load CLEWS embeddings and build full delta matrix.
2. Rank dimensions by mean absolute shift on positive pairs.
3. Define base engineered feature set (manually selected).
4. Construct hybrid feature matrices: base + top-K for K ∈ {256, 512, 1024}.
5. Run the same CV protocol as ablation (StratifiedGroupKFold + XGBoost).
6. Export results and comparison plot.

Inputs
------
    - results/classification/classifier_features.parquet
    - data/clews_embeddings.parquet
    - results/classification/ablation_results.csv (for comparison context)

Outputs
-------
    - results/classification/hybrid_results.csv
    - plots/classification/hybrid_f05_comparison.pdf
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
    PLOT_DPI,
    PLOT_STYLE_PARAMS,
    CLASSIFIER_FEATURE_TABLE,
    CLASSIFICATION_RESULTS_DIR,
    CLASSIFICATION_PLOTS_DIR,
)
from utils.classifier_features import _build_embedding_map, _compute_delta_matrix_for_pairs
from classifier import (
    run_classifier_experiment,
    load_threshold_baselines,
    print_experiment_summary,
)

plt.rcParams.update(PLOT_STYLE_PARAMS)

FEATURE_TABLE   = Path(CLASSIFIER_FEATURE_TABLE)
ABLATION_RESULTS = Path(CLASSIFICATION_RESULTS_DIR) / "ablation_results.csv"
OUTPUT_DIR      = Path(CLASSIFICATION_RESULTS_DIR)
PLOTS_DIR       = Path(CLASSIFICATION_PLOTS_DIR)

# K values to test in hybrid experiments
HYBRID_K_VALUES = [256, 512, 1024]


# Helpers 
def _select_columns(df: pd.DataFrame, pattern: str) -> list[str]:
    return sorted([c for c in df.columns if pattern in c])


# Plotting 
def _plot_hybrid_comparison(
    df_hybrid:    pd.DataFrame,
    df_baselines: pd.DataFrame,
    df_ablation:  pd.DataFrame,
    output_path:  Path,
) -> None:
    """
    Plot comparison: threshold baselines + best ablation configs + hybrid results.
    """
    rows = []

    # Threshold baselines
    for _, r in df_baselines.iterrows():
        rows.append({
            "Method":    r["experiment_name"],
            "F0.5":      r["f05"],
            "Precision": r["precision"],
            "Recall":    r["recall"],
        })

    # Selected ablation results for context
    ablation_selection = [
        "7. All Engineered (No Vocal)",
        "8. All Engineered (Vocal)",
        "10. CLEWS Full Δ (1024D)",
    ]
    for name in ablation_selection:
        match = df_ablation[df_ablation["experiment_name"] == name]
        if not match.empty:
            r = match.iloc[0]
            rows.append({
                "Method":    name,
                "F0.5":      r["f05"],
                "Precision": r["precision"],
                "Recall":    r["recall"],
            })

    # Hybrid results
    for _, r in df_hybrid.iterrows():
        rows.append({
            "Method":    r["experiment_name"],
            "F0.5":      r["f05"],
            "Precision": r["precision"],
            "Recall":    r["recall"],
        })

    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 7))
    x     = np.arange(len(df_plot))
    width = 0.25

    bars = [
        ax.bar(x - width, df_plot["F0.5"],      width, label="F0.5",      color="#2196F3", edgecolor="white"),
        ax.bar(x,         df_plot["Precision"],  width, label="Precision",  color="#4CAF50", edgecolor="white"),
        ax.bar(x + width, df_plot["Recall"],     width, label="Recall",     color="#FF9800", edgecolor="white"),
    ]

    for bar_group in bars:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Method"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Hybrid Experiments: Engineered Features + Top-K CLEWS Dimensions",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {output_path}")


# Main 
def main() -> None:
    print("=" * 70)
    print("HYBRID FEATURE EXPERIMENTS — ENGINEERED + TOP-K CLEWS")
    print("=" * 70)

    # Load feature table 
    if not FEATURE_TABLE.exists():
        print(f"[ERROR] Feature table not found: {FEATURE_TABLE}")
        print("Run classifier_features.py first.")
        return

    print(f"\nLoading feature table from {FEATURE_TABLE}...")
    df     = pd.read_parquet(FEATURE_TABLE)
    y      = df["is_plagiarised"].astype(int).values
    groups = df["filename_ori"].values
    print(f"  Loaded {len(df):,} pairs  ({y.sum():,} pos / {(y == 0).sum():,} neg)")

    # Define base engineered feature set 
    print("\n[1/4] Defining base engineered feature set...")

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

    # Base configuration: All Engineered (No Vocals)
    base_engineered = (
        clews_dist_cols + wealy_dist_cols
        + clews_delta_cols + wealy_delta_cols
    )
    print(f"  Base: All Engineered (No Vocals) — {len(base_engineered)} features")

    # Build CLEWS delta matrix and rank dimensions 
    print("\n[2/4] Building CLEWS delta matrix and ranking dimensions...")
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

    print(f"  CLEWS delta matrix: {delta_matrix.shape}")
    print(f"  Dimensions ranked by mean absolute shift on {pos_mask.sum()} positive pairs")

    # Run hybrid experiments 
    print("\n[3/4] Running hybrid experiments...")
    print("─" * 70)

    base_X = df[base_engineered].values.astype(np.float32)
    base_X = np.nan_to_num(base_X, nan=0.0, posinf=0.0, neginf=0.0)

    hybrid_results: list[dict] = []

    for k in HYBRID_K_VALUES:
        exp_name = f"Hybrid: Engineered (No Vocals) + Top-{k} CLEWS"
        print(f"\n  Running: {exp_name}")

        # Combine base + top-K
        top_k_dims = delta_matrix[:, ranked_idx[:k]]
        X_hybrid   = np.hstack([base_X, top_k_dims]).astype(np.float32)

        res = run_classifier_experiment(
            X=X_hybrid, y=y, groups=groups,
            experiment_name=exp_name,
        )
        print_experiment_summary(res)

        hybrid_results.append({
            "experiment_name": exp_name,
            "classifier":      res["classifier"],
            "n_features":      res["n_features"],
            "f05":             round(res["f05"],            4),
            "f1":              round(res["f1"],             4),
            "precision":       round(res["precision"],      4),
            "recall":          round(res["recall"],         4),
            "accuracy":        round(res["accuracy"],       4),
            "mean_threshold":  round(res["mean_threshold"], 4),
        })

    df_hybrid = pd.DataFrame(hybrid_results)

    # Load ablation results for context 
    print("\n[4/4] Loading ablation results for comparison...")
    df_ablation = pd.DataFrame()
    if ABLATION_RESULTS.exists():
        df_ablation = pd.read_csv(ABLATION_RESULTS)
        print(f"  Loaded {len(df_ablation)} ablation experiments")
    else:
        logger.warning("Ablation results not found. Comparison plot will be incomplete.")

    # Load threshold baselines 
    df_baselines = load_threshold_baselines()

    # Print comparison table 
    print(f"\n{'=' * 100}")
    print(
        f"  {'Experiment':<50} | {'Feats':>6} | {'F0.5':>7} | "
        f"{'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'Acc':>7}"
    )
    print(f"  {'─' * 95}")

    # Print threshold baselines
    if not df_baselines.empty:
        for _, r in df_baselines.iterrows():
            print(
                f"  {r['experiment_name']:<50} | {'–':>6} | {r['f05']:>7.4f} | "
                f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | "
                f"{r['f1']:>7.4f} | {r['accuracy']:>7.4f}"
            )
        print(f"  {'─' * 95}")

    # Print selected ablation results for context
    if not df_ablation.empty:
        ablation_selection = [
            "7. All Engineered (No Vocal)",
            "8. All Engineered (Vocal)",
            "10. CLEWS Full Δ (1024D)",
        ]
        for name in ablation_selection:
            match = df_ablation[df_ablation["experiment_name"] == name]
            if not match.empty:
                r = match.iloc[0]
                print(
                    f"  {name:<50} | {r['n_features']:>6} | {r['f05']:>7.4f} | "
                    f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | "
                    f"{r['f1']:>7.4f} | {r['accuracy']:>7.4f}"
                )
        print(f"  {'─' * 95}")

    # Print hybrid results
    for _, r in df_hybrid.iterrows():
        print(
            f"  {r['experiment_name']:<50} | {r['n_features']:>6} | {r['f05']:>7.4f} | "
            f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | "
            f"{r['f1']:>7.4f} | {r['accuracy']:>7.4f}"
        )

    print(f"{'=' * 100}")

    # Save outputs 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "hybrid_results.csv"
    df_hybrid.to_csv(results_path, index=False)
    print(f"\n  Hybrid results → {results_path}")

    _plot_hybrid_comparison(
        df_hybrid, df_baselines, df_ablation,
        PLOTS_DIR / "hybrid_f05_comparison.pdf",
    )

    print(f"\n  All outputs → {OUTPUT_DIR}/  and  {PLOTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()