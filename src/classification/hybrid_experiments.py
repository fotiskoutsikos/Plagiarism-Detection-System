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

Statistical Rigor
-----------------
All experiments are repeated N_SEEDS times with independent random seeds
to produce mean ± std and 95% confidence intervals, making comparisons
between hybrid configurations statistically defensible.

Set N_SEEDS = 1 for a quick exploratory run; use N_SEEDS = 10 for
publication-grade results.

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

FEATURE_TABLE    = Path(CLASSIFIER_FEATURE_TABLE)
ABLATION_RESULTS = Path(CLASSIFICATION_RESULTS_DIR) / "ablation_results.csv"
OUTPUT_DIR       = Path(CLASSIFICATION_RESULTS_DIR)
PLOTS_DIR        = Path(CLASSIFICATION_PLOTS_DIR)

# K values to test in hybrid experiments
HYBRID_K_VALUES = [256, 512, 1024]

# Statistical rigor 
# Number of independent random seeds for CI estimation.
# Use N_SEEDS = 1 for a quick exploratory run.
# Use N_SEEDS = 10 for publication-grade confidence intervals.
N_SEEDS = 10


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
    Bar chart: threshold baselines + best ablation configs + hybrid results.
    F0.5, Precision and Recall bars; F0.5 carries 95% CI error bars.
    """
    rows = []

    # Threshold baselines (no CI)
    for _, r in df_baselines.iterrows():
        rows.append({
            "Method":    r["experiment_name"],
            "F0.5":      r["f05"],
            "F05_CI95":  0.0,
            "Precision": r["precision"],
            "Prec_CI95": 0.0,
            "Recall":    r["recall"],
            "Rec_CI95":  0.0,
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
                "F05_CI95":  r.get("f05_ci95", 0.0),
                "Precision": r["precision"],
                "Prec_CI95": r.get("precision_ci95", 0.0),
                "Recall":    r["recall"],
                "Rec_CI95":  r.get("recall_ci95", 0.0),
            })

    # Hybrid results (with CI)
    for _, r in df_hybrid.iterrows():
        rows.append({
            "Method":    r["experiment_name"],
            "F0.5":      r["f05"],
            "F05_CI95":  r.get("f05_ci95", 0.0),
            "Precision": r["precision"],
            "Prec_CI95": r.get("precision_ci95", 0.0),
            "Recall":    r["recall"],
            "Rec_CI95":  r.get("recall_ci95", 0.0),
        })

    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 7))
    x     = np.arange(len(df_plot))
    width = 0.25

    ax.bar(
        x - width, df_plot["F0.5"], width,
        label="F0.5", color="#2196F3", edgecolor="white",
        yerr=df_plot["F05_CI95"].values,
        error_kw=dict(elinewidth=1.2, ecolor="#0d47a1", capsize=3, capthick=1.2),
    )
    ax.bar(
        x,          df_plot["Precision"], width,
        label="Precision", color="#4CAF50", edgecolor="white",
        yerr=df_plot["Prec_CI95"].values,
        error_kw=dict(elinewidth=1.2, ecolor="#1b5e20", capsize=3, capthick=1.2),
    )
    ax.bar(
        x + width,  df_plot["Recall"], width,
        label="Recall", color="#FF9800", edgecolor="white",
        yerr=df_plot["Rec_CI95"].values,
        error_kw=dict(elinewidth=1.2, ecolor="#e65100", capsize=3, capthick=1.2),
    )

    # Value labels
    for col, offset, ci_col in [
        ("F0.5",      -width, "F05_CI95"),
        ("Precision",  0,     "Prec_CI95"),
        ("Recall",     width, "Rec_CI95"),
    ]:
        for i, (val, ci) in enumerate(zip(df_plot[col], df_plot[ci_col])):
            if val > 0.01:
                ax.text(
                    i + offset + width / 2, val + ci + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Method"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Hybrid Experiments: Engineered + Top-K CLEWS Dimensions "
        f"({N_SEEDS} seeds, ±95% CI)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 1.15)
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
    print(f"Statistical rigor: {N_SEEDS} seed(s) per experiment")
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
    print(f"\n[3/4] Running hybrid experiments ({N_SEEDS} seed(s) each)...")
    print("─" * 70)

    base_X = df[base_engineered].values.astype(np.float32)
    base_X = np.nan_to_num(base_X, nan=0.0, posinf=0.0, neginf=0.0)

    hybrid_results: list[dict] = []

    for k in HYBRID_K_VALUES:
        exp_name = f"Hybrid: Engineered (No Vocals) + Top-{k} CLEWS"
        print(f"\n  Running: {exp_name}")

        top_k_dims = delta_matrix[:, ranked_idx[:k]]
        X_hybrid   = np.hstack([base_X, top_k_dims]).astype(np.float32)

        res = run_classifier_experiment(
            X=X_hybrid, y=y, groups=groups,
            experiment_name=exp_name,
            n_seeds=N_SEEDS,
        )
        print_experiment_summary(res)

        hybrid_results.append({
            "experiment_name":      exp_name,
            "classifier":           res["classifier"],
            "n_features":           res["n_features"],
            "n_seeds":              res.get("n_seeds", 1),
            "f05":                  round(res["f05"],                          4),
            "f05_std":              round(res.get("f05_std",       0.0),        4),
            "f05_ci95":             round(res.get("f05_ci95",      0.0),        4),
            "f1":                   round(res["f1"],                           4),
            "f1_ci95":              round(res.get("f1_ci95",       0.0),        4),
            "precision":            round(res["precision"],                    4),
            "precision_ci95":       round(res.get("precision_ci95", 0.0),      4),
            "recall":               round(res["recall"],                       4),
            "recall_ci95":          round(res.get("recall_ci95",   0.0),        4),
            "accuracy":             round(res["accuracy"],                     4),
            "mean_threshold":       round(res["mean_threshold"],               4),
            "mean_train_time_sec":  round(res.get("mean_train_time_sec", 0.0),  3),
            "mean_infer_time_ms":   round(res.get("mean_infer_time_ms",  0.0),  4),
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
    print(f"\n{'=' * 115}")
    print(
        f"  {'Experiment':<50} | {'Feats':>6} | {'F0.5':>7} | {'CI95':>7} | "
        f"{'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'Acc':>7} | {'Train(s)':>9}"
    )
    print(f"  {'─' * 110}")

    # Threshold baselines
    if not df_baselines.empty:
        for _, r in df_baselines.iterrows():
            print(
                f"  {r['experiment_name']:<50} | {'–':>6} | {r['f05']:>7.4f} | {'–':>7} | "
                f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | "
                f"{r['f1']:>7.4f} | {r['accuracy']:>7.4f} | {'–':>9}"
            )
        print(f"  {'─' * 110}")

    # Selected ablation reference experiments
    if not df_ablation.empty:
        ablation_selection = [
            "7. All Engineered (No Vocal)",
            "8. All Engineered (Vocal)",
            "10. CLEWS Full Δ (1024D)",
        ]
        for name in ablation_selection:
            match = df_ablation[df_ablation["experiment_name"] == name]
            if not match.empty:
                r      = match.iloc[0]
                ci_str = f"±{r['f05_ci95']:.4f}" if r.get("f05_ci95", 0) > 0 else "–"
                t_str  = f"{r['mean_train_time_sec']:.2f}" if r.get("mean_train_time_sec", 0) > 0 else "–"
                print(
                    f"  {name:<50} | {r['n_features']:>6} | {r['f05']:>7.4f} | {ci_str:>7} | "
                    f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | "
                    f"{r['f1']:>7.4f} | {r['accuracy']:>7.4f} | {t_str:>9}"
                )
        print(f"  {'─' * 110}")

    # Hybrid results
    for _, r in df_hybrid.iterrows():
        ci_str = f"±{r['f05_ci95']:.4f}" if r.get("f05_ci95", 0) > 0 else "–"
        t_str  = f"{r['mean_train_time_sec']:.2f}" if r.get("mean_train_time_sec", 0) > 0 else "–"
        print(
            f"  {r['experiment_name']:<50} | {r['n_features']:>6} | {r['f05']:>7.4f} | {ci_str:>7} | "
            f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | "
            f"{r['f1']:>7.4f} | {r['accuracy']:>7.4f} | {t_str:>9}"
        )

    print(f"{'=' * 115}")

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
