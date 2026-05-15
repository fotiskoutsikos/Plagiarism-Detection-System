"""
Feature Ablation Study for Binary Plagiarism Classification.

Orchestrates multiple classification experiments using the modular
classification engine (classification.py) to answer:

    Q1: Do engineered features (distances + summaries) beat raw embeddings?
    Q2: Can 30 XAI-selected dimensions match the full embedding space?
    Q3: Does XAI feature selection beat blind PCA dimensionality reduction?
    Q4: How much "information" does each reduction method retain?

All experiments use the SAME classifier (Logistic Regression), the SAME
CV protocol (StratifiedGroupKFold), and the SAME threshold optimization
(train-only F0.5). Only the input features change.

Inputs:
    data/classifier_features.parquet
    results/explainability/{model}_topk_dimensions.csv
    data/clews_embeddings.parquet
    data/wealy_embeddings.parquet

Outputs:
    results/classification/ablation_results.csv
    results/classification/ablation_information_retention.csv
    plots/classification/ablation_f05_comparison.pdf
    plots/classification/ablation_information_retention.pdf
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
    EMBEDDING_PATHS, OUTPUT_DIRS, PLOT_DPI, PLOT_STYLE_PARAMS,
    CLASSIFIER_FEATURE_TABLE, CLASSIFICATION_RESULTS_DIR, CLASSIFICATION_PLOTS_DIR,
)

from utils.classifier_features import _build_embedding_map, _compute_delta_matrix_for_pairs

from classification import (
    run_classifier_experiment,
    load_threshold_baselines,
    print_experiment_summary,
)

plt.rcParams.update(PLOT_STYLE_PARAMS)

# Paths 
FEATURE_TABLE = Path(CLASSIFIER_FEATURE_TABLE)
OUTPUT_DIR    = Path(CLASSIFICATION_RESULTS_DIR)
PLOTS_DIR     = Path(CLASSIFICATION_PLOTS_DIR)


# Helpers 
def _select_columns(df: pd.DataFrame, pattern: str) -> list[str]:
    """Select column names matching a substring pattern."""
    return sorted([c for c in df.columns if pattern in c])


def _build_full_delta_matrix(
    df: pd.DataFrame,
    parquet_path: str,
    model_name: str,
) -> np.ndarray:
    """
    Build a full (N_pairs, D) delta matrix using the EXACT same logic
    as classifier_features.py. Rows with missing embeddings get zeros.

    Returns float32 array aligned with df index.
    """
    print(f"  Loading {model_name} embeddings...")
    emb_map = _build_embedding_map(parquet_path)

    print(f"  Computing {model_name} delta matrix...")
    delta_valid, valid_mask = _compute_delta_matrix_for_pairs(df, emb_map)

    n_valid = int(valid_mask.sum())
    n_miss  = int((~valid_mask).sum())

    if delta_valid.size == 0:
        logger.warning(f"  [{model_name}] No valid deltas. Returning zeros.")
        return np.zeros((len(df), 1), dtype=np.float32)

    ndim = delta_valid.shape[1]
    print(f"  [{model_name}] Delta: {n_valid}/{len(df)} valid, {n_miss} missing, dim={ndim}")

    # Build full matrix with zeros for missing pairs
    delta_full = np.zeros((len(df), ndim), dtype=np.float32)
    delta_full[valid_mask] = delta_valid.astype(np.float32)

    return delta_full




# Plotting 
def _plot_ablation_comparison(
    df_results: pd.DataFrame,
    df_baselines: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar chart comparing F0.5, Precision, Recall across all experiments."""
    rows = []
    if not df_baselines.empty:
        for _, r in df_baselines.iterrows():
            rows.append({
                "Method": r["experiment_name"],
                "F0.5": r["f05"],
                "Precision": r["precision"],
                "Recall": r["recall"],
            })

    for _, r in df_results.iterrows():
        rows.append({
            "Method": r["experiment_name"],
            "F0.5": r["f05"],
            "Precision": r["precision"],
            "Recall": r["recall"],
        })

    df_plot = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df_plot))
    width = 0.25

    bars_f05  = ax.bar(x - width, df_plot["F0.5"],      width, label="F0.5",      color="#2196F3", edgecolor="white")
    bars_prec = ax.bar(x,         df_plot["Precision"],  width, label="Precision",  color="#4CAF50", edgecolor="white")
    bars_rec  = ax.bar(x + width, df_plot["Recall"],     width, label="Recall",     color="#FF9800", edgecolor="white")

    for bars in [bars_f05, bars_prec, bars_rec]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Method"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Feature Ablation: Classification Performance Comparison",
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
    print("FEATURE ABLATION STUDY — BINARY PLAGIARISM CLASSIFICATION")
    print("=" * 70)

    # Load Feature Table
    if not FEATURE_TABLE.exists():
        print(f"[ERROR] Feature table not found: {FEATURE_TABLE}")
        print("Run classifier_features.py first.")
        return

    print(f"\nLoading feature table from {FEATURE_TABLE}...")
    df = pd.read_parquet(FEATURE_TABLE)
    y = df["is_plagiarised"].astype(int).values
    groups = df["filename_ori"].values
    print(f"  Loaded {len(df)} pairs ({y.sum()} positives, {(y == 0).sum()} negatives)")

    # Identify Feature Groups
    clews_dist_cols  = [c for c in _select_columns(df, "clews_") if "distance" in c]
    wealy_dist_cols  = [c for c in _select_columns(df, "wealy_") if "distance" in c]
    all_dist_cols    = clews_dist_cols + wealy_dist_cols

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
    all_delta_cols = clews_delta_cols + wealy_delta_cols

    vocal_cols = [
        c for c in df.columns
        if c in {"pair_vocal_valid", "vocal_ratio_ori", "vocal_ratio_mod",
                  "vocal_valid_ori", "vocal_valid_mod"}
        and not df[c].isna().all()
    ]

    clews_xai_cols = _select_columns(df, "clews_xai_dim_")
    wealy_xai_cols = _select_columns(df, "wealy_xai_dim_")

    all_engineered = all_dist_cols + all_delta_cols + vocal_cols

    # Define Phase 1 Experiments
    experiments = [
        ("1. CLEWS Distances",       clews_dist_cols),
        ("2. WEALY Distances",       wealy_dist_cols),
        ("3. All Distances",         all_dist_cols),
        ("4. CLEWS Delta Summaries", clews_delta_cols),
        ("5. WEALY Delta Summaries", wealy_delta_cols),
        ("6. All Delta Summaries",   all_delta_cols),
        ("7. All Engineered",        all_engineered),
        ("8. Engineered - Vocal",    all_dist_cols + all_delta_cols),
        ("9. Vocal Only",            vocal_cols),
        ("10. CLEWS XAI Top-30",     clews_xai_cols),
        ("11. WEALY XAI Top-30",     wealy_xai_cols),
    ]

    # Run Phase 1
    print("\n" + "─" * 70)
    print("PHASE 1: ENGINEERED + XAI FEATURE EXPERIMENTS")
    print("─" * 70)

    all_summaries = []
    for name, cols in experiments:
        if not cols:
            print(f"\n  ⚠ {name}: No features found. Skipping.")
            continue

        print(f"\n  Running: {name} ({len(cols)} features)")
        result = run_classifier_experiment(
            X=df[cols], y=y, groups=groups,
            experiment_name=name,
        )
        print_experiment_summary(result)
        all_summaries.append(result)

    # Run Phase 2: Raw Embedding Experiments
    print("\n" + "─" * 70)
    print("PHASE 2: RAW EMBEDDING EXPERIMENTS")
    print("─" * 70)

    exp_counter = 12
    embedding_deltas = {}  # Store for later concatenation

    for model_name, parquet_path in EMBEDDING_PATHS.items():
        if not Path(parquet_path).exists():
            print(f"\n {model_name} embeddings not found. Skipping.")
            continue

        # Build delta matrix using EXACT same logic as classifier_features.py
        delta_matrix = _build_full_delta_matrix(df, parquet_path, model_name)
        ndim = delta_matrix.shape[1]
        embedding_deltas[model_name] = delta_matrix

        # Experiment: Full embedding delta
        exp_name_full = f"{exp_counter}. {model_name} Full Δ ({ndim}D)"
        print(f"\n  Running: {exp_name_full}")
        result_full = run_classifier_experiment(
            X=delta_matrix, y=y, groups=groups,
            experiment_name=exp_name_full,
        )
        print_experiment_summary(result_full)
        all_summaries.append(result_full)
        exp_counter += 1

    # Build Results Table
    print("\n" + "─" * 70)
    print("RESULTS SUMMARY")
    print("─" * 70)

    result_rows = []
    for s in all_summaries:
        result_rows.append({
            "experiment_name": s["experiment_name"],
            "classifier": s["classifier"],
            "n_features_in": s["n_features_in"],
            "n_features_model": s["n_features_model"],
            "f05": round(s["f05"], 4),
            "f1": round(s["f1"], 4),
            "precision": round(s["precision"], 4),
            "recall": round(s["recall"], 4),
            "accuracy": round(s["accuracy"], 4),
            "mean_threshold": round(s["mean_threshold"], 4),
        })

    df_results = pd.DataFrame(result_rows)

    # Load Threshold Baselines
    print("\nLoading threshold baselines for comparison...")
    df_baselines = load_threshold_baselines()
    if not df_baselines.empty:
        for _, r in df_baselines.iterrows():
            print(f"  {r['experiment_name']}: F0.5={r['f05']:.4f}")

    # Print Final Comparison Table
    print(f"\n{'=' * 100}")
    print(f"  {'Experiment':<35} | {'Feats':>5} | {'F0.5':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'Acc':>7}")
    print(f"  {'─' * 93}")

    if not df_baselines.empty:
        for _, r in df_baselines.iterrows():
            print(
                f"  {r['experiment_name']:<35} | {'–':>5} | {r['f05']:>7.4f} | "
                f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | {r['f1']:>7.4f} | {r['accuracy']:>7.4f}"
            )
        print(f"  {'─' * 93}")

    for _, r in df_results.iterrows():
        print(
            f"  {r['experiment_name']:<35} | {r['n_features_model']:>5} | {r['f05']:>7.4f} | "
            f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | {r['f1']:>7.4f} | {r['accuracy']:>7.4f}"
        )

    print(f"{'=' * 100}")

    # Save Results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "ablation_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n  Results saved → {results_path}")

    # Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_ablation_comparison(
        df_results, df_baselines,
        PLOTS_DIR / "ablation_f05_comparison.pdf",
    )

    print(f"\n  All outputs → {OUTPUT_DIR}/ and {PLOTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()