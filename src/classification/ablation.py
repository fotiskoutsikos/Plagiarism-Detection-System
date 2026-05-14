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

from utils.constants import EMBEDDING_PATHS, OUTPUT_DIRS, PLOT_DPI, PLOT_STYLE_PARAMS

from utils.classifier_features import _build_embedding_map, _compute_delta_matrix_for_pairs

from classification import (
    run_classifier_experiment,
    load_threshold_baselines,
    print_experiment_summary,
)

plt.rcParams.update(PLOT_STYLE_PARAMS)

# Paths 
FEATURE_TABLE = Path("data/classifier_features.parquet")
OUTPUT_DIR    = Path("results/classification")
PLOTS_DIR     = Path("plots/classification")


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


def _compute_xai_shift_energy(
    delta_matrix: np.ndarray,
    xai_col_indices: list[int],
    y: np.ndarray,
) -> float:
    """
    Compute fraction of total shift energy retained by XAI-selected dims.
    Uses ONLY positive pairs for the energy calculation.
    """
    pos_mask = y == 1
    if pos_mask.sum() == 0:
        return 0.0

    mean_delta = np.mean(delta_matrix[pos_mask], axis=0)
    total_energy = float(np.sum(mean_delta ** 2))
    if total_energy == 0:
        return 0.0

    xai_energy = float(np.sum(mean_delta[xai_col_indices] ** 2))
    return xai_energy / total_energy


def _get_xai_dim_indices_from_cols(
    feature_cols: list[str],
    prefix: str,
) -> list[int]:
    """Extract actual embedding dimension indices from XAI column names."""
    indices = []
    tag = f"{prefix}_xai_dim_"
    for col in feature_cols:
        if col.startswith(tag):
            try:
                indices.append(int(col.replace(tag, "")))
            except ValueError:
                pass
    return sorted(indices)


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


def _plot_information_retention(
    df_info: pd.DataFrame,
    output_path: Path,
) -> None:
    """Side-by-side bar chart: XAI Shift Energy vs PCA Variance Retained."""
    if df_info.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4CAF50" if "XAI" in m else "#FF9800" for m in df_info["method"]]

    bars = ax.bar(df_info["method"], df_info["retention_pct"], color=colors, edgecolor="black")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% Information Retained", fontsize=11)
    ax.set_title(
        "Information Retention: XAI (Shift Energy) vs PCA (Variance)",
        fontsize=13, fontweight="bold",
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 1,
            f"{h:.1f}%", ha="center", fontweight="bold", fontsize=9,
        )

    plt.xticks(rotation=20, ha="right", fontsize=9)
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
        ("1. CLEWS Distances",       clews_dist_cols,  False),
        ("2. WEALY Distances",       wealy_dist_cols,  False),
        ("3. All Distances",         all_dist_cols,    False),
        ("4. CLEWS Delta Summaries", clews_delta_cols, False),
        ("5. WEALY Delta Summaries", wealy_delta_cols, False),
        ("6. All Delta Summaries",   all_delta_cols,   False),
        ("7. All Engineered",        all_engineered,   False),
        ("8. Engineered - Vocal",    all_dist_cols + all_delta_cols, False),
        ("9. Vocal Only",            vocal_cols,       False),
        ("10. CLEWS XAI Top-30",     clews_xai_cols,   False),
        ("11. WEALY XAI Top-30",     wealy_xai_cols,   False),
    ]

    # Run Phase 1
    print("\n" + "─" * 70)
    print("PHASE 1: ENGINEERED + XAI FEATURE EXPERIMENTS")
    print("─" * 70)

    all_summaries = []
    for name, cols, use_pca in experiments:
        if not cols:
            print(f"\n  ⚠ {name}: No features found. Skipping.")
            continue

        print(f"\n  Running: {name} ({len(cols)} features)")
        result = run_classifier_experiment(
            X=df[cols], y=y, groups=groups,
            experiment_name=name, use_pca=use_pca,
        )
        print_experiment_summary(result)
        all_summaries.append(result)

    # Run Phase 2: Raw Embedding Experiments
    print("\n" + "─" * 70)
    print("PHASE 2: RAW EMBEDDING EXPERIMENTS (on-the-fly)")
    print("─" * 70)

    info_retention_rows = []
    exp_counter = 12

    for model_name, parquet_path in EMBEDDING_PATHS.items():
        prefix = model_name.lower()

        if not Path(parquet_path).exists():
            print(f"\n  ⚠ {model_name} embeddings not found. Skipping.")
            continue

        # Build delta matrix using EXACT same logic as classifier_features.py
        delta_matrix = _build_full_delta_matrix(df, parquet_path, model_name)
        ndim = delta_matrix.shape[1]

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

        # Experiment: PCA-30 embedding delta
        exp_name_pca = f"{exp_counter}. {model_name} PCA-30 Δ"
        print(f"\n  Running: {exp_name_pca}")
        result_pca = run_classifier_experiment(
            X=delta_matrix, y=y, groups=groups,
            experiment_name=exp_name_pca,
            use_pca=True, n_pca_components=30,
        )
        print_experiment_summary(result_pca)
        all_summaries.append(result_pca)
        exp_counter += 1

        # Information Retention: PCA
        if result_pca["pca_explained_variance"] is not None:
            info_retention_rows.append({
                "method": f"{model_name} PCA-30",
                "type": "Variance Explained",
                "retention_pct": round(result_pca["pca_explained_variance"] * 100, 2),
            })

        # Information Retention: XAI
        xai_cols_for_model = clews_xai_cols if prefix == "clews" else wealy_xai_cols
        xai_dim_indices = _get_xai_dim_indices_from_cols(xai_cols_for_model, prefix)

        if xai_dim_indices:
            valid_indices = [d for d in xai_dim_indices if d < ndim]
            if valid_indices:
                energy = _compute_xai_shift_energy(delta_matrix, valid_indices, y)
                info_retention_rows.append({
                    "method": f"{model_name} XAI Top-30",
                    "type": "Shift Energy Retained",
                    "retention_pct": round(energy * 100, 2),
                })

        # Free memory after each model
        del delta_matrix

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
            "use_pca": s["use_pca"],
            "f05": round(s["f05"], 4),
            "f1": round(s["f1"], 4),
            "precision": round(s["precision"], 4),
            "recall": round(s["recall"], 4),
            "accuracy": round(s["accuracy"], 4),
            "mean_threshold": round(s["mean_threshold"], 4),
            "pca_variance": round(s["pca_explained_variance"], 4) if s["pca_explained_variance"] else None,
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
        pca_tag = " *" if r["use_pca"] else ""
        print(
            f"  {r['experiment_name']:<35} | {r['n_features_model']:>5} | {r['f05']:>7.4f} | "
            f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | {r['f1']:>7.4f} | {r['accuracy']:>7.4f}{pca_tag}"
        )

    print(f"{'=' * 100}")

    # Save Results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "ablation_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n  Results saved → {results_path}")

    if info_retention_rows:
        df_info = pd.DataFrame(info_retention_rows)
        info_path = OUTPUT_DIR / "ablation_information_retention.csv"
        df_info.to_csv(info_path, index=False)
        print(f"  Information retention saved → {info_path}")
    else:
        df_info = pd.DataFrame()

    # Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_ablation_comparison(
        df_results, df_baselines,
        PLOTS_DIR / "ablation_f05_comparison.pdf",
    )

    if not df_info.empty:
        _plot_information_retention(
            df_info,
            PLOTS_DIR / "ablation_information_retention.pdf",
        )

    print(f"\n  All outputs → {OUTPUT_DIR}/ and {PLOTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()