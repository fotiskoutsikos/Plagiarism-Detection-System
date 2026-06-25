"""
Feature Ablation Study for Binary Plagiarism Classification.

Orchestrates multiple classification experiments using the modular
classification engine (classification.py) to answer:

    Q1: Do engineered features (distances + summaries) beat raw embeddings?
    Q2: How many actual top-K CLEWS dimensions are needed to converge?
    Q3: Does adding vocal metadata help or hurt?

All experiments use the SAME classifier (XGBClassifier),
the SAME CV protocol (StratifiedGroupKFold, 5 folds), and the SAME
threshold optimisation (train-only F0.5). Only the input features change.

Statistical Rigor
-----------------
Every experiment is repeated N_SEEDS times with independent random seeds
(controlled via the N_SEEDS constant below). This produces mean ± std and
95% confidence intervals for all reported metrics, making performance
differences between feature sets statistically defensible.

Set N_SEEDS = 1 for a quick exploratory run; use N_SEEDS = 10 for
publication-grade results with confidence intervals.

Phases
------
    Phase 1 — Engineered Features
        Experiments over distances, delta summaries, vocal metadata,
        and their combinations, against CLEWS and WEALY separately and
        jointly. Establishes which engineered feature families contribute.

    Phase 2 — Raw Embedding Deltas
        Full-dimensional delta vectors for CLEWS and WEALY.
        Establishes whether engineered features out-perform raw embeddings.
        Training time and inference latency are recorded here and reused
        in Phase 3 for the full-D baseline point.

    Phase 3 — Top-K Convergence Curve
        Ranks CLEWS delta dimensions by mean absolute shift on positive
        pairs and evaluates classifiers trained on the top-K dimensions
        for K ∈ {10, 30, 50, 100, 256, 512}, appending the precomputed
        Full-1024D result to complete the convergence picture.
        The convergence plot shows F0.5 (with CI band) on the left axis
        and mean training time on the right axis, making the
        performance/cost trade-off immediately visible.

Outputs
-------
    results/classification/ablation_results.csv
    results/classification/ablation_topk_curve.csv
    plots/classification/ablation_f05_comparison.pdf
    plots/classification/ablation_topk_convergence_curve.pdf
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

FEATURE_TABLE = Path(CLASSIFIER_FEATURE_TABLE)
OUTPUT_DIR    = Path(CLASSIFICATION_RESULTS_DIR)
PLOTS_DIR     = Path(CLASSIFICATION_PLOTS_DIR)

# K values for the Top-K convergence curve (Phase 3)
TOPK_VALUES = [10, 30, 50, 100, 256, 512]

# Statistical rigor 
# Number of independent random seeds for CI estimation.
# Use N_SEEDS = 1 for a quick exploratory run.
# Use N_SEEDS = 10 for publication-grade confidence intervals.
N_SEEDS = 10


# Helpers
def _select_columns(df: pd.DataFrame, pattern: str) -> list[str]:
    return sorted([c for c in df.columns if pattern in c])


def _build_full_delta_matrix(
    df: pd.DataFrame,
    parquet_path: str,
    model_name: str,
) -> np.ndarray:
    """
    Build a full (N_pairs, D) delta matrix aligned with df.
    Rows with missing embeddings receive zero vectors.
    """
    print(f"  Loading {model_name} embeddings...")
    emb_map = _build_embedding_map(parquet_path)

    print(f"  Computing {model_name} delta matrix...")
    delta_valid, valid_mask = _compute_delta_matrix_for_pairs(df, emb_map)
    del emb_map

    n_valid = int(valid_mask.sum())
    n_miss  = int((~valid_mask).sum())

    if delta_valid.size == 0:
        logger.warning("[%s] No valid deltas. Returning zeros.", model_name)
        return np.zeros((len(df), 1), dtype=np.float32)

    ndim = delta_valid.shape[1]
    print(
        f"  [{model_name}] Delta: {n_valid}/{len(df)} valid, "
        f"{n_miss} missing, dim={ndim}"
    )

    delta_full = np.zeros((len(df), ndim), dtype=np.float32)
    delta_full[valid_mask] = delta_valid.astype(np.float32)
    return delta_full


# Plotting
def _plot_ablation_comparison(
    df_results:   pd.DataFrame,
    df_baselines: pd.DataFrame,
    output_path:  Path,
) -> None:
    """
    Bar chart comparing F0.5, Precision and Recall for all experiments.
    F0.5 bars carry 95% CI error bars where available.
    """
    rows = []
    for _, r in df_baselines.iterrows():
        rows.append({
            "Method":       r["experiment_name"],
            "F0.5":         r["f05"],
            "F05_CI95":     r.get("f05_ci95", 0.0),
            "Precision":    r["precision"],
            "Prec_CI95":    r.get("precision_ci95", 0.0),
            "Recall":       r["recall"],
            "Rec_CI95":     r.get("recall_ci95", 0.0),
        })
    for _, r in df_results.iterrows():
        rows.append({
            "Method":       r["experiment_name"],
            "F0.5":         r["f05"],
            "F05_CI95":     r.get("f05_ci95", 0.0),
            "Precision":    r["precision"],
            "Prec_CI95":    r.get("precision_ci95", 0.0),
            "Recall":       r["recall"],
            "Rec_CI95":     r.get("recall_ci95", 0.0),
        })

    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(16, 7))
    x     = np.arange(len(df_plot))
    width = 0.25

    # F0.5 bars with 95% CI error bars
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

    # Value labels on top of bars
    ci_col_map = {
        "F0.5":      "F05_CI95",
        "Precision": "Prec_CI95",
        "Recall":    "Rec_CI95",
    }
    for col, offset in [("F0.5", -width), ("Precision", 0), ("Recall", width)]:
        ci_col = ci_col_map[col]
        for i, (val, ci) in enumerate(zip(df_plot[col], df_plot[ci_col])):
            if val > 0.01:
                ax.text(
                    i + offset + width / 2, val + ci + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Method"], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Feature Ablation: Classification Performance (XGBoost, {N_SEEDS} seeds)",
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


def _plot_topk_convergence(df_res: pd.DataFrame, output_path: Path) -> None:
    """
    Dual-axis convergence plot.

    Left axis  : F0.5 score with shaded 95% CI band.
    Right axis : Mean training time per fold (seconds).
    """
    x_labels  = [str(k) for k in df_res["K"]]
    x_pos     = np.arange(len(x_labels))

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Left axis: F0.5 with CI band 
    f05_vals = df_res["F05"].values
    ci95     = df_res.get("F05_CI95", pd.Series(np.zeros(len(df_res)))).values

    ax1.plot(
        x_pos, f05_vals,
        marker="o", linewidth=2.5, color="#4CAF50",
        label="F0.5 (mean ± 95% CI)",
        zorder=3,
    )
    ax1.fill_between(
        x_pos,
        f05_vals - ci95,
        f05_vals + ci95,
        color="#4CAF50", alpha=0.18, zorder=2,
        label="_nolegend_",
    )
    for i, (val, ci) in enumerate(zip(f05_vals, ci95)):
        offset = ci + 0.007
        ax1.annotate(
            f"{val:.3f}", (i, val + offset),
            ha="center", fontsize=8, color="darkgreen", fontweight="bold",
        )

    f05_min = max(0.0, (f05_vals - ci95).min() - 0.06)
    f05_max = (f05_vals + ci95).max() + 0.08
    ax1.set_ylim(f05_min, f05_max)
    ax1.set_xlabel("Number of Retained Dimensions (K)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("F0.5 Score",                        fontsize=11, fontweight="bold", color="#2e7d32")
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=9)

    # Right axis: Training time 
    if "TrainTimeSec" in df_res.columns and df_res["TrainTimeSec"].notna().any():
        ax2 = ax1.twinx()
        train_times = df_res["TrainTimeSec"].values

        ax2.plot(
            x_pos, train_times,
            marker="s", linewidth=2.0, linestyle="--", color="#E53935",
            label="Train time / fold",
            zorder=2,
        )
        for i, t in enumerate(train_times):
            ax2.annotate(
                f"{t:.1f}s", (i, t + max(train_times) * 0.02),
                ha="center", fontsize=7.5, color="#b71c1c",
            )
        ax2.set_ylabel("Mean Training Time per Fold (s)", fontsize=10,
                       fontweight="bold", color="#b71c1c")
        ax2.tick_params(axis="y", labelcolor="#b71c1c")
        ax2.set_ylim(0, max(train_times) * 1.25)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="lower right", fontsize=9)
    else:
        ax1.legend(loc="lower right", fontsize=10)

    ax1.set_title(
        f"Top-K CLEWS Convergence: F0.5 & Training Cost ({N_SEEDS} seeds, ±95% CI)",
        fontsize=13, fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {output_path}")


# Main
def main() -> None:
    print("=" * 70)
    print("FEATURE ABLATION STUDY — BINARY PLAGIARISM CLASSIFICATION")
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

    # Identify feature columns
    clews_dist_cols = [c for c in _select_columns(df, "clews_") if "distance" in c]
    wealy_dist_cols = [c for c in _select_columns(df, "wealy_") if "distance" in c]
    all_dist_cols   = clews_dist_cols + wealy_dist_cols

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
                 "vocal_valid_ori",  "vocal_valid_mod"}
        and not df[c].isna().all()
    ]

    all_engineered_no_vocal = all_dist_cols + all_delta_cols
    all_engineered          = all_engineered_no_vocal + vocal_cols

    # PHASE 1: Engineered Feature Experiments
    print("\n" + "─" * 70)
    print("PHASE 1 — ENGINEERED FEATURE EXPERIMENTS")
    print("─" * 70)

    phase1_experiments = [
        ("1. CLEWS Distance (Winning)",    clews_dist_cols),
        ("2. WEALY Distance (Winning)",    wealy_dist_cols),
        ("3. Both Distances (Winning)",    all_dist_cols),
        ("4. CLEWS Delta Summaries",       clews_delta_cols),
        ("5. WEALY Delta Summaries",       wealy_delta_cols),
        ("6. All Delta Summaries",         all_delta_cols),
        ("7. All Engineered (No Vocal)",   all_engineered_no_vocal),
        ("8. All Engineered (Vocal)",      all_engineered),
        ("9. Vocal Only",                  vocal_cols),
    ]

    all_summaries: list[dict] = []

    for name, cols in phase1_experiments:
        if not cols:
            print(f"\n  ⚠  {name}: No features found. Skipping.")
            continue
        print(f"\n  Running: {name}  ({len(cols)} features, {N_SEEDS} seed(s))")
        res = run_classifier_experiment(
            X=df[cols], y=y, groups=groups,
            experiment_name=name,
            n_seeds=N_SEEDS,
        )
        print_experiment_summary(res)
        all_summaries.append(res)

    # PHASE 2: Raw Embedding Delta Experiments
    print("\n" + "─" * 70)
    print("PHASE 2 — RAW EMBEDDING DELTA EXPERIMENTS")
    print("─" * 70)

    embedding_deltas: dict[str, np.ndarray] = {}
    exp_counter = 10

    for model_name, parquet_path in EMBEDDING_PATHS.items():
        if not Path(parquet_path).exists():
            print(f"\n  ⚠  {model_name} embeddings not found. Skipping.")
            continue

        delta_matrix = _build_full_delta_matrix(df, parquet_path, model_name)
        ndim         = delta_matrix.shape[1]
        embedding_deltas[model_name] = delta_matrix

        exp_name = f"{exp_counter}. {model_name} Full Δ ({ndim}D)"
        print(f"\n  Running: {exp_name}  ({N_SEEDS} seed(s))")
        res = run_classifier_experiment(
            X=delta_matrix, y=y, groups=groups,
            experiment_name=exp_name,
            n_seeds=N_SEEDS,
        )
        print_experiment_summary(res)
        all_summaries.append(res)
        exp_counter += 1

    # PHASE 3: Top-K Convergence Curve (CLEWS)
    print("\n" + "─" * 70)
    print("PHASE 3 — TOP-K CLEWS DIMENSION CONVERGENCE CURVE")
    print("─" * 70)

    # Retrieve or rebuild the CLEWS delta matrix
    if "CLEWS" in embedding_deltas:
        clews_delta = embedding_deltas["CLEWS"]
    else:
        clews_delta = _build_full_delta_matrix(df, EMBEDDING_PATHS["CLEWS"], "CLEWS")

    ndim_clews = clews_delta.shape[1]
    print(f"  Full CLEWS delta matrix: {clews_delta.shape}")

    # Rank dimensions by mean absolute shift on positive pairs only
    print("  Ranking dimensions by mean absolute shift on positive pairs...")
    pos_mask    = y == 1
    mean_shifts = np.mean(np.abs(clews_delta[pos_mask]), axis=0)
    ranked_idx  = np.argsort(mean_shifts)[::-1]

    topk_rows: list[dict] = []

    for k in TOPK_VALUES:
        exp_name = f"Top-{k} CLEWS Dims"
        print(f"\n  Running: {exp_name}  ({N_SEEDS} seed(s))")
        X_topk = clews_delta[:, ranked_idx[:k]]
        res = run_classifier_experiment(
            X=X_topk, y=y, groups=groups,
            experiment_name=exp_name,
            n_seeds=N_SEEDS,
        )

        ci_str = f" ±{res['f05_ci95']:.4f}" if N_SEEDS > 1 else ""
        print(f"  {exp_name:25} → F0.5: {res['f05']:.4f}{ci_str}  "
              f"| Train: {res['mean_train_time_sec']:.2f}s/fold  "
              f"| Infer: {res['mean_infer_time_ms']:.4f}ms/sample")

        topk_rows.append({
            "K":             k,
            "F05":           res["f05"],
            "F05_Std":       res["f05_std"],
            "F05_CI95":      res["f05_ci95"],
            "Precision":     res["precision"],
            "Prec_CI95":     res["precision_ci95"],
            "Recall":        res["recall"],
            "Rec_CI95":      res["recall_ci95"],
            "F1":            res["f1"],
            "Accuracy":      res["accuracy"],
            "Threshold":     res["mean_threshold"],
            "TrainTimeSec":  res["mean_train_time_sec"],
            "InferTimeMs":   res["mean_infer_time_ms"],
        })

    # Append full-D result already computed in Phase 2
    clews_full_res = next(
        (s for s in all_summaries if "CLEWS Full Δ" in s["experiment_name"]),
        None,
    )
    if clews_full_res is not None:
        topk_rows.append({
            "K":             ndim_clews,
            "F05":           clews_full_res["f05"],
            "F05_Std":       clews_full_res.get("f05_std",       0.0),
            "F05_CI95":      clews_full_res.get("f05_ci95",      0.0),
            "Precision":     clews_full_res["precision"],
            "Prec_CI95":     clews_full_res.get("precision_ci95", 0.0),
            "Recall":        clews_full_res["recall"],
            "Rec_CI95":      clews_full_res.get("recall_ci95",    0.0),
            "F1":            clews_full_res["f1"],
            "Accuracy":      clews_full_res["accuracy"],
            "Threshold":     clews_full_res["mean_threshold"],
            "TrainTimeSec":  clews_full_res.get("mean_train_time_sec", float("nan")),
            "InferTimeMs":   clews_full_res.get("mean_infer_time_ms",  float("nan")),
        })
        ci_str = (
            f" ±{clews_full_res['f05_ci95']:.4f}" if N_SEEDS > 1 else ""
        )
        print(
            f"\n  Appended precomputed Full-{ndim_clews}D result → "
            f"F0.5: {clews_full_res['f05']:.4f}{ci_str}"
        )

    df_topk = pd.DataFrame(topk_rows).sort_values("K").reset_index(drop=True)

    # Build aggregate results table
    result_rows = []
    for s in all_summaries:
        result_rows.append({
            "experiment_name":      s["experiment_name"],
            "classifier":           s["classifier"],
            "n_features":           s["n_features"],
            "n_seeds":              s.get("n_seeds", 1),
            "f05":                  round(s["f05"],                         4),
            "f05_std":              round(s.get("f05_std",       0.0),       4),
            "f05_ci95":             round(s.get("f05_ci95",      0.0),       4),
            "f1":                   round(s["f1"],                          4),
            "f1_ci95":              round(s.get("f1_ci95",       0.0),       4),
            "precision":            round(s["precision"],                   4),
            "precision_ci95":       round(s.get("precision_ci95", 0.0),     4),
            "recall":               round(s["recall"],                      4),
            "recall_ci95":          round(s.get("recall_ci95",   0.0),       4),
            "accuracy":             round(s["accuracy"],                    4),
            "mean_threshold":       round(s["mean_threshold"],              4),
            "mean_train_time_sec":  round(s.get("mean_train_time_sec", 0.0), 3),
            "mean_infer_time_ms":   round(s.get("mean_infer_time_ms",  0.0), 4),
        })

    df_results = pd.DataFrame(result_rows)

    # Load threshold baselines
    print("\n\nLoading threshold baselines for comparison...")
    df_baselines = load_threshold_baselines()

    # Print final comparison table
    print(f"\n{'=' * 110}")
    print(
        f"  {'Experiment':<42} | {'Feats':>6} | {'F0.5':>7} | {'CI95':>7} | "
        f"{'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'Acc':>7} | {'Train(s)':>9}"
    )
    print(f"  {'─' * 105}")

    if not df_baselines.empty:
        for _, r in df_baselines.iterrows():
            print(
                f"  {r['experiment_name']:<42} | {'–':>6} | {r['f05']:>7.4f} | {'–':>7} | "
                f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | "
                f"{r['f1']:>7.4f} | {r['accuracy']:>7.4f} | {'–':>9}"
            )
        print(f"  {'─' * 105}")

    for _, r in df_results.iterrows():
        ci_str  = f"±{r['f05_ci95']:.4f}" if r.get("f05_ci95", 0) > 0 else "–"
        t_str   = f"{r['mean_train_time_sec']:.2f}" if r.get("mean_train_time_sec", 0) > 0 else "–"
        print(
            f"  {r['experiment_name']:<42} | {r['n_features']:>6} | {r['f05']:>7.4f} | {ci_str:>7} | "
            f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | "
            f"{r['f1']:>7.4f} | {r['accuracy']:>7.4f} | {t_str:>9}"
        )
    print(f"{'=' * 110}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "ablation_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n  Ablation results  → {results_path}")

    topk_path = OUTPUT_DIR / "ablation_topk_curve.csv"
    df_topk.to_csv(topk_path, index=False)
    print(f"  Top-K curve       → {topk_path}")

    _plot_ablation_comparison(
        df_results, df_baselines,
        PLOTS_DIR / "ablation_f05_comparison.pdf",
    )
    _plot_topk_convergence(
        df_topk,
        PLOTS_DIR / "ablation_topk_convergence_curve.pdf",
    )

    print(f"\n  All outputs → {OUTPUT_DIR}/  and  {PLOTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()