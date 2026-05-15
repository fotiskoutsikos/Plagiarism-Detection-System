"""
Top-K Feature Convergence Curve.

Executes a classification experiment across varying numbers of retained
actual embedding dimensions:

    K = 10, 30, 50, 100, 256, 512

and appends the already computed Full 1024D CLEWS result from the
ablation study, avoiding a second memory-heavy full-dimension run.

Goal:
    To quantify how many actual CLEWS delta dimensions are needed
    to approach the classification performance of the full 1024-D
    representation.

Feature selection strategy:
    - Dimensions are ranked by mean absolute delta on POSITIVE pairs only.
    - The top-K actual dimensions are retained (no PCA, no projection).

Outputs:
    results/classification/topk_curve_results.csv
    plots/classification/topk_convergence_curve.pdf
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve repository root
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root / "src"))

from utils.constants import (
    EMBEDDING_PATHS, PLOT_DPI, PLOT_STYLE_PARAMS,
    CLASSIFIER_FEATURE_TABLE, CLASSIFICATION_RESULTS_DIR, CLASSIFICATION_PLOTS_DIR,
)
from utils.classifier_features import _build_embedding_map, _compute_delta_matrix_for_pairs
from classification import run_classifier_experiment

plt.rcParams.update(PLOT_STYLE_PARAMS)

FEATURE_TABLE   = Path(CLASSIFIER_FEATURE_TABLE)
ABLATION_TABLE  = Path("results/classification/ablation_results.csv")
OUTPUT_DIR      = Path(CLASSIFICATION_RESULTS_DIR)
PLOTS_DIR       = Path(CLASSIFICATION_PLOTS_DIR)

K_VALUES = [10, 30, 50, 100, 256, 512]


def plot_topk_curve(df_res: pd.DataFrame, output_path: Path) -> None:
    """Generate the Top-K convergence curve."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = [str(k) for k in df_res["K"]]

    ax.plot(
        x_labels,
        df_res["F05"],
        marker="o",
        linewidth=2.5,
        color="#4CAF50",
        label="Top-K Actual Dimensions",
    )

    for i, txt in enumerate(df_res["F05"]):
        ax.annotate(
            f"{txt:.3f}",
            (i, df_res["F05"].iloc[i] + 0.006),
            ha="center",
            fontsize=8,
            color="darkgreen",
            fontweight="bold",
        )

    ax.set_ylim(0.3, df_res["F05"].max() + 0.06)
    ax.set_xlabel("Number of Retained Dimensions (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("F0.5 Score", fontsize=11, fontweight="bold")
    ax.set_title(
        "Classification Convergence with Top-K Actual CLEWS Dimensions",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {output_path}")


def _load_full_1024_result(ablation_csv: Path) -> dict | None:
    """
    Load the already computed CLEWS Full 1024D result from ablation_results.csv.
    """
    if not ablation_csv.exists():
        print(f"[WARNING] Ablation results not found: {ablation_csv}")
        return None

    df_ab = pd.read_csv(ablation_csv)

    match = df_ab[df_ab["experiment_name"].astype(str).str.contains("CLEWS Full Δ", regex=False)]
    if match.empty:
        print("[WARNING] Could not find 'CLEWS Full Δ' row in ablation_results.csv")
        return None

    row = match.iloc[0]
    return {
        "K": 1024,
        "F05": float(row["f05"]),
        "Precision": float(row["precision"]),
        "Recall": float(row["recall"]),
        "F1": float(row["f1"]),
        "Accuracy": float(row["accuracy"]),
        "Threshold": float(row["mean_threshold"]) if "mean_threshold" in row else np.nan,
    }


def main() -> None:
    print("=" * 70)
    print("TOP-K FEATURE CONVERGENCE CURVE")
    print("=" * 70)

    # Load Pair Table
    print("\nLoading classifier feature table...")
    df = pd.read_parquet(FEATURE_TABLE)
    y = df["is_plagiarised"].astype(int).values
    groups = df["filename_ori"].values

    # Build Full CLEWS Delta Matrix
    print("\nLoading CLEWS raw embeddings...")
    emb_map = _build_embedding_map(EMBEDDING_PATHS["CLEWS"])
    delta_valid, valid_mask = _compute_delta_matrix_for_pairs(df, emb_map)

    ndim = delta_valid.shape[1]
    delta_matrix = np.zeros((len(df), ndim), dtype=np.float32)
    delta_matrix[valid_mask] = delta_valid.astype(np.float32)

    print(f"  Full CLEWS delta matrix: {delta_matrix.shape}")

    # Rank dimensions using positives only
    print("Ranking dimensions by mean absolute shift on positive pairs...")
    pos_deltas = delta_matrix[y == 1]
    mean_shifts = np.mean(pos_deltas, axis=0)

    ranked_indices = np.argsort(mean_shifts)[::-1]

    # Run Top-K experiments
    results = []
    for k in K_VALUES:
        print(f"\nEvaluating Top-{k}...")

        selected_dims = ranked_indices[:k]
        X = delta_matrix[:, selected_dims]
        exp_name = f"Top-{k}"

        res = run_classifier_experiment(
            X=X,
            y=y,
            groups=groups,
            experiment_name=exp_name,
        )

        print(f"  {exp_name:12} -> F0.5: {res['f05']:.4f}")

        results.append({
            "K": k,
            "F05": res["f05"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1": res["f1"],
            "Accuracy": res["accuracy"],
            "Threshold": res["mean_threshold"],
        })

    # Append Full 1024D result from ablation study
    full_res = _load_full_1024_result(ABLATION_TABLE)
    if full_res is not None:
        print(f"\nAppending precomputed Full-1024 result -> F0.5: {full_res['F05']:.4f}")
        results.append(full_res)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results).sort_values("K").reset_index(drop=True)

    csv_path = OUTPUT_DIR / "topk_curve_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")

    # Plot
    plot_topk_curve(df_res, PLOTS_DIR / "topk_convergence_curve.pdf")

    print("\nDone!")


if __name__ == "__main__":
    main()