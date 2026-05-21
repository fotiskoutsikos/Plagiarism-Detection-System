"""
Final Supervised Plagiarism Classifier (Gradient Boosted Trees).

Trains an XGBoost-style classifier (HistGradientBoostingClassifier) on
pairwise engineered features to produce a binary plagiarism decision.

Experiments:
    A. CLEWS Engineered         — CLEWS distances + CLEWS delta summaries
    B. All Engineered           — CLEWS + WEALY distances + all delta summaries
    C. All Engineered + Vocals  — B + vocal metadata
    D. All Engineered + Top-256 — B + top-256 actual CLEWS delta dimensions
    E. All Engineered + Top-512 — B + top-512 actual CLEWS delta dimensions
    F. All Engineered + Top-1024 — B + top-1024 actual CLEWS delta dimensions

Evaluation:
    - StratifiedGroupKFold (groups=filename_ori, 5 folds)
    - Train-only F0.5 threshold optimization
    - Comparison with threshold baselines and LR ablation results

Inputs:
    results/classification/classifier_features.parquet
    results/classification/ablation_results.csv

Outputs:
    results/classification/xgboost_results.csv
    results/classification/xgboost_comparison.csv
    plots/classification/xgboost_comparison.pdf
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold

# Resolve repository root
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent
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
from utils.categorization import fbeta_score_curve
from utils.classifier_features import _build_embedding_map, _compute_delta_matrix_for_pairs
from classification import load_threshold_baselines

plt.rcParams.update(PLOT_STYLE_PARAMS)

FEATURE_TABLE  = Path(CLASSIFIER_FEATURE_TABLE)
ABLATION_TABLE  = Path("results/classification/ablation_results.csv")
OUTPUT_DIR     = Path(CLASSIFICATION_RESULTS_DIR)
PLOTS_DIR      = Path(CLASSIFICATION_PLOTS_DIR)

# K values to evaluate in the convergence study
K_VALUES = [256, 512, 1024]


# Helpers
def _select_columns(df: pd.DataFrame, pattern: str) -> list[str]:
    return sorted([c for c in df.columns if pattern in c])


def _find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    scores = fbeta_score_curve(precision, recall, BETA)
    if len(scores) <= 1:
        return 0.5
    return float(thresholds[int(np.argmax(scores[:-1]))])


def _run_xgboost_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    experiment_name: str,
) -> dict:
    """
    Run StratifiedGroupKFold CV with HistGradientBoostingClassifier.
    Same protocol as classification.py but with gradient boosted trees.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_features = X.shape

    sgkf = StratifiedGroupKFold(
        n_splits=NUM_K_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=20,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=0,
        )

        clf.fit(X_train, y_train)

        prob_train = clf.predict_proba(X_train)[:, 1]
        prob_test  = clf.predict_proba(X_test)[:, 1]

        opt_thresh = _find_optimal_threshold(y_train, prob_train)
        y_pred = (prob_test >= opt_thresh).astype(int)

        fold_metrics.append({
            "fold": fold_idx,
            "threshold": opt_thresh,
            "f05": fbeta_score(y_test, y_pred, beta=BETA, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred),
        })

        print(
            f"    Fold {fold_idx}: F0.5={fold_metrics[-1]['f05']:.4f}  "
            f"P={fold_metrics[-1]['precision']:.4f}  "
            f"R={fold_metrics[-1]['recall']:.4f}  "
            f"thresh={opt_thresh:.4f}"
        )

    df_folds = pd.DataFrame(fold_metrics)
    mean = df_folds.mean()

    return {
        "experiment_name": experiment_name,
        "classifier": "HistGradientBoosting",
        "n_features": n_features,
        "f05": mean["f05"],
        "f1": mean["f1"],
        "precision": mean["precision"],
        "recall": mean["recall"],
        "accuracy": mean["accuracy"],
        "mean_threshold": mean["threshold"],
        "fold_results": df_folds,
    }


def _print_summary(res: dict) -> None:
    name = res["experiment_name"]
    feats = res["n_features"]
    print(f"\n  ► {name:<40} | Features: {feats}")
    print(
        f"    F0.5: {res['f05']:.4f}  |  "
        f"Prec: {res['precision']:.4f}  |  "
        f"Rec: {res['recall']:.4f}  |  "
        f"F1: {res['f1']:.4f}  |  "
        f"Acc: {res['accuracy']:.4f}"
    )


# Plotting
def _plot_comparison(df_comp: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(df_comp))
    width = 0.22

    ax.bar(x - width, df_comp["f05"],       width, label="F0.5",      color="#2196F3", edgecolor="white")
    ax.bar(x,         df_comp["precision"],  width, label="Precision",  color="#4CAF50", edgecolor="white")
    ax.bar(x + width, df_comp["recall"],     width, label="Recall",     color="#FF9800", edgecolor="white")

    for offset, col in [(-width, "f05"), (0, "precision"), (width, "recall")]:
        for i, val in enumerate(df_comp[col]):
            if val > 0.01:
                ax.text(
                    i + offset, val + 0.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["method"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "XGBoost Convergence Study: Engineered + Top-K Actual Dimensions",
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
    print(f"\n  Plot saved → {output_path}")


# Main
def main() -> None:
    print("=" * 70)
    print("FINAL SUPERVISED CLASSIFIER (GRADIENT BOOSTED TREES)")
    print("=" * 70)

    # Load Feature Table
    print("\nLoading feature table...")
    df = pd.read_parquet(FEATURE_TABLE)
    y = df["is_plagiarised"].astype(int).values
    groups = df["filename_ori"].values
    print(f"  Loaded {len(df)} pairs ({y.sum()} pos, {(y == 0).sum()} neg)")

    # Identify Feature Groups
    clews_dist_cols  = [c for c in _select_columns(df, "clews_") if "distance" in c]
    wealy_dist_cols  = [c for c in _select_columns(df, "wealy_") if "distance" in c]

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

    vocal_cols = [
        c for c in df.columns
        if c in {"pair_vocal_valid", "vocal_ratio_ori", "vocal_ratio_mod",
                 "vocal_valid_ori", "vocal_valid_mod"}
        and not df[c].isna().all()
    ]

    clews_xai_cols = _select_columns(df, "clews_xai_dim_")

    # Feature Sets
    clews_engineered_no_vocal = clews_dist_cols + clews_delta_cols
    all_engineered_no_vocal   = (clews_dist_cols + wealy_dist_cols
                                 + clews_delta_cols + wealy_delta_cols)
    all_engineered            = all_engineered_no_vocal + vocal_cols

    # ── Build XAI enrichment for ALL K values in one pass ─────────────────────
    print("\nBuilding XAI enrichment dimensions (256 / 512 / 1024)...")
    emb_map = _build_embedding_map(EMBEDDING_PATHS["CLEWS"])
    delta_valid, valid_mask = _compute_delta_matrix_for_pairs(df, emb_map)

    ndim = delta_valid.shape[1]
    delta_matrix = np.zeros((len(df), ndim), dtype=np.float32)
    delta_matrix[valid_mask] = delta_valid.astype(np.float32)
    del delta_valid, valid_mask

    # Rank by positive-pair mean shift
    pos_deltas   = delta_matrix[y == 1]
    mean_shifts  = np.mean(pos_deltas, axis=0)
    ranked_idx    = np.argsort(mean_shifts)[::-1]
    del pos_deltas, mean_shifts

    # Pre-compute all K enrichments
    enrichments = {}
    for k_val in K_VALUES:
        enrichments[k_val] = delta_matrix[:, ranked_idx[:k_val]]
        print(f"  Top-{k_val} enrichment: {enrichments[k_val].shape}")

    # Free delta matrix — we only need the enrichments from now on
    del delta_matrix, emb_map

    # ── Define Experiments ─────────────────────────────────────────────────────
    experiments = [
        (
            "A. CLEWS Engineered (No Vocals)",
            df[clews_engineered_no_vocal].values,
        ),
        (
            "B. All Engineered (No Vocals)",
            df[all_engineered_no_vocal].values,
        ),
        (
            "C. All Engineered (With Vocals)",
            df[all_engineered].values,
        ),
        (
            "D. All Engineered (No Vocals) + Top-256",
            np.hstack([df[all_engineered_no_vocal].values, enrichments[256]]),
        ),
        (
            "E. All Engineered (No Vocals) + Top-512",
            np.hstack([df[all_engineered_no_vocal].values, enrichments[512]]),
        ),
        (
            "F. All Engineered (No Vocals) + Top-1024",
            np.hstack([df[all_engineered_no_vocal].values, enrichments[1024]]),
        ),
    ]

    # ── Run Experiments ────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("XGBOOST EXPERIMENTS")
    print("─" * 70)

    xgb_results = []
    for name, X in experiments:
        print(f"\n  Running: {name} ({X.shape[1]} features)")
        res = _run_xgboost_cv(X, y, groups, name)
        _print_summary(res)
        xgb_results.append(res)

    # ── Build Comparison Table ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("COMPARISON WITH BASELINES")
    print("─" * 70)

    comp_rows = []

    # Threshold baselines
    df_baselines = load_threshold_baselines()
    for _, r in df_baselines.iterrows():
        comp_rows.append({
            "method": r["experiment_name"],
            "f05": r["f05"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "accuracy": r["accuracy"],
        })

    # LR best result from ablation
    if ABLATION_TABLE.exists():
        df_ab = pd.read_csv(ABLATION_TABLE)
        if not df_ab.empty:
            best_lr = df_ab.loc[df_ab["f05"].idxmax()]
            comp_rows.append({
                "method": f"LR Best ({best_lr['experiment_name']})",
                "f05": float(best_lr["f05"]),
                "precision": float(best_lr["precision"]),
                "recall": float(best_lr["recall"]),
                "f1": float(best_lr["f1"]),
                "accuracy": float(best_lr["accuracy"]),
            })

    # XGBoost results
    for res in xgb_results:
        comp_rows.append({
            "method": f"XGB {res['experiment_name']}",
            "f05": round(res["f05"], 4),
            "precision": round(res["precision"], 4),
            "recall": round(res["recall"], 4),
            "f1": round(res["f1"], 4),
            "accuracy": round(res["accuracy"], 4),
        })

    df_comp = pd.DataFrame(comp_rows)

    # Print
    print(f"\n{'=' * 100}")
    print(f"  {'Method':<47} | {'F0.5':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'Acc':>7}")
    print(f"  {'─' * 96}")
    for _, r in df_comp.iterrows():
        print(
            f"  {r['method']:<47} | {r['f05']:>7.4f} | "
            f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | "
            f"{r['f1']:>7.4f} | {r['accuracy']:>7.4f}"
        )
    print(f"{'=' * 100}")

    # ── Save Results ───────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xgb_rows = []
    for res in xgb_results:
        xgb_rows.append({
            "experiment_name": res["experiment_name"],
            "classifier": res["classifier"],
            "n_features": res["n_features"],
            "f05": round(res["f05"], 4),
            "f1": round(res["f1"], 4),
            "precision": round(res["precision"], 4),
            "recall": round(res["recall"], 4),
            "accuracy": round(res["accuracy"], 4),
            "mean_threshold": round(res["mean_threshold"], 4),
        })
    pd.DataFrame(xgb_rows).to_csv(OUTPUT_DIR / "xgboost_results.csv", index=False)

    df_comp.to_csv(OUTPUT_DIR / "xgboost_comparison.csv", index=False)

    _plot_comparison(df_comp, PLOTS_DIR / "xgboost_comparison.pdf")

    print(f"\n  Results → {OUTPUT_DIR}/")
    print(f"  Plots   → {PLOTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()