"""
Binary Plagiarism Classifiers using Pre-computed Features.

Trains two classifiers on the unified feature table produced by
classifier_features.py and compares them against the existing
threshold-based baselines (CLEWS, WEALY, FUSION):

    1. Logistic Regression (interpretable linear baseline)
    2. MLPClassifier      (small non-linear classifier)

Evaluation protocol:
    - StratifiedGroupKFold (groups = filename_ori) to prevent leakage
    - Per-fold median imputation + scaling fitted on TRAIN only
    - F0.5-optimized probability threshold per fold (train-side)
    - Reports: F0.5, F1, Precision, Recall, Accuracy

Input:
    results/classification/classifier_features.parquet
    results/threshold/threshold_analysis_summary.csv

Output:
    results/classification/classifier_cv_results.csv
    results/classification/classifier_comparison.csv
    plots/classification/classifier_comparison.pdf
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

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    PLOT_DPI,
    PLOT_STYLE_PARAMS,
)
from utils.categorization import fbeta_score_curve

plt.rcParams.update(PLOT_STYLE_PARAMS)


# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_TABLE = Path("results/classification/classifier_features.parquet")
OUTPUT_DIR    = Path("results/classification")
PLOTS_DIR     = Path("plots/classification")

# Columns that are metadata / labels — NOT features
METADATA_COLS = {
    "pair_id",
    "time",
    "filename_ori",
    "filename_mod",
    "final_mod_type",
    "negative_tier",
    "is_plagiarised",
    "clean_mod_type",
    "category_grouped",
    "source_key_ori",
    "source_key_mod",
}

# Vocal columns that ARE usable features
VOCAL_FEATURE_COLS = {
    "pair_vocal_valid",
    "vocal_ratio_ori",
    "vocal_ratio_mod",
    "vocal_valid_ori",
    "vocal_valid_mod",
}


# ── Feature selection ─────────────────────────────────────────────────────────
def _get_feature_columns(df: pd.DataFrame) -> list:
    """
    Identify classifier feature columns by exclusion.
    Everything that is not metadata and not all-NaN is a feature.
    """
    feature_cols = []
    for col in df.columns:
        if col in METADATA_COLS:
            continue
        if col in VOCAL_FEATURE_COLS or col.startswith(("clews_", "wealy_")):
            if not df[col].isna().all():
                feature_cols.append(col)
    return sorted(feature_cols)


# ── Threshold optimization ────────────────────────────────────────────────────
def _find_optimal_probability_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = BETA,
) -> float:
    """
    Find the probability threshold that maximizes F-beta on the given data.
    Uses the same PR-curve sweep approach as optimal_threshold.py.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fbeta_scores = fbeta_score_curve(precision, recall, beta)

    if len(fbeta_scores) <= 1:
        return 0.5

    optimal_idx = int(np.argmax(fbeta_scores[:-1]))
    return float(thresholds[optimal_idx])


# ── Model factory ─────────────────────────────────────────────────────────────
def _build_estimator(classifier_type: str):
    """
    Build a sklearn Pipeline with:
        1. median imputation
        2. standard scaling
        3. classifier
    """
    classifier_type = classifier_type.lower()

    if classifier_type == "lr":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
        )
    elif classifier_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ── Cross-validation ──────────────────────────────────────────────────────────
def run_classifier_cv(
    df: pd.DataFrame,
    feature_cols: list,
    classifier_type: str,
    n_splits: int = NUM_K_FOLDS,
    random_state: int = RANDOM_STATE,
    beta: float = BETA,
) -> pd.DataFrame:
    """
    Run StratifiedGroupKFold cross-validation for a classifier.

    Groups by filename_ori to prevent data leakage.
    Threshold is optimized on TRAIN probabilities for F-beta.

    Returns:
        DataFrame with per-fold metrics.
    """
    y      = df["is_plagiarised"].astype(int).values
    X      = df[feature_cols].values.astype(np.float64)
    groups = df["filename_ori"].values

    # Just for logging — actual handling happens fold-wise via the Pipeline
    nan_rows = int(np.isnan(X).any(axis=1).sum())
    if nan_rows > 0:
        logger.warning(
            f"[{classifier_type.upper()}] Found NaN in {nan_rows} rows across features. "
            f"Will impute with train-fold medians."
        )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    estimator    = _build_estimator(classifier_type)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator.fit(X_train, y_train)

        # Probabilities
        y_prob_train = estimator.predict_proba(X_train)[:, 1]
        y_prob_test  = estimator.predict_proba(X_test)[:, 1]

        # Optimize threshold on TRAIN
        opt_threshold = _find_optimal_probability_threshold(y_train, y_prob_train, beta=beta)

        # Apply threshold on TEST
        y_pred = (y_prob_test >= opt_threshold).astype(int)

        fold_results.append({
            "classifier": classifier_type.upper(),
            "fold": fold_idx,
            "threshold": round(opt_threshold, 4),
            "f05": round(fbeta_score(y_test, y_pred, beta=beta, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_pos_test": int(y_test.sum()),
            "n_neg_test": int((y_test == 0).sum()),
        })

        print(
            f"  [{classifier_type.upper()}] Fold {fold_idx}: "
            f"F0.5={fold_results[-1]['f05']:.4f}  "
            f"P={fold_results[-1]['precision']:.4f}  "
            f"R={fold_results[-1]['recall']:.4f}  "
            f"thresh={opt_threshold:.4f}"
        )

    return pd.DataFrame(fold_results)


# ── Baseline loading ──────────────────────────────────────────────────────────
def _load_threshold_baselines() -> pd.DataFrame:
    """
    Load threshold-based baselines from threshold_analysis_summary.csv.
    """
    rows = []
    thresh_path = Path(SUMMARY_FILES["threshold_analysis"])

    if thresh_path.exists():
        df_thresh = pd.read_csv(thresh_path)
        for _, row in df_thresh.iterrows():
            model = str(row.get("model", "")).upper()
            rows.append({
                "method": f"{model} threshold",
                "f05": round(float(row.get("fbeta", 0)), 4),
                "f1": round(float(row.get("f1", 0)), 4),
                "precision": round(float(row.get("precision", 0)), 4),
                "recall": round(float(row.get("recall", 0)), 4),
                "accuracy": round(float(row.get("accuracy", 0)), 4),
                "metric": str(row.get("metric", "")),
            })

    if not rows:
        logger.warning("No threshold baselines found for comparison.")

    return pd.DataFrame(rows)


# ── Comparison table ──────────────────────────────────────────────────────────
def _build_comparison(
    df_cv_all: pd.DataFrame,
    df_baselines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a unified comparison table:
        baselines + LR + MLP averages.
    """
    rows = []

    if not df_baselines.empty:
        for _, row in df_baselines.iterrows():
            rows.append(row.to_dict())

    for clf_name in ["LR", "MLP"]:
        df_sub = df_cv_all[df_cv_all["classifier"] == clf_name].copy()
        if df_sub.empty:
            continue

        rows.append({
            "method": f"Meta-classifier ({clf_name})",
            "f05": round(df_sub["f05"].mean(), 4),
            "f1": round(df_sub["f1"].mean(), 4),
            "precision": round(df_sub["precision"].mean(), 4),
            "recall": round(df_sub["recall"].mean(), 4),
            "accuracy": round(df_sub["accuracy"].mean(), 4),
            "metric": "distances + Δ summaries + vocal",
        })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────
def _plot_comparison(df_comp: pd.DataFrame, output_dir: Path) -> None:
    """
    Grouped bar chart comparing F0.5, Precision, Recall across methods.
    """
    if df_comp.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    methods = df_comp["method"].values
    f05     = df_comp["f05"].values
    prec    = df_comp["precision"].values
    rec     = df_comp["recall"].values

    x     = np.arange(len(methods))
    width = 0.25

    bars_f05  = ax.bar(x - width, f05,  width, label="F0.5",     color="#2196F3", edgecolor="white")
    bars_prec = ax.bar(x,         prec, width, label="Precision", color="#4CAF50", edgecolor="white")
    bars_rec  = ax.bar(x + width, rec,  width, label="Recall",    color="#FF9800", edgecolor="white")

    for bars in [bars_f05, bars_prec, bars_rec]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Plagiarism Detection: Threshold Baselines vs Learned Classifiers",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "classifier_comparison.pdf"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ── Console output ────────────────────────────────────────────────────────────
def _print_comparison(df_comp: pd.DataFrame) -> None:
    """Print formatted comparison table."""
    print(f"\n{'=' * 90}")
    print(" CLASSIFIERS vs THRESHOLD BASELINES")
    print(f"{'=' * 90}")
    print(
        f"  {'Method':<32} | {'F0.5':>7} | {'Prec':>7} | "
        f"{'Recall':>7} | {'F1':>7} | {'Acc':>7}"
    )
    print(f"  {'-' * 85}")

    for _, row in df_comp.iterrows():
        is_clf = "Meta-classifier" in str(row["method"])
        prefix = "► " if is_clf else "  "
        print(
            f"{prefix}{row['method']:<30} | {row['f05']:>7.4f} | "
            f"{row['precision']:>7.4f} | {row['recall']:>7.4f} | "
            f"{row['f1']:>7.4f} | {row['accuracy']:>7.4f}"
        )

    print(f"{'=' * 90}")


def _print_cv_summary(df_cv: pd.DataFrame, clf_name: str) -> None:
    """Print mean ± std summary for one classifier."""
    print(f"\n  [{clf_name}] CV Average:")
    print(f"    F0.5      = {df_cv['f05'].mean():.4f} ± {df_cv['f05'].std():.4f}")
    print(f"    F1        = {df_cv['f1'].mean():.4f} ± {df_cv['f1'].std():.4f}")
    print(f"    Precision = {df_cv['precision'].mean():.4f} ± {df_cv['precision'].std():.4f}")
    print(f"    Recall    = {df_cv['recall'].mean():.4f} ± {df_cv['recall'].std():.4f}")
    print(f"    Accuracy  = {df_cv['accuracy'].mean():.4f} ± {df_cv['accuracy'].std():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("BINARY PLAGIARISM CLASSIFIERS")
    print("=" * 60)

    # Load feature table
    if not FEATURE_TABLE.exists():
        print(
            f"\n[ERROR] Feature table not found: {FEATURE_TABLE}\n"
            f"Run classifier_features.py first."
        )
        return

    print(f"\nLoading features from {FEATURE_TABLE}…")
    df = pd.read_parquet(FEATURE_TABLE)
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")

    # Identify features
    feature_cols = _get_feature_columns(df)
    print(f"  Feature columns: {len(feature_cols)}")
    for col in feature_cols:
        print(f"    - {col}")

    if not feature_cols:
        print("[ERROR] No feature columns found.")
        return

    # Validate required columns
    if "is_plagiarised" not in df.columns:
        print("[ERROR] 'is_plagiarised' column not found.")
        return
    if "filename_ori" not in df.columns:
        print("[ERROR] 'filename_ori' column not found (needed for grouping).")
        return

    print(f"\nRunning {NUM_K_FOLDS}-Fold StratifiedGroupKFold CV…")
    print("  Groups: filename_ori")
    print(f"  Threshold optimization: F{BETA}")

    all_cv_results = []

    # ── Logistic Regression ──
    print("\n[LR] Model: LogisticRegression(class_weight='balanced')")
    df_cv_lr = run_classifier_cv(df, feature_cols, classifier_type="lr")
    _print_cv_summary(df_cv_lr, "LR")
    all_cv_results.append(df_cv_lr)

    # ── MLP ──
    print("\n[MLP] Model: MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True)")
    df_cv_mlp = run_classifier_cv(df, feature_cols, classifier_type="mlp")
    _print_cv_summary(df_cv_mlp, "MLP")
    all_cv_results.append(df_cv_mlp)

    # Save CV results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_cv_all = pd.concat(all_cv_results, ignore_index=True)

    cv_path = OUTPUT_DIR / "classifier_cv_results.csv"
    df_cv_all.to_csv(cv_path, index=False)
    print(f"\n  CV results saved → {cv_path}")

    # Load threshold baselines
    print("\nLoading threshold baselines…")
    df_baselines = _load_threshold_baselines()
    if not df_baselines.empty:
        for _, row in df_baselines.iterrows():
            print(f"  {row['method']}: F0.5={row['f05']:.4f}")

    # Build comparison
    df_comp = _build_comparison(df_cv_all, df_baselines)

    comp_path = OUTPUT_DIR / "classifier_comparison.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"  Comparison saved → {comp_path}")

    # Print comparison
    _print_comparison(df_comp)

    # Plot
    _plot_comparison(df_comp, PLOTS_DIR)

    print(f"\nResults → {OUTPUT_DIR}/")
    print(f"Plots   → {PLOTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()