"""
Binary Classification Analysis for Plagiarism Detection (Pairwise).
Evaluates the model using FIXED thresholds and WINNING METRICS imported
from 'threshold_analysis_summary.csv'.
Categorizes results into Human, Original+DSP, AI(Base), and AI+DSP.
Calculates F0.5-score and exports a False Positive Tier Breakdown.
"""

import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score, f1_score

# Resolve repository root and load logging_util
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
setup_logging = logging_util.setup_logging

# Import centralized utilities
sys.path.insert(0, str(repo_root / "src"))
from utils.constants import MODEL_PATHS, OUTPUT_DIRS, SUMMARY_FILES
from utils.categorization import get_ground_truth_label, clean_mod_type, categorize_modification

setup_logging(__file__)


# METRICS HELPER
def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard classification metrics for a binary prediction array.
    Uses sklearn consistently — no manual reimplementation.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Precision':  precision_score(y_true, y_pred, zero_division=0),
        'Recall':     recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':   f1_score(y_true, y_pred, zero_division=0),
        'F0.5-Score': fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    }


# MAIN EVALUATION
def run_binary_classification_evaluation(
    df_all:           pd.DataFrame,
    model_name:       str,
    fixed_threshold:  float,
    target_metric:    str,
) -> pd.DataFrame:
    """
    Evaluate pairwise similarity using a predefined distance threshold and metric.

    Steps:
      1. Assign ground truth labels.
      2. Apply threshold to produce binary predictions.
      3. Compute metrics per broad category (positives vs all negatives).
      4. Compute metrics per granular mod-type.
      5. Compute overall metrics.
      6. Save all results and False Positive tier breakdown.
    """
    output_dir = Path(OUTPUT_DIRS['binary_classification'])
    output_dir.mkdir(parents=True, exist_ok=True)

    df_eval = df_all.copy()

    # Ground truth & categorization
    df_eval['y_true']           = df_eval['final_mod_type'].apply(get_ground_truth_label)
    df_eval['clean_mod_type']   = df_eval['final_mod_type'].apply(clean_mod_type)
    df_eval['category_grouped'] = df_eval['clean_mod_type'].apply(categorize_modification)

    # Group all negatives under one label for display
    df_eval.loc[df_eval['y_true'] == 0, 'category_grouped'] = 'Negative Pairs'

    # Validate target metric exists
    if target_metric not in df_eval.columns:
        raise ValueError(
            f"Winning metric '{target_metric}' not found in {model_name} dataset."
        )

    # Binary prediction: lower distance = more similar → predict positive
    df_eval['y_pred'] = (df_eval[target_metric] <= fixed_threshold).astype(int)

    # Per-category metrics (each positive category vs all negatives)
    metrics_list = []
    positive_categories = sorted(df_eval[df_eval['y_true'] == 1]['category_grouped'].unique())

    for cat in positive_categories:
        mask   = (df_eval['category_grouped'] == cat) | (df_eval['y_true'] == 0)
        df_sub = df_eval[mask]
        m      = _compute_binary_metrics(df_sub['y_true'].values, df_sub['y_pred'].values)
        metrics_list.append({
            'Category':      cat,
            'Total_Queries': m['TP'] + m['FN'],
            **m,
        })

    # Per-mod-type detailed metrics
    detailed_list = []
    for mod in sorted(df_eval[df_eval['y_true'] == 1]['clean_mod_type'].unique()):
        mask   = (df_eval['clean_mod_type'] == mod) | (df_eval['y_true'] == 0)
        df_sub = df_eval[mask]
        if len(df_sub['y_true'].unique()) < 2:
            continue
        m = _compute_binary_metrics(df_sub['y_true'].values, df_sub['y_pred'].values)
        detailed_list.append({'Modification_Type': mod, **m})

    df_detailed = pd.DataFrame(detailed_list)
    df_detailed.to_csv(
        output_dir / f'{model_name.lower()}_detailed_metrics.csv', index=False
    )

    # Overall metrics (all positives vs all negatives)
    m_overall = _compute_binary_metrics(df_eval['y_true'].values, df_eval['y_pred'].values)
    metrics_list.append({
        'Category':      'OVERALL',
        'Total_Queries': m_overall['TP'] + m_overall['FN'],
        **m_overall,
    })

    df_metrics = pd.DataFrame(metrics_list)

    # Print formatted table
    print(f"\n{'=' * 115}")
    print(f" BINARY CLASSIFICATION PERFORMANCE: {model_name.upper()}")
    print(f" Metric: {target_metric} | Threshold: <= {fixed_threshold:.4f}")
    print(f"{'=' * 115}")
    print(
        f"{'Modification Category':<30} | {'Precision':>9} | {'Recall':>8} | "
        f"{'F1-Score':>8} | {'F0.5-Score':>9} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>6}"
    )
    print(f"{'-' * 115}")

    for _, row in df_metrics.iterrows():
        is_overall = row['Category'] == 'OVERALL'
        prefix     = "► " if is_overall else "  "
        print(
            f"{prefix}{row['Category']:<28} | {row['Precision']:>8.1%} | "
            f"{row['Recall']:>7.1%} | {row['F1-Score']:>7.1%} | "
            f"{row['F0.5-Score']:>8.1%} | {row['TP']:>5} | "
            f"{row['FP']:>5} | {row['FN']:>5} | {row['TN']:>6}"
        )
    print(f"{'=' * 115}\n")

    # Save per-model CSV
    df_metrics.to_csv(
        output_dir / f'{model_name.lower()}_binary_metrics.csv', index=False
    )

    # Append to shared summary CSV
    summary_path = Path(SUMMARY_FILES['binary_summary'])
    df_summary   = df_metrics.copy()
    df_summary.insert(0, 'Model', model_name)
    df_summary.insert(1, 'Metric', target_metric)
    df_summary['Applied_Threshold'] = fixed_threshold
    df_summary.to_csv(
        summary_path, index=False,
        mode='a', header=not summary_path.exists()
    )

    # False Positive Tier Breakdown
    if 'negative_tier' in df_eval.columns:
        fp_df = df_eval[(df_eval['y_true'] == 0) & (df_eval['y_pred'] == 1)]
        if not fp_df.empty:
            fp_breakdown = (
                fp_df
                .groupby(['category_grouped', 'negative_tier'])
                .size()
                .reset_index(name='FP_Count')
            )
            fp_path = output_dir / f"{model_name.lower()}_{target_metric}_fp_tier_breakdown.csv"
            fp_breakdown.to_csv(fp_path, index=False)
            print(f"False Positive Tier Breakdown saved → {fp_path}")
        else:
            print("No False Positives found — perfect separation!")
    else:
        print("'negative_tier' column not present; skipping FP tier analysis.")

    return df_metrics


# ENTRY POINT
def main():
    print("=" * 80)
    print("STARTING BINARY CLASSIFICATION WITH OPTIMAL THRESHOLDS")
    print("=" * 80)

    # Clear old summary so we start fresh on each run
    summary_path = Path(SUMMARY_FILES['binary_summary'])
    if summary_path.exists():
        summary_path.unlink()

    # Load optimal thresholds from ablation study
    threshold_file = Path(SUMMARY_FILES['threshold_analysis'])
    thresholds_dict: dict = {}

    if threshold_file.exists():
        df_thresh = pd.read_csv(threshold_file)
        print(f"Loaded {len(df_thresh)} optimal configurations from {threshold_file}:")
        for _, row in df_thresh.iterrows():
            model_key  = str(row['model']).upper()
            opt_metric = str(row['metric'])
            opt_thresh = float(row['optimal_threshold'])
            thresholds_dict[model_key] = {'metric': opt_metric, 'threshold': opt_thresh}
            print(f"  → {model_key}: {opt_metric} (Threshold: {opt_thresh:.4f})")
    else:
        print(
            f"\n[ERROR] Required threshold file '{threshold_file}' not found. "
            "Run optimal_threshold.py first."
        )
        return

    # Run evaluation for each model
    processed_any = False
    for model, csv_path in MODEL_PATHS.items():
        csv_path_obj = Path(csv_path)

        if not csv_path_obj.exists():
            print(f"Warning: Distance file not found → {csv_path}")
            continue

        if model not in thresholds_dict:
            print(f"Warning: No threshold mapped for {model}. Skipping.")
            continue

        fixed_thresh  = thresholds_dict[model]['threshold']
        target_metric = thresholds_dict[model]['metric']
        df_all        = pd.read_csv(csv_path_obj)

        try:
            run_binary_classification_evaluation(df_all, model, fixed_thresh, target_metric)
            processed_any = True
        except ValueError as e:
            print(f"\n[ERROR] Skipping {model}: {e}")

    if processed_any:
        print(f"\n[SUCCESS] Results saved → {OUTPUT_DIRS['binary_classification']}/")
    else:
        print("\n[FAILED] No distance CSV files found to process.")


if __name__ == "__main__":
    main()