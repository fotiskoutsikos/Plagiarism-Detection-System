"""
Supervised Classification Results — Presentation Layer.

This script presents the final supervised model evaluation results
in the same format as binary_classification.py to enable direct
apples-to-apples comparison between threshold-based and supervised approaches.

Key Design Principles
---------------------
1. Pure presentation — no model training, no CV, no computation.
2. Reads pre-computed results from selected_model_evaluation.py.
3. Prints with identical formatting to binary_classification.py.
4. Saves outputs to a separate directory for clarity.

Configuration
-------------
Edit SELECTED_CONFIG below to specify which evaluated configuration
to present as the final supervised model.

Inputs
------
    - results/classification/{config}_broad_metrics.csv
    - results/classification/{config}_detailed_metrics.csv
    - results/classification/{config}_fp_tier_breakdown.csv

Outputs (mirroring binary_classification.py structure)
-------
    - results/binary_supervised_classification/supervised_broad_metrics.csv
    - results/binary_supervised_classification/supervised_detailed_metrics.csv
    - results/binary_supervised_classification/supervised_fp_tier_breakdown.csv
"""

import sys
import importlib.util
from pathlib import Path
import pandas as pd

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

sys.path.insert(0, str(repo_root / "src"))
from utils.constants import CLASSIFICATION_RESULTS_DIR

# CONFIGURATION: Select which evaluated model to present
SELECTED_CONFIG = "hybrid_top512"

INPUT_DIR  = Path(CLASSIFICATION_RESULTS_DIR)
OUTPUT_DIR = Path("results/binary_supervised_classification")


def _safe_name(text: str) -> str:
    """Convert config name to filesystem-safe identifier."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def present_supervised_results(config_name: str) -> None:
    """
    Load and present supervised classification results with the same
    formatting as binary_classification.py.

    Args:
        config_name: The configuration identifier (e.g., "hybrid_top512").
    """
    safe = _safe_name(config_name)

    # Load CSVs 
    broad_path    = INPUT_DIR / f"{safe}_broad_metrics.csv"
    detailed_path = INPUT_DIR / f"{safe}_detailed_metrics.csv"
    fp_path       = INPUT_DIR / f"{safe}_fp_tier_breakdown.csv"

    if not broad_path.exists():
        print(f"\n[ERROR] Broad metrics not found: {broad_path}")
        print(f"Run selected_model_evaluation.py with --config {config_name} first.")
        return

    df_broad = pd.read_csv(broad_path)

    # Print formatted table
    print(f"\n{'=' * 115}")
    print(f" SUPERVISED CLASSIFICATION PERFORMANCE: {config_name.upper()}")
    print(f" Model: XGBoost | Out-of-Fold Predictions (5-Fold StratifiedGroupKFold)")
    print(f"{'=' * 115}")
    print(
        f"{'Modification Category':<30} | {'Precision':>9} | {'Recall':>8} | "
        f"{'F1-Score':>8} | {'F0.5-Score':>9} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>6}"
    )
    print(f"{'-' * 115}")

    for _, row in df_broad.iterrows():
        is_overall = row['Category'] == 'OVERALL'
        prefix     = "► " if is_overall else "  "
        print(
            f"{prefix}{row['Category']:<28} | {row['Precision']:>8.1%} | "
            f"{row['Recall']:>7.1%} | {row['F1-Score']:>7.1%} | "
            f"{row['F0.5-Score']:>8.1%} | {row['TP']:>5} | "
            f"{row['FP']:>5} | {row['FN']:>5} | {row['TN']:>6}"
        )
    print(f"{'=' * 115}\n")

    # Save to supervised output directory 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_broad = OUTPUT_DIR / "supervised_broad_metrics.csv"
    df_broad.to_csv(out_broad, index=False)
    print(f"  Broad metrics saved → {out_broad}")

    if detailed_path.exists():
        df_detailed = pd.read_csv(detailed_path)
        out_detailed = OUTPUT_DIR / "supervised_detailed_metrics.csv"
        df_detailed.to_csv(out_detailed, index=False)
        print(f"  Detailed metrics saved → {out_detailed}")

    if fp_path.exists():
        df_fp = pd.read_csv(fp_path)
        if not df_fp.empty:
            out_fp = OUTPUT_DIR / "supervised_fp_tier_breakdown.csv"
            df_fp.to_csv(out_fp, index=False)
            print(f"  FP tier breakdown saved → {out_fp}")
            print(f"\n  False Positive Breakdown:")
            print(df_fp.to_string(index=False))
        else:
            print(f"  No False Positives in OOF predictions — perfect separation!")
    else:
        print(f"  'negative_tier' column not present; FP tier analysis unavailable.")

    print(f"\n{'=' * 115}")
    print(f"SUPERVISED RESULTS PRESENTATION COMPLETE")
    print(f"  Configuration: {config_name}")
    print(f"  Results saved → {OUTPUT_DIR}/")
    print(f"{'=' * 115}")


def main():
    print("=" * 80)
    print("SUPERVISED CLASSIFICATION RESULTS — PRESENTATION")
    print("=" * 80)

    print(f"\nSelected configuration: {SELECTED_CONFIG}")
    print("(Edit SELECTED_CONFIG at the top of the script to change)")

    present_supervised_results(SELECTED_CONFIG)


if __name__ == "__main__":
    main()