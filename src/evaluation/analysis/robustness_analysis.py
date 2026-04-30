"""
Robustness Analysis for Plagiarism Detection (Strictly Distance-Based):
- Attack model: Reference vs Plagiarized version (+DSP Modification)
- Human: Reference ↔ Cover(+Pitch/Tempo)
- AI: Reference ↔ MusicGen/AudioLDM2/MGE-LDM(+Pitch/Tempo)
- Visualizes:
  1. Distance Trends vs Modification Intensity
  2. Extreme Stress Tests (combined AI+DSP) via distance boxplots
  * The Optimal Threshold is drawn ONLY as a visual reference line,
    not used for metric calculations.

Outputs saved to OUTPUT_DIRS["robustness"].
"""

import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

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

setup_logging(__file__)
logger = logging.getLogger(__name__)

# Import centralized utilities
sys.path.insert(0, str(repo_root / "src"))
from utils.constants import (
    MODEL_PATHS, OUTPUT_DIRS, SUMMARY_FILES,
    PLOT_COLORS, PLOT_DPI, PLOT_STYLE_PARAMS,
    AUDIO_SOURCES,
)
from utils.categorization import extract_features, get_ground_truth_label

plt.rcParams.update(PLOT_STYLE_PARAMS)


# THRESHOLD LOADING
def load_threshold_summary(summary_csv: str = None) -> dict:
    """
    Loads optimal thresholds from the ablation study.
    Returns dict: {model: {'metric': str, 'threshold': float}}
    """
    if summary_csv is None:
        summary_csv = SUMMARY_FILES['threshold_analysis']

    thresholds = {}
    summary_path = Path(summary_csv)

    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
        for _, row in df_summary.iterrows():
            model  = str(row.get('model', '')).upper()
            metric = row.get('metric', '')
            th     = float(row.get('optimal_threshold', np.nan))
            if not np.isnan(th):
                thresholds[model] = {'metric': metric, 'threshold': th}
    else:
        logger.warning(
            f"Threshold summary not found at {summary_csv}. "
            "Plots will run without threshold lines."
        )
    return thresholds


# DATA LOADING
def get_dynamic_metrics(df: pd.DataFrame) -> list:
    """
    Identify distance/composite metrics dynamically,
    excluding bookkeeping columns.
    """
    invalid_cols = {'filename_ori', 'filename_mod', 'final_mod_type', 'pair_id', 'time'}
    return [
        col for col in df.columns
        if (col.endswith('_distance') or '+' in col) and col not in invalid_cols
    ]


def process_model_data(csv_path: str, model_name: str):
    """
    Loads CSV, filters to Positive plagiarism pairs only,
    extracts DSP/intensity features, and returns (df, metrics).
    """
    if not Path(csv_path).exists():
        logger.warning(f"File not found: {csv_path}. Skipping {model_name}.")
        return None, []

    df = pd.read_csv(csv_path)

    # Keep ONLY positive plagiarism pairs (robustness stress test)
    df['is_plagiarised'] = df['final_mod_type'].apply(get_ground_truth_label)
    df = df[df['is_plagiarised'] == 1].copy()
    if df.empty:
        logger.warning(f"No positive pairs found in {model_name}. Skipping.")
        return None, []

    metrics = get_dynamic_metrics(df)
    if not metrics:
        logger.warning(f"No valid metric columns found in {model_name}.")
        return None, []

    # Extract Source + DSP intensity metadata for line plots
    features = df['final_mod_type'].apply(extract_features)
    features.columns = ['Source', 'Pitch_Intensity', 'Tempo_Intensity', 'Is_Extreme', 'DSP_Category']
    df = pd.concat([df, features], axis=1)

    # Exclude Ignore categories
    df = df[df['Source'] != 'Ignore'].copy()

    print(f"{model_name}: {len(df)} positive pairs loaded (Reference ↔ Plagiarized).")
    return df, metrics


# PLOTTING HELPERS
def _add_origin_anchors(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Adds synthetic anchor rows at DSP intensity = 0 with distance = 0.
    Represents ori vs ori (no DSP applied), trivially 0 by definition.
    Gives the trend line a meaningful starting point at the origin.
    """
    anchor_rows = []
    for source in df['Source'].unique():
        anchor_rows.append({
            'Pitch_Intensity': 0.0,
            'Tempo_Intensity': 1.0,
            metric:            0.0,
            'Source':          source,
            'Is_Extreme':      False,
            'DSP_Category':    'Base Generation',
        })

    anchors = pd.DataFrame(anchor_rows)
    for col in df.columns:
        if col not in anchors.columns:
            anchors[col] = np.nan

    return pd.concat([df, anchors], ignore_index=True)


# PLOTS
def plot_distance_trends(
    df:           pd.DataFrame,
    model_name:   str,
    metric:       str,
    threshold:    float,
    output_dir:   str,
) -> None:
    """
    Line plots of Distance vs Pitch/Tempo intensity, split by Source.
    Threshold drawn as visual reference only.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    display_metric = metric.replace("_distance", "").replace("+", " + ").title()

    # Preserve display order defined in constants
    existing_sources = [s for s in AUDIO_SOURCES if s in df['Source'].unique()]
    if not existing_sources:
        logger.warning(f"No valid sources in {model_name} for distance trends.")
        plt.close(fig)
        return

    # Plot 1: Pitch
    pitch_df = df[
        ((df['Tempo_Intensity'] == 1.0) & (~df['Is_Extreme'])) |
        (df['DSP_Category'] == 'Base Generation')
    ].copy()
    pitch_df = _add_origin_anchors(pitch_df, metric)

    if len(pitch_df) > 0 and len(pitch_df['Pitch_Intensity'].unique()) > 1:
        sns.lineplot(
            data=pitch_df, x='Pitch_Intensity', y=metric, hue='Source',
            hue_order=existing_sources, palette=PLOT_COLORS,
            marker='o', markersize=8, ax=axes[0], errorbar=None, linewidth=2,
        )
        axes[0].set_title('Distance vs Pitch Shift', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Pitch Shift (Semitones)', fontsize=11)
        axes[0].set_ylabel(f'{display_metric} Distance', fontsize=11)
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.3, label='Base (0)')
        axes[0].axhline(
            y=threshold, color='red', linestyle='--', linewidth=2,
            label=f'Threshold Line ({threshold:.3f})'
        )
        axes[0].grid(alpha=0.3)
        axes[0].legend(title='Source', loc='upper left')

    # Plot 2: Tempo
    tempo_df = df[
        ((df['Pitch_Intensity'] == 0.0) & (~df['Is_Extreme'])) |
        (df['DSP_Category'] == 'Base Generation')
    ].copy()
    tempo_df = _add_origin_anchors(tempo_df, metric)

    if len(tempo_df) > 0 and len(tempo_df['Tempo_Intensity'].unique()) > 1:
        sns.lineplot(
            data=tempo_df, x='Tempo_Intensity', y=metric, hue='Source',
            hue_order=existing_sources, palette=PLOT_COLORS,
            marker='o', markersize=8, ax=axes[1], errorbar=None, linewidth=2,
        )
        axes[1].set_title('Distance vs Tempo Shift', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Tempo Factor', fontsize=11)
        axes[1].set_ylabel(f'{display_metric} Distance', fontsize=11)
        axes[1].axvline(x=1.0, color='black', linestyle='--', alpha=0.3, label='Base (1.0x)')
        axes[1].axhline(
            y=threshold, color='red', linestyle='--', linewidth=2,
            label=f'Threshold Line ({threshold:.3f})'
        )
        axes[1].grid(alpha=0.3)
        axes[1].legend(title='Source', loc='upper left')

    plt.suptitle(
        f'{model_name} — Robustness Analysis: Distance vs Modification Intensity',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()

    safe_metric = metric.replace("+", "_PLUS_")
    plot_path   = Path(output_dir) / f"{model_name.lower()}_{safe_metric}_distance_trends.pdf"
    fig.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Distance Trends → {plot_path}")


def plot_extreme_scenarios(
    df:         pd.DataFrame,
    model_name: str,
    metric:     str,
    output_dir: str,
) -> None:
    """
    Boxplots comparing Distances for Base vs Extreme Down vs Extreme Up.
    """
    target_cats = ["Base Generation", "Extreme Down", "Extreme Up"]
    df_extremes = df[df['DSP_Category'].isin(target_cats)].copy()

    if df_extremes.empty:
        print(f"No extreme scenarios for {model_name} ({metric}). Skipping.")
        return

    display_metric = metric.replace("_distance", "").replace("+", " + ").title()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df_extremes, x='DSP_Category', y=metric, hue='Source',
        order=target_cats, palette=PLOT_COLORS, ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    ax.set_title(
        f'{model_name} — Distance Under Extreme Stress (AI+DSP Combinations)',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylabel(f'{display_metric} Distance', fontsize=12)
    ax.set_xlabel('Modification Scenario', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    safe_metric = metric.replace("+", "_PLUS_")
    plot_path   = Path(output_dir) / f"{model_name.lower()}_{safe_metric}_extreme_stress_test.pdf"
    fig.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Extreme Stress Test → {plot_path}")



def main():
    print("=" * 70)
    print("ROBUSTNESS ANALYSIS (Distances Only — Reference ↔ Plagiarized)")
    print("=" * 70)

    output_dir = Path(OUTPUT_DIRS["robustness"])
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_map = load_threshold_summary()

    for model_name, csv_path in MODEL_PATHS.items():
        print(f"\nProcessing Robustness for {model_name}...")
        df_processed, found_metrics = process_model_data(csv_path, model_name)

        if df_processed is None or not found_metrics:
            continue

        winning_info = threshold_map.get(model_name)
        if not winning_info:
            logger.warning(
                f"No threshold found for {model_name}. "
                "Run optimal_threshold.py first. Skipping."
            )
            continue

        target_metric = winning_info['metric']
        opt_thresh    = winning_info['threshold']
        print(f"Using {target_metric} (Threshold visual line = {opt_thresh:.4f})")

        if target_metric not in df_processed.columns:
            logger.warning(
                f"Metric '{target_metric}' not in dataframe for {model_name}. Skipping."
            )
            continue

        plot_distance_trends(
            df_processed, model_name, target_metric, opt_thresh, str(output_dir)
        )
        plot_extreme_scenarios(
            df_processed, model_name, target_metric, str(output_dir)
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE. Distance robustness plots ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()