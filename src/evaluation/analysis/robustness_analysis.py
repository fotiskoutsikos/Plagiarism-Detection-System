"""
Robustness Analysis for Plagiarism Detection (Strictly Distance-Based):
- Attack model: Reference vs Plagiarized version (+DSP Modification)
- Human: Reference ↔ Cover(+Pitch/Tempo)
- AI: Reference ↔ MusicGen/AudioLDM2/MGE-LDM(+Pitch/Tempo)
- Visualizes:
  1. Distance Trends vs Modification Intensity 
  2. Extreme Stress Tests (combined AI+DSP) via distance boxplots
  * The Optimal Threshold is drawn ONLY as a visual reference line, not used for metric calculations.

Outputs saved to OUTPUT_DIRS["robustness"].
"""

import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
import logging
logger = logging.getLogger(__name__)

# Import centralized utilities
sys.path.insert(0, str(repo_root / "src"))
from utils.constants import MODEL_PATHS, OUTPUT_DIRS, SUMMARY_FILES, PLOT_COLORS, PLOT_DPI
from utils.categorization import extract_features, get_ground_truth_label


def load_threshold_summary(summary_csv=None):
    """
    Loads optimal thresholds from the ablation study.
    Returns dict: {model: {'metric': str, 'threshold': float}}
    """
    if summary_csv is None:
        summary_csv = SUMMARY_FILES['threshold_analysis']

    thresholds = {}
    if os.path.exists(summary_csv):
        df_summary = pd.read_csv(summary_csv)
        for _, row in df_summary.iterrows():
            model = str(row.get('model', '')).upper()
            metric = row.get('metric', '')
            th = float(row.get('optimal_threshold', np.nan))
            if not np.isnan(th):
                thresholds[model] = {'metric': metric, 'threshold': th}
    else:
        logger.warning(f"Threshold summary not found at {summary_csv}. Plots will run without threshold lines.")
    return thresholds


def get_dynamic_metrics(df: pd.DataFrame) -> list:
    """
    Identify distance/composite metrics dynamically, excluding bookkeeping columns.
    """
    metrics = [col for col in df.columns if col.endswith('_distance') or '+' in col]
    invalid_cols = ['filename_ori', 'filename_mod', 'final_mod_type', 'pair_id', 'time']
    return [m for m in metrics if m not in invalid_cols]


def process_model_data(csv_path: str, model_name: str):
    """
    Loads CSV, filters to Positive plagiarism pairs, extracts DSP/intensity features,
    and returns (df, metrics).
    """
    if not os.path.exists(csv_path):
        logger.warning(f"File not found: {csv_path}. Skipping {model_name}.")
        return None, []

    df = pd.read_csv(csv_path)

    # Use ground truth label to keep ONLY positive plagiarism pairs (robustness stress test)
    df['is_plagiarised'] = df['final_mod_type'].apply(get_ground_truth_label)
    df = df[df['is_plagiarised'] == 1].copy()
    if df.empty:
        logger.warning(f"No positive plagiarism pairs found in {model_name}. Skipping.")
        return None, []

    metrics = get_dynamic_metrics(df)
    if not metrics:
        logger.warning(f"No valid metric columns found in {model_name}.")
        return None, []

    # Extract Source + DSP intensity metadata (for line plots)
    features = df['final_mod_type'].apply(extract_features)
    features.columns = ['Source', 'Pitch_Intensity', 'Tempo_Intensity', 'Is_Extreme', 'DSP_Category']
    df = pd.concat([df, features], axis=1)

    # Exclude "Ignore" categories if present
    df = df[df['Source'] != 'Ignore'].copy()

    logger.info(f"{model_name}: {len(df)} positive pairs loaded (Reference ↔ Plagiarized).")
    return df, metrics


def _add_origin_anchors(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Adds synthetic anchor rows at DSP intensity = 0 with distance = 0 for
    the 'Original' source. This represents ori vs ori (no DSP applied),
    which is trivially 0 by definition and does not need to be computed.
    The anchor gives the trend line a meaningful starting point.
    """
    sources_present = df['Source'].unique()
    anchor_rows = []

    for source in sources_present:
        anchor_rows.append({
            'Pitch_Intensity': 0.0,
            'Tempo_Intensity': 1.0,
            metric: 0.0,
            'Source': source,
            'Is_Extreme': False,
            'DSP_Category': 'Base Generation',
        })

    anchors = pd.DataFrame(anchor_rows)
    # Fill any other columns present in df with NaN so concat works cleanly
    for col in df.columns:
        if col not in anchors.columns:
            anchors[col] = np.nan

    return pd.concat([df, anchors], ignore_index=True)


def plot_distance_trends(df: pd.DataFrame, model_name: str, metric: str, threshold: float, output_dir: str):
    """
    Plots Distance trends over Pitch/Tempo intensity for Human vs AI sources.
    Original source anchored at (0, 0) — distance is trivially 0 when no DSP is applied.
    The threshold is plotted ONLY as a visual reference line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    display_metric = metric.replace("_distance", "").replace("+", " + ").title()

    existing_sources = [s for s in ['Original', 'Cover', 'MusicGen', 'AudioLDM2', 'MGE-LDM'] if s in df['Source'].unique()]
    if not existing_sources:
        logger.warning(f"No valid sources in {model_name} for distance trends.")
        plt.close(fig)
        return

    # ── Plot 1: Pitch Modification Trend ──────────────────────────────────────
    pitch_df = df[
        (df['Tempo_Intensity'] == 1.0) & (~df['Is_Extreme']) |
        (df['DSP_Category'] == 'Base Generation')
    ].copy()

    # Anchor: every source starts at pitch=0, distance=0
    pitch_df = _add_origin_anchors(pitch_df, metric)

    if len(pitch_df) > 0 and len(pitch_df['Pitch_Intensity'].unique()) > 1:
        sns.lineplot(
            data=pitch_df, x='Pitch_Intensity', y=metric, hue='Source',
            hue_order=existing_sources, palette=PLOT_COLORS,
            marker='o', markersize=8, ax=axes[0], errorbar=None, linewidth=2
        )
        axes[0].set_title('Distance vs Pitch Shift', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Pitch Shift (Semitones)', fontsize=11)
        axes[0].set_ylabel(f'{display_metric} Distance', fontsize=11)
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.3, label='Base (0)')
        axes[0].axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                        label=f'Threshold Line ({threshold:.3f})')
        axes[0].grid(alpha=0.3)
        axes[0].legend(title='Source', loc='upper left')

    # ── Plot 2: Tempo Modification Trend ──────────────────────────────────────
    tempo_df = df[
        (df['Pitch_Intensity'] == 0.0) & (~df['Is_Extreme']) |
        (df['DSP_Category'] == 'Base Generation')
    ].copy()

    # Anchor: every source starts at tempo=1.0, distance=0
    tempo_df = _add_origin_anchors(tempo_df, metric)

    if len(tempo_df) > 0 and len(tempo_df['Tempo_Intensity'].unique()) > 1:
        sns.lineplot(
            data=tempo_df, x='Tempo_Intensity', y=metric, hue='Source',
            hue_order=existing_sources, palette=PLOT_COLORS,
            marker='o', markersize=8, ax=axes[1], errorbar=None, linewidth=2
        )
        axes[1].set_title('Distance vs Tempo Shift', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Tempo Factor', fontsize=11)
        axes[1].set_ylabel(f'{display_metric} Distance', fontsize=11)
        axes[1].axvline(x=1.0, color='black', linestyle='--', alpha=0.3, label='Base (1.0x)')
        axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                        label=f'Threshold Line ({threshold:.3f})')
        axes[1].grid(alpha=0.3)
        axes[1].legend(title='Source', loc='upper left')

    plt.suptitle(
        f'{model_name} - Robustness Analysis: Distance vs Modification Intensity',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()

    safe_metric_name = metric.replace("+", "_PLUS_")
    plot_path = Path(output_dir) / f"{model_name.lower()}_{safe_metric_name}_distance_trends.png"
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Saved Distance Trends for {model_name} ({metric})")


def plot_extreme_scenarios(df: pd.DataFrame, model_name: str, metric: str, output_dir: str):
    """
    Boxplots comparing Distances for Base vs Extreme Down vs Extreme Up.
    """
    target_cats = ["Base Generation", "Extreme Down", "Extreme Up"]
    df_extremes = df[df['DSP_Category'].isin(target_cats)].copy()

    if df_extremes.empty:
        logger.info(f"No extreme scenarios found for {model_name} ({metric}). Skipping extreme boxplot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    display_metric = metric.replace("_distance", "").replace("+", " + ").title()

    sns.boxplot(
        data=df_extremes, x='DSP_Category', y=metric, hue='Source',
        order=target_cats, palette=PLOT_COLORS, ax=ax,
        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"}
    )

    ax.set_title(
        f'{model_name} - Distance Under Extreme Stress (AI+DSP Combinations)',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylabel(f'{display_metric} Distance', fontsize=12)
    ax.set_xlabel('Modification Scenario', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    safe_metric_name = metric.replace("+", "_PLUS_")
    plot_path = Path(output_dir) / f"{model_name.lower()}_{safe_metric_name}_extreme_stress_test.png"
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Saved Extreme Stress Test for {model_name} ({metric})")


def main():
    logger.info("=" * 70)
    logger.info("ROBUSTNESS ANALYSIS (Distances Only - Reference ↔ Plagiarized)")
    logger.info("=" * 70)

    output_dir = OUTPUT_DIRS["robustness"]
    os.makedirs(output_dir, exist_ok=True)

    # Load thresholds strictly for drawing the visual reference line
    threshold_map = load_threshold_summary()

    for model_name, csv_path in MODEL_PATHS.items():
        logger.info(f"\nProcessing Robustness for {model_name}...")
        df_processed, found_metrics = process_model_data(csv_path, model_name)

        if df_processed is None or not found_metrics:
            continue

        # Get the official winning metric and threshold for this model
        winning_info = threshold_map.get(model_name)
        if not winning_info:
            logger.warning(f"No threshold found for {model_name}. Run optimal_threshold.py first. Skipping plots.")
            continue

        target_metric = winning_info['metric']
        opt_thresh = winning_info['threshold']
        logger.info(f"Using {target_metric} (Threshold visual line = {opt_thresh:.4f})")

        if target_metric not in df_processed.columns:
            logger.warning(f"Metric '{target_metric}' not found in dataframe for {model_name}. Skipping.")
            continue

        # Plot purely objective Distance Trends
        plot_distance_trends(df_processed, model_name, target_metric, opt_thresh, output_dir)

        # Plot objective Distances for Extreme scenarios
        plot_extreme_scenarios(df_processed, model_name, target_metric, output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE. Distance robustness plots ready.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()