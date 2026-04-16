"""
Robustness Analysis for Plagiarism Detection: 
1. Direct modification (Original + DSP)
2. Generative modification (AI + DSP)

Visualizes:
- How raw distances change (Distance Trends).
- How the actual detection performance drops (F1-Score Degradation).
"""

import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

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
    """Loads the optimal thresholds from the ablation study."""
    if summary_csv is None:
        summary_csv = SUMMARY_FILES['threshold_analysis']
    
    thresholds = {}
    if os.path.exists(summary_csv):
        df_summary = pd.read_csv(summary_csv)
        for _, row in df_summary.iterrows():
            model = str(row.get('model', '')).upper()
            metric = row.get('metric', '')
            th = float(row.get('optimal_threshold', np.nan))
            thresholds[model] = {'metric': metric, 'threshold': th}
    return thresholds


def get_dynamic_metrics(df: pd.DataFrame) -> list:
    metrics = [col for col in df.columns if col.endswith('_distance') or '+' in col]
    invalid_cols = ['filename_ori', 'filename_mod', 'final_mod_type', 'pair_id', 'time']
    return [m for m in metrics if m not in invalid_cols]


def process_model_data(csv_path: str, model_name: str):
    if not os.path.exists(csv_path):
        logger.warning(f"File not found: {csv_path}. Skipping {model_name}.")
        return None, []

    df = pd.read_csv(csv_path)
    df['is_plagiarised'] = df['final_mod_type'].apply(get_ground_truth_label)
    metrics = get_dynamic_metrics(df)
    
    if not metrics:
        logger.warning(f"No valid metric columns found in {model_name}.")
        return None, []

    features = df['final_mod_type'].apply(extract_features)
    features.columns = ['Source', 'Pitch_Intensity', 'Tempo_Intensity', 'Is_Extreme', 'DSP_Category']
    df = pd.concat([df, features], axis=1)
    
    df = df[df['Source'] != 'Ignore'].copy()
    return df, metrics


def plot_distance_trends(df: pd.DataFrame, model_name: str, metric: str, output_dir: str):
    """Plots lines showing raw Distance trends over intensity."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    display_metric = metric.replace("_distance", "").replace("+", " + ").title()
    
    existing_sources = [s for s in ['Original', 'MusicGen', 'AudioLDM2', 'MGE-LDM'] if s in df['Source'].unique()]
    
    # Plot 1: Pure Pitch Trend
    pitch_df = df[(df['Tempo_Intensity'] == 1.0) & (~df['Is_Extreme']) | (df['DSP_Category'] == 'Base Generation')].copy()
    if len(pitch_df['Pitch_Intensity'].unique()) > 1:
        sns.lineplot(data=pitch_df, x='Pitch_Intensity', y=metric, hue='Source', 
                     hue_order=existing_sources, palette=PLOT_COLORS, 
                     marker='o', markersize=8, ax=axes[0], errorbar=None, linewidth=2)
        
        axes[0].set_title('Pitch Modification (Distance Impact)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Pitch Shift (Semitones)', fontsize=11)
        axes[0].set_ylabel(f'{display_metric} Distance', fontsize=11)
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.3, label='Base (0)')
        axes[0].grid(alpha=0.3)
        
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(title='Source Generation', loc='upper left')

    # Plot 2: Pure Tempo Trend
    tempo_df = df[(df['Pitch_Intensity'] == 0.0) & (~df['Is_Extreme']) | (df['DSP_Category'] == 'Base Generation')].copy()
    if len(tempo_df['Tempo_Intensity'].unique()) > 1:
        sns.lineplot(data=tempo_df, x='Tempo_Intensity', y=metric, hue='Source', 
                     hue_order=existing_sources, palette=PLOT_COLORS, 
                     marker='o', markersize=8, ax=axes[1], errorbar=None, linewidth=2)
        
        axes[1].set_title('Tempo Modification (Distance Impact)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Tempo Factor (e.g., 0.90x, 1.10x)', fontsize=11)
        axes[1].set_ylabel(f'{display_metric} Distance', fontsize=11)
        axes[1].axvline(x=1.0, color='black', linestyle='--', alpha=0.3, label='Base (1.0x)')
        axes[1].grid(alpha=0.3)
        axes[1].legend(title='Source Generation', loc='upper left')

    plt.suptitle(f'{model_name} - Distance Shift vs Modification Intensity', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    safe_metric_name = metric.replace("+", "_PLUS_")
    plot_path = Path(output_dir) / f"{model_name.lower()}_{safe_metric_name}_distance_trends.png"
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_f1_degradation_trends(df: pd.DataFrame, model_name: str, metric: str, threshold: float, output_dir: str):
    """Plots lines showing F1-Score degradation over modification intensity."""
    
    # Invert logic: for distance, prediction is 1 if distance <= threshold
    invert_logic = 'distance' in metric.lower() or '+' in metric.lower()
    df['Prediction'] = (df[metric] <= threshold).astype(int) if invert_logic else (df[metric] >= threshold).astype(int)

    # Calculate F1 grouped by Source and Intensity
    def calc_f1(group):
        if len(group['is_plagiarised'].unique()) < 2:
            return np.nan # Need both positive and negative to calculate proper F1
        return f1_score(group['is_plagiarised'], group['Prediction'], zero_division=0)

    # Pitch Aggregation
    pitch_df = df[(df['Tempo_Intensity'] == 1.0) & (~df['Is_Extreme']) | (df['DSP_Category'] == 'Base Generation')]
    pitch_f1 = pitch_df.groupby(['Source', 'Pitch_Intensity']).apply(calc_f1, include_groups=False).reset_index(name='F1_Score').dropna()

    # Tempo Aggregation
    tempo_df = df[(df['Pitch_Intensity'] == 0.0) & (~df['Is_Extreme']) | (df['DSP_Category'] == 'Base Generation')]
    tempo_f1 = tempo_df.groupby(['Source', 'Tempo_Intensity']).apply(calc_f1, include_groups=False).reset_index(name='F1_Score').dropna()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    existing_sources = [s for s in ['Original', 'MusicGen', 'AudioLDM2', 'MGE-LDM'] if s in df['Source'].unique()]

    # Plot 1: Pitch F1 Trend
    if not pitch_f1.empty:
        sns.lineplot(data=pitch_f1, x='Pitch_Intensity', y='F1_Score', hue='Source', 
                     hue_order=existing_sources, palette=PLOT_COLORS, 
                     marker='s', markersize=9, ax=axes[0], linewidth=2.5)
        
        axes[0].set_title('Performance Degradation (Pitch)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Pitch Shift (Semitones)', fontsize=11)
        axes[0].set_ylabel('F1-Score', fontsize=11)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.3, label='Base (0)')
        axes[0].grid(alpha=0.3)
        axes[0].legend(title='Source Generation', loc='lower left')

    # Plot 2: Tempo F1 Trend
    if not tempo_f1.empty:
        sns.lineplot(data=tempo_f1, x='Tempo_Intensity', y='F1_Score', hue='Source', 
                     hue_order=existing_sources, palette=PLOT_COLORS, 
                     marker='s', markersize=9, ax=axes[1], linewidth=2.5)
        
        axes[1].set_title('Performance Degradation (Tempo)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Tempo Factor', fontsize=11)
        axes[1].set_ylabel('F1-Score', fontsize=11)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].axvline(x=1.0, color='black', linestyle='--', alpha=0.3, label='Base (1.0x)')
        axes[1].grid(alpha=0.3)
        axes[1].legend(title='Source Generation', loc='lower left')

    plt.suptitle(f'{model_name} - System Robustness (F1-Score Degradation)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    safe_metric_name = metric.replace("+", "_PLUS_")
    plot_path = Path(output_dir) / f"{model_name.lower()}_{safe_metric_name}_f1_degradation.png"
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Saved F1 Degradation plots for {model_name}")


def plot_extreme_scenarios(df: pd.DataFrame, model_name: str, metric: str, output_dir: str):
    """Boxplots comparing Base vs Extreme Up vs Extreme Down."""
    target_cats = ["Base Generation", "Extreme Down", "Extreme Up"]
    df_extremes = df[df['DSP_Category'].isin(target_cats)].copy()
    
    if df_extremes.empty:
        return
        
    fig, ax = plt.subplots(figsize=(12, 6))
    display_metric = metric.replace("_distance", "").replace("+", " + ").title()
    
    sns.boxplot(data=df_extremes, x='DSP_Category', y=metric, hue='Source',
                order=target_cats, palette=PLOT_COLORS, ax=ax,
                showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    
    ax.set_title(f'{model_name} - Stress Testing (Extreme DSP Combinations)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{display_metric} Distance', fontsize=12)
    ax.set_xlabel('Modification Scenario', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(title='Generative Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    safe_metric_name = metric.replace("+", "_PLUS_")
    plot_path = Path(output_dir) / f"{model_name.lower()}_{safe_metric_name}_extreme_stress_test.png"
    fig.savefig(plot_path, dpi=PLOT_DPI)
    plt.close(fig)


def main():
    logger.info("=" * 70)
    logger.info("ROBUSTNESS & F1-DEGRADATION ANALYSIS")
    logger.info("=" * 70)

    output_dir = OUTPUT_DIRS["robustness"]
    os.makedirs(output_dir, exist_ok=True)

    # Load thresholds from previous step
    threshold_map = load_threshold_summary()

    for model_name, csv_path in MODEL_PATHS.items():
        logger.info(f"\nProcessing EDA for {model_name}...")
        df_processed, found_metrics = process_model_data(csv_path, model_name)
        
        if df_processed is None or not found_metrics:
            continue

        # Get the official winning metric and threshold for this model
        winning_info = threshold_map.get(model_name)
        if winning_info:
            target_metric = winning_info['metric']
            opt_thresh = winning_info['threshold']
            logger.info(f"Using official threshold for {model_name}: {target_metric} (th={opt_thresh:.4f})")
            
            if target_metric in df_processed.columns:
                # Plot Distance Trends
                plot_distance_trends(df_processed, model_name, target_metric, output_dir)
                
                # Plot F1 Degradation
                plot_f1_degradation_trends(df_processed, model_name, target_metric, opt_thresh, output_dir)
                
                # Plot Extreme Stress Test Boxplots
                plot_extreme_scenarios(df_processed, model_name, target_metric, output_dir)
            else:
                logger.warning(f"Metric '{target_metric}' not found in dataframe.")
        else:
            logger.warning(f"No threshold found for {model_name}. Run optimal_threshold.py first.")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE. F1 Degradation plots ready.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()