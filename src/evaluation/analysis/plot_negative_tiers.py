"""
Negative Tier Distance Distribution Visualization:
- Reads distance CSV files from results/distances/
- Creates publication-quality boxplots for negative sampling tier distributions
- Exports to plots/negative_tiers/ in PDF format
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
from utils.constants import OUTPUT_DIRS, PLOT_COLORS, PLOT_DPI


# Define paths
DISTANCES_DIR = repo_root / "results" / "distances"
OUTPUT_NEGATIVE_TIERS_DIR = OUTPUT_DIRS["negative_tiers"]

# Target negative tiers in specific order (global_nearest -> intra_category_nearest -> random)
TARGET_TIERS = ['global_nearest', 'intra_category_nearest', 'random']

# Distance metrics to plot
DISTANCE_METRICS = ['euclidean_distance', 'cosine_distance', 'manhattan_distance', 'pearson_distance']

# Model files to process
DISTANCE_FILES = {
    'CLEWS': 'clews_distances.csv',
    'Wealy': 'wealy_distances.csv',
}


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads distance CSV file and returns filtered DataFrame with only negative pairs.
    
    Args:
        csv_path: Path to the distance CSV file
        
    Returns:
        Filtered DataFrame with only negative pairs and valid tiers
    """
    if not os.path.exists(csv_path):
        logger.warning(f"File not found: {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} total rows from {os.path.basename(csv_path)}")
    
    # Filter 1: Keep only negative pairs (final_mod_type starts with 'Negative_')
    df = df[df['final_mod_type'].str.startswith('Negative_', na=False)].copy()
    logger.info(f"After filtering negative pairs: {len(df)} rows")
    
    # Filter 2: Keep only the three specific negative tiers
    df = df[df['negative_tier'].isin(TARGET_TIERS)].copy()
    logger.info(f"After filtering to target tiers: {len(df)} rows")
    
    # Remove any rows with null negative_tier
    df = df.dropna(subset=['negative_tier'])
    
    return df


def plot_metric_tiers(df: pd.DataFrame, model_name: str, metric: str, output_dir: str):
    """
    Creates a boxplot for a specific distance metric across negative tiers.
    
    Args:
        df: Filtered DataFrame with negative pairs
        model_name: Name of the model (CLEWS/Wealy)
        metric: Distance metric column name
        output_dir: Directory to save the plot
    """
    if df.empty or metric not in df.columns:
        logger.warning(f"No data for metric {metric}")
        return
    
    # Check if all tiers exist in data
    tiers_present = [t for t in TARGET_TIERS if t in df['negative_tier'].unique()]
    if len(tiers_present) < 2:
        logger.warning(f"Not enough tiers present for {metric}. Found: {tiers_present}")
        return
    
    # Set up the figure with academic style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': PLOT_DPI,
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot with ordered tiers
    # Use specific order: global_nearest -> intra_category_nearest -> random
    tier_order = [t for t in TARGET_TIERS if t in tiers_present]
    
    # Create display name mapping
    tier_display_names = {
        'global_nearest': 'Global Nearest',
        'intra_category_nearest': 'Intra-Category Nearest',
        'random': 'Random'
    }
    
    # Map tier names for display
    df_plot = df.copy()
    df_plot['negative_tier_display'] = df_plot['negative_tier'].map(tier_display_names)
    
    # Define colors for each tier
    tier_colors = {
        'global_nearest': PLOT_COLORS.get('Global Nearest', '#1f77b4'),
        'intra_category_nearest': PLOT_COLORS.get('Intra-Category Nearest', '#ff7f0e'),
        'random': PLOT_COLORS.get('Random', '#2ca02c'),
    }
    palette = [tier_colors[t] for t in tier_order]
    
    # Create the boxplot
    sns.boxplot(
        data=df_plot,
        x='negative_tier_display',
        y=metric,
        order=[tier_display_names[t] for t in tier_order],
        palette=palette,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
        linewidth=1.5,
        flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 4, "alpha": 0.5},
    )
    
    # Format metric name for title
    metric_display = metric.replace('_distance', '').replace('_', ' ').title()
    
    # Set titles and labels
    ax.set_title(
        f'Distribution of Negative Tiers - {model_name} ({metric_display} Distance)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('Negative Sampling Tier', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_display} Distance', fontsize=12, fontweight='bold')
    
    # Add grid for readability
    ax.grid(alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Adjust y-axis label format for large numbers
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    
    plt.tight_layout()
    
    # Save as PDF
    safe_metric_name = metric.replace('_distance', '')
    plot_path = Path(output_dir) / f"{model_name.lower()}_{safe_metric_name}_negative_tiers.pdf"
    fig.savefig(plot_path, dpi=PLOT_DPI, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved: {plot_path.name}")


def process_model_distances(model_name: str, csv_filename: str, output_dir: str):
    """
    Processes all distance metrics for a single model.
    
    Args:
        model_name: Name of the model (CLEWS/Wealy)
        csv_filename: Name of the CSV file
        output_dir: Directory to save plots
    """
    csv_path = DISTANCES_DIR / csv_filename
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {model_name} distances from {csv_filename}")
    logger.info(f"{'='*60}")
    
    # Load and filter data
    df = load_data(str(csv_path))
    
    if df.empty:
        logger.warning(f"No valid data found for {model_name}. Skipping.")
        return
    
    # Log tier distribution
    logger.info(f"Tier distribution:")
    for tier in TARGET_TIERS:
        count = len(df[df['negative_tier'] == tier])
        logger.info(f"  - {tier}: {count} pairs")
    
    # Plot each distance metric
    for metric in DISTANCE_METRICS:
        if metric in df.columns:
            plot_metric_tiers(df, model_name, metric, output_dir)
        else:
            logger.warning(f"Metric {metric} not found in {model_name} data")


def main():
    """
    Main function to orchestrate the negative tier visualization.
    """
    logger.info("=" * 70)
    logger.info("NEGATIVE TIER DISTRIBUTION ANALYSIS")
    logger.info("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_NEGATIVE_TIERS_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_NEGATIVE_TIERS_DIR}")
    
    # Process each model's distance data
    for model_name, csv_filename in DISTANCE_FILES.items():
        process_model_distances(model_name, csv_filename, str(OUTPUT_NEGATIVE_TIERS_DIR))
    
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE. Negative tier plots saved to:")
    logger.info(f"  {OUTPUT_NEGATIVE_TIERS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()