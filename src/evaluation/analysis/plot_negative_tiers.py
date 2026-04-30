"""
Negative Tier Distance Distribution Visualization:
- Reads distance CSV files from results/distances/
- Creates publication-quality boxplots for negative sampling tier distributions
- Exports to plots/negative_tiers/ in PDF format
"""

import sys
import importlib.util
import logging
from pathlib import Path
import pandas as pd
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
logger = logging.getLogger(__name__)

sys.path.insert(0, str(repo_root / "src"))
from utils.constants import (
    OUTPUT_DIRS, MODEL_PATHS,
    DISTANCE_METRICS, PLOT_DPI, PLOT_STYLE_PARAMS,
)

plt.rcParams.update(PLOT_STYLE_PARAMS)
sns.set_style("whitegrid")

# CONFIG
OUTPUT_DIR = Path(OUTPUT_DIRS["negative_tiers"])

TARGET_TIERS = ['global_nearest', 'intra_category_nearest', 'random']

TIER_DISPLAY_NAMES = {
    'global_nearest':         'Global Nearest',
    'intra_category_nearest': 'Intra-Category Nearest',
    'random':                 'Random',
}

# Tier-specific colors (not source-based, so defined locally)
TIER_COLORS = {
    'global_nearest':         '#1f77b4',
    'intra_category_nearest': '#ff7f0e',
    'random':                 '#2ca02c',
}


# DATA LOADING
def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load distance CSV and return only negative pair rows for target tiers.
    """
    if not csv_path.exists():
        logger.warning(f"File not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total rows from {csv_path.name}")

    # Keep only negative pairs
    df = df[df['final_mod_type'].str.startswith('Negative_', na=False)].copy()
    print(f"After filtering negatives: {len(df)} rows")

    # Keep only the three target tiers
    df = df[df['negative_tier'].isin(TARGET_TIERS)].copy()
    df = df.dropna(subset=['negative_tier'])
    print(f"After filtering to target tiers: {len(df)} rows")

    return df


# PLOTTING
def plot_metric_tiers(
    df:         pd.DataFrame,
    model_name: str,
    metric:     str,
    output_dir: Path,
) -> None:
    """
    Boxplot for a single distance metric across the three negative tiers.
    """
    if df.empty or metric not in df.columns:
        logger.warning(f"No data for metric '{metric}' in {model_name}.")
        return

    tiers_present = [t for t in TARGET_TIERS if t in df['negative_tier'].unique()]
    if len(tiers_present) < 2:
        logger.warning(
            f"Not enough tiers for {metric} in {model_name}. Found: {tiers_present}"
        )
        return

    df_plot = df.copy()
    df_plot['tier_display'] = df_plot['negative_tier'].map(TIER_DISPLAY_NAMES)

    tier_order_display = [TIER_DISPLAY_NAMES[t] for t in tiers_present]
    palette            = [TIER_COLORS[t] for t in tiers_present]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_plot,
        x='tier_display',
        y=metric,
        order=tier_order_display,
        palette=palette,
        ax=ax,
        showmeans=True,
        meanprops={
            "marker": "o", "markerfacecolor": "white",
            "markeredgecolor": "black", "markersize": 8,
        },
        linewidth=1.5,
        flierprops={
            "marker": "o", "markerfacecolor": "gray",
            "markersize": 4, "alpha": 0.5,
        },
    )

    metric_display = metric.replace('_distance', '').replace('_', ' ').title()
    ax.set_title(
        f'Negative Tier Distribution — {model_name} ({metric_display} Distance)',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set_xlabel('Negative Sampling Tier', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_display} Distance', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    safe_metric = metric.replace('_distance', '')
    plot_path   = output_dir / f"{model_name.lower()}_{safe_metric}_negative_tiers.pdf"
    fig.savefig(plot_path, dpi=PLOT_DPI, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {plot_path.name}")


# PER-MODEL PIPELINE
def process_model(model_name: str, csv_path: Path, output_dir: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"Processing {model_name} from {csv_path.name}")
    print(f"{'=' * 60}")

    df = load_data(csv_path)
    if df.empty:
        logger.warning(f"No valid data for {model_name}. Skipping.")
        return

    print("Tier distribution:")
    for tier in TARGET_TIERS:
        count = len(df[df['negative_tier'] == tier])
        print(f"  {tier}: {count} pairs")

    for metric in DISTANCE_METRICS:
        if metric in df.columns:
            plot_metric_tiers(df, model_name, metric, output_dir)
        else:
            logger.warning(f"Metric '{metric}' not found in {model_name} data.")


def main() -> None:
    print("=" * 70)
    print("NEGATIVE TIER DISTRIBUTION ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    for model_name, csv_path_str in MODEL_PATHS.items():
        # Skip FUSION — it has fused distances, not raw tier distances
        if model_name == "FUSION":
            continue
        process_model(model_name, Path(csv_path_str), OUTPUT_DIR)

    print("\n" + "=" * 70)
    print(f"ANALYSIS COMPLETE. Plots saved → {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()