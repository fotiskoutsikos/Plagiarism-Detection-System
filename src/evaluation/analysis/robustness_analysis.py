"""
Robustness Analysis for Plagiarism Detection (Strictly Distance-Based).

Attack model: Reference vs Plagiarized version (+DSP Modification)
  - Human  : Reference ↔ Cover (+Pitch/Tempo)
  - AI     : Reference ↔ MusicGen/AudioLDM2/MGE-LDM (+Pitch/Tempo)

Two analysis layers are produced:

  Layer A — Source-level (existing behaviour, unchanged):
      Plots all sources (Original, Cover, MusicGen, AudioLDM2, MGE-LDM)
      as separate lines/boxes.

  Layer B — MGE-LDM stem-level (new supplementary plots):
      Filters to MGE-LDM rows only and decomposes by stem
      (bass, drums, other). No synthetic zero-baseline is added;
      trendlines start from the lowest DSP intensity present in the data.
      The model-level threshold is drawn as a visual reference line on
      all stem plots, enabling statements such as
      "drums stem evades detection while bass remains detectable."

The Optimal Threshold is drawn ONLY as a visual reference line
in both layers; it is never used for metric calculations.

Outputs saved to:
    plots  → OUTPUT_DIRS["robustness"]
    CSVs   → OUTPUT_DIRS["robustness_stem_csv"]
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

sys.path.insert(0, str(repo_root / "src"))
from utils.constants import (
    MODEL_PATHS, OUTPUT_DIRS, SUMMARY_FILES,
    PLOT_COLORS, PLOT_DPI, PLOT_STYLE_PARAMS,
    AUDIO_SOURCES,
    MGELDM_STEM_ORDER, MGELDM_STEM_COLORS,
)
from utils.categorization import (
    extract_features,
    extract_features_with_stem,
    get_ground_truth_label,
)

plt.rcParams.update(PLOT_STYLE_PARAMS)


# Constants
_THRESHOLD_LABEL = "Model Threshold (ref.)"


# Threshold loading
def load_threshold_summary(summary_csv: str = None) -> dict:
    """
    Loads optimal thresholds from the ablation study.
    Returns dict: {model: {'metric': str, 'threshold': float}}
    """
    if summary_csv is None:
        summary_csv = SUMMARY_FILES['threshold_analysis']

    thresholds  = {}
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


# Data loading
def get_dynamic_metrics(df: pd.DataFrame) -> list:
    """Identify distance/composite metric columns dynamically."""
    invalid_cols = {'filename_ori', 'filename_mod', 'final_mod_type', 'pair_id', 'time'}
    return [
        col for col in df.columns
        if (col.endswith('_distance') or '+' in col) and col not in invalid_cols
    ]


def process_model_data(csv_path: str, model_name: str):
    """
    Load CSV, filter to positive plagiarism pairs only, extract DSP/source
    features (source-level, 5 columns).  Returns (df, metrics).
    """
    if not Path(csv_path).exists():
        logger.warning(f"File not found: {csv_path}. Skipping {model_name}.")
        return None, []

    df = pd.read_csv(csv_path)

    df['is_plagiarised'] = df['final_mod_type'].apply(get_ground_truth_label)
    df = df[df['is_plagiarised'] == 1].copy()
    if df.empty:
        logger.warning(f"No positive pairs found in {model_name}. Skipping.")
        return None, []

    metrics = get_dynamic_metrics(df)
    if not metrics:
        logger.warning(f"No valid metric columns found in {model_name}.")
        return None, []

    # Source-level features (legacy 5-column extractor — unchanged)
    features = df['final_mod_type'].apply(extract_features)
    features.columns = ['Source', 'Pitch_Intensity', 'Tempo_Intensity',
                        'Is_Extreme', 'DSP_Category']
    df = pd.concat([df, features], axis=1)
    df = df[df['Source'] != 'Ignore'].copy()

    print(f"{model_name}: {len(df)} positive pairs loaded (Reference ↔ Plagiarized).")
    return df, metrics


def _build_stem_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the full positive-pairs dataframe, extract only MGE-LDM rows
    and enrich with stem-level features (6-column extractor).

    Returns a dataframe with columns:
        Stem, Pitch_Intensity, Tempo_Intensity, Is_Extreme, DSP_Category
    plus all original distance / metadata columns.

    Rows where the stem could not be determined are dropped.
    """
    df_mgeldm = df[df['Source'] == 'MGE-LDM'].copy()
    if df_mgeldm.empty:
        return pd.DataFrame()

    # Apply stem-aware extractor
    stem_features = df_mgeldm['final_mod_type'].apply(extract_features_with_stem)
    stem_features.columns = [
        'Source', 'Stem', 'Pitch_Intensity', 'Tempo_Intensity',
        'Is_Extreme', 'DSP_Category',
    ]

    # Drop the old Source-level feature columns before concat to avoid duplicates
    drop_cols = ['Source', 'Pitch_Intensity', 'Tempo_Intensity',
                 'Is_Extreme', 'DSP_Category']
    df_mgeldm = df_mgeldm.drop(columns=[c for c in drop_cols if c in df_mgeldm.columns])
    df_mgeldm = pd.concat(
        [df_mgeldm.reset_index(drop=True), stem_features.reset_index(drop=True)],
        axis=1,
    )

    # Keep only rows with a valid stem
    df_mgeldm = df_mgeldm[df_mgeldm['Stem'].isin(MGELDM_STEM_ORDER)].copy()
    return df_mgeldm


# Plotting helpers
def _add_origin_anchors(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Adds synthetic anchor rows at DSP intensity = 0 / tempo = 1.0
    with distance = 0 (represents ori vs ori, trivially identical).
    Used ONLY in source-level trend plots.
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


def _draw_threshold(ax, threshold: float) -> None:
    """Draw the model-level threshold as a red dashed reference line."""
    ax.axhline(
        y=threshold,
        color='red',
        linestyle='--',
        linewidth=1.8,
        label=f'{_THRESHOLD_LABEL} ({threshold:.3f})',
        zorder=3,
    )


def _source_palette_with_original_dsp() -> dict:
    palette = dict(PLOT_COLORS)
    if 'Original + DSP' not in palette:
        palette['Original + DSP'] = palette.get('Original', 'blue')
    return palette


# Source-level plots
def plot_distance_trends(
    df:         pd.DataFrame,
    model_name: str,
    metric:     str,
    threshold:  float,
    output_dir: str,
) -> None:
    """
    Line plots of Distance vs Pitch/Tempo intensity, split by Source.
    Threshold drawn as visual reference only.
    Saves: {model}_{metric}_distance_trends.pdf
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    display_metric = metric.replace("_distance", "").replace("+", " + ").title()

    unique_sources = [s for s in df['Source'].dropna().unique()]
    existing_sources = [s for s in AUDIO_SOURCES if s in unique_sources]
    existing_sources.extend([s for s in unique_sources if s not in existing_sources])
    if not existing_sources:
        logger.warning(f"No valid sources in {model_name} for distance trends.")
        plt.close(fig)
        return

    # Pitch
    pitch_df = df[
        ((df['Tempo_Intensity'] == 1.0) & (~df['Is_Extreme'])) |
        (df['DSP_Category'] == 'Base Generation')
    ].copy()
    pitch_df = _add_origin_anchors(pitch_df, metric)

    if len(pitch_df) > 0 and len(pitch_df['Pitch_Intensity'].unique()) > 1:
        sns.lineplot(
            data=pitch_df, x='Pitch_Intensity', y=metric, hue='Source',
            hue_order=existing_sources, palette=_source_palette_with_original_dsp(),
            marker='o', markersize=8, ax=axes[0], errorbar=None, linewidth=2,
        )
        axes[0].set_title('Distance vs Pitch Shift', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Pitch Shift (Semitones)', fontsize=11)
        axes[0].set_ylabel(f'{display_metric} Distance', fontsize=11)
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.3, label='Base (0)')
        _draw_threshold(axes[0], threshold)
        axes[0].grid(alpha=0.3)
        axes[0].legend(title='Source', loc='upper left')

    # Tempo
    tempo_df = df[
        ((df['Pitch_Intensity'] == 0.0) & (~df['Is_Extreme'])) |
        (df['DSP_Category'] == 'Base Generation')
    ].copy()
    tempo_df = _add_origin_anchors(tempo_df, metric)

    if len(tempo_df) > 0 and len(tempo_df['Tempo_Intensity'].unique()) > 1:
        sns.lineplot(
            data=tempo_df, x='Tempo_Intensity', y=metric, hue='Source',
            hue_order=existing_sources, palette=_source_palette_with_original_dsp(),
            marker='o', markersize=8, ax=axes[1], errorbar=None, linewidth=2,
        )
        axes[1].set_title('Distance vs Tempo Shift', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Tempo Factor', fontsize=11)
        axes[1].set_ylabel(f'{display_metric} Distance', fontsize=11)
        axes[1].axvline(x=1.0, color='black', linestyle='--', alpha=0.3, label='Base (1.0x)')
        _draw_threshold(axes[1], threshold)
        axes[1].grid(alpha=0.3)
        axes[1].legend(title='Source', loc='upper left')

    plt.suptitle(
        f'{model_name} — Robustness Analysis: Distance vs Modification Intensity',
        fontsize=15, fontweight='bold',
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
    threshold:  float,
    output_dir: str,
) -> None:
    """
    Boxplots comparing distances for Base vs Extreme Down vs Extreme Up,
    grouped by Source. Threshold drawn as visual reference only.
    Saves: {model}_{metric}_extreme_stress_test.pdf
    """
    target_cats = ["Base Generation", "Extreme Down", "Extreme Up"]
    df_extremes = df[df['DSP_Category'].isin(target_cats)].copy()
    if df_extremes.empty:
        print(f"No extreme scenarios for {model_name} ({metric}). Skipping.")
        return

    display_metric = metric.replace("_distance", "").replace("+", " + ").title()

    # Clip to 99th percentile to remove outliers that compress visualization
    if metric in df_extremes.columns:
        p99 = np.percentile(df_extremes[metric].dropna(), 99)
        df_extremes = df_extremes[df_extremes[metric] <= p99].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df_extremes, x='DSP_Category', y=metric, hue='Source',
        order=target_cats, palette=_source_palette_with_original_dsp(), ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white",
                   "markeredgecolor": "black"},
    )
    _draw_threshold(ax, threshold)
    ax.set_title(
        f'{model_name} — Distance Under Extreme Stress (AI+DSP Combinations)',
        fontsize=14, fontweight='bold',
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


# MGE-LDM stem-level plots
def plot_stem_distance_trends(
    df_stem:    pd.DataFrame,
    model_name: str,
    metric:     str,
    threshold:  float,
    output_dir: str,
    csv_dir:    str,
) -> None:
    """
    Supplementary stem-level trendlines for MGE-LDM rows only.

    One line per stem (bass, drums, other) on the same axes.
    No synthetic zero-baseline is added; trendlines start from the
    lowest DSP intensity present in the data.

    Saves:
        plots/{model}_{metric}_mgeldm_stem_distance_trends.pdf
        csv/  {model}_{metric}_mgeldm_stem_distance_trends.csv
    """
    display_metric = metric.replace("_distance", "").replace("+", " + ").title()
    fig, axes      = plt.subplots(1, 2, figsize=(16, 6))
    has_content    = False

    # ── Pitch ──
    pitch_df = df_stem[
        (df_stem['Tempo_Intensity'] == 1.0) & (~df_stem['Is_Extreme'])
    ].copy()

    # Include base generation rows (pitch=0, tempo=1) so the "no-pitch-shift"
    # anchor is the real measured base-generation distance, not a synthetic 0.
    base_df  = df_stem[df_stem['DSP_Category'] == 'Base Generation'].copy()
    pitch_df = pd.concat([pitch_df, base_df], ignore_index=True).drop_duplicates()

    if not pitch_df.empty and len(pitch_df['Pitch_Intensity'].unique()) > 1:
        sns.lineplot(
            data=pitch_df,
            x='Pitch_Intensity', y=metric, hue='Stem',
            hue_order=MGELDM_STEM_ORDER,
            palette=MGELDM_STEM_COLORS,
            marker='o', markersize=8, ax=axes[0],
            errorbar=None, linewidth=2,
        )
        axes[0].set_title('MGE-LDM Stem — Distance vs Pitch Shift',
                          fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Pitch Shift (Semitones)', fontsize=11)
        axes[0].set_ylabel(f'{display_metric} Distance', fontsize=11)
        _draw_threshold(axes[0], threshold)
        axes[0].grid(alpha=0.3)
        axes[0].legend(title='Stem', loc='upper left')
        has_content = True

    # Tempo
    tempo_df = df_stem[
        (df_stem['Pitch_Intensity'] == 0.0) & (~df_stem['Is_Extreme'])
    ].copy()
    tempo_df = pd.concat([tempo_df, base_df], ignore_index=True).drop_duplicates()

    if not tempo_df.empty and len(tempo_df['Tempo_Intensity'].unique()) > 1:
        sns.lineplot(
            data=tempo_df,
            x='Tempo_Intensity', y=metric, hue='Stem',
            hue_order=MGELDM_STEM_ORDER,
            palette=MGELDM_STEM_COLORS,
            marker='o', markersize=8, ax=axes[1],
            errorbar=None, linewidth=2,
        )
        axes[1].set_title('MGE-LDM Stem — Distance vs Tempo Shift',
                          fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Tempo Factor', fontsize=11)
        axes[1].set_ylabel(f'{display_metric} Distance', fontsize=11)
        _draw_threshold(axes[1], threshold)
        axes[1].grid(alpha=0.3)
        axes[1].legend(title='Stem', loc='upper left')
        has_content = True

    if not has_content:
        plt.close(fig)
        return

    plt.suptitle(
        f'{model_name} — MGE-LDM Stem Robustness: Distance vs Modification Intensity\n'
        f'(Threshold line = model-level reference, not stem-specific)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    safe_metric = metric.replace("+", "_PLUS_")
    plot_path   = (
        Path(output_dir)
        / f"{model_name.lower()}_{safe_metric}_mgeldm_stem_distance_trends.pdf"
    )
    fig.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved MGE-LDM Stem Distance Trends → {plot_path}")

    # ── Summary CSV ──
    _save_stem_summary_csv(
        df_stem, metric,
        group_cols=['Stem', 'Pitch_Intensity', 'Tempo_Intensity', 'DSP_Category'],
        csv_dir=csv_dir,
        filename=f"{model_name.lower()}_{safe_metric}_mgeldm_stem_distance_trends.csv",
    )


def plot_stem_extreme_scenarios(
    df_stem:    pd.DataFrame,
    model_name: str,
    metric:     str,
    threshold:  float,
    output_dir: str,
    csv_dir:    str,
) -> None:
    """
    Supplementary stem-level extreme stress boxplots for MGE-LDM rows only.

    Compares Base Generation vs Extreme Down vs Extreme Up,
    with one box per stem per DSP category.

    Saves:
        plots/{model}_{metric}_mgeldm_stem_extreme_stress_test.pdf
        csv/  {model}_{metric}_mgeldm_stem_extreme_stress_test.csv
    """
    target_cats = ["Base Generation", "Extreme Down", "Extreme Up"]
    df_extremes = df_stem[df_stem['DSP_Category'].isin(target_cats)].copy()
    if df_extremes.empty:
        print(f"No MGE-LDM extreme scenarios for {model_name} ({metric}). Skipping.")
        return

    display_metric = metric.replace("_distance", "").replace("+", " + ").title()

    # Clip to 99th percentile to remove outliers that compress visualization
    if metric in df_extremes.columns:
        p99 = np.percentile(df_extremes[metric].dropna(), 99)
        df_extremes = df_extremes[df_extremes[metric] <= p99].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df_extremes,
        x='DSP_Category', y=metric, hue='Stem',
        order=target_cats,
        hue_order=MGELDM_STEM_ORDER,
        palette=MGELDM_STEM_COLORS,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white",
                   "markeredgecolor": "black"},
    )
    _draw_threshold(ax, threshold)
    ax.set_title(
        f'{model_name} — MGE-LDM Stem: Distance Under Extreme Stress\n'
        f'(Threshold line = model-level reference, not stem-specific)',
        fontsize=13, fontweight='bold',
    )
    ax.set_ylabel(f'{display_metric} Distance', fontsize=12)
    ax.set_xlabel('Modification Scenario', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(title='Stem', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    safe_metric = metric.replace("+", "_PLUS_")
    plot_path   = (
        Path(output_dir)
        / f"{model_name.lower()}_{safe_metric}_mgeldm_stem_extreme_stress_test.pdf"
    )
    fig.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved MGE-LDM Stem Extreme Stress Test → {plot_path}")

    # ── Summary CSV ──
    _save_stem_summary_csv(
        df_extremes, metric,
        group_cols=['Stem', 'DSP_Category'],
        csv_dir=csv_dir,
        filename=f"{model_name.lower()}_{safe_metric}_mgeldm_stem_extreme_stress_test.csv",
    )


# CSV helper for stem-level summaries
def _save_stem_summary_csv(
    df:         pd.DataFrame,
    metric:     str,
    group_cols: list,
    csv_dir:    str,
    filename:   str,
) -> None:
    """
    Save a grouped summary (mean, std, n) for the stem-level analysis.
    """
    if metric not in df.columns:
        return
    Path(csv_dir).mkdir(parents=True, exist_ok=True)
    summary = (
        df.groupby(group_cols)[metric]
        .agg(mean_distance='mean', std_distance='std', n='count')
        .reset_index()
        .round(6)
    )
    out = Path(csv_dir) / filename
    summary.to_csv(out, index=False)
    print(f"Saved Stem Summary CSV → {out}")


def main():
    print("=" * 70)
    print("ROBUSTNESS ANALYSIS (Distances Only — Reference ↔ Plagiarized)")
    print("=" * 70)

    plot_dir = Path(OUTPUT_DIRS["robustness"])
    csv_dir  = Path(OUTPUT_DIRS["robustness_stem_csv"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    threshold_map = load_threshold_summary()

    for model_name, csv_path in MODEL_PATHS.items():
        print(f"\nProcessing Robustness for {model_name} …")

        df_processed, _ = process_model_data(csv_path, model_name)
        if df_processed is None:
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
            df_processed, model_name, target_metric, opt_thresh, str(plot_dir)
        )
        plot_extreme_scenarios(
            df_processed, model_name, target_metric, opt_thresh, str(plot_dir)
        )

        # MGE-LDM stem-level supplementary plots
        df_stem = _build_stem_df(df_processed)
        if df_stem.empty:
            print(f"  No MGE-LDM rows found for {model_name} — stem plots skipped.")
            continue

        print(f"  MGE-LDM stem rows: {len(df_stem)} "
              f"(stems: {df_stem['Stem'].value_counts().to_dict()})")

        plot_stem_distance_trends(
            df_stem, model_name, target_metric, opt_thresh,
            str(plot_dir), str(csv_dir),
        )
        plot_stem_extreme_scenarios(
            df_stem, model_name, target_metric, opt_thresh,
            str(plot_dir), str(csv_dir),
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE. Robustness plots and stem CSVs ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()