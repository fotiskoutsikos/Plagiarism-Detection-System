"""
MGE-LDM Stem Base Generation Analysis.

Compares the inherent distance of each MGE-LDM stem-conditioned generation
(bass, drums, other) from the original segment, BEFORE any DSP is applied.

This answers the question:
    "Which musical component, when used as the inpainting seed,
     produces outputs closest to / farthest from the original?"

Reads:
    - Distance CSVs from results/distances/ and results/fusion/
    - Winning metrics + thresholds from threshold_analysis_summary.csv

Outputs per model:
    - Boxplot with threshold reference line
    - Summary CSV with mean, std, n per stem

Position in pipeline: Supplementary / Diagnostic
    (runs after metrics.py + optimal_threshold.py)
"""

import sys
import importlib.util
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    MODEL_PATHS, SUMMARY_FILES,
    PLOT_DPI, PLOT_STYLE_PARAMS,
    MGELDM_STEM_ORDER, MGELDM_STEM_COLORS,
)
from utils.categorization import (
    get_ground_truth_label,
    extract_features_with_stem,
)

plt.rcParams.update(PLOT_STYLE_PARAMS)
sns.set_style("whitegrid")


# Output paths 
OUTPUT_CSV_DIR  = Path("results/stem_analysis")
OUTPUT_PLOT_DIR = Path("plots/stem_analysis")

# Display names for boxplot x-axis
STEM_DISPLAY_NAMES = {
    'bass':  'Bass',
    'drums': 'Drums',
    'other': 'Other',
}


# Threshold loading 
def _load_threshold_map() -> dict:
    """
    Load winning metric + threshold per model from threshold_analysis_summary.csv.
    Returns dict: {MODEL: {'metric': str, 'threshold': float}}
    """
    path = Path(SUMMARY_FILES["threshold_analysis"])
    result = {}

    if not path.exists():
        logger.warning(
            f"Threshold summary not found at {path}. "
            "Plots will be generated without threshold reference line."
        )
        return result

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        model     = str(row.get("model", "")).upper()
        metric    = str(row.get("metric", ""))
        threshold = float(row.get("optimal_threshold", float("nan")))
        if model and metric and not np.isnan(threshold):
            result[model] = {"metric": metric, "threshold": threshold}

    return result


# Data loading & filtering 
def _load_base_generation_stems(
    csv_path: str, model_name: str, target_metric: str,
) -> pd.DataFrame:
    """
    Load distance CSV, filter to:
      - positive pairs only (is_plagiarised == 1)
      - MGE-LDM rows only
      - base generation only (no DSP: dsp_category == 'Base Generation')

    Enrich with stem label via extract_features_with_stem().
    """
    if not Path(csv_path).exists():
        logger.warning(f"[{model_name}] Distance CSV not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    df["is_plagiarised"] = df["final_mod_type"].apply(get_ground_truth_label)
    df = df[df["is_plagiarised"] == 1].copy()

    if target_metric not in df.columns:
        logger.warning(
            f"[{model_name}] Metric '{target_metric}' not found in columns. Skipping."
        )
        return pd.DataFrame()

    df_mgeldm = df[
        df["final_mod_type"].str.contains("mgeldm", case=False, na=False)
    ].copy()

    if df_mgeldm.empty:
        return pd.DataFrame()

    stem_features = df_mgeldm["final_mod_type"].apply(extract_features_with_stem)
    stem_features.columns = [
        "source", "stem", "pitch_intensity", "tempo_intensity",
        "is_extreme", "dsp_category",
    ]
    df_mgeldm = pd.concat(
        [df_mgeldm.reset_index(drop=True), stem_features.reset_index(drop=True)],
        axis=1,
    )

    df_mgeldm = df_mgeldm[df_mgeldm["stem"].isin(MGELDM_STEM_ORDER)].copy()
    df_base   = df_mgeldm[df_mgeldm["dsp_category"] == "Base Generation"].copy()

    return df_base


# Analysis 
def _compute_stem_base_stats(
    df_base: pd.DataFrame, metric: str,
) -> pd.DataFrame:
    """
    Compute mean, std, count of the target distance metric per stem.
    """
    stats = (
        df_base
        .groupby("stem")[metric]
        .agg(mean_distance="mean", std_distance="std", n="count")
        .reindex(MGELDM_STEM_ORDER)
        .reset_index()
        .round(6)
    )
    return stats


# Plotting 
def _plot_stem_base_comparison(
    df_base:     pd.DataFrame,
    model_name:  str,
    metric:      str,
    threshold:   float | None,
    output_dir:  Path,
) -> None:
    """
    Boxplot: distance distribution for each stem's base generation,
    with optional model-level threshold reference line.
    Styled consistently with plot_negative_tiers.py.
    """
    display_metric = metric.replace("_distance", "").replace("+", " + ").title()

    # Add display name column for x-axis
    df_plot = df_base.copy()
    df_plot["stem_display"] = df_plot["stem"].map(STEM_DISPLAY_NAMES)

    stem_order_display = [STEM_DISPLAY_NAMES[s] for s in MGELDM_STEM_ORDER]
    palette            = [MGELDM_STEM_COLORS[s] for s in MGELDM_STEM_ORDER]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_plot,
        x="stem_display",
        y=metric,
        order=stem_order_display,
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

    # Threshold reference line
    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label=f"Model Threshold ({threshold:.4f})",
            zorder=3,
        )
        ax.legend(loc="upper right", fontsize=10)

    ax.set_title(
        f"MGE-LDM Base Generation Distance by Stem — {model_name}\n"
        f"({display_metric} Distance — no DSP applied)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("MGE-LDM Target Stem", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{display_metric} Distance", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()

    safe_metric = metric.replace("+", "_PLUS_")
    plot_path   = output_dir / f"{model_name.lower()}_{safe_metric}_stem_base_generation.pdf"
    fig.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {plot_path}")


# Console output 
def _print_stem_base_table(
    stats: pd.DataFrame, model_name: str, metric: str, threshold: float | None,
) -> None:
    """Print stem base generation comparison to console."""
    thresh_str = f"{threshold:.4f}" if threshold is not None else "N/A"

    print(f"\n  {'─' * 60}")
    print(f"  {model_name} — MGE-LDM Base Generation Distance by Stem")
    print(f"  Metric: {metric} | Threshold (ref): {thresh_str}")
    print(f"  {'─' * 60}")
    print(f"  {'Stem':<10} | {'Mean':>10} | {'Std':>10} | {'N':>6} | {'vs Thresh':>10}")
    print(f"  {'─' * 55}")

    for _, row in stats.iterrows():
        stem = row["stem"]
        mean = row["mean_distance"]
        std  = row["std_distance"] if pd.notna(row["std_distance"]) else 0.0
        n    = int(row["n"])

        if threshold is not None:
            verdict = "ABOVE" if mean > threshold else "below"
        else:
            verdict = "—"

        print(
            f"  {stem:<10} | {mean:>10.4f} | {std:>10.4f} | {n:>6} | {verdict:>10}"
        )

    print(f"  {'─' * 60}")


# Per-model pipeline 
def run_stem_base_analysis(
    csv_path:   str,
    model_name: str,
    metric:     str,
    threshold:  float | None,
) -> None:
    """
    Full stem base generation analysis for one model.
    """
    print(f"\n{'─' * 60}")
    print(f"Stem Base Analysis — {model_name}")
    print(f"{'─' * 60}")

    df_base = _load_base_generation_stems(csv_path, model_name, metric)
    if df_base.empty:
        print("  No MGE-LDM base generation rows found. Skipping.")
        return

    print(f"  Base generation rows: {len(df_base)} "
          f"(stems: {df_base['stem'].value_counts().to_dict()})")

    # Compute statistics
    stats = _compute_stem_base_stats(df_base, metric)

    # Console
    _print_stem_base_table(stats, model_name, metric, threshold)

    # CSV
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    safe_metric  = metric.replace("+", "_PLUS_")
    csv_path_out = (
        OUTPUT_CSV_DIR
        / f"{model_name.lower()}_{safe_metric}_stem_base_generation.csv"
    )
    stats.to_csv(csv_path_out, index=False)
    print(f"  CSV saved  → {csv_path_out}")

    # Plot (boxplot — passes raw data, not aggregated stats)
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_stem_base_comparison(df_base, model_name, metric, threshold, OUTPUT_PLOT_DIR)


# Main 
def main() -> None:
    print("=" * 60)
    print("MGE-LDM STEM BASE GENERATION ANALYSIS")
    print("=" * 60)

    threshold_map = _load_threshold_map()

    for model_name, csv_path in MODEL_PATHS.items():
        info = threshold_map.get(model_name)
        if info is None:
            logger.warning(
                f"No threshold/metric found for {model_name}. "
                f"Run optimal_threshold.py first. Skipping."
            )
            continue

        metric    = info["metric"]
        threshold = info["threshold"]
        print(f"\n{model_name}: using {metric} (threshold ref = {threshold:.4f})")

        run_stem_base_analysis(csv_path, model_name, metric, threshold)

    print("\n" + "=" * 60)
    print("STEM BASE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()