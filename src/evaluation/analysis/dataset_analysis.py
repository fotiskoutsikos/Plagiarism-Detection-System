"""
Dataset Statistics & Visualization.

Produces comprehensive descriptive statistics and publication-quality
plots for the plagiarism detection evaluation dataset.

Sections:
    1. Raw Segment Inventory (source models, stems, unique pairs)
    2. Modification Type Breakdown (broad, DSP family, granular)
    3. Evaluation Pair Statistics (positive/negative, tiers, prevalence)
    4. Vocal Validity Analysis (source-level and pair-level)
    5. Publication-Quality PDF Plots

Reads:
    - One distance CSV (default: CLEWS) as the canonical pair universe
    - Vocal metadata CSV (via vocal_metadata utilities)

Outputs:
    results/dataset_stats/dataset_summary.csv
    results/dataset_stats/modification_breakdown.csv
    results/dataset_stats/negative_tier_distribution.csv
    results/dataset_stats/vocal_validity_breakdown.csv
    plots/dataset_stats/segments_by_source.pdf
    plots/dataset_stats/modification_type_breakdown.pdf
    plots/dataset_stats/negative_tier_distribution.pdf
    plots/dataset_stats/vocal_validity_by_source.pdf
"""

import sys
import importlib.util
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    MODEL_PATHS, PLOT_DPI, PLOT_STYLE_PARAMS,
    SOURCE_COLORS, CATEGORY_COLORS, DSP_FAMILY_COLORS,
    MGELDM_STEM_COLORS, MGELDM_STEM_ORDER,
    ATTRIBUTION_TIER_COLORS, ATTRIBUTION_TIER_DISPLAY,
)
from utils.categorization import (
    get_ground_truth_label, clean_mod_type, get_broad_category,
    extract_dsp_and_source_features, get_dsp_family,
    extract_features_with_stem, _get_source_group, _get_dsp_label,
)

# Try to import vocal metadata (optional — script works without it)
try:
    from utils.vocal_metadata import attach_vocal_metadata, get_vocal_summary
    HAS_VOCAL = True
except ImportError:
    HAS_VOCAL = False
    logger.warning("vocal_metadata module not available. Vocal analysis will be skipped.")

# Style 
plt.rcParams.update(PLOT_STYLE_PARAMS)
sns.set_style("whitegrid")

# Output paths 
RESULTS_DIR = Path("results/dataset_stats")
PLOTS_DIR   = Path("plots/dataset_stats")

# Canonical data source 
CANONICAL_CSV = MODEL_PATHS["CLEWS"]


# DATA LOADING & ENRICHMENT

# Sources to exclude from plots (empty without stem qualifier)
_EXCLUDE_SOURCE_GROUPS = {"MGE-LDM"}


def _filter_excluded_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Remove source groups that should not appear in plots/stats."""
    return df[~df["source_group"].isin(_EXCLUDE_SOURCE_GROUPS)]


def load_and_enrich(csv_path: str) -> pd.DataFrame:
    """
    Load the canonical distance CSV and add all categorization columns.
    """
    df = pd.read_csv(csv_path, keep_default_na=False)
    print(f"Loaded {len(df):,} rows from {Path(csv_path).name}")

    # Ground truth
    df["y_true"] = df["final_mod_type"].apply(get_ground_truth_label)

    # Clean mod type (strip Negative_ prefix)
    df["clean_mod_type"] = df["final_mod_type"].apply(clean_mod_type)

    # Broad category
    df["broad_category"] = df["clean_mod_type"].apply(get_broad_category)

    # Source features (source, stem, pitch, tempo, dsp_category)
    feat_cols = ["source", "stem", "pitch_intensity", "tempo_intensity",
                 "is_extreme", "dsp_category"]
    df[feat_cols] = df["clean_mod_type"].apply(extract_features_with_stem)

    # Source group (e.g. MGE-LDM_bass)
    df["source_group"] = df.apply(_get_source_group, axis=1)

    # DSP label (e.g. pitchU4, tempo090, base)
    df["dsp_label"] = df["clean_mod_type"].apply(_get_dsp_label)

    # DSP family (Base, Pitch Only, Tempo Only, Combined)
    df["dsp_family"] = df.apply(
        lambda r: get_dsp_family(r["pitch_intensity"], r["tempo_intensity"]),
        axis=1,
    )

    # Positive / negative label
    df["pair_type"] = df["y_true"].map({1: "Positive", 0: "Negative"})

    return df


# STATISTICS COMPUTATION
def compute_segment_inventory(df: pd.DataFrame) -> dict:
    """Compute raw segment counts."""
    df_pos = df[df["y_true"] == 1]
    df_pos_filtered = _filter_excluded_sources(df_pos)

    stats = {
        "total_pairs": len(df),
        "positive_pairs": int(df["y_true"].sum()),
        "negative_pairs": int((df["y_true"] == 0).sum()),
        "unique_pair_ids": int(df["pair_id"].nunique()),
        "unique_filename_ori": int(df["filename_ori"].nunique()),
        "unique_filename_mod": int(df["filename_mod"].nunique()),
        "prevalence_pct": round(df["y_true"].mean() * 100, 2),
    }

    # Segments by source (from positives, excluding empty groups)
    source_counts = df_pos_filtered["source_group"].value_counts().to_dict()
    stats["segments_by_source"] = source_counts

    # Segments by broad category (from ALL positives — no filter here)
    broad_counts = df_pos["broad_category"].value_counts().to_dict()
    stats["segments_by_broad_category"] = broad_counts

    return stats


def compute_modification_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Detailed modification type breakdown for positive pairs."""
    df_pos = df[df["y_true"] == 1]

    breakdown = (
        df_pos
        .groupby(["broad_category", "source_group", "dsp_family", "dsp_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["broad_category", "source_group", "count"], ascending=[True, True, False])
    )
    return breakdown


def compute_negative_tier_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of negative pairs across sampling tiers."""
    df_neg = df[df["y_true"] == 0].copy()

    tier_dist = (
        df_neg
        .groupby("negative_tier")
        .size()
        .reset_index(name="count")
    )
    tier_dist["percentage"] = (tier_dist["count"] / tier_dist["count"].sum() * 100).round(2)
    return tier_dist


def compute_vocal_validity(df: pd.DataFrame) -> pd.DataFrame | None:
    """Vocal validity breakdown by source group (if metadata available)."""
    if not HAS_VOCAL:
        return None

    try:
        df_enriched = attach_vocal_metadata(df)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not attach vocal metadata: {e}")
        return None

    df_pos = df_enriched[df_enriched["y_true"] == 1]
    df_pos = _filter_excluded_sources(df_pos)

    vocal_cols = {
        "vocal_valid_ori": "ori_vocal_valid",
        "vocal_valid_mod": "mod_vocal_valid",
        "pair_vocal_valid": "pair_vocal_valid",
    }

    # Rename for clarity
    rename_map = {}
    for new_name, old_name in vocal_cols.items():
        if old_name in df_pos.columns:
            rename_map[old_name] = new_name

    df_vocal = df_pos.rename(columns=rename_map)

    if "pair_vocal_valid" not in df_vocal.columns:
        return None

    breakdown = (
        df_vocal
        .groupby("source_group")
        .agg(
            total=("pair_vocal_valid", "size"),
            vocal_valid=("pair_vocal_valid", "sum"),
        )
        .reset_index()
    )
    breakdown["vocal_invalid"] = breakdown["total"] - breakdown["vocal_valid"]
    breakdown["valid_pct"] = (breakdown["vocal_valid"] / breakdown["total"] * 100).round(1)

    return breakdown


# PLOTTING
def plot_segments_by_source(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: number of positive segments per source model."""
    df_pos = df[df["y_true"] == 1]
    df_pos = _filter_excluded_sources(df_pos)
    counts = df_pos["source_group"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [SOURCE_COLORS.get(s.split("_")[0] if "MGE-LDM" not in s else "MGE-LDM", "#999")
              for s in counts.index]

    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_width() + max(counts.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            ha="left", va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Number of Positive Segments", fontsize=11, fontweight="bold")
    ax.set_title("Dataset Composition: Segments by Source Model", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / "segments_by_source.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_broad_category_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: positive pairs per broad category."""
    df_pos = df[df["y_true"] == 1]
    counts = df_pos["broad_category"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [CATEGORY_COLORS.get(cat, "#999999") for cat in counts.index]

    bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Number of Positive Pairs", fontsize=11, fontweight="bold")
    ax.set_title("Positive Pairs by Broad Modification Category", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / "modification_type_breakdown.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_dsp_family_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Stacked bar: DSP family distribution per source group."""
    df_pos = df[df["y_true"] == 1]
    df_pos = _filter_excluded_sources(df_pos)

    pivot = pd.crosstab(df_pos["source_group"], df_pos["dsp_family"])

    # Reorder columns
    family_order = ["Base", "Pitch Only", "Tempo Only", "Combined (Extreme)"]
    pivot = pivot.reindex(columns=[c for c in family_order if c in pivot.columns], fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [DSP_FAMILY_COLORS.get(f, "#999") for f in pivot.columns]
    pivot.plot(kind="barh", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Number of Segments", fontsize=11, fontweight="bold")
    ax.set_title("DSP Family Distribution per Source Model", fontsize=13, fontweight="bold")
    ax.legend(title="DSP Family", loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / "dsp_family_distribution.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_negative_tier_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Pie chart: negative pair distribution across sampling tiers."""
    df_neg = df[df["y_true"] == 0]
    tier_counts = df_neg["negative_tier"].value_counts()

    # Reorder
    tier_order = ["random", "intra_category_nearest", "global_nearest"]
    tier_counts = tier_counts.reindex([t for t in tier_order if t in tier_counts.index])

    fig, ax = plt.subplots(figsize=(8, 8))

    display_names = [ATTRIBUTION_TIER_DISPLAY.get(t, t) for t in tier_counts.index]
    colors = [ATTRIBUTION_TIER_COLORS.get(t, "#999") for t in tier_counts.index]

    total = tier_counts.sum()
    labels = [
        f"{name}\n{count:,} ({count/total*100:.1f}%)"
        for name, count in zip(display_names, tier_counts.values)
    ]

    ax.pie(
        tier_counts.values,
        labels=labels,
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )

    ax.set_title(
        f"Negative Pair Distribution by Sampling Tier\n(Total: {total:,} negative pairs)",
        fontsize=13, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    path = output_dir / "negative_tier_distribution.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_positive_negative_split(df: pd.DataFrame, output_dir: Path) -> None:
    """Pie chart: positive vs negative pair ratio."""
    n_pos = int(df["y_true"].sum())
    n_neg = int((df["y_true"] == 0).sum())

    fig, ax = plt.subplots(figsize=(7, 7))

    sizes = [n_pos, n_neg]
    labels = [f"Positive\n{n_pos:,} ({n_pos/(n_pos+n_neg)*100:.1f}%)",
              f"Negative\n{n_neg:,} ({n_neg/(n_pos+n_neg)*100:.1f}%)"]
    colors = ["#4CAF50", "#E53935"]

    ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="", startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Evaluation Dataset: Positive vs Negative Pairs",
                 fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    path = output_dir / "positive_negative_split.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")

def plot_smp_relation_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Pie chart: distribution of SMP relation types (plag, plag_doubt, remake)."""
    df_pos = df[df["y_true"] == 1]

    # SMP base pairs have final_mod_type starting with 'SMP_'
    df_smp = df_pos[df_pos["clean_mod_type"].str.startswith("SMP_")]

    if df_smp.empty:
        print("  ⚠ No SMP base pairs found. Skipping relation breakdown.")
        return

    # Extract relation from clean_mod_type (e.g. 'SMP_plag' -> 'plag')
    df_smp = df_smp.copy()
    df_smp["relation"] = df_smp["clean_mod_type"].str.replace("SMP_", "", n=1)

    relation_counts = df_smp["relation"].value_counts()

    # Display names
    relation_display = {
        "plag": "Plagiarism",
        "plag_doubt": "Doubtful Plagiarism",
        "remake": "Remake",
    }
    relation_colors = {
        "plag": "#E53935",
        "plag_doubt": "#FF9800",
        "remake": "#4CAF50",
    }

    display_names = [relation_display.get(r, r) for r in relation_counts.index]
    colors = [relation_colors.get(r, "#999") for r in relation_counts.index]

    total = relation_counts.sum()
    labels = [
        f"{name}\n{count:,} ({count/total*100:.1f}%)"
        for name, count in zip(display_names, relation_counts.values)
    ]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(
        relation_counts.values,
        labels=labels,
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )

    ax.set_title(
        f"SMP Human Plagiarism: Relation Type Distribution\n"
        f"(Total: {total:,} base pairs)",
        fontsize=13, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    path = output_dir / "smp_relation_breakdown.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_base_vs_dsp_by_source(df: pd.DataFrame, output_dir: Path) -> None:
    """Stacked bar: base files vs DSP variants per source."""
    df_pos = df[df["y_true"] == 1]
    df_pos = _filter_excluded_sources(df_pos)

    df_pos = df_pos.copy()
    df_pos["is_base"] = df_pos["dsp_family"] == "Base"
    df_pos["variant_type"] = df_pos["is_base"].map({True: "Base", False: "DSP Variant"})

    pivot = pd.crosstab(df_pos["source_group"], df_pos["variant_type"])

    # Ensure column order
    for col in ["Base", "DSP Variant"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Base", "DSP Variant"]]

    pivot = pivot.sort_values("Base", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#4CAF50", "#2196F3"]
    pivot.plot(kind="barh", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)

    # Add total labels
    for i, (source, row) in enumerate(pivot.iterrows()):
        total = row.sum()
        ax.text(
            total + max(pivot.sum(axis=1)) * 0.01, i,
            f"{int(total):,}",
            ha="left", va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Number of Segments", fontsize=11, fontweight="bold")
    ax.set_title("Base Files vs DSP Variants per Source", fontsize=13, fontweight="bold")
    ax.legend(title="Variant Type", loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / "base_vs_dsp_by_source.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_mgeldm_stem_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: MGE-LDM segment counts per stem (bass, drums, other)."""
    df_pos = df[df["y_true"] == 1]

    # Filter to MGE-LDM only (with stem)
    df_mgeldm = df_pos[
        (df_pos["source"] == "MGE-LDM") & (df_pos["stem"].notna())
    ].copy()

    if df_mgeldm.empty:
        print("  ⚠ No MGE-LDM stem data found. Skipping.")
        return

    stem_counts = df_mgeldm["stem"].value_counts()

    # Reorder
    stem_order = [s for s in MGELDM_STEM_ORDER if s in stem_counts.index]
    stem_counts = stem_counts.reindex(stem_order)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [MGELDM_STEM_COLORS.get(s, "#999") for s in stem_counts.index]
    display_names = [s.capitalize() for s in stem_counts.index]

    bars = ax.bar(display_names, stem_counts.values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, stem_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stem_counts.values) * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Number of Segments", fontsize=11, fontweight="bold")
    ax.set_title(
        "MGE-LDM Segments by Preserved Stem\n(Including Base + DSP Variants)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / "mgeldm_stem_breakdown.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_broad_category_pie(df: pd.DataFrame, output_dir: Path) -> None:
    """Pie chart: percentage distribution of the 5 broad categories."""
    df_pos = df[df["y_true"] == 1]
    counts = df_pos["broad_category"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 9))

    colors = [CATEGORY_COLORS.get(cat, "#999") for cat in counts.index]

    total = counts.sum()
    labels = [
        f"{cat}\n{count:,} ({count/total*100:.1f}%)"
        for cat, count in zip(counts.index, counts.values)
    ]

    ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )

    ax.set_title(
        f"Positive Pairs: Broad Category Distribution\n"
        f"(Total: {total:,} pairs)",
        fontsize=13, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    path = output_dir / "broad_category_pie.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_vocal_validity(df_vocal: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: vocal valid vs invalid per source group."""
    if df_vocal is None or df_vocal.empty:
        return

    df_plot = df_vocal.sort_values("source_group")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df_plot))
    width = 0.35

    bars_valid = ax.bar(
        x - width / 2, df_plot["vocal_valid"].values, width,
        label="Vocal Valid", color="#4CAF50", edgecolor="white",
    )
    bars_invalid = ax.bar(
        x + width / 2, df_plot["vocal_invalid"].values, width,
        label="Vocal Invalid", color="#E53935", edgecolor="white",
    )

    for bars in [bars_valid, bars_invalid]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 5,
                    f"{int(h):,}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["source_group"].values, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of Positive Pairs", fontsize=11, fontweight="bold")
    ax.set_title("Vocal Validity by Source Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / "vocal_validity_by_source.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


# CONSOLE OUTPUT
def print_summary(stats: dict, df: pd.DataFrame) -> None:
    """Print formatted dataset summary to console."""
    print(f"\n{'=' * 70}")
    print(f" DATASET SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  Total evaluation pairs : {stats['total_pairs']:,}")
    print(f"  Positive pairs         : {stats['positive_pairs']:,}")
    print(f"  Negative pairs         : {stats['negative_pairs']:,}")
    print(f"  Prevalence             : {stats['prevalence_pct']:.2f}%")
    print(f"  Unique pair IDs        : {stats['unique_pair_ids']}")
    print(f"  Unique originals       : {stats['unique_filename_ori']}")
    print(f"  Unique modified        : {stats['unique_filename_mod']}")

    print(f"\n  Segments by Source (Positives only):")
    for source, count in sorted(stats["segments_by_source"].items()):
        print(f"    {source:<25}: {count:>6,}")

    print(f"\n  Segments by Broad Category (Positives only):")
    for cat, count in sorted(stats["segments_by_broad_category"].items()):
        print(f"    {cat:<35}: {count:>6,}")

    print(f"{'=' * 70}\n")


# MAIN
def main() -> None:
    print("=" * 70)
    print("DATASET STATISTICS & VISUALIZATION")
    print("=" * 70)

    # Load and enrich
    csv_path = Path(CANONICAL_CSV)
    if not csv_path.exists():
        print(f"[ERROR] Canonical CSV not found: {csv_path}")
        print("Run metrics.py first.")
        return

    df = load_and_enrich(str(csv_path))

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Section 1: Segment Inventory 
    print("\n[1/5] Computing segment inventory...")
    stats = compute_segment_inventory(df)
    print_summary(stats, df)

    # Save summary
    summary_rows = [
        {"metric": k, "value": str(v)}
        for k, v in stats.items()
        if not isinstance(v, dict)
    ]
    for source, count in stats["segments_by_source"].items():
        summary_rows.append({"metric": f"source_{source}", "value": count})
    for cat, count in stats["segments_by_broad_category"].items():
        summary_rows.append({"metric": f"broad_{cat}", "value": count})

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    print(f"  Summary saved → {RESULTS_DIR / 'dataset_summary.csv'}")

    # Section 2: Modification Breakdown 
    print("\n[2/5] Computing modification breakdown...")
    df_breakdown = compute_modification_breakdown(df)
    df_breakdown.to_csv(RESULTS_DIR / "modification_breakdown.csv", index=False)
    print(f"  Breakdown saved → {RESULTS_DIR / 'modification_breakdown.csv'}")
    print(f"  Unique modification types: {df_breakdown['dsp_label'].nunique()}")

    # Section 3: Negative Tier Distribution 
    print("\n[3/5] Computing negative tier distribution...")
    df_tiers = compute_negative_tier_distribution(df)
    df_tiers.to_csv(RESULTS_DIR / "negative_tier_distribution.csv", index=False)
    print(df_tiers.to_string(index=False))

    # Section 4: Vocal Validity 
    print("\n[4/5] Computing vocal validity...")
    df_vocal = compute_vocal_validity(df)
    if df_vocal is not None:
        df_vocal.to_csv(RESULTS_DIR / "vocal_validity_breakdown.csv", index=False)
        print(df_vocal.to_string(index=False))
    else:
        print("  Vocal metadata not available. Skipping.")

    # Section 5: Plots 
    print("\n[5/5] Generating plots...")
    plot_segments_by_source(df, PLOTS_DIR)
    plot_broad_category_breakdown(df, PLOTS_DIR)
    plot_dsp_family_distribution(df, PLOTS_DIR)
    plot_negative_tier_distribution(df, PLOTS_DIR)
    plot_positive_negative_split(df, PLOTS_DIR)
    plot_vocal_validity(df_vocal, PLOTS_DIR)
    plot_smp_relation_breakdown(df, PLOTS_DIR)
    plot_base_vs_dsp_by_source(df, PLOTS_DIR)
    plot_mgeldm_stem_breakdown(df, PLOTS_DIR)
    plot_broad_category_pie(df, PLOTS_DIR)

    # Done 
    print(f"\n{'=' * 70}")
    print(f"DATASET STATISTICS COMPLETE")
    print(f"  Results → {RESULTS_DIR}/")
    print(f"  Plots   → {PLOTS_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()