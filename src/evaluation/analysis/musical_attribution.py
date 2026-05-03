"""
4-Way Forced-Choice Musical Source Attribution.

For each modified track (query), ranks 4 candidates by distance:
  1 true positive  +  3 hard negatives (random, intra-category, global nearest)

Evaluates whether the embedding space can identify the correct source
WITHOUT any threshold — purely by relative distance ranking.

Reads:
  - Distance CSVs from results/distances/ and results/fusion/
  - Winning metrics from results/threshold/threshold_analysis_summary.csv

Outputs:
  - Per-model attribution CSVs (overall, broad, detailed, misattribution)
  - Cross-model summary CSV
  - Publication-quality PDF plots
"""

import sys
import importlib.util
import logging
from pathlib import Path

import pandas as pd
import numpy as np
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
setup_logging = logging_util.setup_logging

setup_logging(__file__)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(repo_root / "src"))
from utils.constants import (
    MODEL_PATHS, OUTPUT_DIRS, SUMMARY_FILES,
    PLOT_DPI, PLOT_STYLE_PARAMS, CATEGORY_COLORS,
    ATTRIBUTION_RANK_COLORS, ATTRIBUTION_TIER_COLORS, ATTRIBUTION_TIER_DISPLAY,
)
from utils.categorization import (
    get_ground_truth_label, clean_mod_type, categorize_modification,
)

# Style 
plt.rcParams.update(PLOT_STYLE_PARAMS)
sns.set_style("whitegrid")

# Output paths
ATTRIBUTION_DIR = OUTPUT_DIRS["attribution"]
ATTRIBUTION_PLOTS_DIR = OUTPUT_DIRS["attribution_plots"]
ATTRIBUTION_SUMMARY = SUMMARY_FILES["attribution_summary"]

# Expected negative tiers
NEGATIVE_TIERS = {"random", "intra_category_nearest", "global_nearest"}
POSITIVE_TIER_LABEL = "N/A"
ATTRIBUTION_GROUP_SIZE = 4  # 1 positive + 3 negatives



# LOADING
def load_winning_metrics() -> dict:
    """
    Load winning metric per model from threshold analysis summary.
    """
    thresh_path = Path(SUMMARY_FILES["threshold_analysis"])
    if not thresh_path.exists():
        raise FileNotFoundError(
            f"Threshold summary not found: {thresh_path}\n"
            f"Run optimal_threshold.py first."
        )

    df = pd.read_csv(thresh_path)
    mapping = {}
    for _, row in df.iterrows():
        model = str(row["model"]).upper()
        metric = str(row["metric"])
        mapping[model] = metric
        print(f"Winning metric for {model}: {metric}")

    return mapping


# DATA PREPARATION
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ground truth, cleaned mod type, and broad category columns.
    Uses centralized categorization utilities for consistency.
    """
    df = df.copy()
    df["y_true"] = df["final_mod_type"].apply(get_ground_truth_label)
    df["clean_mod_type"] = df["final_mod_type"].apply(clean_mod_type)
    df["category_grouped"] = df["clean_mod_type"].apply(categorize_modification)
    return df


def build_query_groups(df: pd.DataFrame, metric_col: str):
    """
    Build valid 4-way attribution groups.

    Grouping: (pair_id, time, filename_ori, clean_mod_type)

    When multiple positives share the same group keys (common in SMP pairs
    where one original segment maps to multiple cover timestamps), we keep
    exactly ONE positive per group (deterministic: smallest filename_mod).

    Similarly, if negatives are duplicated across positives, we keep
    exactly one per tier.

    A valid group has exactly 1 positive + 3 negatives (one per tier).
    """
    required_cols = [
        "pair_id", "time", "filename_ori", "filename_mod",
        "clean_mod_type", "y_true", "negative_tier", metric_col,
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    df = df.copy()

    # Composite group key
    group_keys = ["pair_id", "time", "filename_ori", "clean_mod_type"]
    df["query_key"] = (
        df["pair_id"].astype(str) + "|"
        + df["time"].astype(str) + "|"
        + df["filename_ori"].astype(str) + "|"
        + df["clean_mod_type"].astype(str)
    )

    valid_frames = []
    n_skipped = 0
    skip_reasons: dict = {}

    def _skip(reason: str):
        nonlocal n_skipped
        n_skipped += 1
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for key, grp in df.groupby("query_key"):

        # Positives: keep exactly 1 (deterministic)
        pos = grp[grp["y_true"] == 1]
        if len(pos) == 0:
            _skip("no_positive")
            continue

        # Deterministic selection: smallest filename_mod
        pos_selected = pos.sort_values("filename_mod").iloc[[0]]

        # Negatives: keep exactly 1 per expected tier
        neg = grp[grp["y_true"] == 0]
        neg_selected_parts = []
        tier_ok = True

        for tier in sorted(NEGATIVE_TIERS):
            tier_rows = neg[neg["negative_tier"] == tier]
            if len(tier_rows) == 0:
                _skip(f"missing_tier={tier}")
                tier_ok = False
                break
            # Deterministic: smallest filename_mod within tier
            neg_selected_parts.append(
                tier_rows.sort_values("filename_mod").iloc[[0]]
            )

        if not tier_ok:
            continue

        neg_selected = pd.concat(neg_selected_parts, ignore_index=False)

        # Validate metric values
        combined = pd.concat([pos_selected, neg_selected], ignore_index=False)
        if combined[metric_col].isna().any():
            _skip("nan_in_metric")
            continue

        valid_frames.append(combined)

    # Assemble
    if valid_frames:
        df_valid = pd.concat(valid_frames, ignore_index=False).copy()
        # Ensure query_key is set consistently
        df_valid["query_key"] = (
            df_valid["pair_id"].astype(str) + "|"
            + df_valid["time"].astype(str) + "|"
            + df_valid["filename_ori"].astype(str) + "|"
            + df_valid["clean_mod_type"].astype(str)
        )
    else:
        df_valid = pd.DataFrame()

    n_valid = df_valid["query_key"].nunique() if not df_valid.empty else 0

    return df_valid, n_skipped, skip_reasons

# RANKING 
def rank_queries(df_valid: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    For each valid 4-way group, rank candidates by ascending distance
    and record the rank of the true positive.
    """
    results = []

    for query_key, grp in df_valid.groupby("query_key"):
        # Sort by distance ascending (smaller = more similar = better match)
        grp_sorted = grp.sort_values(
            by=[metric_col, "y_true"],
            ascending=[True, False],  # tie-break: favour positive
        ).reset_index(drop=True)

        # Find positive rank (1-indexed)
        pos_mask = grp_sorted["y_true"] == 1
        pos_idx = pos_mask.idxmax()  # first True index in sorted frame
        positive_rank = int(pos_idx) + 1

        # Who won rank 1?
        rank1_row = grp_sorted.iloc[0]
        if rank1_row["y_true"] == 1:
            winner_tier = "correct"
        else:
            winner_tier = str(rank1_row["negative_tier"])

        # Metadata from positive row
        pos_row = grp_sorted.loc[pos_mask].iloc[0]

        results.append({
            "query_key": query_key,
            "pair_id": pos_row["pair_id"],
            "time": pos_row["time"],
            "filename_ori": pos_row["filename_ori"],
            "filename_mod": pos_row["filename_mod"],
            "clean_mod_type": pos_row["clean_mod_type"],
            "category_grouped": pos_row["category_grouped"],
            "positive_rank": positive_rank,
            "winner_tier": winner_tier,
            "positive_distance": float(pos_row[metric_col]),
            "rank1_distance": float(rank1_row[metric_col]),
            "rank1_tier": str(rank1_row["negative_tier"])
                if rank1_row["y_true"] == 0 else POSITIVE_TIER_LABEL,
        })

    return pd.DataFrame(results)


# METRICS 
def compute_attribution_metrics(df_ranked: pd.DataFrame, label: str = "Overall") -> dict:
    """
    Compute retrieval metrics from ranked results.
    """
    n = len(df_ranked)
    if n == 0:
        return {"label": label, "n_queries": 0}

    ranks = df_ranked["positive_rank"].values

    top1_acc = float((ranks == 1).sum()) / n
    mean_rank = float(ranks.mean())
    mrr = float((1.0 / ranks).mean())

    result = {
        "label": label,
        "n_queries": n,
        "top1_accuracy": round(top1_acc, 4),
        "mean_rank": round(mean_rank, 4),
        "mrr": round(mrr, 4),
    }

    for r in range(1, ATTRIBUTION_GROUP_SIZE + 1):
        count = int((ranks == r).sum())
        result[f"rank_{r}_count"] = count
        result[f"rank_{r}_rate"] = round(count / n, 4)

    return result


def compute_grouped_metrics(
    df_ranked: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Compute attribution metrics grouped by a categorical column.
    """
    rows = []
    for group_val in sorted(df_ranked[group_col].unique()):
        subset = df_ranked[df_ranked[group_col] == group_val]
        metrics = compute_attribution_metrics(subset, label=str(group_val))
        rows.append(metrics)
    return pd.DataFrame(rows)


def compute_misattribution_breakdown(df_ranked: pd.DataFrame) -> pd.DataFrame:
    """
    For all misattributions (positive_rank > 1), count which negative tier
    won rank 1. Optionally break down by broad category.
    """
    misses = df_ranked[df_ranked["positive_rank"] > 1].copy()

    if misses.empty:
        return pd.DataFrame(columns=["category_grouped", "winner_tier", "count"])

    breakdown = (
        misses
        .groupby(["category_grouped", "winner_tier"])
        .size()
        .reset_index(name="count")
        .sort_values(["category_grouped", "count"], ascending=[True, False])
    )
    return breakdown


# PLOTTING
def plot_top1_by_category(
    df_broad: pd.DataFrame,
    model_name: str,
    metric_name: str,
    output_dir: Path,
) -> None:
    """Bar chart: Top-1 Attribution Accuracy per Broad Category."""
    if df_broad.empty or "top1_accuracy" not in df_broad.columns:
        return

    df_plot = df_broad[df_broad["label"] != "Overall"].copy()
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [CATEGORY_COLORS.get(cat, "#999999") for cat in df_plot["label"]]

    bars = ax.bar(
        range(len(df_plot)),
        df_plot["top1_accuracy"].values * 100,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    # Value labels on bars
    for bar, val in zip(bars, df_plot["top1_accuracy"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1%}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot["label"].values, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Top-1 Attribution Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Source Attribution Accuracy — {model_name}\n"
        f"(metric: {metric_name})",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / f"{model_name.lower()}_attribution_top1_by_category.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_rank_distribution(
    df_ranked: pd.DataFrame,
    model_name: str,
    metric_name: str,
    output_dir: Path,
) -> None:
    """Stacked bar: Rank Distribution per Broad Category."""
    if df_ranked.empty:
        return

    categories = sorted(df_ranked["category_grouped"].unique())
    if not categories:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    rank_data = {r: [] for r in range(1, ATTRIBUTION_GROUP_SIZE + 1)}

    for cat in categories:
        subset = df_ranked[df_ranked["category_grouped"] == cat]
        n = len(subset)
        for r in range(1, ATTRIBUTION_GROUP_SIZE + 1):
            count = int((subset["positive_rank"] == r).sum())
            rank_data[r].append(count / n * 100 if n > 0 else 0)

    x = np.arange(len(categories))
    width = 0.6
    bottom = np.zeros(len(categories))

    for r in range(1, ATTRIBUTION_GROUP_SIZE + 1):
        values = np.array(rank_data[r])
        ax.bar(
            x, values, width,
            bottom=bottom,
            label=f"Rank {r}",
            color=ATTRIBUTION_RANK_COLORS[r],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add percentage labels for significant segments
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val >= 5:
                ax.text(
                    x[i], bot + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white",
                )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Distribution (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Rank Distribution of True Source — {model_name}\n"
        f"(metric: {metric_name})",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / f"{model_name.lower()}_attribution_rank_distribution.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


def plot_misattribution_breakdown(
    df_breakdown: pd.DataFrame,
    model_name: str,
    metric_name: str,
    output_dir: Path,
) -> None:
    """Bar chart: Which negative tier wins when the system makes an error."""
    if df_breakdown.empty:
        return

    # Aggregate across categories for overall view
    overall = (
        df_breakdown
        .groupby("winner_tier")["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=True)
    )

    if overall.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [ATTRIBUTION_TIER_COLORS.get(t, "#999999") for t in overall["winner_tier"]]
    display_names = [ATTRIBUTION_TIER_DISPLAY.get(t, t) for t in overall["winner_tier"]]

    bars = ax.barh(
        display_names,
        overall["count"].values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    total_errors = overall["count"].sum()
    for bar, val in zip(bars, overall["count"].values):
        pct = val / total_errors * 100 if total_errors > 0 else 0
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({pct:.1f}%)",
            ha="left", va="center",
            fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Number of Misattributions", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Misattribution Error Source — {model_name}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output_dir / f"{model_name.lower()}_attribution_misattribution_breakdown.pdf"
    fig.savefig(path, dpi=PLOT_DPI, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path.name}")


# 6. CONSOLE OUTPUT

def print_attribution_table(
    overall: dict,
    df_broad: pd.DataFrame,
    model_name: str,
    metric_name: str,
) -> None:
    """Print formatted attribution results to console."""
    print(f"\n{'=' * 100}")
    print(f" SOURCE ATTRIBUTION RESULTS: {model_name.upper()}")
    print(f" Metric: {metric_name} | Protocol: 4-Way Forced Choice")
    print(f"{'=' * 100}")

    print(f"\n  Overall:")
    print(f"    Queries     : {overall['n_queries']}")
    print(f"    Top-1 Acc   : {overall['top1_accuracy']:.1%}")
    print(f"    Mean Rank   : {overall['mean_rank']:.3f}")
    print(f"    MRR         : {overall['mrr']:.4f}")
    print(f"    Rank Dist   : ", end="")
    for r in range(1, ATTRIBUTION_GROUP_SIZE + 1):
        print(f"R{r}={overall.get(f'rank_{r}_rate', 0):.1%}  ", end="")
    print()

    if not df_broad.empty:
        print(f"\n  {'Category':<35} | {'Top-1':>7} | {'MRR':>7} | "
              f"{'MeanR':>7} | {'R1':>5} | {'R2':>5} | {'R3':>5} | {'R4':>5} | {'N':>5}")
        print(f"  {'-' * 95}")

        for _, row in df_broad.iterrows():
            print(
                f"  {row['label']:<35} | {row['top1_accuracy']:>6.1%} | "
                f"{row['mrr']:>7.4f} | {row['mean_rank']:>7.3f} | "
                f"{row.get('rank_1_count', 0):>5} | {row.get('rank_2_count', 0):>5} | "
                f"{row.get('rank_3_count', 0):>5} | {row.get('rank_4_count', 0):>5} | "
                f"{row['n_queries']:>5}"
            )

    print(f"{'=' * 100}\n")


# PER-MODEL PIPELINE
def run_model_attribution(
    csv_path: str,
    model_name: str,
    metric_col: str,
    output_dir: Path,
    plots_dir: Path,
) -> dict | None:
    """
    Full attribution pipeline for one model.

    Returns overall metrics dict or None if no valid data.
    """
    csv_path_obj = Path(csv_path)
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    
    if not csv_path_obj.exists():
        logger.warning(f"Distance file not found: {csv_path_obj}")
        return None

    print(f"\n{'─' * 70}")
    print(f"Processing {model_name} | Metric: {metric_col}")
    print(f"{'─' * 70}")

    # Load and prepare
    df = pd.read_csv(csv_path_obj, keep_default_na=False)
    print(f"  Loaded {len(df)} rows from {csv_path_obj.name}")

    if metric_col not in df.columns:
        logger.error(f"  Metric '{metric_col}' not found in {model_name} data. Skipping.")
        return None

    df = prepare_dataframe(df)

    # Build valid 4-way groups
    df_valid, n_skipped, skip_reasons = build_query_groups(df, metric_col)
    n_valid = df_valid["query_key"].nunique()
    print(f"  Valid 4-way groups: {n_valid} | Skipped: {n_skipped}")
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items()):
            print(f"    - {reason}: {count}")

    if n_valid == 0:
        logger.warning(f"  No valid attribution groups for {model_name}.")
        return None

    # Rank
    df_ranked = rank_queries(df_valid, metric_col)
    print(f"  Ranked {len(df_ranked)} query groups")

    # Compute metrics
    overall = compute_attribution_metrics(df_ranked, label="Overall")
    df_broad = compute_grouped_metrics(df_ranked, "category_grouped")
    df_detailed = compute_grouped_metrics(df_ranked, "clean_mod_type")
    df_misattr = compute_misattribution_breakdown(df_ranked)

    # Print to console
    print_attribution_table(overall, df_broad, model_name, metric_col)

    # Misattribution summary
    if not df_misattr.empty:
        print(f"  Misattribution Breakdown:")
        total_errors = df_misattr["count"].sum()
        for _, row in df_misattr.groupby("winner_tier")["count"].sum().items():
            tier_display = ATTRIBUTION_TIER_DISPLAY.get(_, _)
            pct = row / total_errors * 100 if total_errors > 0 else 0
            print(f"    {tier_display:<25}: {row:>4} ({pct:.1f}%)")

    # Save CSVs
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_df = pd.DataFrame([overall])
    overall_df.insert(0, "model", model_name)
    overall_df.insert(1, "metric", metric_col)
    overall_df.to_csv(
        output_dir / f"{model_name.lower()}_attribution_overall.csv", index=False
    )

    df_broad.to_csv(
        output_dir / f"{model_name.lower()}_attribution_broad.csv", index=False
    )

    df_detailed.to_csv(
        output_dir / f"{model_name.lower()}_attribution_detailed.csv", index=False
    )

    if not df_misattr.empty:
        df_misattr.to_csv(
            output_dir / f"{model_name.lower()}_attribution_misattribution.csv",
            index=False,
        )

    # Save ranked raw data for debugging
    df_ranked.to_csv(
        output_dir / f"{model_name.lower()}_attribution_ranked.csv", index=False
    )

    # Plots
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_top1_by_category(df_broad, model_name, metric_col, plots_dir)
    plot_rank_distribution(df_ranked, model_name, metric_col, plots_dir)
    plot_misattribution_breakdown(df_misattr, model_name, metric_col, plots_dir)

    return overall



def main() -> None:
    print("=" * 70)
    print("4-WAY MUSICAL SOURCE ATTRIBUTION")
    print("=" * 70)

    # Load winning metrics
    try:
        winning_metrics = load_winning_metrics()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    print(f"\nWinning metrics loaded:")
    for model, metric in winning_metrics.items():
        print(f"  {model}: {metric}")

    # Clear old summary
    summary_path = Path(ATTRIBUTION_SUMMARY)
    if summary_path.exists():
        summary_path.unlink()

    # Process each model
    all_summaries = []

    for model_name, csv_path in MODEL_PATHS.items():
        if model_name not in winning_metrics:
            print(f"\nWarning: No winning metric for {model_name}. Skipping.")
            continue

        metric_col = winning_metrics[model_name]

        result = run_model_attribution(
            csv_path=csv_path,
            model_name=model_name,
            metric_col=metric_col,
            output_dir=ATTRIBUTION_DIR,
            plots_dir=ATTRIBUTION_PLOTS_DIR,
        )

        if result:
            summary_row = {
                "model": model_name,
                "metric": metric_col,
                **result,
            }
            all_summaries.append(summary_row)

    # Save cross-model summary
    if all_summaries:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df_summary = pd.DataFrame(all_summaries)
        df_summary.to_csv(summary_path, index=False)
        print(f"\n[SUCCESS] Attribution summary saved → {summary_path}")

        # Final comparison table
        print(f"\n{'=' * 80}")
        print(f" CROSS-MODEL ATTRIBUTION COMPARISON (4-Way Forced Choice)")
        print(f"{'=' * 80}")
        print(f"  {'Model':<10} | {'Metric':<25} | {'Top-1':>7} | "
              f"{'MRR':>7} | {'Mean Rank':>10} | {'N':>6}")
        print(f"  {'-' * 75}")
        for _, row in df_summary.iterrows():
            print(
                f"  {row['model']:<10} | {row['metric']:<25} | "
                f"{row['top1_accuracy']:>6.1%} | {row['mrr']:>7.4f} | "
                f"{row['mean_rank']:>10.3f} | {row['n_queries']:>6}"
            )
        print(f"{'=' * 80}")
    else:
        print("\n[FAILED] No models could be processed for attribution.")

    print(f"\nResults → {ATTRIBUTION_DIR}/")
    print(f"Plots   → {ATTRIBUTION_PLOTS_DIR}/")


if __name__ == "__main__":
    main()