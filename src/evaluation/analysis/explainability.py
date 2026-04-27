"""
Explainable AI (XAI) - Full Dimensionality Embedding Analysis.

Analyzes embedding vectors in their NATIVE high-dimensional space (1024/512)
WITHOUT any dimensionality reduction.  Produces:
  - Per-dimension stability analysis  (which dims never change)
  - Per-category top-K most shifted dimensions
  - Cross-category dimension comparison (SMP vs AI)
  - Per-dimension distribution stats   (mean / std)
  - Covariance & redundancy analysis   (Barlow Twins metric)
  - Effective dimensionality           (PCA / participation ratio)
  - DSP dose-response / linearity      (equivariance check)
  - CSV exports + publication-quality plots
"""

import os
import sys
import re
import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)

# Repo-root resolution
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent

_logging_util_path = repo_root / "src" / "utils" / "logging_util.py"
_spec = importlib.util.spec_from_file_location("logging_util", str(_logging_util_path))
if _spec is None or _spec.loader is None:
    raise FileNotFoundError(f"Could not load logging_util from {_logging_util_path}")
_logging_util = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_logging_util)
_logging_util.setup_logging(__file__)

import logging
logger = logging.getLogger(__name__)

sys.path.insert(0, str(repo_root / "src"))

from utils.constants import PLOT_COLORS, CATEGORY_COLORS, OUTPUT_DIRS, PLOT_STYLE_PARAMS
from utils.categorization import clean_mod_type, get_broad_category
from utils.dataset_builder import (
    build_positive_pairs,
    clean_embedding,
    validate_and_filter_embeddings,
)

plt.rcParams.update(PLOT_STYLE_PARAMS)

TOP_K      = 30
RES_SUBDIR = OUTPUT_DIRS["explainability"]
PLT_SUBDIR = OUTPUT_DIRS["explainability_plots"]


# Tiny helpers
def _cat_color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, "gray")


def _short(cat: str) -> str:
    """'3. AI Generation (Base)' → 'AI Generation (Base)'"""
    return cat.split(". ", 1)[1] if ". " in cat else cat


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


def _csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved → {path}")


# DATA LOADING 
def load_pairs(parquet_path: str, pairs_path: str, model_name: str) -> pd.DataFrame:
    """
    Load positive pairs, validate embeddings, attach delta vectors
    AND the raw mod_type string (needed for dose-response analysis).
    """
    print(f"Loading pairs for {model_name}…")
    df = build_positive_pairs(parquet_path, pairs_path)

    if df.empty:
        logger.error(f"No pairs built for {model_name}.")
        return df

    df["clean_mod_type"] = df["final_mod_type"].apply(clean_mod_type)
    df["broad_category"] = df["clean_mod_type"].apply(get_broad_category)

    print("Validating embedding shapes…")
    df, mode_dim = validate_and_filter_embeddings(
        df, emb_cols=["embedding_ori", "embedding_mod"], clean=True
    )

    if df.empty:
        logger.error("No valid pairs after embedding validation.")
        return df

    print(
        f"  {len(df)} valid pairs | dim={mode_dim} | "
        f"cats={sorted(df['broad_category'].unique())}"
    )

    emb_ori = np.stack(df["embedding_ori"].values)
    emb_mod = np.stack(df["embedding_mod"].values)
    df["delta_vector"] = list(np.abs(emb_mod - emb_ori))

    return df


# Dimension Stability
def analyze_dimension_stability(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str
) -> pd.DataFrame:
    """
    Mean |Δ| per dimension across ALL modification types.
    Low-Δ dims → content-invariant (stable); high-Δ → style/timbre.
    """
    print("Analysis 1: Dimension Stability…")

    delta_matrix = np.stack(df["delta_vector"].values)
    ndim = delta_matrix.shape[1]
    categories = sorted(df["broad_category"].unique())

    mean_delta   = np.mean(delta_matrix, axis=0)
    std_delta    = np.std(delta_matrix,  axis=0)
    median_delta = np.median(delta_matrix, axis=0)
    max_delta    = np.max(delta_matrix,  axis=0)

    cat_means: dict[str, np.ndarray] = {}
    for cat in categories:
        mask = df["broad_category"] == cat
        cat_means[cat] = np.mean(delta_matrix[mask.values], axis=0)

    cat_stack     = np.stack(list(cat_means.values()))
    cross_cat_std  = np.std(cat_stack,  axis=0)
    cross_cat_mean = np.mean(cat_stack, axis=0)

    stability_df = pd.DataFrame(
        {
            "dimension":           np.arange(ndim),
            "global_mean_delta":   mean_delta,
            "global_std_delta":    std_delta,
            "global_median_delta": median_delta,
            "global_max_delta":    max_delta,
            "cross_category_std":  cross_cat_std,
            "cross_category_mean": cross_cat_mean,
        }
    )
    for cat in categories:
        stability_df[f"mean_delta_{_short(cat)}"] = cat_means[cat]

    stability_df = stability_df.sort_values("global_mean_delta").reset_index(drop=True)
    stability_df["stability_rank"] = np.arange(1, ndim + 1)
    _csv(stability_df, os.path.join(res_dir, f"{model_name.lower()}_dimension_stability.csv"))

    # plot
    sorted_dims = stability_df["dimension"].values
    n_stable    = min(50, ndim // 10)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]}
    )

    for cat in categories:
        ax.plot(
            cat_means[cat][sorted_dims],
            label=_short(cat), color=_cat_color(cat), alpha=0.75, linewidth=0.8,
        )

    ymax = ax.get_ylim()[1]
    ax.axvspan(0,            n_stable, alpha=0.08, color="green")
    ax.axvspan(ndim - n_stable, ndim, alpha=0.08, color="red")
    ax.text(n_stable // 2,        ymax * 0.9, "STABLE",   ha="center",
            fontsize=8, color="green", fontweight="bold")
    ax.text(ndim - n_stable // 2, ymax * 0.9, "VOLATILE", ha="center",
            fontsize=8, color="red",   fontweight="bold")
    ax.set_ylabel("Mean |Δ| per Dimension")
    ax.set_title(
        f"{model_name} — Dimension Stability Landscape "
        f"(sorted by global mean Δ,  {ndim} dims)",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")

    ax2.bar(np.arange(ndim), cross_cat_std[sorted_dims], color="#9E9E9E", alpha=0.6, width=1.0)
    ax2.set_xlabel(f"Dimensions (sorted by global mean Δ,  leftmost={sorted_dims[0]})")
    ax2.set_ylabel("Cross-Cat σ")
    ax2.set_title(
        "Cross-Category Variability  (high = dimension reacts differently per category)",
        fontsize=9,
    )

    plt.tight_layout()
    _save(fig, os.path.join(plt_dir, f"{model_name.lower()}_dimension_stability.pdf"))

    return stability_df


# Top-K Most Shifted Dimensions per Category
def analyze_topk_dimensions(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str, top_k: int = TOP_K
) -> tuple[pd.DataFrame, dict]:
    """Top-K dims with HIGHEST mean |Δ| per category + overlap heatmap."""
    print(f"Analysis 2: Top-{top_k} shifted dimensions per category…")

    delta_matrix = np.stack(df["delta_vector"].values)
    categories   = sorted(df["broad_category"].unique())

    topk_records: list[dict] = []
    cat_top_dims: dict[str, np.ndarray] = {}

    for cat in categories:
        mask      = df["broad_category"] == cat
        cat_delta = delta_matrix[mask.values]
        cat_mean  = np.mean(cat_delta, axis=0)
        cat_std   = np.std(cat_delta,  axis=0)
        top_idx   = np.argsort(cat_mean)[::-1][:top_k]

        cat_top_dims[cat] = top_idx
        for rank, dim in enumerate(top_idx, 1):
            topk_records.append(
                {
                    "category":   cat,
                    "rank":       rank,
                    "dimension":  int(dim),
                    "mean_delta": float(cat_mean[dim]),
                    "std_delta":  float(cat_std[dim]),
                }
            )

    topk_df = pd.DataFrame(topk_records)
    _csv(topk_df, os.path.join(res_dir, f"{model_name.lower()}_topk_dimensions.csv"))

    n_cats = len(categories)
    fig, axes = plt.subplots(n_cats, 1, figsize=(16, 4 * n_cats))
    if n_cats == 1:
        axes = [axes]

    for idx, cat in enumerate(categories):
        ax   = axes[idx]
        cd   = topk_df[topk_df["category"] == cat]
        dims = cd["dimension"].values
        ax.bar(
            np.arange(top_k), cd["mean_delta"].values,
            yerr=cd["std_delta"].values, capsize=2,
            color=_cat_color(cat), alpha=0.8, edgecolor="black", linewidth=0.3,
        )
        ax.set_xticks(np.arange(top_k))
        ax.set_xticklabels([str(d) for d in dims], rotation=45, fontsize=7)
        ax.set_ylabel("Mean |Δ|")
        ax.set_title(f"{_short(cat)} — Top-{top_k} Most Affected Dimensions", fontweight="bold")

    plt.suptitle(
        f"{model_name} — Dimensions Most Affected per Category",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, os.path.join(plt_dir, f"{model_name.lower()}_topk_dimensions.pdf"))

    # overlap heatmap
    overlap = np.zeros((n_cats, n_cats), dtype=int)
    for i, ci in enumerate(categories):
        for j, cj in enumerate(categories):
            overlap[i, j] = len(set(cat_top_dims[ci]) & set(cat_top_dims[cj]))

    fig_ov, ax_ov = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        overlap, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=[_short(c) for c in categories],
        yticklabels=[_short(c) for c in categories],
        ax=ax_ov,
    )
    ax_ov.set_title(
        f"{model_name} — Top-{top_k} Dimension Overlap Between Categories", fontweight="bold"
    )
    plt.tight_layout()
    _save(fig_ov, os.path.join(plt_dir, f"{model_name.lower()}_topk_overlap.pdf"))

    return topk_df, cat_top_dims


# SMP vs AI Comparison
def analyze_smp_vs_ai(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str
) -> pd.DataFrame | None:
    """Dimension-by-dimension discrimination: Human Plagiarism vs AI Base."""
    print("Analysis 3: SMP vs AI comparison…")

    categories   = sorted(df["broad_category"].unique())
    smp_cats     = [c for c in categories if "Human" in c]
    ai_base_cats = [c for c in categories if "AI Generation" in c]

    if not smp_cats or not ai_base_cats:
        logger.warning(f"Required categories not found. Available: {categories}")
        return None

    smp_label = smp_cats[0]
    ai_label  = ai_base_cats[0]

    delta_matrix = np.stack(df["delta_vector"].values)
    ndim         = delta_matrix.shape[1]
    cat_labels   = df["broad_category"].values

    smp_delta = delta_matrix[cat_labels == smp_label]
    ai_delta  = delta_matrix[cat_labels == ai_label]

    smp_mean = np.mean(smp_delta, axis=0)
    ai_mean  = np.mean(ai_delta,  axis=0)
    smp_std  = np.std(smp_delta,  axis=0)
    ai_std   = np.std(ai_delta,   axis=0)

    diff       = ai_mean - smp_mean
    abs_diff   = np.abs(diff)
    pooled_std = np.sqrt((smp_std**2 + ai_std**2) / 2 + 1e-10)
    cohens_d   = diff / pooled_std

    comparison_df = pd.DataFrame(
        {
            "dimension":               np.arange(ndim),
            "smp_mean_delta":          smp_mean,
            "smp_std_delta":           smp_std,
            "ai_base_mean_delta":      ai_mean,
            "ai_base_std_delta":       ai_std,
            "difference_ai_minus_smp": diff,
            "abs_difference":          abs_diff,
            "cohens_d":                cohens_d,
        }
    )
    comparison_df = comparison_df.sort_values("abs_difference", ascending=False).reset_index(drop=True)
    comparison_df["discrimination_rank"] = np.arange(1, ndim + 1)
    _csv(comparison_df, os.path.join(res_dir, f"{model_name.lower()}_smp_vs_ai_comparison.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # A – scatter
    ax = axes[0]
    ax.scatter(smp_mean, ai_mean, alpha=0.3, s=15, c="#616161", edgecolors="none")
    lim = max(smp_mean.max(), ai_mean.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="red", linewidth=1, alpha=0.6, label="y = x")
    top_disc = comparison_df.head(TOP_K)["dimension"].values.astype(int)
    ax.scatter(
        smp_mean[top_disc], ai_mean[top_disc],
        alpha=0.8, s=40, c="#FF6F00", edgecolors="black", linewidths=0.5,
        label=f"Top-{TOP_K} discriminating", zorder=5,
    )
    for dim in top_disc[:10]:
        ax.annotate(str(dim), (smp_mean[dim], ai_mean[dim]), fontsize=6, alpha=0.7)
    ax.set_xlabel("SMP Mean |Δ|")
    ax.set_ylabel("AI Base Mean |Δ|")
    ax.set_title("Dimension-wise: SMP vs AI Base", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # B – difference profile
    ax2 = axes[1]
    bar_colors = np.where(diff > 0, "#E53935", "#2196F3")
    ax2.bar(np.arange(ndim), diff, color=bar_colors, alpha=0.6, width=1.0)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Dimension Index")
    ax2.set_ylabel("Δ(AI) − Δ(SMP)")
    ax2.set_title(
        "Per-Dimension Difference  (Red = AI shifts more, Blue = SMP shifts more)",
        fontweight="bold", fontsize=9,
    )

    # C – Cohen's d histogram
    ax3 = axes[2]
    ax3.hist(cohens_d, bins=60, color="#7E57C2", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax3.axvline(0, color="red", linewidth=1, linestyle="--")
    ax3.set_xlabel("Cohen's d  (AI − SMP)")
    ax3.set_ylabel("# Dimensions")
    ax3.set_title("Effect-Size Distribution Across Dimensions", fontweight="bold")
    n_large = int(np.sum(np.abs(cohens_d) > 0.8))
    ax3.text(
        0.95, 0.95,
        f"AI > SMP:   {np.sum(diff > 0)} dims\n"
        f"SMP > AI:   {np.sum(diff < 0)} dims\n"
        f"|d| > 0.8:  {n_large} dims",
        transform=ax3.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(
        f"{model_name} — SMP vs AI Base: Dimension-Level Comparison",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, os.path.join(plt_dir, f"{model_name.lower()}_smp_vs_ai_comparison.pdf"))

    return comparison_df


# Per-Dimension Distribution (signed shift heatmap)
def analyze_dimension_distributions(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str
) -> pd.DataFrame:
    """Mean / std of raw embedding values and SIGNED shift per category."""
    print("Analysis 4: Per-dimension value distributions…")

    emb_ori    = np.stack(df["embedding_ori"].values)
    emb_mod    = np.stack(df["embedding_mod"].values)
    ndim       = emb_ori.shape[1]
    categories = sorted(df["broad_category"].unique())
    cat_labels = df["broad_category"].values

    ori_mean = np.mean(emb_ori, axis=0)
    ori_std  = np.std(emb_ori,  axis=0)

    records: list[dict] = []
    for d in range(ndim):
        rec = {"dimension": d, "ori_mean": float(ori_mean[d]), "ori_std": float(ori_std[d])}
        for cat in categories:
            mask  = cat_labels == cat
            c_ori = emb_ori[mask, d]
            c_mod = emb_mod[mask, d]
            key   = _short(cat).replace(" ", "_").replace("/", "_")
            rec[f"mod_mean_{key}"]   = float(np.mean(c_mod))
            rec[f"mod_std_{key}"]    = float(np.std(c_mod))
            rec[f"shift_mean_{key}"] = float(np.mean(c_mod - c_ori))
            rec[f"shift_std_{key}"]  = float(np.std(c_mod  - c_ori))
        records.append(rec)

    dist_df = pd.DataFrame(records)
    _csv(dist_df, os.path.join(res_dir, f"{model_name.lower()}_dimension_distributions.csv"))

    shift_cols   = [c for c in dist_df.columns if c.startswith("shift_mean_")]
    shift_matrix = dist_df[shift_cols].values.T
    row_labels   = [c.replace("shift_mean_", "").replace("_", " ") for c in shift_cols]

    if ndim > 400:
        show_idx  = np.concatenate([np.arange(200), np.arange(ndim - 200, ndim)])
        shift_vis = shift_matrix[:, show_idx]
        dim_lbls  = [str(i) for i in show_idx]
        subtitle  = f"(first 200 + last 200 of {ndim} dims)"
    else:
        shift_vis = shift_matrix
        dim_lbls  = [str(i) for i in range(ndim)]
        subtitle  = f"(all {ndim} dims)"

    fig, ax = plt.subplots(figsize=(20, 4 + 0.6 * len(categories)))
    vmax = float(np.percentile(np.abs(shift_vis), 97))
    im   = ax.imshow(shift_vis, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    step = max(1, len(dim_lbls) // 40)
    ax.set_xticks(np.arange(0, len(dim_lbls), step))
    ax.set_xticklabels([dim_lbls[i] for i in range(0, len(dim_lbls), step)],
                       rotation=45, fontsize=6)
    ax.set_xlabel("Dimension Index")
    ax.set_title(f"{model_name} — Signed Mean Shift per Dimension  {subtitle}", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mean Signed Shift (mod − ori)")
    plt.tight_layout()
    _save(fig, os.path.join(plt_dir, f"{model_name.lower()}_dimension_distributions.pdf"))

    return dist_df


# Summary Dashboard
def generate_summary_dashboard(
    df: pd.DataFrame,
    stability_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    comparison_df: pd.DataFrame | None,
    model_name: str,
    res_dir: str,
    plt_dir: str,
) -> None:
    """6-panel overview figure."""
    print("Analysis 5: Summary Dashboard…")

    delta_matrix = np.stack(df["delta_vector"].values)
    ndim         = delta_matrix.shape[1]
    categories   = sorted(df["broad_category"].unique())
    cat_labels   = df["broad_category"].values

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # A – overall shift distribution
    ax_a = fig.add_subplot(gs[0, 0])
    total_shift = np.mean(delta_matrix, axis=1)
    for cat in categories:
        vals = total_shift[cat_labels == cat]
        ax_a.hist(vals, bins=40, alpha=0.5, color=_cat_color(cat),
                  label=_short(cat), density=True)
    ax_a.set_xlabel("Mean |Δ| across all dims")
    ax_a.set_ylabel("Density")
    ax_a.set_title("A) Overall Shift Distribution", fontweight="bold")
    ax_a.legend(fontsize=7)

    # B – active dimensions per category
    ax_b = fig.add_subplot(gs[0, 1])
    threshold = float(np.percentile(np.mean(delta_matrix, axis=0), 75))
    act_labels, act_vals, act_colors = [], [], []
    for cat in categories:
        mask = cat_labels == cat
        act_labels.append(_short(cat))
        act_vals.append(int(np.sum(np.mean(delta_matrix[mask], axis=0) > threshold)))
        act_colors.append(_cat_color(cat))

    bars = ax_b.barh(act_labels, act_vals, color=act_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, act_vals):
        ax_b.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                  str(val), va="center", fontsize=8)
    ax_b.set_xlabel(f"# Dims  mean|Δ| > {threshold:.4f}  (75th pct)")
    ax_b.set_title("B) Active Dimensions per Category", fontweight="bold")

    # C – top-10 stable
    ax_c = fig.add_subplot(gs[0, 2])
    ts = stability_df.head(10)
    ax_c.barh([f"Dim {d:.0f}" for d in ts["dimension"]], ts["global_mean_delta"],
              color="#4CAF50", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax_c.set_xlabel("Global Mean |Δ|")
    ax_c.set_title("C) Top-10 Most Stable Dimensions", fontweight="bold")
    ax_c.invert_yaxis()

    # D – top-10 volatile
    ax_d = fig.add_subplot(gs[1, 0])
    tv = stability_df.tail(10).iloc[::-1]
    ax_d.barh([f"Dim {d:.0f}" for d in tv["dimension"]], tv["global_mean_delta"],
              color="#F44336", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax_d.set_xlabel("Global Mean |Δ|")
    ax_d.set_title("D) Top-10 Most Volatile Dimensions", fontweight="bold")
    ax_d.invert_yaxis()

    # E – SMP vs AI discrimination
    ax_e = fig.add_subplot(gs[1, 1])
    if comparison_df is not None and not comparison_df.empty:
        td = comparison_df.head(15)
        ec = ["#E53935" if d > 0 else "#2196F3" for d in td["difference_ai_minus_smp"]]
        ax_e.barh([f"Dim {d:.0f}" for d in td["dimension"]], td["abs_difference"],
                  color=ec, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax_e.set_xlabel("|Δ(AI) − Δ(SMP)|")
        ax_e.set_title("E) Top-15 Discriminating Dims  (Red=AI, Blue=SMP)",
                        fontweight="bold", fontsize=9)
        ax_e.invert_yaxis()
    else:
        ax_e.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
        ax_e.set_title("E) SMP vs AI  (not available)")

    # F – text summary
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    unique_topk: set[int] = set()
    for cat in categories:
        unique_topk.update(topk_df[topk_df["category"] == cat]["dimension"].values[:TOP_K])

    txt = (
        f"Model        : {model_name}\n"
        f"Embed. dim   : {ndim}\n"
        f"Total pairs  : {len(df)}\n"
        f"──────────────────────────\n"
        f"Global mean |Δ|  : {np.mean(delta_matrix):.4f}\n"
        f"Most stable   dim: {stability_df.iloc[0]['dimension']:.0f}"
        f"  (Δ={stability_df.iloc[0]['global_mean_delta']:.4f})\n"
        f"Most volatile dim: {stability_df.iloc[-1]['dimension']:.0f}"
        f"  (Δ={stability_df.iloc[-1]['global_mean_delta']:.4f})\n"
        f"──────────────────────────\n"
        f"Unique top-{TOP_K} dims\n"
        f"  across all cats : {len(unique_topk)}/{ndim}\n"
    )
    if comparison_df is not None:
        n_large = int(np.sum(np.abs(comparison_df["cohens_d"]) > 0.8))
        txt += (
            f"──────────────────────────\n"
            f"SMP vs AI:\n"
            f"  Large effect (|d|>0.8): {n_large} dims\n"
            f"  Top discrim. dim: {comparison_df.iloc[0]['dimension']:.0f}\n"
        )

    ax_f.text(0.05, 0.95, txt, transform=ax_f.transAxes,
              fontsize=9, va="top", ha="left", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax_f.set_title("F) Summary", fontweight="bold")

    fig.suptitle(
        f"{model_name} — Embedding Dimension Dashboard",
        fontsize=16, fontweight="bold", y=0.98,
    )
    _save(fig, os.path.join(plt_dir, f"{model_name.lower()}_dimension_dashboard.pdf"))


# Covariance & Redundancy  (Barlow Twins metric)
def analyze_covariance_redundancy(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str
) -> pd.DataFrame:
    """
    Compute the cross-correlation matrix of the ORIGINAL embeddings.

    Barlow Twins metric: how far is C from the identity matrix?
      - Diagonal elements: should be 1  (variance normalised)
      - Off-diagonal:      should be 0  (no redundancy)

    High off-diagonal magnitude → dimensions encode redundant information
    (dimensional collapse in disguise).

    Outputs
    -------
    CSV  : per-dimension pair (i, j) with |corr|, flagged if |corr| > 0.8
    Plot : correlation heatmap + off-diagonal distribution
    """
    print("6: Covariance & Redundancy (Barlow Twins metric)…")

    emb_ori = np.stack(df["embedding_ori"].values)   # (N, D)
    N, D    = emb_ori.shape

    # Z-score normalise each dimension (necessary for correlation)
    mu  = emb_ori.mean(axis=0, keepdims=True)
    sig = emb_ori.std(axis=0,  keepdims=True) + 1e-8
    emb_z = (emb_ori - mu) / sig                      # (N, D)

    # Full cross-correlation matrix  C = (1/N) * Z^T Z   → shape (D, D)
    C = (emb_z.T @ emb_z) / N

    # Barlow Twins scalar metric 
    off_diag_mask  = ~np.eye(D, dtype=bool)
    bt_metric      = float(np.mean(C[off_diag_mask] ** 2))   # mean squared off-diag
    diag_dev       = float(np.mean((np.diag(C) - 1) ** 2))   # diagonal should be 1

    print(
        f"  Barlow-Twins off-diag metric: {bt_metric:.6f}  "
        f"(0 = no redundancy)  |  diag deviation: {diag_dev:.6f}"
    )

    # Per-dimension redundancy score 
    # For each dim d: mean |corr| with all OTHER dims
    abs_C                  = np.abs(C)
    np.fill_diagonal(abs_C, 0.0)
    per_dim_redundancy     = abs_C.mean(axis=1)         # (D,)
    per_dim_max_corr       = abs_C.max(axis=1)          # worst-case partner
    per_dim_n_high_corr    = (abs_C > 0.8).sum(axis=1) # dims with |r|>0.8

    redund_df = pd.DataFrame(
        {
            "dimension":          np.arange(D),
            "mean_abs_corr_others": per_dim_redundancy,
            "max_abs_corr":       per_dim_max_corr,
            "n_partners_above_0.8": per_dim_n_high_corr.astype(int),
        }
    ).sort_values("mean_abs_corr_others", ascending=False).reset_index(drop=True)
    redund_df["redundancy_rank"] = np.arange(1, D + 1)

    _csv(redund_df, os.path.join(res_dir, f"{model_name.lower()}_covariance_redundancy.csv"))

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel A – heatmap (subsample dims for readability if D > 200)
    ax = axes[0]
    if D > 200:
        # Show the 100 most redundant dims + 100 least redundant
        top_r  = redund_df.head(100)["dimension"].values
        bot_r  = redund_df.tail(100)["dimension"].values
        show   = np.concatenate([top_r, bot_r])
        C_show = C[np.ix_(show, show)]
        subtitle_h = "(top-100 + bottom-100 redundant dims)"
    else:
        C_show    = C
        subtitle_h = f"(all {D} dims)"

    im = ax.imshow(C_show, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(f"Cross-Correlation Matrix  {subtitle_h}", fontweight="bold")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Dimension")

    # Panel B – off-diagonal distribution
    ax2 = axes[1]
    off_vals = C[off_diag_mask]
    ax2.hist(off_vals, bins=100, color="#7E57C2", alpha=0.75,
             edgecolor="black", linewidth=0.2)
    ax2.axvline(0,    color="black", linewidth=1)
    ax2.axvline( 0.8, color="red",   linewidth=1, linestyle="--", label="|r|=0.8")
    ax2.axvline(-0.8, color="red",   linewidth=1, linestyle="--")
    ax2.set_xlabel("Pearson r  (off-diagonal)")
    ax2.set_ylabel("# Dimension Pairs")
    ax2.set_title("Off-Diagonal Correlation Distribution", fontweight="bold")
    n_high = int(np.sum(np.abs(off_vals) > 0.8))
    ax2.text(
        0.97, 0.97,
        f"BT metric : {bt_metric:.5f}\n"
        f"Diag dev  : {diag_dev:.5f}\n"
        f"|r|>0.8 pairs: {n_high}",
        transform=ax2.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax2.legend(fontsize=8)

    # Panel C – top-20 most redundant dims
    ax3 = axes[2]
    top20 = redund_df.head(20)
    ax3.barh(
        [f"Dim {d}" for d in top20["dimension"]],
        top20["mean_abs_corr_others"],
        color="#E53935", alpha=0.8, edgecolor="black", linewidth=0.4,
    )
    ax3.set_xlabel("Mean |r| with other dims")
    ax3.set_title("Top-20 Most Redundant Dimensions", fontweight="bold")
    ax3.invert_yaxis()

    plt.suptitle(
        f"{model_name} — Covariance & Redundancy Analysis  "
        f"(Barlow Twins metric = {bt_metric:.5f})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, os.path.join(plt_dir, f"{model_name.lower()}_covariance_redundancy.pdf"))

    return redund_df


# Effective Dimensionality  (PCA / Participation Ratio)
def analyze_effective_dimensionality(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str
) -> pd.DataFrame:
    """
    How many of the D dimensions are ACTUALLY used?

    Participation Ratio (PR) = (Σλᵢ)² / Σ(λᵢ²)
      - PR ≈ D  → uniform variance across all dims (ideal, no collapse)
      - PR ≈ 1  → single dominant direction (total collapse)

    Run separately on:
      - All original embeddings  (what the model encodes in general)
      - Delta vectors            (what CHANGES across modifications)
      - Per-category subsets     (does AI use more/fewer dims than SMP?)
    """
    print("Analysis 7: Effective Dimensionality (PCA / Participation Ratio)…")

    emb_ori      = np.stack(df["embedding_ori"].values)
    delta_matrix = np.stack(df["delta_vector"].values)
    categories   = sorted(df["broad_category"].unique())
    cat_labels   = df["broad_category"].values
    D            = emb_ori.shape[1]

    def _participation_ratio(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Returns (PR, eigenvalues, cumulative_explained_variance)."""
        pca      = PCA()
        pca.fit(matrix)
        lam      = pca.explained_variance_          # eigenvalues
        pr       = float((lam.sum() ** 2) / (lam ** 2).sum())
        cum_var  = np.cumsum(pca.explained_variance_ratio_)
        return pr, lam, cum_var

    records: list[dict] = []

    # Global – original embeddings
    pr_global, lam_global, cum_global = _participation_ratio(emb_ori)
    records.append({"subset": "All_Originals", "participation_ratio": pr_global,
                    "n_dims": D, "pr_fraction": pr_global / D})

    # Global – delta vectors
    pr_delta, lam_delta, cum_delta = _participation_ratio(delta_matrix)
    records.append({"subset": "All_Deltas", "participation_ratio": pr_delta,
                    "n_dims": D, "pr_fraction": pr_delta / D})

    # Per-category delta
    cat_pr: dict[str, float] = {}
    cat_cum: dict[str, np.ndarray] = {}
    for cat in categories:
        mask        = cat_labels == cat
        pr_c, _, cum_c = _participation_ratio(delta_matrix[mask])
        cat_pr[cat]  = pr_c
        cat_cum[cat] = cum_c
        records.append({"subset": _short(cat), "participation_ratio": pr_c,
                        "n_dims": D, "pr_fraction": pr_c / D})

    pr_df = pd.DataFrame(records)
    _csv(pr_df, os.path.join(res_dir, f"{model_name.lower()}_effective_dimensionality.csv"))

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel A – cumulative explained variance (original embeddings)
    ax = axes[0]
    ax.plot(np.arange(1, len(cum_global) + 1), cum_global,
            color="#2196F3", linewidth=1.5, label="Originals")
    ax.plot(np.arange(1, len(cum_delta) + 1), cum_delta,
            color="#E53935", linewidth=1.5, linestyle="--", label="Deltas")
    for thresh in [0.50, 0.80, 0.95]:
        n_orig = int(np.searchsorted(cum_global, thresh)) + 1
        n_delt = int(np.searchsorted(cum_delta,  thresh)) + 1
        ax.axhline(thresh, color="gray", linewidth=0.5, linestyle=":")
        ax.text(D * 0.6, thresh + 0.005,
                f"{int(thresh*100)}%: orig={n_orig}, delta={n_delt}", fontsize=7)
    ax.set_xlabel("# Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("A) Cumulative Variance  (Originals vs Deltas)", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel B – participation ratio per subset (bar chart)
    ax2 = axes[1]
    subsets = pr_df["subset"].tolist()
    pr_vals = pr_df["participation_ratio"].tolist()
    colors  = (
        ["#2196F3", "#E53935"]
        + [_cat_color(cat) for cat in categories]
    )
    bars = ax2.barh(subsets, pr_vals, color=colors, alpha=0.8,
                    edgecolor="black", linewidth=0.4)
    ax2.axvline(D, color="gray", linewidth=1, linestyle="--", label=f"Max possible = {D}")
    for bar, val in zip(bars, pr_vals):
        ax2.text(bar.get_width() + D * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}  ({val/D*100:.1f}%)", va="center", fontsize=7)
    ax2.set_xlabel("Participation Ratio  (higher = more dims used)")
    ax2.set_title("B) Effective Dimensionality per Subset", fontweight="bold")
    ax2.legend(fontsize=8)

    # Panel C – eigenvalue spectrum (log scale) of original embeddings
    ax3 = axes[2]
    ax3.plot(np.arange(1, len(lam_global) + 1), lam_global,
             color="#2196F3", linewidth=1.0, label="Originals")
    ax3.plot(np.arange(1, len(lam_delta) + 1), lam_delta,
             color="#E53935", linewidth=1.0, linestyle="--", label="Deltas")
    ax3.set_yscale("log")
    ax3.set_xlabel("Principal Component Index")
    ax3.set_ylabel("Eigenvalue  (log scale)")
    ax3.set_title("C) Eigenvalue Spectrum", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.text(
        0.97, 0.97,
        f"PR originals : {pr_global:.1f}/{D}  ({pr_global/D*100:.1f}%)\n"
        f"PR deltas    : {pr_delta:.1f}/{D}  ({pr_delta/D*100:.1f}%)",
        transform=ax3.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(
        f"{model_name} — Effective Dimensionality  "
        f"(Participation Ratio: originals={pr_global:.1f}/{D}, "
        f"deltas={pr_delta:.1f}/{D})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, os.path.join(plt_dir, f"{model_name.lower()}_effective_dimensionality.pdf"))

    return pr_df


# DSP Dose-Response  (Linearity / Equivariance Check)
def analyze_dsp_dose_response(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str
) -> pd.DataFrame:
    """
    Does the mean |Δ| grow proportionally with DSP intensity?

    Groups pairs by DSP type and intensity level, then plots:
      - Mean |Δ| vs pitch semitones  (pitch_u2, pitch_u4, pitch_u12 …)
      - Mean |Δ| vs tempo ratio      (tempo90, tempo110 …)

    A linear relationship → equivariant (interpretable) dimension.
    Saturation / non-linearity → entangled representation.

    Also finds which INDIVIDUAL DIMENSIONS are most linear
    using R² of a linear fit across intensity levels.
    """
    print("Analysis 8: DSP Dose-Response (Equivariance Check)…")

    delta_matrix = np.stack(df["delta_vector"].values)
    ndim         = delta_matrix.shape[1]
    mod_types    = df["clean_mod_type"].values

    # ── Parse intensity from mod_type string ─────────────────────────
    pitch_re = re.compile(r"pitch([ud])(\d+)")
    tempo_re = re.compile(r"tempo(\d+)")

    rows: list[dict] = []
    for i, mod in enumerate(mod_types):
        m_p = pitch_re.search(str(mod))
        m_t = tempo_re.search(str(mod))

        pitch_val = None
        tempo_val = None

        if m_p:
            sign      = 1 if m_p.group(1) == "u" else -1
            pitch_val = sign * int(m_p.group(2))

        if m_t:
            tempo_val = int(m_t.group(1))   # e.g. 90, 110, 85

        if pitch_val is not None or tempo_val is not None:
            rows.append({
                "pair_idx":   i,
                "pitch_semi": pitch_val,
                "tempo_bpm":  tempo_val,
                "mean_delta": float(np.mean(delta_matrix[i])),
                "broad_cat":  df["broad_category"].iloc[i],
            })

    if not rows:
        logger.warning("No DSP-modified pairs found – skipping dose-response.")
        return pd.DataFrame()

    dose_df = pd.DataFrame(rows)
    _csv(dose_df, os.path.join(res_dir, f"{model_name.lower()}_dsp_dose_response.csv"))

    # Per-dimension linearity (R²) for pitch
    pitch_pairs = dose_df.dropna(subset=["pitch_semi"])
    tempo_pairs = dose_df.dropna(subset=["tempo_bpm"])

    def _dim_r2(intensities: np.ndarray, dim_idx: int,
                pair_indices: np.ndarray) -> float:
        """R² of linear fit: intensity → delta[dim_idx]."""
        y = delta_matrix[pair_indices, dim_idx]
        x = intensities
        if x.std() < 1e-8:
            return 0.0
        coeffs  = np.polyfit(x, y, 1)
        y_pred  = np.polyval(coeffs, x)
        ss_res  = np.sum((y - y_pred) ** 2)
        ss_tot  = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-12))

    r2_records: list[dict] = []
    if len(pitch_pairs) > 4:
        p_idx   = pitch_pairs["pair_idx"].values
        p_semi  = pitch_pairs["pitch_semi"].values.astype(float)
        for d in range(ndim):
            r2_records.append({
                "dimension": d,
                "dsp_type":  "pitch",
                "r2_linear": _dim_r2(p_semi, d, p_idx),
            })

    if len(tempo_pairs) > 4:
        t_idx   = tempo_pairs["pair_idx"].values
        t_bpm   = tempo_pairs["tempo_bpm"].values.astype(float)
        for d in range(ndim):
            r2_records.append({
                "dimension": d,
                "dsp_type":  "tempo",
                "r2_linear": _dim_r2(t_bpm, d, t_idx),
            })

    if r2_records:
        r2_df = pd.DataFrame(r2_records)
        _csv(r2_df, os.path.join(res_dir, f"{model_name.lower()}_dsp_linearity_r2.csv"))
    else:
        r2_df = pd.DataFrame()

    # Plots
    n_panels = 2 + (1 if not r2_df.empty else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    # Panel A – mean |Δ| vs pitch semitones
    ax = axes[0]
    if not pitch_pairs.empty:
        grouped = pitch_pairs.groupby("pitch_semi")["mean_delta"].agg(["mean", "std"])
        ax.errorbar(
            grouped.index, grouped["mean"], yerr=grouped["std"],
            fmt="o-", color="#E53935", capsize=4, linewidth=1.5, markersize=6,
        )
        # Fit line
        x_fit = grouped.index.values.astype(float)
        y_fit = grouped["mean"].values
        if len(x_fit) > 2:
            coeffs = np.polyfit(x_fit, y_fit, 1)
            ax.plot(x_fit, np.polyval(coeffs, x_fit),
                    "--", color="gray", linewidth=1, label="Linear fit")
        ax.set_xlabel("Pitch Shift (semitones, negative=down)")
        ax.set_ylabel("Mean |Δ| across all dims")
        ax.set_title("A) Dose-Response: Pitch Shift", fontweight="bold")
        ax.legend(fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    else:
        ax.text(0.5, 0.5, "No pitch data", ha="center", va="center")
        ax.set_title("A) Pitch (no data)")

    # Panel B – mean |Δ| vs tempo
    ax2 = axes[1]
    if not tempo_pairs.empty:
        grouped_t = tempo_pairs.groupby("tempo_bpm")["mean_delta"].agg(["mean", "std"])
        ax2.errorbar(
            grouped_t.index, grouped_t["mean"], yerr=grouped_t["std"],
            fmt="s-", color="#2196F3", capsize=4, linewidth=1.5, markersize=6,
        )
        x_fit_t = grouped_t.index.values.astype(float)
        y_fit_t = grouped_t["mean"].values
        if len(x_fit_t) > 2:
            coeffs_t = np.polyfit(x_fit_t, y_fit_t, 1)
            ax2.plot(x_fit_t, np.polyval(coeffs_t, x_fit_t),
                     "--", color="gray", linewidth=1, label="Linear fit")
        ax2.set_xlabel("Tempo  (% of original, 100 = unchanged)")
        ax2.set_ylabel("Mean |Δ| across all dims")
        ax2.set_title("B) Dose-Response: Tempo Change", fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.axvline(100, color="black", linewidth=0.5, linestyle=":")
    else:
        ax2.text(0.5, 0.5, "No tempo data", ha="center", va="center")
        ax2.set_title("B) Tempo (no data)")

    # Panel C – R² distribution (which dims respond linearly to DSP?)
    if not r2_df.empty and n_panels == 3:
        ax3 = axes[2]
        for dsp_type, color in [("pitch", "#E53935"), ("tempo", "#2196F3")]:
            subset = r2_df[r2_df["dsp_type"] == dsp_type]
            if not subset.empty:
                ax3.hist(subset["r2_linear"], bins=50, alpha=0.55,
                         color=color, label=dsp_type, edgecolor="black", linewidth=0.2)
        ax3.set_xlabel("R²  of linear fit  (intensity → dim shift)")
        ax3.set_ylabel("# Dimensions")
        ax3.set_title(
            "C) Per-Dimension Linearity  (R² distribution)\n"
            "High R² → equivariant / interpretable dim",
            fontweight="bold",
        )
        ax3.legend(fontsize=8)

        # Annotate top-5 most linear dims per DSP type
        for dsp_type in r2_df["dsp_type"].unique():
            top5 = (r2_df[r2_df["dsp_type"] == dsp_type]
                    .sort_values("r2_linear", ascending=False)
                    .head(5)["dimension"].tolist())
            print(f"  Top-5 most linear dims for {dsp_type}: {top5}")

    plt.suptitle(
        f"{model_name} — DSP Dose-Response & Equivariance Analysis",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, os.path.join(plt_dir, f"{model_name.lower()}_dsp_dose_response.pdf"))

    return dose_df


def process_model(
    parquet_path: str, pairs_path: str,
    model_name: str, res_dir: str, plt_dir: str,
) -> None:
    print(f"{'='*60}")
    print(f"  FULL DIMENSION ANALYSIS: {model_name}")
    print(f"{'='*60}")

    df = load_pairs(parquet_path, pairs_path, model_name)
    if df.empty:
        logger.error(f"Empty dataset for {model_name} – aborting.")
        return

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    # Original 5 analyses
    stability_df  = analyze_dimension_stability(df,  model_name, res_dir, plt_dir)
    topk_df, _    = analyze_topk_dimensions(df,      model_name, res_dir, plt_dir)
    comparison_df = analyze_smp_vs_ai(df,             model_name, res_dir, plt_dir)
    _             = analyze_dimension_distributions(df, model_name, res_dir, plt_dir)
    generate_summary_dashboard(df, stability_df, topk_df, comparison_df,
                                model_name, res_dir, plt_dir)

    # New analyses
    analyze_covariance_redundancy(df,        model_name, res_dir, plt_dir)
    analyze_effective_dimensionality(df,     model_name, res_dir, plt_dir)
    analyze_dsp_dose_response(df,            model_name, res_dir, plt_dir)

    print(f"All analyses complete for {model_name}.")


def main() -> None:
    print("=" * 80)
    print("STARTING FULL EMBEDDING DIMENSION ANALYSIS ")
    print("=" * 80)

    PAIRS_CSV     = "data/Final_dataset_pairs.csv"
    CLEWS_PARQUET = "data/clews_embeddings.parquet"
    WEALY_PARQUET = "data/wealy_embeddings.parquet"

    for model_name, parquet in [("CLEWS", CLEWS_PARQUET), ("WEALY", WEALY_PARQUET)]:
        if os.path.exists(parquet) and os.path.exists(PAIRS_CSV):
            process_model(parquet, PAIRS_CSV, model_name, RES_SUBDIR, PLT_SUBDIR)
        else:
            logger.warning(f"{model_name}: data not found – skipping.")

    print("=" * 80)
    print("DIMENSION ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()