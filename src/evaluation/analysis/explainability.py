"""
Explainable AI (XAI) - Embedding Plagiarism Signature Analysis.

Analyzes embedding vectors in their NATIVE high-dimensional space (1024/512).
Focused on the core research question: "How does plagiarism manifest in latent space?"

Generates:
  1. Overall Shift Distribution & Active Dimensions
  2. Top-30 Dimensions Most Affected per Category
  3. Top-30 Dimension Overlap Between Categories (Heatmap)
  4. Signed Mean Shift per Dimension (Fingerprints)
  5. SMP vs AI Base: Dimension-Level Comparison
  6. Stable Core Preservation (Plagiarism definition: Content stays, Style changes)
"""

import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)

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
logging_util.setup_logging(__file__)

logger = logging.getLogger(__name__)

sys.path.insert(0, str(repo_root / "src"))
from utils.constants import (
    PLOT_STYLE_PARAMS, PLOT_DPI,
    CATEGORY_COLORS, DSP_FAMILY_COLORS, SOURCE_COLORS,
    OUTPUT_DIRS, SMP_CSV, EMBEDDING_PATHS,
)
from utils.categorization import clean_mod_type, get_broad_category, extract_dsp_and_source_features, get_dsp_family
from utils.dataset_builder import build_positive_pairs, validate_and_filter_embeddings

plt.rcParams.update(PLOT_STYLE_PARAMS)

# CONFIG
TOP_K       = 30
RES_SUBDIR  = OUTPUT_DIRS["explainability"]
PLT_SUBDIR  = OUTPUT_DIRS["explainability_plots"]


# SMALL HELPERS
def _cat_color(cat: str) -> str:
    return (
        CATEGORY_COLORS.get(cat)
        or DSP_FAMILY_COLORS.get(cat)
        or SOURCE_COLORS.get(cat)
        or "gray"
    )


def _short(cat: str) -> str:
    """Strip leading numeric prefix (e.g. '1a. Human …' → 'Human …')."""
    return cat.split(". ", 1)[1] if ". " in cat else cat


def _save(fig: plt.Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved Plot → {path}")
    plt.close(fig)


def _csv(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved CSV  → {path}")


# DATA LOADING
def load_pairs(parquet_path: str, pairs_path: str, model_name: str) -> pd.DataFrame:
    print(f"Loading pairs for {model_name}…")
    df = build_positive_pairs(parquet_path, pairs_path)

    if df.empty:
        logger.error(f"No pairs built for {model_name}.")
        return df

    df["clean_mod_type"] = df["final_mod_type"].apply(clean_mod_type)
    df["broad_category"] = df["clean_mod_type"].apply(get_broad_category)

    # Extract DSP/source features via existing util
    feat = df["clean_mod_type"].apply(extract_dsp_and_source_features)
    feat_df = pd.DataFrame(feat.tolist())
    df["source"] = feat_df["source"]
    df["pitch_intensity"] = feat_df["pitch_intensity"]
    df["tempo_intensity"] = feat_df["tempo_intensity"]
    df["dsp_category"] = feat_df["dsp_category"]
    df["dsp_family"] = df.apply(
        lambda r: get_dsp_family(r["pitch_intensity"], r["tempo_intensity"]),
        axis=1,
    )

    df, mode_dim = validate_and_filter_embeddings(
        df, emb_cols=["embedding_ori", "embedding_mod"], clean=True
    )

    if df.empty:
        return df

    emb_ori = np.stack(df["embedding_ori"].values)
    emb_mod = np.stack(df["embedding_mod"].values)
    df["delta_vector"] = list(np.abs(emb_mod - emb_ori))

    print(f"Loaded {len(df)} pairs | Dim: {mode_dim}")
    return df

def export_pairwise_delta_features(
    df: pd.DataFrame,
    model_name: str,
    res_dir: str,
    data_dir: str = "data",
) -> None:
    """
    Export pair-level delta summary features derived from delta_vector = |emb_mod - emb_ori|.

    Note:
        This export contains POSITIVE pairs only, because explainability.py
        is built on top of build_positive_pairs().

    Saved files:
        - {model}_pairwise_delta_features.csv
        - {model}_pairwise_delta_features.parquet
    """
    print("Exporting pairwise delta summary features…")

    if df.empty or "delta_vector" not in df.columns:
        logger.warning(f"No delta vectors available for {model_name}. Skipping export.")
        return

    delta_matrix = np.stack(df["delta_vector"].values)

    # Global threshold for "active dimensions"
    global_q75 = float(np.percentile(np.mean(delta_matrix, axis=0), 75))

    # Stable / volatile dimensions (same logic as plot_stable_core)
    global_mean       = np.mean(delta_matrix, axis=0)
    stability_ranking = np.argsort(global_mean)
    n_core            = min(100, delta_matrix.shape[1] // 5)
    stable_dims       = stability_ranking[:n_core]
    volatile_dims     = stability_ranking[-n_core:]

    # Pair-level summaries
    delta_mean   = np.mean(delta_matrix, axis=1)
    delta_std    = np.std(delta_matrix, axis=1)
    delta_median = np.median(delta_matrix, axis=1)
    delta_max    = np.max(delta_matrix, axis=1)
    delta_l2     = np.linalg.norm(delta_matrix, axis=1)
    delta_p90    = np.percentile(delta_matrix, 90, axis=1)
    delta_p95    = np.percentile(delta_matrix, 95, axis=1)

    active_dims_q75_global = np.sum(delta_matrix > global_q75, axis=1)

    stable_core_mean_delta   = np.mean(delta_matrix[:, stable_dims], axis=1)
    volatile_shell_mean_delta = np.mean(delta_matrix[:, volatile_dims], axis=1)

    stable_to_volatile_ratio = np.divide(
        stable_core_mean_delta,
        volatile_shell_mean_delta,
        out=np.zeros_like(stable_core_mean_delta),
        where=volatile_shell_mean_delta != 0,
    )

    df_export = pd.DataFrame({
        "pair_id": df["pair_id"].values,
        "time": df["time"].values,
        "final_mod_type": df["final_mod_type"].values,
        "clean_mod_type": df["clean_mod_type"].values,
        "broad_category": df["broad_category"].values,
        "filename_ori": df["filename_ori"].values,
        "filename_mod": df["filename_mod"].values,
        "source": df["source"].values,
        "dsp_category": df["dsp_category"].values,
        "dsp_family": df["dsp_family"].values,
        "pitch_intensity": df["pitch_intensity"].values,
        "tempo_intensity": df["tempo_intensity"].values,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_median": delta_median,
        "delta_max": delta_max,
        "delta_l2": delta_l2,
        "delta_p90": delta_p90,
        "delta_p95": delta_p95,
        "active_dims_q75_global": active_dims_q75_global,
        "stable_core_mean_delta": stable_core_mean_delta,
        "volatile_shell_mean_delta": volatile_shell_mean_delta,
        "stable_to_volatile_ratio": stable_to_volatile_ratio,
    })

    # Save CSV
    csv_path = Path(res_dir) / f"{model_name.lower()}_pairwise_delta_features.csv"
    _csv(df_export, csv_path)

    # Save parquet
    pq_path = Path(res_dir) / f"{model_name.lower()}_pairwise_delta_features.parquet"
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    df_export.to_parquet(pq_path, index=False)
    print(f"Saved Parquet → {pq_path}")


# PLOT 1: Overall Shifts & Active Dimensions
def plot_overall_shifts(df, model_name, plt_dir, group_col="broad_category", suffix=""):
    print("Generating Plot 1: Overall Shifts & Active Dimensions…")

    delta_matrix = np.stack(df["delta_vector"].values)
    categories = sorted(df[group_col].dropna().unique())
    cat_labels = df[group_col].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # A) Overall Shift Distribution
    ax1           = axes[0]
    total_shift   = np.mean(delta_matrix, axis=1)
    for cat in categories:
        vals = total_shift[cat_labels == cat]
        ax1.hist(vals, bins=40, alpha=0.5, color=_cat_color(cat),
                 label=_short(cat), density=True)
    ax1.set_xlim(0, np.percentile(total_shift, 99) * 1.05)
    ax1.set_xlabel("Mean |Δ| across all dimensions")
    ax1.set_ylabel("Density")
    ax1.set_title("A) Overall Shift Distribution", fontweight="bold")
    ax1.legend(fontsize=9)

    # B) Active Dimensions per Category
    ax2           = axes[1]
    threshold     = float(np.percentile(np.mean(delta_matrix, axis=0), 75))
    act_labels, act_vals, act_colors = [], [], []
    for cat in categories:
        mask = cat_labels == cat
        act_labels.append(_short(cat))
        act_vals.append(int(np.sum(np.mean(delta_matrix[mask], axis=0) > threshold)))
        act_colors.append(_cat_color(cat))

    bars = ax2.barh(act_labels, act_vals, color=act_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, act_vals):
        ax2.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9
        )
    ax2.set_xlabel(f"# Dimensions with mean |Δ| > {threshold:.4f} (75th pct)")
    ax2.set_title("B) Active Dimensions per Category", fontweight="bold")

    plt.suptitle(
        f"{model_name} — Shift Magnitudes & Active Dimensions",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, Path(plt_dir) / f"{model_name.lower()}_overall_shifts{suffix}.pdf")


# PLOTS 2 & 3: Top-K Affected Dimensions & Overlap
def plot_topk_and_overlap(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str,
    group_col: str = "broad_category", suffix: str = "",
) -> None:
    print(f"Generating Plots 2 & 3: Top-{TOP_K} Dimensions & Overlap…")

    delta_matrix = np.stack(df["delta_vector"].values)
    categories   = sorted(df[group_col].dropna().unique())
    cat_top_dims = {}
    topk_records = []

    for cat in categories:
        mask     = df[group_col] == cat
        cat_mean = np.mean(delta_matrix[mask.values], axis=0)
        cat_std  = np.std(delta_matrix[mask.values],  axis=0)
        top_idx  = np.argsort(cat_mean)[::-1][:TOP_K]
        cat_top_dims[cat] = top_idx

        for rank, dim in enumerate(top_idx, 1):
            topk_records.append({
                "category":   cat,
                "rank":       rank,
                "dimension":  int(dim),
                "mean_delta": float(cat_mean[dim]),
                "std_delta":  float(cat_std[dim]),
            })

    topk_df = pd.DataFrame(topk_records)
    _csv(topk_df, Path(res_dir) / f"{model_name.lower()}_topk_dimensions{suffix}.csv")

    # Plot 2: Bar charts per category
    n_cats    = len(categories)
    fig, axes = plt.subplots(n_cats, 1, figsize=(16, 3.5 * n_cats))
    if n_cats == 1:
        axes = [axes]

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        cd = topk_df[topk_df["category"] == cat]
        ax.bar(
            np.arange(TOP_K), cd["mean_delta"].values,
            yerr=cd["std_delta"].values, capsize=2,
            color=_cat_color(cat), alpha=0.8, edgecolor="black", linewidth=0.3
        )
        ax.set_xticks(np.arange(TOP_K))
        ax.set_xticklabels(
            [str(d) for d in cd["dimension"].values], rotation=45, fontsize=8
        )
        ax.set_ylabel("Mean |Δ|")
        ax.set_title(
            f"{_short(cat)} — Top-{TOP_K} Most Affected Dimensions",
            fontweight="bold"
        )

    plt.suptitle(
        f"{model_name} — Dimensions Most Affected per Category",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, Path(plt_dir) / f"{model_name.lower()}_topk_dimensions{suffix}.pdf")

    # Plot 3: Overlap Heatmap
    overlap = np.zeros((n_cats, n_cats), dtype=int)
    for i, ci in enumerate(categories):
        for j, cj in enumerate(categories):
            overlap[i, j] = len(set(cat_top_dims[ci]) & set(cat_top_dims[cj]))

    fig_ov, ax_ov = plt.subplots(figsize=(8, 6))
    labels = [_short(c) for c in categories]
    sns.heatmap(
        overlap, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax_ov,
        cbar_kws={"label": f"Shared Dimensions (out of Top {TOP_K})"},
    )
    ax_ov.set_title(
        f"{model_name} — Top-{TOP_K} Dimension Overlap Between Categories",
        fontweight="bold", pad=15
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig_ov, Path(plt_dir) / f"{model_name.lower()}_topk_overlap{suffix}.pdf")


# PLOT 4: Signed Shift Heatmap
def plot_signed_shift(
    df: pd.DataFrame, model_name: str, plt_dir: str,
    group_col: str = "broad_category", suffix: str = "",
) -> None:
    print("Generating Plot 4: Signed Mean Shift per Dimension…")

    emb_ori    = np.stack(df["embedding_ori"].values)
    emb_mod    = np.stack(df["embedding_mod"].values)
    ndim       = emb_ori.shape[1]
    categories = sorted(df[group_col].dropna().unique())

    shift_matrix = np.zeros((len(categories), ndim))
    for i, cat in enumerate(categories):
        mask = df[group_col] == cat
        shift_matrix[i, :] = np.mean(emb_mod[mask] - emb_ori[mask], axis=0)

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
    vmax    = float(np.percentile(np.abs(shift_vis), 97))
    im      = ax.imshow(
        shift_vis, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest"
    )

    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels([_short(c) for c in categories], fontsize=10)

    step = max(1, len(dim_lbls) // 40)
    ax.set_xticks(np.arange(0, len(dim_lbls), step))
    ax.set_xticklabels(
        [dim_lbls[i] for i in range(0, len(dim_lbls), step)],
        rotation=45, fontsize=8
    )
    ax.set_xlabel("Dimension Index")
    ax.set_title(
        f"{model_name} — Signed Mean Shift per Dimension {subtitle}",
        fontweight="bold", fontsize=13
    )
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mean Signed Shift (mod − ori)")
    plt.tight_layout()
    _save(fig, Path(plt_dir) / f"{model_name.lower()}_signed_shift_heatmap{suffix}.pdf")


# PLOT 5: SMP vs AI Base Discrimination
def plot_smp_vs_ai(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str
) -> None:
    print("Generating Plot 5: SMP vs AI Base Discrimination…")

    categories = sorted(df["broad_category"].unique())
    smp_cats   = [c for c in categories if "Human" in c]
    ai_cats    = [c for c in categories if "AI Generation" in c]

    if not smp_cats or not ai_cats:
        logger.warning("Required categories missing for SMP vs AI plot.")
        return

    smp_label, ai_label = smp_cats[0], ai_cats[0]
    delta_matrix = np.stack(df["delta_vector"].values)
    cat_labels   = df["broad_category"].values

    smp_delta    = delta_matrix[cat_labels == smp_label]
    ai_delta     = delta_matrix[cat_labels == ai_label]
    smp_mean     = np.mean(smp_delta, axis=0)
    ai_mean      = np.mean(ai_delta,  axis=0)
    pooled_std   = np.sqrt(
        (np.std(smp_delta, axis=0)**2 + np.std(ai_delta, axis=0)**2) / 2 + 1e-10
    )
    cohens_d = (ai_mean - smp_mean) / pooled_std

    comparison_df = pd.DataFrame({
        "dimension":         np.arange(delta_matrix.shape[1]),
        "smp_mean_delta":    smp_mean,
        "ai_base_mean_delta":ai_mean,
        "difference":        ai_mean - smp_mean,
        "cohens_d":          cohens_d,
    }).sort_values("cohens_d", key=abs, ascending=False)
    _csv(comparison_df, Path(res_dir) / f"{model_name.lower()}_smp_vs_ai.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # A) Scatter
    ax1 = axes[0]
    ax1.scatter(smp_mean, ai_mean, alpha=0.4, s=20, c="#607D8B", edgecolors="none")
    lim = max(smp_mean.max(), ai_mean.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "--", color="red", linewidth=1.2,
             label="y=x (Perfect Alignment)")
    ax1.set_xlabel(f"Mean Shift: {_short(smp_label)}", fontsize=10)
    ax1.set_ylabel(f"Mean Shift: {_short(ai_label)}",  fontsize=10)
    ax1.set_title("A) Dimension-Level Shift Comparison", fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # B) Cohen's d distribution
    ax2   = axes[1]
    n_sig = int(np.sum(np.abs(cohens_d) > 0.8))
    ax2.hist(cohens_d, bins=50, color="darkblue", alpha=0.8,
             edgecolor="black", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax2.text(
        0.95, 0.95,
        f"Large Effect (|d| > 0.8):\n{n_sig} dimensions",
        transform=ax2.transAxes, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )
    ax2.set_xlabel("Cohen's d (AI − SMP)", fontsize=10)
    ax2.set_ylabel("Number of Dimensions", fontsize=10)
    ax2.set_title("B) Effect Size Distribution (Separability)", fontweight="bold")

    plt.suptitle(
        f"{model_name} — SMP vs AI Base: Dimension-Level Comparison",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, Path(plt_dir) / f"{model_name.lower()}_smp_vs_ai.pdf")


# PLOT 6: Stable Core Preservation
def plot_stable_core(df: pd.DataFrame, model_name: str, plt_dir: str) -> None:
    print("Generating Plot 6: Stable Core Preservation…")

    delta_matrix      = np.stack(df["delta_vector"].values)
    global_mean       = np.mean(delta_matrix, axis=0)
    stability_ranking = np.argsort(global_mean)
    n_core            = min(100, delta_matrix.shape[1] // 5)
    stable_dims       = stability_ranking[:n_core]
    volatile_dims     = stability_ranking[-n_core:]

    categories = sorted(df["broad_category"].unique())
    cat_labels = df["broad_category"].values
    records    = []

    for cat in categories:
        mask            = cat_labels == cat
        stable_shifts   = delta_matrix[mask][:, stable_dims].mean(axis=1)
        volatile_shifts = delta_matrix[mask][:, volatile_dims].mean(axis=1)
        records.append({
            "category":                  cat,
            "Stable Core (Content)":     np.mean(stable_shifts),
            "Volatile Shell (Style/Noise)": np.mean(volatile_shifts),
        })

    core_df    = pd.DataFrame(records)
    fig, ax    = plt.subplots(figsize=(10, 6))
    x          = np.arange(len(categories))
    width      = 0.35
    bar_colors = [_cat_color(cat) for cat in categories]

    ax.bar(x - width / 2, core_df["Stable Core (Content)"], width,
           color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.bar(x + width / 2, core_df["Volatile Shell (Style/Noise)"], width,
           color=bar_colors, edgecolor="black", linewidth=0.8, hatch="///", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short(c) for c in categories], rotation=15, ha="right", fontsize=10
    )
    ax.set_ylabel("Mean Absolute Shift (|Δ|)", fontsize=11)
    ax.set_title(
        f"{model_name} — Stable Core Preservation\n"
        "(Plagiarism retains content while disrupting style)",
        fontweight="bold", fontsize=13
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    legend_elements = [
        mpatches.Patch(facecolor="gray", edgecolor="black",
                       label=f"Stable Core (Top {n_core} Dims)"),
        mpatches.Patch(facecolor="gray", edgecolor="black", hatch="///", alpha=0.7,
                       label=f"Volatile Shell (Top {n_core} Dims)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

    _save(fig, Path(plt_dir) / f"{model_name.lower()}_stable_core.pdf")



def process_model(parquet_path: str, pairs_path: str, model_name: str) -> None:
    print("=" * 60)
    print(f"ANALYZING PLAGIARISM SIGNATURES: {model_name}")
    print("=" * 60)

    df = load_pairs(parquet_path, pairs_path, model_name)
    if df.empty:
        return

    export_pairwise_delta_features(df, model_name, RES_SUBDIR)

    # Run 1: broad_category analysis
    plot_overall_shifts(df, model_name, PLT_SUBDIR)
    plot_topk_and_overlap(df, model_name, RES_SUBDIR, PLT_SUBDIR)
    plot_signed_shift(df, model_name, PLT_SUBDIR)
    plot_stable_core(df, model_name, PLT_SUBDIR)

    # Run 2: DSP Family (Pitch vs Tempo)
    plot_overall_shifts(df, model_name, PLT_SUBDIR, group_col="dsp_family", suffix="_by_dsp_family")
    plot_topk_and_overlap(df, model_name, RES_SUBDIR, PLT_SUBDIR, group_col="dsp_family", suffix="_by_dsp_family")
    plot_signed_shift(df, model_name, PLT_SUBDIR, group_col="dsp_family", suffix="_by_dsp_family")

    # Run 3: Source (MusicGen vs AudioLDM2 vs MGE-LDM vs Human)
    plot_overall_shifts(df, model_name, PLT_SUBDIR, group_col="source", suffix="_by_source")
    plot_topk_and_overlap(df, model_name, RES_SUBDIR, PLT_SUBDIR, group_col="source", suffix="_by_source")
    plot_signed_shift(df, model_name, PLT_SUBDIR, group_col="source", suffix="_by_source")

    print(f"Done processing {model_name}.\n")


def main() -> None:
    pairs_csv = SMP_CSV

    for model_name, parquet in EMBEDDING_PATHS.items():
        parquet_path = Path(parquet)
        pairs_path   = Path(pairs_csv)

        if parquet_path.exists() and pairs_path.exists():
            process_model(str(parquet_path), str(pairs_path), model_name)
        else:
            logger.warning(f"Data missing for {model_name}. Skipping.")


if __name__ == "__main__":
    main()