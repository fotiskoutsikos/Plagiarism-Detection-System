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
  6. Stable Core vs Volatile Shell Analysis
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
import re

warnings.filterwarnings("ignore", category=FutureWarning)

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
    MGELDM_STEM_COLORS,
    OUTPUT_DIRS, SMP_CSV, EMBEDDING_PATHS,
)
from utils.categorization import (
    clean_mod_type, get_broad_category,
    extract_dsp_and_source_features, get_dsp_family,
    _get_source_group, _get_dsp_label,
)
from utils.dataset_builder import build_positive_pairs, validate_and_filter_embeddings

plt.rcParams.update(PLOT_STYLE_PARAMS)

# CONFIG 
TOP_K      = 30
RES_SUBDIR = OUTPUT_DIRS["explainability"]
PLT_SUBDIR = OUTPUT_DIRS["explainability_plots"]

# RUN CONFIG KEYS 
# Each run is a dict with these keys:
#   group_col   : column name used to split df into categories
#   suffix      : appended to every output filename  (safe for all OS)
#   filter_col  : if not None, restrict df to rows where filter_col == filter_val
#   filter_val  : value to filter on
#   context     : human-readable string used in plot titles
#   include_plots: set of plot IDs to run  ("shifts","topk","signed","core","smp_ai")
#                  None → run all that apply

_ALL_PLOTS   = {"shifts", "topk", "signed", "core", "smp_ai"}
_NO_CORE_SMP = {"shifts", "topk", "signed"}   # subgroup runs skip core & smp_ai


# COLOR HELPERS 
_SOURCE_GROUP_COLORS = {
    **SOURCE_COLORS,
    "MGE-LDM_bass":  MGELDM_STEM_COLORS["bass"],
    "MGE-LDM_drums": MGELDM_STEM_COLORS["drums"],
    "MGE-LDM_other": MGELDM_STEM_COLORS["other"],
}


def _directional_color(label: str) -> str:
    lbl = label.lower()
    if lbl == "base":
        return "#9E9E9E"
    if "pitch" in lbl and "tempo" in lbl:
        return "#E53935"
    if lbl.startswith("pitchu"):
        val = int(re.search(r"\d+", lbl).group())
        return "#1565C0" if val >= 4 else "#64B5F6"
    if lbl.startswith("pitchd"):
        val = int(re.search(r"\d+", lbl).group())
        return "#6A1B9A" if val >= 4 else "#CE93D8"
    if lbl.startswith("tempo"):
        val = int(re.search(r"\d+", lbl).group())
        if val > 100:
            return "#E65100" if val >= 110 else "#FFCC80"
        return "#00695C" if val <= 90 else "#80CBC4"
    return "#9E9E9E"


def _cat_color(cat: str) -> str:
    for d in (CATEGORY_COLORS, DSP_FAMILY_COLORS, _SOURCE_GROUP_COLORS, MGELDM_STEM_COLORS):
        if cat in d:
            return d[cat]
    return _directional_color(cat)


def _short(cat: str) -> str:
    return cat.split(". ", 1)[1] if ". " in cat else cat


# PATH / TITLE HELPERS
_UNSAFE = re.compile(r'[<>:"/\\|?*+\s]+')

def _safe_suffix(text: str) -> str:
    """
    Convert any string to a filesystem-safe suffix fragment.
    Replaces whitespace and characters illegal on Windows with underscores,
    then strips leading/trailing underscores.

    Examples:
        "Original + DSP" → "original_dsp"
        "MGE-LDM_bass"   → "mge-ldm_bass"
    """
    return _UNSAFE.sub("_", text).strip("_").lower()


_GROUP_COL_DISPLAY = {
    "broad_category":      "Broad Category",
    "dsp_label":           "DSP Label",
    "source_group_merged": "Source",
    "source_group":        "Source",
    "stem":                "MGE-LDM Stem",
}


def _make_suptitle(model_name: str, base_title: str, context: str) -> str:
    """
    Builds a consistent two-line suptitle.
      Line 1: "{model_name} — {base_title}"
      Line 2: context  (e.g. "All sources · Grouped by: Broad Category")
    """
    return f"{model_name} — {base_title}\n{context}"


def _run_context(run: dict) -> str:
    """
    Human-readable context string for a run config dict.
    Used in plot titles and logged output.
    """
    group_display = _GROUP_COL_DISPLAY.get(run["group_col"], run["group_col"])
    source_display = run.get("context", "All sources")
    return f"{source_display} -- Grouped by: {group_display}"


# I/O HELPERS 
def _save(fig: plt.Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"  Saved Plot → {path}")
    plt.close(fig)


def _csv(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  Saved CSV  → {path}")


# MGE-LDM STEM MERGING
def _merge_mgeldm_stems(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `source_group_merged`: MGE-LDM_bass/drums/other → "MGE-LDM".
    Used for all cross-source plots so MGE-LDM appears as one entry.
    """
    df = df.copy()
    df["source_group_merged"] = df["source_group"].apply(
        lambda sg: "MGE-LDM" if str(sg).startswith("MGE-LDM") else sg
    )
    return df


# DATA LOADING 
def load_pairs(parquet_path: str, pairs_path: str, model_name: str) -> pd.DataFrame:
    print(f"Loading pairs for {model_name}…")
    df = build_positive_pairs(parquet_path, pairs_path)

    if df.empty:
        logger.error(f"No pairs built for {model_name}.")
        return df

    df["clean_mod_type"] = df["final_mod_type"].apply(clean_mod_type)
    df["broad_category"] = df["clean_mod_type"].apply(get_broad_category)

    feat_df = pd.DataFrame(
        df["clean_mod_type"].apply(extract_dsp_and_source_features).tolist()
    )
    df["source"]          = feat_df["source"]
    df["stem"]            = feat_df["stem"]
    df["pitch_intensity"] = feat_df["pitch_intensity"]
    df["tempo_intensity"] = feat_df["tempo_intensity"]
    df["dsp_category"]    = feat_df["dsp_category"]
    df["dsp_family"]      = df.apply(
        lambda r: get_dsp_family(r["pitch_intensity"], r["tempo_intensity"]), axis=1
    )
    df["source_group"] = df.apply(_get_source_group, axis=1)
    df["dsp_label"]    = df["clean_mod_type"].apply(_get_dsp_label)
    df = _merge_mgeldm_stems(df)

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


# EXPORT 
def export_pairwise_delta_features(
    df: pd.DataFrame, model_name: str, res_dir: str, data_dir: str = "data"
) -> None:
    """Export pair-level delta summary features (positive pairs only)."""
    print("Exporting pairwise delta summary features…")
    if df.empty or "delta_vector" not in df.columns:
        logger.warning(f"No delta vectors for {model_name}. Skipping export.")
        return

    delta_matrix = np.stack(df["delta_vector"].values)
    global_q75   = float(np.percentile(np.mean(delta_matrix, axis=0), 75))

    global_mean       = np.mean(delta_matrix, axis=0)
    stability_ranking = np.argsort(global_mean)
    n_core            = min(100, delta_matrix.shape[1] // 5)
    stable_dims       = stability_ranking[:n_core]
    volatile_dims     = stability_ranking[-n_core:]

    s_core = np.mean(delta_matrix[:, stable_dims],   axis=1)
    s_vol  = np.mean(delta_matrix[:, volatile_dims], axis=1)

    df_export = pd.DataFrame({
        "pair_id":                   df["pair_id"].values,
        "time":                      df["time"].values,
        "final_mod_type":            df["final_mod_type"].values,
        "clean_mod_type":            df["clean_mod_type"].values,
        "broad_category":            df["broad_category"].values,
        "filename_ori":              df["filename_ori"].values,
        "filename_mod":              df["filename_mod"].values,
        "source":                    df["source"].values,
        "stem":                      df["stem"].values,
        "source_group":              df["source_group"].values,
        "source_group_merged":       df["source_group_merged"].values,
        "dsp_label":                 df["dsp_label"].values,
        "dsp_category":              df["dsp_category"].values,
        "dsp_family":                df["dsp_family"].values,
        "pitch_intensity":           df["pitch_intensity"].values,
        "tempo_intensity":           df["tempo_intensity"].values,
        "delta_mean":                np.mean(delta_matrix, axis=1),
        "delta_std":                 np.std(delta_matrix,  axis=1),
        "delta_median":              np.median(delta_matrix, axis=1),
        "delta_max":                 np.max(delta_matrix, axis=1),
        "delta_l2":                  np.linalg.norm(delta_matrix, axis=1),
        "delta_p90":                 np.percentile(delta_matrix, 90, axis=1),
        "delta_p95":                 np.percentile(delta_matrix, 95, axis=1),
        "active_dims_q75_global":    np.sum(delta_matrix > global_q75, axis=1),
        "stable_core_mean_delta":    s_core,
        "volatile_shell_mean_delta": s_vol,
        "stable_to_volatile_ratio":  np.divide(
            s_core, s_vol,
            out=np.zeros_like(s_core), where=s_vol != 0
        ),
    })

    _csv(df_export, Path(res_dir) / f"{model_name.lower()}_pairwise_delta_features.csv")
    pq = Path(data_dir) / f"{model_name.lower()}_pairwise_delta_features.parquet"
    pq.parent.mkdir(parents=True, exist_ok=True)
    df_export.to_parquet(pq, index=False)
    print(f"  Saved Parquet → {pq}")


# PLOT 1: Overall Shifts & Active Dimensions 
def plot_overall_shifts(
    df: pd.DataFrame, model_name: str, plt_dir: str, run: dict
) -> None:
    print(f"  [Plot 1] Overall Shifts & Active Dimensions — {_run_context(run)}")

    group_col    = run["group_col"]
    suffix       = run["suffix"]
    delta_matrix = np.stack(df["delta_vector"].values)
    categories   = sorted(df[group_col].dropna().unique())
    cat_labels   = df[group_col].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # A) Shift distribution
    total_shift = np.mean(delta_matrix, axis=1)
    for cat in categories:
        ax1.hist(
            total_shift[cat_labels == cat], bins=40, alpha=0.5,
            color=_cat_color(cat), label=_short(cat), density=True,
        )
    ax1.set_xlim(0, np.percentile(total_shift, 99) * 1.05)
    ax1.set_xlabel("Mean |Δ| across all dimensions")
    ax1.set_ylabel("Density")
    ax1.set_title("A) Overall Shift Distribution", fontweight="bold")
    ax1.legend(fontsize=9)

    # B) Active dimensions
    threshold = float(np.percentile(np.mean(delta_matrix, axis=0), 75))
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
            str(val), va="center", fontsize=9,
        )
    ax2.set_xlabel(f"# Dimensions with mean |Δ| > {threshold:.4f} (75th pct)")
    ax2.set_title("B) Active Dimensions per Category", fontweight="bold")

    fig.suptitle(
        _make_suptitle(model_name, "Shift Magnitudes & Active Dimensions", _run_context(run)),
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, Path(plt_dir) / f"{model_name.lower()}_overall_shifts{suffix}.pdf")


# PLOTS 2 & 3: Top-K Dimensions & Overlap 
def plot_topk_and_overlap(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str, run: dict
) -> None:
    print(f"  [Plot 2/3] Top-{TOP_K} Dimensions & Overlap — {_run_context(run)}")

    group_col    = run["group_col"]
    suffix       = run["suffix"]
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

    # Plot 2: bar charts
    n_cats = len(categories)
    fig, axes = plt.subplots(n_cats, 1, figsize=(16, 3.5 * n_cats))
    if n_cats == 1:
        axes = [axes]

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        cd = topk_df[topk_df["category"] == cat]
        ax.bar(
            np.arange(TOP_K), cd["mean_delta"].values,
            yerr=cd["std_delta"].values, capsize=2,
            color=_cat_color(cat), alpha=0.8, edgecolor="black", linewidth=0.3,
        )
        ax.set_xticks(np.arange(TOP_K))
        ax.set_xticklabels([str(d) for d in cd["dimension"].values], rotation=45, fontsize=8)
        ax.set_ylabel("Mean |Δ|")
        ax.set_title(f"{_short(cat)} — Top-{TOP_K} Most Affected Dimensions", fontweight="bold")

    fig.suptitle(
        _make_suptitle(model_name, f"Top-{TOP_K} Dimensions Most Affected per Category", _run_context(run)),
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, Path(plt_dir) / f"{model_name.lower()}_topk_dimensions{suffix}.pdf")

    # Plot 3: overlap heatmap
    overlap = np.zeros((n_cats, n_cats), dtype=int)
    for i, ci in enumerate(categories):
        for j, cj in enumerate(categories):
            overlap[i, j] = len(set(cat_top_dims[ci]) & set(cat_top_dims[cj]))

    fig_ov, ax_ov = plt.subplots(figsize=(max(6, n_cats * 1.2), max(5, n_cats)))
    labels = [_short(c) for c in categories]
    sns.heatmap(
        overlap, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax_ov,
        cbar_kws={"label": f"Shared Dimensions (out of Top {TOP_K})"},
    )
    ax_ov.set_title(
        _make_suptitle(model_name, f"Top-{TOP_K} Dimension Overlap Between Categories", _run_context(run)),
        fontweight="bold", pad=15,
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig_ov, Path(plt_dir) / f"{model_name.lower()}_topk_overlap{suffix}.pdf")


# PLOT 4: Signed Shift Heatmap 
def plot_signed_shift(
    df: pd.DataFrame, model_name: str, plt_dir: str, run: dict
) -> None:
    print(f"  [Plot 4] Signed Mean Shift — {_run_context(run)}")

    group_col  = run["group_col"]
    suffix     = run["suffix"]
    emb_ori    = np.stack(df["embedding_ori"].values)
    emb_mod    = np.stack(df["embedding_mod"].values)
    ndim       = emb_ori.shape[1]
    categories = sorted(df[group_col].dropna().unique())

    shift_matrix = np.zeros((len(categories), ndim))
    for i, cat in enumerate(categories):
        mask = df[group_col] == cat
        shift_matrix[i] = np.mean(emb_mod[mask] - emb_ori[mask], axis=0)

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
    im   = ax.imshow(
        shift_vis, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels([_short(c) for c in categories], fontsize=10)

    step = max(1, len(dim_lbls) // 40)
    ax.set_xticks(np.arange(0, len(dim_lbls), step))
    ax.set_xticklabels([dim_lbls[i] for i in range(0, len(dim_lbls), step)], rotation=45, fontsize=8)
    ax.set_xlabel("Dimension Index")
    ax.set_title(
        _make_suptitle(
            model_name,
            f"Signed Mean Shift per Dimension {subtitle}",
            _run_context(run),
        ),
        fontweight="bold", fontsize=12,
    )
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mean Signed Shift (mod − ori)")
    plt.tight_layout()
    _save(fig, Path(plt_dir) / f"{model_name.lower()}_signed_shift_heatmap{suffix}.pdf")


# PLOT 5: SMP vs AI Base 
def plot_smp_vs_ai(
    df: pd.DataFrame, model_name: str, res_dir: str, plt_dir: str, run: dict
) -> None:
    print(f"  [Plot 5] SMP vs AI Base — {_run_context(run)}")

    suffix     = run["suffix"]
    categories = sorted(df["broad_category"].unique())
    smp_cats   = [c for c in categories if "Human" in c]
    ai_cats    = [c for c in categories if "AI Generation" in c]

    if not smp_cats or not ai_cats:
        logger.warning("Required categories missing for SMP vs AI plot.")
        return

    smp_label, ai_label = smp_cats[0], ai_cats[0]
    delta_matrix = np.stack(df["delta_vector"].values)
    cat_labels   = df["broad_category"].values

    smp_mean   = np.mean(delta_matrix[cat_labels == smp_label], axis=0)
    ai_mean    = np.mean(delta_matrix[cat_labels == ai_label],  axis=0)
    pooled_std = np.sqrt(
        (np.std(delta_matrix[cat_labels == smp_label], axis=0) ** 2
         + np.std(delta_matrix[cat_labels == ai_label],  axis=0) ** 2) / 2 + 1e-10
    )
    cohens_d = (ai_mean - smp_mean) / pooled_std

    comparison_df = pd.DataFrame({
        "dimension":          np.arange(delta_matrix.shape[1]),
        "smp_mean_delta":     smp_mean,
        "ai_base_mean_delta": ai_mean,
        "difference":         ai_mean - smp_mean,
        "cohens_d":           cohens_d,
    }).sort_values("cohens_d", key=abs, ascending=False)
    _csv(comparison_df, Path(res_dir) / f"{model_name.lower()}_smp_vs_ai{suffix}.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    lim = max(smp_mean.max(), ai_mean.max()) * 1.05
    ax1.scatter(smp_mean, ai_mean, alpha=0.4, s=20, c="#607D8B", edgecolors="none")
    ax1.plot([0, lim], [0, lim], "--", color="red", linewidth=1.2, label="y=x (Perfect Alignment)")
    ax1.set_xlabel(f"Mean Shift: {_short(smp_label)}", fontsize=10)
    ax1.set_ylabel(f"Mean Shift: {_short(ai_label)}",  fontsize=10)
    ax1.set_title("A) Dimension-Level Shift Comparison", fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    n_sig = int(np.sum(np.abs(cohens_d) > 0.8))
    ax2.hist(cohens_d, bins=50, color="darkblue", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax2.text(
        0.95, 0.95, f"Large Effect (|d| > 0.8):\n{n_sig} dimensions",
        transform=ax2.transAxes, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax2.set_xlabel("Cohen's d (AI − SMP)", fontsize=10)
    ax2.set_ylabel("Number of Dimensions", fontsize=10)
    ax2.set_title("B) Effect Size Distribution (Separability)", fontweight="bold")

    fig.suptitle(
        _make_suptitle(model_name, "SMP vs AI Base: Dimension-Level Comparison", _run_context(run)),
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, Path(plt_dir) / f"{model_name.lower()}_smp_vs_ai{suffix}.pdf")


# PLOT 6: Stable Core vs Volatile Shell 
def plot_stable_core(
    df: pd.DataFrame, model_name: str, plt_dir: str, run: dict
) -> None:
    print(f"  [Plot 6] Stable Core vs Volatile Shell — {_run_context(run)}")

    suffix       = run["suffix"]
    delta_matrix = np.stack(df["delta_vector"].values)
    global_mean  = np.mean(delta_matrix, axis=0)
    ranking      = np.argsort(global_mean)
    n_core       = min(100, delta_matrix.shape[1] // 5)
    stable_dims  = ranking[:n_core]
    volatile_dims = ranking[-n_core:]

    categories = sorted(df["broad_category"].unique())
    cat_labels = df["broad_category"].values

    records = []
    for cat in categories:
        mask = cat_labels == cat
        records.append({
            "category":      cat,
            "Stable Core":   delta_matrix[mask][:, stable_dims].mean(axis=1).mean(),
            "Volatile Shell": delta_matrix[mask][:, volatile_dims].mean(axis=1).mean(),
        })

    core_df    = pd.DataFrame(records)
    fig, ax    = plt.subplots(figsize=(10, 6))
    x          = np.arange(len(categories))
    width      = 0.35
    bar_colors = [_cat_color(cat) for cat in categories]

    ax.bar(x - width / 2, core_df["Stable Core"],   width, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.bar(x + width / 2, core_df["Volatile Shell"], width, color=bar_colors, edgecolor="black", linewidth=0.8, hatch="///", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([_short(c) for c in categories], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean Absolute Shift (|Δ|)", fontsize=11)
    ax.set_title(
        _make_suptitle(model_name, "Stable Core vs Volatile Shell", _run_context(run)),
        fontweight="bold", fontsize=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(handles=[
        mpatches.Patch(facecolor="gray", edgecolor="black", label=f"Stable Core (Top {n_core} Dims)"),
        mpatches.Patch(facecolor="gray", edgecolor="black", hatch="///", alpha=0.7, label=f"Volatile Shell (Top {n_core} Dims)"),
    ], fontsize=10, loc="upper left")

    _save(fig, Path(plt_dir) / f"{model_name.lower()}_stable_core{suffix}.pdf")


# RUN DISPATCHER 
def _dispatch_run(
    df: pd.DataFrame, model_name: str,
    res_dir: str, plt_dir: str, run: dict,
) -> None:
    """Execute all plots requested in a run config dict."""
    plots = run.get("include_plots", _ALL_PLOTS)

    if "shifts"  in plots:
        plot_overall_shifts(df, model_name, plt_dir, run)
    if "topk"    in plots:
        plot_topk_and_overlap(df, model_name, res_dir, plt_dir, run)
    if "signed"  in plots:
        plot_signed_shift(df, model_name, plt_dir, run)
    if "core"    in plots:
        plot_stable_core(df, model_name, plt_dir, run)
    if "smp_ai"  in plots:
        plot_smp_vs_ai(df, model_name, res_dir, plt_dir, run)


# MAIN PIPELINE 
def _build_run_configs(df: pd.DataFrame) -> list[dict]:
    """
    Builds the full list of run configs from the loaded dataframe.
    All suffix strings are passed through _safe_suffix so they are
    guaranteed to be filesystem-safe on Windows and Unix alike.
    """
    runs = []

    # Run 1: broad_category (all plots) 
    runs.append({
        "group_col":     "broad_category",
        "suffix":        "",
        "context":       "All sources",
        "include_plots": _ALL_PLOTS,
    })

    # Run 2: DSP label cross-source 
    if df["dsp_label"].nunique() > 1:
        runs.append({
            "group_col":     "dsp_label",
            "suffix":        "_by_dsp",
            "context":       "All sources",
            "include_plots": _NO_CORE_SMP,
        })

    # Run 3: Source group (MGE-LDM merged) 
    if df["source_group_merged"].nunique() > 1:
        runs.append({
            "group_col":     "source_group_merged",
            "suffix":        "_by_source",
            "context":       "All sources",
            "include_plots": _NO_CORE_SMP,
        })

    # Run 4a: Per non-MGE-LDM source, grouped by dsp_label 
    for src in sorted(df["source_group_merged"].dropna().unique()):
        if src == "MGE-LDM":
            continue
        df_src = df[df["source_group_merged"] == src]
        if df_src["dsp_label"].nunique() < 2:
            continue
        runs.append({
            "group_col":     "dsp_label",
            "suffix":        f"_by_dsp_{_safe_suffix(src)}",
            "context":       src,
            "include_plots": _NO_CORE_SMP,
            "_df_filter":    ("source_group_merged", src),
        })

    # Run 4b: Per MGE-LDM stem, grouped by dsp_label (bass, drums, other)
    df_mgeldm = df[df["source_group_merged"] == "MGE-LDM"]
    if not df_mgeldm.empty:
        for stem in ["bass", "drums", "other"]:
            df_stem = df_mgeldm[df_mgeldm["stem"] == stem]
            if df_stem["dsp_label"].nunique() < 2:
                continue
            stem_display = stem.capitalize()
            runs.append({
                "group_col":     "dsp_label",
                "suffix":        f"_by_dsp_mgeldm_{stem}",
                "context":       f"MGE-LDM {stem_display}",
                "include_plots": _NO_CORE_SMP,
                "_df_filter":    [("source_group_merged", "MGE-LDM"), ("stem", stem)],
            })

    return runs


def process_model(parquet_path: str, pairs_path: str, model_name: str) -> None:
    print("=" * 60)
    print(f"ANALYZING PLAGIARISM SIGNATURES: {model_name}")
    print("=" * 60)

    df = load_pairs(parquet_path, pairs_path, model_name)
    if df.empty:
        return

    export_pairwise_delta_features(df, model_name, RES_SUBDIR)

    runs = _build_run_configs(df)
    print(f"\n{len(runs)} analysis runs scheduled.\n")

    for run in runs:
        # Apply optional dataframe filter(s) defined in the run config
        filter_spec = run.get("_df_filter")
        if filter_spec:
            # Support both single filter (col, val) and multiple filters [(col1, val1), (col2, val2)]
            if isinstance(filter_spec, list):
                df_run = df.copy()
                for col, val in filter_spec:
                    df_run = df_run[df_run[col] == val]
            else:
                df_run = df[df[filter_spec[0]] == filter_spec[1]].copy()
        else:
            df_run = df
        print(f"Run: suffix='{run['suffix']}' | n={len(df_run)}")
        _dispatch_run(df_run, model_name, RES_SUBDIR, PLT_SUBDIR, run)

    print(f"\nDone processing {model_name}.\n")


def main() -> None:
    for model_name, parquet in EMBEDDING_PATHS.items():
        parquet_path = Path(parquet)
        pairs_path   = Path(SMP_CSV)
        if parquet_path.exists() and pairs_path.exists():
            process_model(str(parquet_path), str(pairs_path), model_name)
        else:
            logger.warning(f"Data missing for {model_name}. Skipping.")


if __name__ == "__main__":
    main()