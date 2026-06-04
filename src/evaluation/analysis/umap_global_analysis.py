"""
Global UMAP Latent Space Visualization for Plagiarism Detection:
- Projects ALL modification embeddings into a shared 2D space via UMAP
- Two color schemes per model:
    (1) broad_category  — Original Track / Human Plagiarism / AI Generation / AI+DSP / Negative
    (2) source_group    — MusicGen / AudioLDM2 / MGE-LDM stems / Cover / Negative
- Includes original track embeddings as a reference anchor class
- Stratified subsampling per group for visual balance and UMAP speed
- Runs for both CLEWS and WEALY models

Outputs saved to plots/umap/
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import umap

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# Resolve repository root 
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent

sys.path.insert(0, str(repo_root / "src"))

from utils.constants import (
    SOURCE_COLORS, CATEGORY_COLORS,
    PLOT_STYLE_PARAMS, PLOT_DPI,
)
from utils.categorization import (
    clean_mod_type, get_broad_category,
    extract_features_with_stem, _get_source_group,
)
from utils.dataset_builder import validate_and_filter_embeddings

plt.rcParams.update(PLOT_STYLE_PARAMS)


# CONFIG 
MAX_SAMPLES_PER_GROUP = 800   # stratified cap; keeps UMAP tractable and plot balanced

UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
UMAP_METRIC       = "cosine"
UMAP_RANDOM_STATE = 42

POINT_ALPHA    = 0.60
POINT_SIZE     = 18
ORI_POINT_SIZE = 22   # original anchors slightly larger than mod points

# COLOR MAPS 
_ORI_ANCHOR_COLOR = SOURCE_COLORS.get("Original", "#1565C0")

_BROAD_CATEGORY_COLORS = {
    **CATEGORY_COLORS,                          # covers AI Generation, AI+DSP, Human Plagiarism, …
    "Original Track": _ORI_ANCHOR_COLOR,
    "Negative":       "#9E9E9E",
}

_SOURCE_GROUP_COLORS = {
    **SOURCE_COLORS,                            # MusicGen, AudioLDM2, Cover, Original + DSP
    "MGE-LDM_bass":   "#AB47BC",
    "MGE-LDM_drums":  "#7B1FA2",
    "MGE-LDM_other":  "#4A148C",
    "Original Track": _ORI_ANCHOR_COLOR,
    "Negative":       "#9E9E9E",
}


# DATA LOADING 
def load_and_prepare_global(distances_csv: str, embeddings_parquet: str) -> pd.DataFrame:
    """
    Load distance CSV and embeddings parquet, enrich with broad_category /
    source_group, deduplicate on filename, and attach embedding vectors.

    Returns a DataFrame with columns:
        filename, embedding, broad_category, source_group, point_type
    where point_type ∈ {'mod', 'ori'}.
    """
    # 1. Load distances & merge SMP subtypes 
    df = pd.read_csv(distances_csv, keep_default_na=False)
    print(f"  Loaded {len(df):,} rows from {Path(distances_csv).name}")

    smp_mask = df["final_mod_type"].isin(["SMP_plag", "SMP_plag_doubt", "SMP_remake"])
    df.loc[smp_mask, "final_mod_type"] = "SMP_plag"

    # 2. Enrich with categorisation labels 
    df["clean_mod_type"] = df["final_mod_type"].apply(clean_mod_type)
    df["broad_category"] = df["clean_mod_type"].apply(get_broad_category)

    feat_cols = ["source", "stem", "pitch_intensity", "tempo_intensity",
                 "is_extreme", "dsp_category"]
    df[feat_cols] = df["clean_mod_type"].apply(extract_features_with_stem)
    df["source_group"] = df.apply(_get_source_group, axis=1)

    # Flatten all negatives into a single source_group label
    neg_mask = df["broad_category"] == "Negative"
    df.loc[neg_mask, "source_group"] = "Negative"

    # 3. Load embeddings lookup 
    df_emb = pd.read_parquet(embeddings_parquet)
    df_emb = df_emb.drop_duplicates(subset=["filename"])[["filename", "embedding"]]
    emb_lookup: dict = dict(zip(df_emb["filename"], df_emb["embedding"]))

    # 4. MOD embeddings (one row per unique file × category) 
    df_mod = (
        df[["filename_mod", "broad_category", "source_group"]]
        .drop_duplicates(subset=["filename_mod", "broad_category"])
        .copy()
    )
    df_mod["embedding"]  = df_mod["filename_mod"].map(emb_lookup)
    df_mod["point_type"] = "mod"
    df_mod = df_mod.rename(columns={"filename_mod": "filename"})
    df_mod = df_mod.dropna(subset=["embedding"])

    # 5. ORI embeddings — unique originals from positive pairs only 
    pos_mask = ~df["final_mod_type"].str.startswith("Negative", na=False)
    ori_files = df.loc[pos_mask, "filename_ori"].drop_duplicates()

    df_ori = pd.DataFrame({"filename": ori_files})
    df_ori["embedding"]       = df_ori["filename"].map(emb_lookup)
    df_ori["broad_category"]  = "Original Track"
    df_ori["source_group"]    = "Original Track"
    df_ori["point_type"]      = "ori"
    df_ori = df_ori.dropna(subset=["embedding"])

    # 6. Combine & validate embeddings 
    keep_cols = ["filename", "embedding", "broad_category", "source_group", "point_type"]
    df_all = pd.concat([df_mod[keep_cols], df_ori[keep_cols]], ignore_index=True)

    df_all, mode_dim = validate_and_filter_embeddings(
        df_all, emb_cols=["embedding"], clean=True
    )

    _print_composition(df_all)
    print(f"  Embedding dimension : {mode_dim}")
    return df_all


def _print_composition(df: pd.DataFrame) -> None:
    """Print per-group counts for transparency."""
    total = len(df)
    print(f"\n  Embedding composition ({total:,} unique points):")
    for cat, count in df["broad_category"].value_counts().items():
        print(f"    {cat:<35}: {count:>6,}")
    print()


# SUBSAMPLING 
def subsample_balanced(
    df: pd.DataFrame,
    group_col: str,
    max_per_group: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified subsampling: cap each group at max_per_group points.
    Groups smaller than the cap are kept in full.
    """
    parts = []
    for _, group_df in df.groupby(group_col):
        if len(group_df) > max_per_group:
            group_df = group_df.sample(n=max_per_group, random_state=random_state)
        parts.append(group_df)
    result = pd.concat(parts, ignore_index=True)
    print(f"  After subsampling ({max_per_group}/group): {len(result):,} points")
    return result


# UMAP FITTING 
def fit_global_umap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a single UMAP on all embeddings and attach umap_x / umap_y.
    L2-normalises embeddings before fitting (cosine metric equivalent).
    """
    X = np.stack(df["embedding"].values)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    print(f"  Fitting UMAP on {len(X):,} points …")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        low_memory=True,
    )
    X_2d = reducer.fit_transform(X)

    df = df.copy()
    df["umap_x"] = X_2d[:, 0]
    df["umap_y"] = X_2d[:, 1]
    return df


# PLOTTING HELPERS 
def _scatter_group(ax, df_group, color, is_ori: bool) -> None:
    """Scatter a single category group with consistent styling."""
    rgba = to_rgba(color)
    rgb  = rgba[:3]

    ax.scatter(
        df_group["umap_x"], df_group["umap_y"],
        marker="o", color=rgb,
        s=ORI_POINT_SIZE if is_ori else POINT_SIZE,
        edgecolors="none",
        alpha=POINT_ALPHA,
        zorder=5 if is_ori else 3,
    )


def _apply_axis_style(ax, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)


def _save_fig(fig, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.close(fig)


# PLOT 1: BROAD CATEGORY 
def plot_by_broad_category(
    df_2d: pd.DataFrame,
    output_path: str,
    model_name: str,
) -> None:
    """
    Global UMAP coloured by broad modification category:
    Original Track / Human Plagiarism / AI Generation / AI+DSP / Negative.
    """
    categories = df_2d["broad_category"].unique()
    # Sort for consistent legend order; put Original Track first, Negative last
    priority = {"Original Track": 0, "Negative": 99}
    categories = sorted(categories, key=lambda c: (priority.get(c, 1), c))

    fig, ax = plt.subplots(figsize=(12, 9))

    legend_elems = []
    for cat in categories:
        sub = df_2d[df_2d["broad_category"] == cat]
        color = _BROAD_CATEGORY_COLORS.get(cat, "#BDBDBD")
        is_ori = (cat == "Original Track")
        _scatter_group(ax, sub, color, is_ori)

        # Legend entry
        rgba = to_rgba(color)
        legend_elems.append(Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=rgba,
            markeredgecolor="none",
            markersize=9,
            linestyle="None",
            label=f"{cat}  ({len(sub):,})",
        ))

    _apply_axis_style(
        ax, f"{model_name} — Global Latent Space (by Broad Category)"
    )

    fig.legend(
        handles=legend_elems,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(legend_elems), 3),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Global UMAP Projection of Embedding Space",
        fontsize=16, fontweight="bold", y=0.97,
    )

    plt.subplots_adjust(bottom=0.14)
    _save_fig(fig, output_path)


# PLOT 2: SOURCE GROUP 
def plot_by_source_group(
    df_2d: pd.DataFrame,
    output_path: str,
    model_name: str,
) -> None:
    """
    Global UMAP coloured by granular source group:
    MusicGen / AudioLDM2 / MGE-LDM_{bass,drums,other} / Cover / Negative / …
    """
    source_groups = df_2d["source_group"].unique()
    priority = {"Original Track": 0, "Negative": 99}
    source_groups = sorted(source_groups, key=lambda s: (priority.get(s, 1), s))

    fig, ax = plt.subplots(figsize=(13, 9))

    legend_elems = []
    for sg in source_groups:
        sub = df_2d[df_2d["source_group"] == sg]
        color = _SOURCE_GROUP_COLORS.get(sg, "#BDBDBD")
        is_ori = (sg == "Original Track")
        _scatter_group(ax, sub, color, is_ori)

        rgba = to_rgba(color)
        legend_elems.append(Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=rgba,
            markeredgecolor="none",
            markersize=9,
            linestyle="None",
            label=f"{sg}  ({len(sub):,})",
        ))

    _apply_axis_style(
        ax, f"{model_name} — Global Latent Space (by Source Group)"
    )

    n_cols = min(len(legend_elems), 4)
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=n_cols,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        "Global UMAP Projection of Embedding Space",
        fontsize=16, fontweight="bold", y=0.97,
    )

    plt.subplots_adjust(bottom=0.14)
    _save_fig(fig, output_path)


# MAIN 
if __name__ == "__main__":
    models = {
        "CLEWS": {
            "distances":   "results/distances/clews_distances.csv",
            "embeddings":  "data/clews_embeddings.parquet",
            "out_broad":   "plots/umap/clews_umap_global_broad_category.pdf",
            "out_source":  "plots/umap/clews_umap_global_source_group.pdf",
        },
        "WEALY": {
            "distances":   "results/distances/wealy_distances.csv",
            "embeddings":  "data/wealy_embeddings.parquet",
            "out_broad":   "plots/umap/wealy_umap_global_broad_category.pdf",
            "out_source":  "plots/umap/wealy_umap_global_source_group.pdf",
        },
    }

    for model_name, cfg in models.items():
        print(f"\n{'=' * 60}")
        print(f"=== {model_name} ===")
        print(f"{'=' * 60}")

        if not os.path.exists(cfg["distances"]):
            print(f"  [skip] {cfg['distances']} not found.")
            continue
        if not os.path.exists(cfg["embeddings"]):
            print(f"  [skip] {cfg['embeddings']} not found.")
            continue

        # Load & prepare
        df_raw = load_and_prepare_global(cfg["distances"], cfg["embeddings"])

        # Fit UMAP on all points — no subsampling
        df_2d = fit_global_umap(df_raw)

        # Plot 1 — broad category
        print(f"\n  [1/2] Plotting by broad category …")
        plot_by_broad_category(df_2d, cfg["out_broad"], model_name)

        # Plot 2 — source group
        print(f"  [2/2] Plotting by source group …")
        plot_by_source_group(df_2d, cfg["out_source"], model_name)

    print(f"\n{'=' * 60}")
    print("Global UMAP analysis complete.")
    print(f"{'=' * 60}")
