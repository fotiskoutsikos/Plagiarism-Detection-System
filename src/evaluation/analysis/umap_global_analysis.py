"""
Consistent Global UMAP Latent Space Visualization for Plagiarism Detection.

Design goal:
    Match the semantic logic of the local UMAP script:
    - keep only the positive plagiarism ecosystem
    - exclude Negative pairs
    - exclude cross-type SMP+DSP pairs (smp_*)
    - merge SMP_plag / SMP_plag_doubt / SMP_remake into one
      "Human Plagiarism (SMP)" class
    - include original-track anchors once
    - project all unique points into one shared 2D space

Runs for:
    - CLEWS
    - WEALY

Outputs:
    plots/umap/clews_umap_global_broad.pdf
    plots/umap/clews_umap_global_source.pdf
    plots/umap/wealy_umap_global_broad.pdf
    plots/umap/wealy_umap_global_source.pdf
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
    SOURCE_COLORS,
    PLOT_COLORS,
    CATEGORY_COLORS,
    PLOT_STYLE_PARAMS,
    PLOT_DPI,
)
from utils.categorization import (
    clean_mod_type,
    extract_dsp_and_source_features,
)
from utils.dataset_builder import validate_and_filter_embeddings

plt.rcParams.update(PLOT_STYLE_PARAMS)

# CONFIG
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
UMAP_METRIC       = "cosine"
UMAP_RANDOM_STATE = 42

# Visual hierarchy: base points are vivid, DSP faded
BASE_ALPHA     = 0.85
DSP_FADE_ALPHA = 0.80

# Point sizes
POINT_SIZE     = 14
ORI_POINT_SIZE = 30
HUMAN_SIZE     = 22

# Subsampling: cap the dominant groups for visual balance
# Groups smaller than this are kept in full
MAX_SAMPLES_PER_GROUP = 2000

# Original Track: subtle grey circle instead of black X
_ORI_COLOR = (0.6, 0.6, 0.6)   # medium grey

# Consistent with local UMAP visual semantics
_HUMAN_COLOR      = to_rgba(PLOT_COLORS.get("Original", "blue"))[:3]
_ORIGINAL_DSP_COLOR = to_rgba(PLOT_COLORS.get("Original", "blue"))[:3]

_SOURCE_GROUP_COLORS = {
    "Human Plagiarism (SMP)": _HUMAN_COLOR,
    "Original + DSP":         _ORIGINAL_DSP_COLOR,
    "MusicGen":               to_rgba(SOURCE_COLORS.get("MusicGen", "#E53935"))[:3],
    "AudioLDM2":              to_rgba(SOURCE_COLORS.get("AudioLDM2", "#4CAF50"))[:3],
    "MGE-LDM_bass":           to_rgba("#AB47BC")[:3],
    "MGE-LDM_drums":          to_rgba("#7B1FA2")[:3],
    "MGE-LDM_other":          to_rgba("#4A148C")[:3],
    "Original Track":         _ORI_COLOR,
}

_BROAD_COLORS = {
    "Original Track":            _ORI_COLOR,
    "Human Plagiarism (SMP)":    _HUMAN_COLOR,
    "Original + DSP":            _ORIGINAL_DSP_COLOR,
    "AI Generation (Base)":      to_rgba(CATEGORY_COLORS.get("3. AI Generation (Base)", "#E53935"))[:3],
    "AI + DSP":                  to_rgba(CATEGORY_COLORS.get("4. AI + DSP", "#EF9A9A"))[:3],
}


# SEMANTIC HELPER — SAME LOGIC AS LOCAL UMAP
def _get_umap_visual_meta(mod_type: str):
    """
    Consistent with the local UMAP script.

    Returns:
        source      : Human Plagiarism (SMP) | Original + DSP | MusicGen | AudioLDM2 |
                      MGE-LDM_bass | MGE-LDM_drums | MGE-LDM_other
        is_base     : True if no DSP applied
        is_human    : True only for merged SMP base label
        broad_group : Original + DSP | AI Generation (Base) | AI + DSP | Human Plagiarism (SMP)
    """
    mod_str = str(mod_type)

    # Explicit merged human class
    if mod_str == "Human Plagiarism (SMP)":
        return "Human Plagiarism (SMP)", True, True, "Human Plagiarism (SMP)"

    cleaned = clean_mod_type(mod_str)
    feats   = extract_dsp_and_source_features(cleaned)

    source = feats["source"]
    stem   = feats.get("stem")
    is_base = (feats["dsp_category"] == "Base Generation")

    # MGE-LDM stem-aware
    if source == "MGE-LDM" and stem:
        source = f"MGE-LDM_{stem}"

    # Broad group
    if source == "Original + DSP":
        broad_group = "Original + DSP"
    elif source in ["MusicGen", "AudioLDM2", "MGE-LDM_bass", "MGE-LDM_drums", "MGE-LDM_other"]:
        broad_group = "AI Generation (Base)" if is_base else "AI + DSP"
    else:
        broad_group = "Other"

    return source, is_base, False, broad_group


# DATA PREPARATION — GLOBAL BUT CONSISTENT WITH LOCAL SCRIPT
def load_and_prepare_global(distances_csv: str, embeddings_parquet: str) -> pd.DataFrame:
    """
    Build the exact same semantic universe as the local UMAP script, but globally.

    Keeps:
        - Human Plagiarism (SMP) base pairs
        - Original + DSP
        - AI bases
        - AI + DSP
        - Original anchors

    Excludes:
        - Negative pairs
        - cross-type SMP+DSP pairs (smp_*)
    """
    df = pd.read_csv(distances_csv, keep_default_na=False)
    print(f"  Loaded {len(df):,} rows from {Path(distances_csv).name}")

    # 1) Exclude negatives
    df = df[~df["final_mod_type"].str.startswith("Negative", na=False)].copy()

    # 2) Exclude cross-type SMP+DSP pairs (ori_base ↔ comp+DSP)
    df = df[~df["final_mod_type"].str.startswith("smp_", na=False)].copy()

    # 3) Merge SMP relation types into one label
    smp_mask = df["final_mod_type"].isin(["SMP_plag", "SMP_plag_doubt", "SMP_remake"])
    df.loc[smp_mask, "final_mod_type"] = "Human Plagiarism (SMP)"

    # 4) Same dedup logic as local script
    mask_human = df["final_mod_type"] == "Human Plagiarism (SMP)"

    df_human = df[mask_human].drop_duplicates(
        subset=["pair_id", "time", "filename_mod", "final_mod_type"]
    ).copy()

    df_ai = df[~mask_human].drop_duplicates(
        subset=["pair_id", "time", "final_mod_type"]
    ).copy()

    df = pd.concat([df_human, df_ai], ignore_index=True)

    print(f"  After filtering/dedup: {len(df):,} positive ecosystem rows")

    # 5) Load embeddings lookup
    df_emb = pd.read_parquet(embeddings_parquet)
    df_emb = df_emb.drop_duplicates(subset=["filename"])[["filename", "embedding"]]
    emb_lookup = dict(zip(df_emb["filename"], df_emb["embedding"]))

    # 6) Unique modified points
    df_mod = df[["filename_mod", "final_mod_type"]].drop_duplicates().copy()
    df_mod["embedding"] = df_mod["filename_mod"].map(emb_lookup)
    df_mod["point_type"] = "mod"
    df_mod = df_mod.rename(columns={"filename_mod": "filename"})
    df_mod = df_mod.dropna(subset=["embedding"]).reset_index(drop=True)

    # 7) Unique original anchors
    ori_files = df["filename_ori"].drop_duplicates()
    df_ori = pd.DataFrame({"filename": ori_files})
    df_ori["embedding"] = df_ori["filename"].map(emb_lookup)
    df_ori["point_type"] = "ori"
    df_ori["final_mod_type"] = "Original Track"
    df_ori = df_ori.dropna(subset=["embedding"]).reset_index(drop=True)

    # 8) Combine
    df_all = pd.concat([df_mod, df_ori], ignore_index=True)

    # 9) Add semantic visual metadata
    rows = []
    for _, row in df_all.iterrows():
        if row["point_type"] == "ori":
            rows.append({
                "filename":      row["filename"],
                "embedding":     row["embedding"],
                "point_type":    "ori",
                "broad_group":   "Original Track",
                "source_group":  "Original Track",
                "is_base":       True,
                "is_human":      False,
                "point_alpha":   BASE_ALPHA,
                "marker":        "x",
            })
        else:
            source, is_base, is_human, broad_group = _get_umap_visual_meta(row["final_mod_type"])
            rows.append({
                "filename":      row["filename"],
                "embedding":     row["embedding"],
                "point_type":    "mod",
                "broad_group":   broad_group,
                "source_group":  source,
                "is_base":       is_base,
                "is_human":      is_human,
                "point_alpha":   BASE_ALPHA if (is_base or is_human) else DSP_FADE_ALPHA,
                "marker":        "D" if is_human else "o",
            })

    df_all = pd.DataFrame(rows)

    # 10) Validate embeddings
    df_all, mode_dim = validate_and_filter_embeddings(
        df_all,
        emb_cols=["embedding"],
        clean=True,
    )

    _print_composition(df_all)
    print(f"  Embedding dimension: {mode_dim}")
    return df_all


def _print_composition(df: pd.DataFrame) -> None:
    print(f"\n  Embedding composition ({len(df):,} unique points):")
    print("    By broad group:")
    for cat, count in df["broad_group"].value_counts().items():
        print(f"      {cat:<28}: {count:>6,}")
    print("    By source group:")
    for cat, count in df["source_group"].value_counts().items():
        print(f"      {cat:<28}: {count:>6,}")
    print()


# UMAP
def fit_global_umap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit one UMAP on all unique points.
    """
    X = np.stack(list(df["embedding"].values))
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    print(f"  Fitting UMAP on {len(X):,} points ...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        low_memory=True,
    )
    X_2d = np.asarray(reducer.fit_transform(X))

    df = df.copy()
    df["umap_x"] = X_2d[:, 0]
    df["umap_y"] = X_2d[:, 1]
    return df


# PLOTTING
def _scatter_subset(ax, df_sub: pd.DataFrame, color_rgb, marker: str) -> None:
    """Scatter a group with consistent styling. Plots DSP first, then base on top."""
    if df_sub.empty:
        return

    if marker == "o" and "point_alpha" in df_sub.columns:
        # Split into base (vivid, on top) and DSP (faded, behind)
        df_base = df_sub[df_sub["is_base"] == True]
        df_dsp  = df_sub[df_sub["is_base"] == False]

        # DSP first (behind)
        if not df_dsp.empty:
            ax.scatter(
                df_dsp["umap_x"], df_dsp["umap_y"],
                marker="o", c=[color_rgb],
                s=POINT_SIZE * 0.7,
                edgecolors="none",
                alpha=DSP_FADE_ALPHA,
                zorder=2,
            )

        # Base on top
        if not df_base.empty:
            ax.scatter(
                df_base["umap_x"], df_base["umap_y"],
                marker="o", c=[color_rgb],
                s=POINT_SIZE,
                edgecolors="none",
                alpha=BASE_ALPHA,
                zorder=4,
            )
    elif marker == "D":
        ax.scatter(
            df_sub["umap_x"], df_sub["umap_y"],
            marker="D", c=[color_rgb],
            s=HUMAN_SIZE,
            edgecolors="black",
            linewidths=0.6,
            alpha=BASE_ALPHA,
            zorder=5,
        )
    elif marker == "ori":
        # Original Track: subtle filled circle, NOT black X
        ax.scatter(
            df_sub["umap_x"], df_sub["umap_y"],
            marker="o", c=[color_rgb],
            s=ORI_POINT_SIZE,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.70,
            zorder=6,
        )
    else:
        ax.scatter(
            df_sub["umap_x"], df_sub["umap_y"],
            marker=marker, c=[color_rgb],
            s=POINT_SIZE,
            edgecolors="none",
            alpha=BASE_ALPHA,
            zorder=3,
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

def _subsample_for_visual_balance(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Cap each group at MAX_SAMPLES_PER_GROUP for visual balance.
    Smaller groups are kept in full. Original Track is never subsampled.
    """
    parts = []
    for group_name, group_df in df.groupby(group_col):
        if group_name == "Original Track":
            parts.append(group_df)
        elif len(group_df) > MAX_SAMPLES_PER_GROUP:
            parts.append(group_df.sample(n=MAX_SAMPLES_PER_GROUP, random_state=42))
        else:
            parts.append(group_df)

    result = pd.concat(parts, ignore_index=True)
    print(f"  Subsampled for visual balance: {len(df):,} → {len(result):,} points")
    return result


def plot_by_broad_group(df_2d: pd.DataFrame, output_path: str, model_name: str) -> None:
    """
    Global UMAP colored by broad plagiarism ecosystem groups.
    """
    order = [
        "Original Track",
        "Human Plagiarism (SMP)",
        "Original + DSP",
        "AI Generation (Base)",
        "AI + DSP",
    ]

    fig, ax = plt.subplots(figsize=(12, 9))
    legend_elems = []

    for cat in order:
        sub = df_2d[df_2d["broad_group"] == cat]
        if sub.empty:
            continue

        color = _BROAD_COLORS.get(cat, (0.7, 0.7, 0.7))
        marker = "x" if cat == "Original Track" else ("D" if cat == "Human Plagiarism (SMP)" else "o")

        _scatter_subset(ax, sub, color, marker)

        if marker == "x":
            legend_elems.append(Line2D(
                [0], [0], marker="x", color="black", linestyle="None",
                markersize=10, markeredgewidth=1.8,
                label=f"{cat} ({len(sub):,})"
            ))
        elif marker == "D":
            legend_elems.append(Line2D(
                [0], [0], marker="D", color="w", linestyle="None",
                markerfacecolor=color, markeredgecolor="black", markersize=9,
                label=f"{cat} ({len(sub):,})"
            ))
        else:
            legend_elems.append(Line2D(
                [0], [0], marker="o", color="w", linestyle="None",
                markerfacecolor=color, markeredgecolor="none", markersize=9,
                label=f"{cat} ({len(sub):,})"
            ))

    _apply_axis_style(ax, f"{model_name} — Global Latent Space (Consistent Broad View)")
    fig.suptitle("Global UMAP Projection of the Positive Plagiarism Ecosystem",
                 fontsize=16, fontweight="bold", y=0.97)

    fig.legend(
        handles=legend_elems,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(legend_elems), 3),
        frameon=False,
        fontsize=10,
    )

    plt.subplots_adjust(bottom=0.14)
    _save_fig(fig, output_path)


def plot_by_source_group(df_2d: pd.DataFrame, output_path: str, model_name: str) -> None:
    """Global UMAP colored by source group."""
    df_2d = _subsample_for_visual_balance(df_2d, "source_group")

    # Draw order: large/faded groups first, small/vivid last
    order = [
        "MGE-LDM_bass", "MGE-LDM_drums", "MGE-LDM_other",
        "AudioLDM2", "MusicGen",
        "Original + DSP",
        "Human Plagiarism (SMP)",
        "Original Track",
    ]

    fig, ax = plt.subplots(figsize=(13, 9))
    legend_elems = []

    for sg in order:
        sub = df_2d[df_2d["source_group"] == sg]
        if sub.empty:
            continue

        color = _SOURCE_GROUP_COLORS.get(sg, (0.7, 0.7, 0.7))

        if sg == "Original Track":
            marker = "ori"
        elif sg == "Human Plagiarism (SMP)":
            marker = "D"
        else:
            marker = "o"

        _scatter_subset(ax, sub, color, marker)

        # Legend
        if sg == "Original Track":
            legend_elems.append(Line2D(
                [0], [0], marker="o", color="w", linestyle="None",
                markerfacecolor=color, markeredgecolor="white",
                markersize=10, label=f"{sg} ({len(sub):,})"
            ))
        elif sg == "Human Plagiarism (SMP)":
            legend_elems.append(Line2D(
                [0], [0], marker="D", color="w", linestyle="None",
                markerfacecolor=color, markeredgecolor="black",
                markersize=9, label=f"{sg} ({len(sub):,})"
            ))
        else:
            legend_elems.append(Line2D(
                [0], [0], marker="o", color="w", linestyle="None",
                markerfacecolor=color, markeredgecolor="none",
                markersize=9, label=f"{sg} ({len(sub):,})"
            ))

    _apply_axis_style(ax, f"{model_name} — Global Latent Space (Source View)")
    fig.suptitle("Global UMAP Projection of the Positive Plagiarism Ecosystem",
                 fontsize=16, fontweight="bold", y=0.97)

    fig.legend(
        handles=legend_elems, loc="lower center",
        bbox_to_anchor=(0.5, 0.01), ncol=4,
        frameon=False, fontsize=9.5,
    )

    plt.subplots_adjust(bottom=0.14)
    _save_fig(fig, output_path)

# MAIN
if __name__ == "__main__":
    models = {
        "CLEWS": {
            "distances":  "results/distances/clews_distances.csv",
            "embeddings": "data/clews_embeddings.parquet",
            "out_broad":  "plots/umap/clews_umap_global_broad.pdf",
            "out_source": "plots/umap/clews_umap_global_source.pdf",
        },
        "WEALY": {
            "distances":  "results/distances/wealy_distances.csv",
            "embeddings": "data/wealy_embeddings.parquet",
            "out_broad":  "plots/umap/wealy_umap_global_broad.pdf",
            "out_source": "plots/umap/wealy_umap_global_source.pdf",
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

        df_raw = load_and_prepare_global(
            cfg["distances"],
            cfg["embeddings"],
        )

        df_2d = fit_global_umap(df_raw)

        print(f"\n  [1/2] Plotting by broad group ...")
        plot_by_broad_group(df_2d, cfg["out_broad"], model_name)

        print(f"  [2/2] Plotting by source group ...")
        plot_by_source_group(df_2d, cfg["out_source"], model_name)

    print(f"\n{'=' * 60}")
    print("Consistent global UMAP analysis complete.")
    print(f"{'=' * 60}")