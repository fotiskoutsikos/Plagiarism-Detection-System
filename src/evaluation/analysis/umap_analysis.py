import os
import sys
import ast
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import umap
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# Resolve repository root and import centralized utilities
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent

sys.path.insert(0, str(repo_root / "src"))
from utils.constants import PLOT_COLORS
from utils.categorization import extract_dsp_and_source_features, clean_mod_type

plt.rcParams.update({
    "figure.dpi": 200,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.facecolor": "white",
    "axes.edgecolor": "#d0d0d0",
    "axes.grid": True,
    "grid.color": "#f0f0f0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
})

# CONFIG
BASE_ALPHA      = 1.00   # full opacity for base (model / Human SMP)
DSP_FADE_ALPHA  = 0.35   # faded opacity for DSP variants


# DATA LOADING
def load_and_clean_data(distances_csv, embeddings_parquet):
    df = pd.read_csv(distances_csv)
    df = df[~df['final_mod_type'].str.startswith('Negative')].copy()

    # Συγχώνευση όλων των SMP κατηγοριών
    smp_conditions = df['final_mod_type'].isin(['SMP_plag', 'SMP_plag_doubt', 'SMP_remake'])
    df.loc[smp_conditions, 'final_mod_type'] = 'Human Plagiarism (SMP)'

    mask_human = df['final_mod_type'] == 'Human Plagiarism (SMP)'
    df_human = df[mask_human].copy()
    df_ai = df[~mask_human].copy()

    df_ai = df_ai.drop_duplicates(subset=['pair_id', 'time', 'final_mod_type']).copy()
    df_human = df_human.drop_duplicates(subset=['pair_id', 'time', 'filename_mod', 'final_mod_type']).copy()

    df = pd.concat([df_human, df_ai], ignore_index=True)

    df_emb = pd.read_parquet(embeddings_parquet)
    df_emb = df_emb.drop_duplicates(subset=['filename']).copy()

    df = df.merge(df_emb[['filename', 'embedding']], left_on='filename_ori', right_on='filename', how='inner')
    df = df.rename(columns={'embedding': 'embedding_ori'}).drop(columns=['filename'])

    df = df.merge(df_emb[['filename', 'embedding']], left_on='filename_mod', right_on='filename', how='inner')
    df = df.rename(columns={'embedding': 'embedding_mod'}).drop(columns=['filename'])

    df = df.reset_index(drop=True)

    def clean_embedding(emb):
        if isinstance(emb, str):
            try:
                emb = ast.literal_eval(emb)
            except Exception:
                return np.array([], dtype=np.float32)
        def extract_numbers(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                flat = []
                for x in item:
                    flat.extend(extract_numbers(x))
                return flat
            elif item is not None and not pd.isna(item):
                return [float(item)]
            return []
        try:
            return np.array(extract_numbers(emb), dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

    print("Cleaning and validating embeddings...")
    df['embedding_ori'] = df['embedding_ori'].apply(clean_embedding)
    df['embedding_mod'] = df['embedding_mod'].apply(clean_embedding)

    lens_ori = np.array([len(x) for x in df['embedding_ori']])
    lens_mod = np.array([len(x) for x in df['embedding_mod']])

    values, counts = np.unique(lens_ori, return_counts=True)
    mode_dim = values[np.argmax(counts)]

    valid_mask = (lens_ori == mode_dim) & (lens_mod == mode_dim)
    df = df.iloc[valid_mask].reset_index(drop=True)
    print(f"Kept {len(df)} completely valid modification pairs (Dimension: {mode_dim}).")

    return df


# CATEGORIZATION HELPERS
def _categorize_mod(mod_type: str):
    """
    Returns (source, is_base, is_human) for a given mod_type.
      - source : 'Original' | 'MusicGen' | 'AudioLDM2' | 'MGE-LDM'
      - is_base: True if no DSP is applied (clean model/original)
      - is_human: True if it's Human Plagiarism (SMP)
    """
    mod_str  = str(mod_type)
    is_human = ('Human Plagiarism' in mod_str) or mod_str.lower().startswith('smp_')
    cleaned  = clean_mod_type(mod_str)
    feats    = extract_dsp_and_source_features(cleaned)
    return feats['source'], (feats['dsp_category'] == 'Base Generation'), is_human


def _build_color_alpha_maps(mod_types):
    """
    Builds color and alpha maps for each mod_type:
      - Color: PLOT_COLORS[source]  (Original=blue, MusicGen=red, AudioLDM2=green, MGE-LDM=purple)
      - Alpha:
          * Human Plagiarism (SMP)  -> BASE_ALPHA  (full blue - "real" plagiarism)
          * Original + DSP          -> DSP_FADE_ALPHA (faded blue)
          * AI base (e.g. musicgen_none) -> BASE_ALPHA
          * AI + DSP                -> DSP_FADE_ALPHA
    """
    color_map = {}
    alpha_map = {}

    for mod_type in mod_types:
        source, is_base, is_human = _categorize_mod(mod_type)

        # Color from PLOT_COLORS based on source
        if source in PLOT_COLORS:
            color = to_rgba(PLOT_COLORS[source])
        else:
            print(f"[warn] Unmapped source for mod_type='{mod_type}' (source={source})")
            color = to_rgba('lightgray')

        # Alpha: full for base/human, else faded
        if is_human or is_base:
            alpha = BASE_ALPHA
        else:
            alpha = DSP_FADE_ALPHA

        color_map[mod_type] = color
        alpha_map[mod_type] = alpha

    return color_map, alpha_map


def plot_trajectories_grid(df_all, output_path, title):
    df_all = df_all.copy()

    # Precompute categorization columns so we don't have to do regex inside the loop
    cat_results        = df_all['final_mod_type'].apply(_categorize_mod)
    df_all['source']   = cat_results.apply(lambda x: x[0])
    df_all['is_base']  = cat_results.apply(lambda x: x[1])
    df_all['is_human'] = cat_results.apply(lambda x: x[2])

    df_originals = df_all[df_all['filename_ori'].str.contains('_ori_')]
    unique_segments = df_originals[['pair_id', 'time']].drop_duplicates()
    sampled_segments = unique_segments.sample(n=9).reset_index(drop=True)

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = axes.flatten()

    mod_types = sorted(df_all['final_mod_type'].unique())
    color_map, alpha_map = _build_color_alpha_maps(mod_types)

    # Plot grid 
    for i, (_, seg) in enumerate(sampled_segments.iterrows()):
        ax = axes[i]
        pair_id = seg['pair_id']
        time_val = seg['time']
        pair_data = df_all[(df_all['pair_id'] == pair_id) & (df_all['time'] == time_val)].copy()

        if pair_data.empty:
            continue

        # UMAP transform
        X_ori_local = np.stack(pair_data['embedding_ori'].values)
        X_mod_local = np.stack(pair_data['embedding_mod'].values)
        X_local = np.vstack([X_ori_local[0:1], X_mod_local])
        X_local = X_local / (np.linalg.norm(X_local, axis=1, keepdims=True) + 1e-8)

        reducer = umap.UMAP(n_neighbors=4, min_dist=0.8, metric='cosine', random_state=40)
        X_2d = reducer.fit_transform(X_local)

        ori_x, ori_y = X_2d[0, 0], X_2d[0, 1]
        pair_data['umap_mod_x'] = X_2d[1:, 0]
        pair_data['umap_mod_y'] = X_2d[1:, 1]

        # Original (X marker)
        ax.scatter(ori_x, ori_y, marker="x", c="black", s=120, zorder=10, linewidths=2.5)

        # Find AI base coordinates (DSP variants connect to their AI base) 
        ai_base_coords = {}
        for ai_source in ['MusicGen', 'AudioLDM2', 'MGE-LDM']:
            base_rows = pair_data[(pair_data['source'] == ai_source) & (pair_data['is_base'])]
            if not base_rows.empty:
                r = base_rows.iloc[0]
                ai_base_coords[ai_source] = (r['umap_mod_x'], r['umap_mod_y'])

        # Plot lines & markers
        for _, row in pair_data.iterrows():
            mod      = row['final_mod_type']
            mx, my   = row['umap_mod_x'], row['umap_mod_y']
            color    = color_map[mod]
            alpha    = alpha_map[mod]
            source   = row['source']
            is_base  = row['is_base']
            is_human = row['is_human']

            # Determine starting point of edge
            if (source in ai_base_coords) and (not is_base):
                # AI DSP variants -> start from their AI base
                start_x, start_y = ai_base_coords[source]
            else:
                # All other variants (Human, Original+DSP, AI base) start from the Original X
                start_x, start_y = ori_x, ori_y

            # Visual styling per category
            if is_human:
                # Real human plagiarism -> solid blue (Diamond)
                marker_shape, marker_size = 'D', 95
                edge_width                = 1.2
                z_marker, z_line          = 6, 4
                line_style                = '-'
                line_alpha                = 0.7 * alpha
            elif is_base:
                # AI base (musicgen_none, audioldm2_none, mgeldm_none) -> solid circle
                marker_shape, marker_size = 'o', 100
                edge_width                = 1.2
                z_marker, z_line          = 6, 4
                line_style                = '-'
                line_alpha                = 0.7 * alpha
            else:
                # DSP variant (Original+DSP or AI+DSP) -> faded, smaller circle
                marker_shape, marker_size = 'o', 60
                edge_width                = 0.6
                z_marker, z_line          = 4, 2
                # Distinct linestyle for the Original+DSP to distinguish it from Human
                line_style                = '--' if source == 'Original' else '-'
                line_alpha                = 0.45 * alpha

            line_color_rgb = (color[0], color[1], color[2])
            ax.plot([start_x, mx], [start_y, my],
                    color=line_color_rgb, alpha=line_alpha,
                    linewidth=1.8, linestyle=line_style, zorder=z_line)

            ax.scatter(mx, my, marker=marker_shape, color=line_color_rgb, alpha=alpha,
                       s=marker_size, edgecolors="black", linewidths=edge_width,
                       zorder=z_marker)

        ax.set_title(f"Pair {pair_id} | Segment {time_val}s", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.95)

    # LEGEND - matches color scheme
    legend_elems = []

    # 1) Original Track
    legend_elems.append(Line2D([0], [0], marker="x", color='w',
                               markeredgecolor='black', markersize=12, markeredgewidth=2.5,
                               label="Original Track", linestyle='None'))

    # 2) Original family (blue): Human Plagiarism (solid) + Original+DSP (faded)
    orig_color = to_rgba(PLOT_COLORS['Original'])
    legend_elems.append(Line2D([0], [0], marker="D", color='w',
                               markerfacecolor=orig_color, markeredgecolor='black',
                               markersize=11, label="Human Plagiarism (SMP)",
                               linestyle='None'))
    faded_orig = (orig_color[0], orig_color[1], orig_color[2], DSP_FADE_ALPHA)
    legend_elems.append(Line2D([0], [0], marker="o", color='w',
                               markerfacecolor=faded_orig, markeredgecolor='black',
                               markersize=10, label="Original + DSP",
                               linestyle='None'))

    # 3) AI families: base (solid) + DSP variants (faded)
    for base_model in ['MusicGen', 'AudioLDM2', 'MGE-LDM']:
        if base_model not in PLOT_COLORS:
            continue
        c = to_rgba(PLOT_COLORS[base_model])
        legend_elems.append(Line2D([0], [0], marker="o", color='w',
                                   markerfacecolor=c, markeredgecolor='black',
                                   markersize=11, label=f"{base_model}",
                                   linestyle='None'))
        faded = (c[0], c[1], c[2], DSP_FADE_ALPHA)
        legend_elems.append(Line2D([0], [0], marker="o", color='w',
                                   markerfacecolor=faded, markeredgecolor='black',
                                   markersize=10, label=f"{base_model} + DSP",
                                   linestyle='None'))

    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False, fontsize=10)

    plt.subplots_adjust(bottom=0.13, hspace=0.2, wspace=0.15)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Success! Plot saved at: {output_path}")
    plt.close()


if __name__ == '__main__':
    CLEWS_DISTANCES = 'results/distances/clews_distances.csv'
    CLEWS_EMBEDDINGS = 'data/clews_embeddings.parquet'

    print("=== Analyzing CLEWS ===")
    if os.path.exists(CLEWS_DISTANCES) and os.path.exists(CLEWS_EMBEDDINGS):
        df_clews = load_and_clean_data(CLEWS_DISTANCES, CLEWS_EMBEDDINGS)
        plot_trajectories_grid(df_clews,
                               output_path='plots/umap/clews_umap_plot.png',
                               title="CLEWS Latent Space Topology (Local Ecosystem per Track)")