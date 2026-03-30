import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import umap
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

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

def load_and_clean_data(distances_csv, embeddings_parquet):
    df = pd.read_csv(distances_csv)
    
    df = df[df['final_mod_type'] != 'Negative_Baseline'].copy()
    
    smp_conditions = df['final_mod_type'].isin(['SMP_plag', 'SMP_plag_doubt', 'SMP_remake'])
    df.loc[smp_conditions, 'final_mod_type'] = 'Human Plagiarism (SMP)'
    
    df_emb = pd.read_parquet(embeddings_parquet)
    
    df = df.merge(df_emb[['filename', 'embedding']], left_on='filename_ori', right_on='filename', how='inner')\
           .rename(columns={'embedding': 'embedding_ori'}).drop(columns=['filename'])
    df = df.merge(df_emb[['filename', 'embedding']], left_on='filename_mod', right_on='filename', how='inner')\
           .rename(columns={'embedding': 'embedding_mod'}).drop(columns=['filename'])

    def clean_embedding(emb):
        if isinstance(emb, str):
            try: emb = ast.literal_eval(emb)
            except: return np.array([], dtype=np.float32)
        def extract_numbers(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                flat = []
                for x in item: flat.extend(extract_numbers(x))
                return flat
            elif item is not None and not pd.isna(item):
                return [float(item)]
            return []
        try: return np.array(extract_numbers(emb), dtype=np.float32)
        except: return np.array([], dtype=np.float32)

    df['embedding_ori'] = df['embedding_ori'].apply(clean_embedding)
    df['embedding_mod'] = df['embedding_mod'].apply(clean_embedding)

    mode_dim = pd.Series([len(x) for x in df['embedding_ori']]).mode()[0]
    valid_mask = (df['embedding_ori'].apply(len) == mode_dim) & (df['embedding_mod'].apply(len) == mode_dim)
    df = df[valid_mask].reset_index(drop=True)
    
    return df

def plot_academic_trajectories_grid(df_all, output_path, title):
    
    unique_segments = df_all[['pair_id', 'time']].drop_duplicates()
    sampled_segments = unique_segments.sample(n=9, random_state=42).reset_index(drop=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = axes.flatten()
    
    # Fixed colors for consistency across subplots
    mod_types = sorted(df_all['final_mod_type'].unique())
    cmap = plt.get_cmap("tab20")
    color_map = {mod: cmap(i % 20) for i, mod in enumerate(mod_types)}

    color_map['Human Plagiarism (SMP)'] = to_rgba('gray')  # send human samples to the background
    color_map['musicgen'] = to_rgba('blue')                # keep musicgen strong and identifiable

    for i, (_, seg) in enumerate(sampled_segments.iterrows()):
        ax = axes[i]
        pair_id = seg['pair_id']
        time_val = seg['time']
        pair_data = df_all[(df_all['pair_id'] == pair_id) & (df_all['time'] == time_val)].copy()
        
        if pair_data.empty: continue
            
        X_ori_local = np.stack(pair_data['embedding_ori'].values)
        X_mod_local = np.stack(pair_data['embedding_mod'].values)
        
        X_local = np.vstack([X_ori_local[0:1], X_mod_local]) 
        X_local = X_local / (np.linalg.norm(X_local, axis=1, keepdims=True) + 1e-8)
        
        reducer = umap.UMAP(n_neighbors=4, min_dist=0.8, metric='cosine', random_state=42)
        X_2d = reducer.fit_transform(X_local)
        
        ori_x, ori_y = X_2d[0, 0], X_2d[0, 1]
        pair_data['umap_mod_x'] = X_2d[1:, 0]
        pair_data['umap_mod_y'] = X_2d[1:, 1]
        
        # Original
        ax.scatter(ori_x, ori_y, marker="x", c="black", s=100, zorder=4, linewidths=2.5)
        
        mg_row = pair_data[pair_data['final_mod_type'] == 'musicgen']
        has_mg = not mg_row.empty
        if has_mg:
            mg_x, mg_y = mg_row.iloc[0]['umap_mod_x'], mg_row.iloc[0]['umap_mod_y']
        
        for _, row in pair_data.iterrows():
            mod = row['final_mod_type']
            mx, my = row['umap_mod_x'], row['umap_mod_y']
            c = color_map[mod]
            
            if mod.startswith('musicgen_') and has_mg:
                start_x, start_y = mg_x, mg_y
            else:
                start_x, start_y = ori_x, ori_y
            
            if mod == 'Human Plagiarism (SMP)':
                line_alpha = 0.3
                marker_alpha = 0.5
                z_line = 0
                z_marker = 2
            else:
                line_alpha = 0.6
                marker_alpha = 1.0
                z_line = 1
                z_marker = 3

            # Edge
            ax.plot([start_x, mx], [start_y, my], color=c, alpha=line_alpha, linewidth=2, zorder=z_line)

            # Node
            ax.scatter(mx, my, color=c, s=80, alpha=marker_alpha, edgecolors="black", linewidths=0.8, zorder=z_marker)

        ax.set_title(f"Pair {pair_id} | Segment {time_val}s", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.95)
    
    # Global Legend
    legend_elems = [Line2D([0],[0], marker="x", color='w', markeredgecolor='black', markersize=12, markeredgewidth=2.5, label="Original Track")]
    for mod in mod_types:
        legend_elems.append(Line2D([0],[0], marker="o", color='w', markerfacecolor=color_map[mod], markeredgecolor='black', markersize=12, label=mod))
        
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=11)
    
    plt.subplots_adjust(bottom=0.15, hspace=0.2, wspace=0.15)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"[OK] Saved High-Resolution Local Network Grid to: {output_path}")
    plt.close()

if __name__ == '__main__':
    CLEWS_DISTANCES = 'data/clews_distances.csv'
    CLEWS_EMBEDDINGS = 'data/clews_embeddings.parquet'
    
    print("=== Analyzing CLEWS ===")
    if os.path.exists(CLEWS_DISTANCES) and os.path.exists(CLEWS_EMBEDDINGS):
        df_clews = load_and_clean_data(CLEWS_DISTANCES, CLEWS_EMBEDDINGS)
        plot_academic_trajectories_grid(df_clews, 
                                        output_path='plots/clews_umap_grid.png', 
                                        title="CLEWS Latent Space Topology (Local Ecosystem per Track)")