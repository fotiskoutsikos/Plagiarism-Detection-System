import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import plotly.graph_objects as go


def load_and_prepare_data(distances_csv_path, embeddings_parquet_path, segments_metadata_path):
    """Load distances, merge with parquet embeddings, and attach song titles."""

    # Load distances CSV
    df = pd.read_csv(distances_csv_path)

    # Load embeddings from Parquet
    df_emb = pd.read_parquet(embeddings_parquet_path)
    
    # Ensure embeddings are numpy arrays for UMAP
    df_emb['embedding'] = df_emb['embedding'].apply(lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else x)

    # Merge original embeddings
    df = df.merge(
        df_emb[['filename', 'embedding']],
        left_on='filename_ori',
        right_on='filename',
        how='left'
    ).rename(columns={'embedding': 'embedding_ori'}).drop(columns=['filename'])

    # Merge modified embeddings
    df = df.merge(
        df_emb[['filename', 'embedding']],
        left_on='filename_mod',
        right_on='filename',
        how='left'
    ).rename(columns={'embedding': 'embedding_mod'}).drop(columns=['filename'])

    # Drop rows with missing embeddings
    df = df.dropna(subset=['embedding_ori', 'embedding_mod'])

    # Load metadata and remove duplicates
    df_meta = pd.read_csv(segments_metadata_path)
    df_meta = df_meta.drop_duplicates(subset=['pair_number'], keep='first')
    df_meta = df_meta.rename(columns={'pair_number': 'pair_id', 'title': 'song_title'})

    # Final merge for song titles
    df = df.merge(df_meta[['pair_id', 'song_title']], on='pair_id', how='left')

    return df


def fit_umap(df):
    """Fit UMAP on combined embeddings and project to 2D."""

    # Keep stable ordering to preserve ori/mod alignment in split coordinates
    df = df.sort_values(['pair_id', 'time', 'ori_comp']).reset_index(drop=True)

    # Stack all embeddings into a single array to ensure shared projection space
    embedding_vectors = []

    for _, row in df.iterrows():
        embedding_vectors.append(row['embedding_ori'])
        embedding_vectors.append(row['embedding_mod'])

    X_all = np.vstack(embedding_vectors)

    # Normalize embedding vectors before UMAP (cosine metric assumes normalized data)
    X_all = X_all / (np.linalg.norm(X_all, axis=1, keepdims=True) + 1e-8)

    # Fit UMAP reducer
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    X_2d = reducer.fit_transform(X_all)

    # Split back into original and modified coordinates
    umap_ori_coords = X_2d[::2]  # Every other point starting from 0
    umap_mod_coords = X_2d[1::2]  # Every other point starting from 1

    df['umap_ori_x'] = umap_ori_coords[:, 0]
    df['umap_ori_y'] = umap_ori_coords[:, 1]
    df['umap_mod_x'] = umap_mod_coords[:, 0]
    df['umap_mod_y'] = umap_mod_coords[:, 1]

    return df


def plot_static_trajectories(df, output_path):
    """Create static matplotlib plot showing latent space trajectories."""

    fig, ax = plt.subplots(figsize=(14, 10))

    # Define color palette for modification types
    mod_types = df['final_mod_type'].unique()
    colors = sns.color_palette('husl', len(mod_types))
    mod_type_colors = {mod_type: colors[i] for i, mod_type in enumerate(mod_types)}

    # Plot original points as black stars
    ax.scatter(
        df['umap_ori_x'],
        df['umap_ori_y'],
        marker='*',
        s=300,
        color='black',
        label='Original',
        zorder=3,
    )

    # Plot modified points colored by final_mod_type
    for mod_type in mod_types:
        mask = df['final_mod_type'] == mod_type
        subset = df[mask]
        ax.scatter(
            subset['umap_mod_x'],
            subset['umap_mod_y'],
            s=100,
            color=mod_type_colors[mod_type],
            label=mod_type,
            alpha=0.7,
            zorder=2,
        )

    # Draw trajectory arrows from original to modified
    for _, row in df.iterrows():
        ax.annotate(
            '',
            xy=(row['umap_mod_x'], row['umap_mod_y']),
            xytext=(row['umap_ori_x'], row['umap_ori_y']),
            arrowprops=dict(arrowstyle='->', lw=0.8, alpha=0.3, color='gray'),
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Latent Space Trajectories (Original → Modified)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save with high resolution
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Static plot saved to: {output_path}")
    plt.close(fig)


def plot_interactive_trajectories(df, output_path):
    """Create interactive plotly plot showing latent space trajectories."""

    fig = go.Figure()

    # Add scatter trace for Original points
    fig.add_trace(
        go.Scatter(
            x=df['umap_ori_x'],
            y=df['umap_ori_y'],
            mode='markers',
            marker=dict(size=12, color='black', symbol='star'),
            text=df['song_title'].fillna('Unknown'),
            hovertemplate='<b>Original</b><br>Title: %{text}<extra></extra>',
            name='Original',
        )
    )

    # Define colors for each modification type
    mod_types = df['final_mod_type'].unique()
    colors = sns.color_palette('husl', len(mod_types))
    mod_type_colors = {mod_type: f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c, mod_type in zip(colors, mod_types)}

    # Add scatter traces for Modified points by modification type
    for mod_type in mod_types:
        mask = df['final_mod_type'] == mod_type
        subset = df[mask]

        hover_text = subset['song_title'].fillna('Unknown') + '<br>' + subset['final_mod_type']

        fig.add_trace(
            go.Scatter(
                x=subset['umap_mod_x'],
                y=subset['umap_mod_y'],
                mode='markers',
                marker=dict(size=8, color=mod_type_colors[mod_type]),
                text=hover_text,
                hovertemplate='<b>Modified</b><br>%{text}<extra></extra>',
                name=mod_type,
            )
        )

    # Add trajectory lines from original to modified
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['umap_ori_x'], row['umap_mod_x']],
                y=[row['umap_ori_y'], row['umap_mod_y']],
                mode='lines',
                line=dict(color='rgba(128, 128, 128, 0.2)', width=1),
                hoverinfo='skip',
                showlegend=False,
                name='',
            )
        )

    fig.update_layout(
        title='Latent Space Trajectories (Original → Modified)',
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        hovermode='closest',
        template='plotly_white',
        width=1200,
        height=800,
        font=dict(size=12),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"Interactive plot saved to: {output_path}")


if __name__ == '__main__':
# Paths relative to project root
    DISTANCES_CSV = 'data/clews_distances.csv'  
    EMBEDDINGS_PARQUET = 'data/clews_embeddings.parquet'
    SEGMENTS_METADATA_CSV = 'data/segments_metadata.csv'
    OUTPUT_STATIC_PNG = 'plots/umap_static_trajectories.png'
    OUTPUT_INTERACTIVE_HTML = 'plots/umap_interactive_trajectories.html'

    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data(DISTANCES_CSV, EMBEDDINGS_PARQUET, SEGMENTS_METADATA_CSV)

    # Fit UMAP
    print("Fitting UMAP...")
    df = fit_umap(df)

    # Generate plots
    print("Generating static plot...")
    plot_static_trajectories(df, OUTPUT_STATIC_PNG)

    print("Generating interactive plot...")
    plot_interactive_trajectories(df, OUTPUT_INTERACTIVE_HTML)

    print("Done!")
