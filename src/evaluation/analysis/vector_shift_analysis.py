"""
Explainable AI (XAI) - Vector Dimensionality Shift Analysis.
Strictly replicates the precise positive pairing logic from metrics.py.
Computes the Delta (Δ) vectors between Original and Modified embeddings
to guarantee comparisons are identical to the distance metrics evaluation.
"""

import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve repository root and load utils
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent

# Setup Logging
logging_util_path = repo_root / "src" / "utils" / "logging_util.py"
spec = importlib.util.spec_from_file_location("logging_util", str(logging_util_path))
if spec is None or spec.loader is None:
    raise FileNotFoundError(f"Could not load logging_util from {logging_util_path}")
logging_util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_util)
setup_logging = logging_util.setup_logging
setup_logging(__file__)

import logging
logger = logging.getLogger(__name__)

# Import Centralized Utils
sys.path.insert(0, str(repo_root / "src"))
from utils.constants import OUTPUT_DIRS
from utils.categorization import clean_mod_type, get_broad_category
from utils.dataset_builder import build_positive_pairs


def process_and_plot_shift(parquet_path: str, pairs_path: str, model_name: str, output_dir: str):
    logger.info(f"--- Processing {model_name} ---")
    
    # Build positive pairs using centralized utility
    df_positives = build_positive_pairs(parquet_path, pairs_path)
    
    if df_positives.empty:
        logger.error(f"Failed to extract pairs for {model_name}.")
        return

    # Apply strict categorization using utils
    df_positives['clean_mod_type'] = df_positives['final_mod_type'].apply(clean_mod_type)
    df_positives['broad_category'] = df_positives['clean_mod_type'].apply(get_broad_category)
    
    # Calculate Absolute Dimensionality Shift (Δ-vectors)
    logger.info("Computing Δ-vectors (Absolute Shift per dimension)...")
    emb_ori_stack = np.stack(df_positives['embedding_ori'].values)
    emb_mod_stack = np.stack(df_positives['embedding_mod'].values)
    
    delta_vectors = np.abs(emb_mod_stack - emb_ori_stack)
    df_positives['delta_vector'] = list(delta_vectors)
    
    # Generate the Plot
    logger.info(f"Aggregating shifts and saving plot...")
    categories = sorted(df_positives['broad_category'].unique())
    
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="darkgrid")
    
    colors = {
        '1. Human Plagiarism (SMP)': '#2ecc71',
        '2. Original + DSP': '#3498db',
        '3. AI Generation (Base)': '#e67e22',
        '4. AI + DSP': '#e74c3c'
    }
    
    for cat in categories:
        cat_vectors = np.stack(df_positives[df_positives['broad_category'] == cat]['delta_vector'].values)
        mean_shift = np.mean(cat_vectors, axis=0)
        
        plt.plot(
            mean_shift, 
            label=cat, 
            color=colors.get(cat, 'gray'), 
            alpha=0.8, 
            linewidth=1.5
        )
    
    plt.title(f"Latent Space Shift Analysis (Explainability) - {model_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Embedding Dimensions", fontsize=12)
    plt.ylabel("Mean Absolute Shift Magnitude (Δ)", fontsize=12)
    plt.legend(title="Modification Type", fontsize=11, title_fontsize=12, loc='upper right')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{model_name.lower()}_vector_shift_explainability.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved Explainability Plot to: {plot_path}")
    plt.close()


def main():
    logger.info("=" * 80)
    logger.info("STARTING NATIVE VECTOR DIMENSIONALITY SHIFT ANALYSIS (XAI)")
    logger.info("=" * 80)
    
    PAIRS_CSV = "data/Final_dataset_pairs.csv"
    CLEWS_PARQUET = "data/clews_embeddings.parquet"
    WEALY_PARQUET = "data/wealy_embeddings.parquet"
    out_dir = "plots/explainability"
    
    if os.path.exists(CLEWS_PARQUET) and os.path.exists(PAIRS_CSV):
        process_and_plot_shift(CLEWS_PARQUET, PAIRS_CSV, "CLEWS", out_dir)
    else:
        logger.error("CLEWS Parquet or Pairs CSV not found.")

    if os.path.exists(WEALY_PARQUET) and os.path.exists(PAIRS_CSV):
        process_and_plot_shift(WEALY_PARQUET, PAIRS_CSV, "WEALY", out_dir)

if __name__ == "__main__":
    main()