"""
Centralized constants for plagiarism detection evaluation pipeline.
Eliminates hardcoded paths and metric definitions across all analysis scripts.

Updated to support source-level vocal validity metadata and vocal-aware evaluation.
"""

import numpy as np

# DATA SOURCE PATHS
SMP_CSV = "data/Final_dataset_pairs.csv"
VOCAL_RATIOS_CSV = "results/vocal_detection/vocal_ratios_source.csv"

EMBEDDING_PATHS = {
    "CLEWS": "data/clews_embeddings.parquet",
    "WEALY": "data/wealy_embeddings.parquet",
}


# VOCAL VALIDITY / SOURCE-LEVEL METADATA
# Matches:
#   pair_9_comp_51s.wav
#   pair_9_comp_51s_musicgen.wav
#   pair_9_comp_51s_pitchU4.wav
#   pair_9_comp_51s_musicgen_pitchD4_tempo090.wav
SOURCE_KEY_REGEX = r"(pair_\d+_(?:ori|comp)_\d+s)"

# Base metadata columns from results/vocal_detection/vocal_ratios_source.csv
VOCAL_METADATA_COLUMNS = {
    "source_key": "source_key",
    "source_filename": "source_filename",
    "stem_relpath": "stem_relpath",
    "duration_sec": "duration_sec",
    "vocal_rms_db": "vocal_rms_db",
    "vocal_ratio": "vocal_ratio",
    "vocal_valid": "vocal_valid",
}

# Pair-level columns that will be created after merging vocal metadata
PAIR_VOCAL_COLUMNS = {
    "ori_key": "source_key_ori",
    "mod_key": "source_key_mod",
    "ori_ratio": "vocal_ratio_ori",
    "mod_ratio": "vocal_ratio_mod",
    "ori_valid": "vocal_valid_ori",
    "mod_valid": "vocal_valid_mod",
    "pair_valid": "pair_vocal_valid",
}

# Policy:
# A pair is vocal-valid only if BOTH sides are vocal-valid.
PAIR_VOCAL_VALIDITY_POLICY = "both"

# Which models should definitely support / use the vocal-valid subset
VOCAL_FILTER_SUPPORTED_MODELS = ["CLEWS", "WEALY", "FUSION"]
VOCAL_FILTER_REQUIRED_MODELS = ["WEALY", "FUSION"]

# Global analysis subsets
ANALYSIS_SUBSETS = {
    "full": {
        "label": "Full Dataset",
        "suffix": "",
        "apply_pair_vocal_filter": False,
    },
    "vocal_valid": {
        "label": "Vocal-Valid Pairs Only",
        "suffix": "_vocal_valid",
        "apply_pair_vocal_filter": True,
    },
}

DEFAULT_ANALYSIS_SUBSET = "full"


# DISTANCE METRICS
DISTANCE_METRICS = [
    "cosine_distance",
    "euclidean_distance",
    "manhattan_distance",
    "pearson_distance",
]


# MERGE / KEEP KEYS FOR DATA ALIGNMENT
MERGE_KEYS = ['pair_id', 'time', 'filename_ori', 'filename_mod', 'final_mod_type', 'negative_tier']

# MODEL PATHS (Distance CSV files)
MODEL_PATHS = {
    "CLEWS": "results/distances/clews_distances.csv",
    "WEALY": "results/distances/wealy_distances.csv",
    "FUSION": "results/fusion/optimal_fused_distances.csv",
}


# OUTPUT DIRECTORIES
OUTPUT_DIRS = {
    "binary_classification": "results/binary_classification",
    "robustness": "plots/robustness",
    "fusion": "results/fusion",
    "fusion_plots": "plots/fusion",
    "threshold": "results/threshold",
    "threshold_plots": "plots/threshold",
    "explainability": "results/explainability",
    "explainability_plots": "plots/explainability",
    "negative_tiers": "plots/negative_tiers",
    "attribution": "results/attribution",
    "attribution_plots": "plots/attribution"
}

# SUMMARY / ANALYSIS FILES
SUMMARY_FILES = {
    "threshold_analysis": "results/threshold/threshold_analysis_summary.csv",
    "all_metrics_detailed": "results/threshold/all_metrics_detailed_summary.csv",
    "binary_summary": "results/binary_classification/binary_summary.csv",
    "fusion_grid_search": "results/fusion/fusion_grid_search_results.csv",
    "optimal_fusion_config": "results/fusion/optimal_fusion_config.csv",
    "optimal_fused_distances": "results/fusion/optimal_fused_distances.csv",
    "attribution_summary": "results/attribution/attribution_summary.csv"
}


# FUSION OPTIMIZATION PARAMETERS
ALPHA_VALUES = np.arange(0.0, 1.01, 0.05)  # 21 steps for fusion weights
NUM_K_FOLDS = 5  # Number of folds for cross-validation
TEST_SIZE = 0.2  # Train-test split ratio for fusion optimization
RANDOM_STATE = 42  # Random seed for reproducibility
BETA = 0.5  # Precision-weighted F-beta score


# AUDIO PROCESSING PARAMETERS
AUDIO_SOURCES = ["Original", "Cover", "MusicGen", "AudioLDM2", "MGE-LDM"]

AUDIO_SOURCES_PATTERNS = {
    "Original": ["smp_", "none_"],
    "Cover": ["cover"],
    "MusicGen": ["musicgen"],
    "AudioLDM2": ["audioldm2"],
    "MGE-LDM": ["mgeldm", "mge-ldm"],
}


# MODIFICATION CATEGORIES
BROAD_CATEGORIES = {
    "1. Human Plagiarism (SMP)": "smp_",
    "2. Original + DSP": "none_",
    "3. AI Generation (Base)": "ai_base",
    "4. AI + DSP": "ai_dsp",
}


# LEGEND / DISPLAY NAMES
CATEGORY_DISPLAY_NAMES = {
    "Human Plagiarism (SMP)": "1. Human Plagiarism (SMP)",
    "Original + DSP": "2. Original + DSP",
    "AI Generation (Base)": "3. AI Generation (Base)",
    "AI + DSP": "4. AI + DSP",
    "Negative Pairs": "Negative Pairs",
}


# VISUALIZATION SETTINGS
PLOT_COLORS = {
    "Original": "blue",
    "Cover": "orange",
    "MusicGen": "red",
    "AudioLDM2": "green",
    "MGE-LDM": "purple",
}

PLOT_LINE_STYLES = ["-", "--", "-.", ":"]
PLOT_MARKERS = ["o", "s", "^", "D"]
PLOT_DPI = 300

CATEGORY_COLORS = {
    "1a. Human Plagiarism (Base)": "#2196F3",
    "1b. Human Plagiarism + DSP": "#64B5F6",
    "2. Original + DSP": "#4CAF50",
    "3. AI Generation (Base)": "#E53935",
    "4. AI + DSP": "#EF9A9A",
    "1. Human Plagiarism (SMP)": "#2196F3",  # Fallback
}

PLOT_STYLE_PARAMS = {
    "figure.dpi": 200,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.facecolor": "white",
    "axes.edgecolor": "#d0d0d0",
    "axes.grid": True,
    "grid.color": "#f0f0f0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
}

# Tier display names & colors for attribution analysis
ATTRIBUTION_TIER_DISPLAY = {
    "random": "Random",
    "intra_category_nearest": "Intra-Category",
    "global_nearest": "Global Nearest",
}
ATTRIBUTION_TIER_COLORS = {
    "random": "#2ca02c",
    "intra_category_nearest": "#ff7f0e",
    "global_nearest": "#1f77b4",
}

ATTRIBUTION_RANK_COLORS = {
    1: "#4CAF50",
    2: "#FFC107",
    3: "#FF9800",
    4: "#F44336",
}