"""
Centralized constants for plagiarism detection evaluation pipeline.
Eliminates hardcoded paths and metric definitions across all analysis scripts.
"""

import numpy as np

# ============================================================================
# DISTANCE METRICS
# ============================================================================
DISTANCE_METRICS = [
    'cosine_distance',
    'euclidean_distance',
    'manhattan_distance',
    'pearson_distance'
]

# ============================================================================
# MERGE/KEEP KEYS FOR DATA ALIGNMENT
# ============================================================================
MERGE_KEYS = ['pair_id', 'time', 'filename_ori', 'filename_mod', 'final_mod_type']

# ============================================================================
# MODEL PATHS (Distance CSV files)
# ============================================================================
MODEL_PATHS = {
    "CLEWS": "results/distances/clews_distances.csv",
    "WEALY": "results/distances/wealy_distances.csv",
    "FUSION": "results/fusion/optimal_fused_distances.csv"
}

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
OUTPUT_DIRS = {
    "binary_classification": "results/binary_classification",
    "robustness": "plots/robustness",
    "fusion": "results/fusion",
    "fusion_plots": "plots/fusion",
    "threshold": "results/threshold",
    "threshold_plots": "plots/threshold",
    "explainability": "results/explainability",
    "explainability_plots": "plots/explainability"
}

# ============================================================================
# SUMMARY/ANALYSIS FILES
# ============================================================================
SUMMARY_FILES = {
    "threshold_analysis": "results/threshold/threshold_analysis_summary.csv",
    "all_metrics_detailed": "results/threshold/all_metrics_detailed_summary.csv",
    "binary_summary": "results/binary_classification/binary_summary.csv",
    "fusion_grid_search": "results/fusion/fusion_grid_search_results.csv",
    "optimal_fusion_config": "results/fusion/optimal_fusion_config.csv",
    "optimal_fused_distances": "results/fusion/optimal_fused_distances.csv"
}

# ============================================================================
# FUSION OPTIMIZATION PARAMETERS
# ============================================================================
ALPHA_VALUES = np.arange(0.0, 1.01, 0.05)  # 21 steps for fusion weights
NUM_K_FOLDS = 5  # Number of folds for cross-validation
TEST_SIZE = 0.2  # Train-test split ratio for fusion optimization
RANDOM_STATE = 42  # Random seed for reproducibility

# ============================================================================
# AUDIO PROCESSING PARAMETERS
# ============================================================================
AUDIO_SOURCES = ['Original', 'MusicGen', 'AudioLDM2', 'MGE-LDM']
AUDIO_SOURCES_PATTERNS = {
    'Original': ['smp_', 'none_'],
    'MusicGen': ['musicgen'],
    'AudioLDM2': ['audioldm2'],
    'MGE-LDM': ['mgeldm', 'mge-ldm']
}

# ============================================================================
# MODIFICATION CATEGORIES
# ============================================================================
BROAD_CATEGORIES = {
    "1. Human Plagiarism (SMP)": "smp_",
    "2. Original + DSP": "none_",
    "3. AI Generation (Base)": "ai_base",
    "4. AI + DSP": "ai_dsp"
}

# ============================================================================
# LEGEND/DISPLAY NAMES
# ============================================================================
CATEGORY_DISPLAY_NAMES = {
    'Human Plagiarism (SMP)': '1. Human Plagiarism (SMP)',
    'Original + DSP': '2. Original + DSP',
    'AI Generation (Base)': '3. AI Generation (Base)',
    'AI + DSP': '4. AI + DSP',
    'Negative Pairs': 'Negative Pairs'
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
PLOT_COLORS = {
    'Original': 'blue',
    'Cover': 'orange',
    'MusicGen': 'red',
    'AudioLDM2': 'green',
    'MGE-LDM': 'purple'
}

PLOT_LINE_STYLES = ['-', '--', '-.', ':']
PLOT_MARKERS = ['o', 's', '^', 'D']
PLOT_DPI = 300

CATEGORY_COLORS = {
    '1a. Human Plagiarism (Base)': '#2196F3',
    '1b. Human Plagiarism + DSP':  '#64B5F6',
    '2. Original + DSP':           '#4CAF50',
    '3. AI Generation (Base)':     '#E53935',
    '4. AI + DSP':                 '#EF9A9A',
    '1. Human Plagiarism (SMP)':   '#2196F3'  # Fallback
}

PLOT_STYLE_PARAMS = {
    "figure.dpi": 300,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.facecolor": "white",
    "axes.edgecolor": "#d0d0d0",
    "axes.grid": True,
    "grid.color": "#f0f0f0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]
}