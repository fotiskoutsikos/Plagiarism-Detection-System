# Music Plagiarism Detection System

An end-to-end, scientifically rigorous research framework and deployment pipeline for detecting music plagiarism in modern audio content. This repository addresses both human-made covers/derivatives and AI-generated music across various Digital Signal Processing (DSP) and generative modifications using multimodal latent space representations (**CLEWS** for acoustic/melodic features and **WEALY** for semantic/vocal features).

---

## Table of Contents
1. [System Architecture & Overview](#-system-architecture--overview)
2. [Directory & File Structure](#-directory--file-structure)
3. [Execution Order & Pipeline Workflow](#-execution-order--pipeline-workflow)
   - [Phase 1: Feature Extraction & Data Preparation](#phase-1-feature-extraction--data-preparation)
   - [Phase 2: Baseline Unsupervised Evaluation (Distance & Thresholding)](#phase-2-baseline-unsupervised-evaluation-distance--thresholding)
   - [Phase 3: Supervised Machine Learning Pipeline](#phase-3-supervised-machine-learning-pipeline)
   - [Phase 4: Diagnostic, Robustness & XAI Analyses](#phase-4-diagnostic-robustness--xai-analyses)
   - [Phase 5: Production Training & Real-Time Inference](#phase-5-production-training--real-time-inference)
4. [File Breakdown & Responsibilities](#-file-breakdown--responsibilities)
5. [Reproducibility Guide](#-reproducibility-guide)

---

## [System Architecture & Overview](#-system-architecture--overview)

The framework evaluates music plagiarism through a multi-tiered approach:
1. **Multimodal Embedding Extraction**:
   - **CLEWS (Acoustic Branch)**: CQT-based ResNet50 backbone extracting 1024-dimensional acoustic/melodic representations.
   - **WEALY (Semantic Branch)**: Whisper Decoder Latent Adaptations via a Transformer Encoder extracting 512-dimensional vocal/semantic representations.
2. **Metric Learning & Distance Computation**: Evaluates Cosine, Euclidean, Manhattan, and Pearson metrics across pairs with varying difficulty (Random, Intra-Category, Global Hard Negatives).
3. **Score-Level Fusion**: Late-fusion strategy combining acoustic and semantic metrics using dynamic vocal-aware fallback policies.
4. **Supervised Classification (XGBoost)**: Feature engineering (distances, delta summary statistics, Top-K XAI dimensions) and hybrid XGBoost modeling optimized strictly for $F_{0.5}$-Score (precision-heavy) with Stratified Group K-Fold cross-validation to prevent data leakage.
5. **Explainable AI (XAI)**: Latent space drift, stable-core preservation, and Cohen's $d$ feature effect analysis.

---

## [Directory & File Structure](#-directory--file-structure)

```text
Plagiarism-Detection-System/
├── configs/                  # Model configurations
│   └── extraction/
│       ├── clews.yaml        # CLEWS architecture & settings
│       └── wealy.yaml        # WEALY Transformer & Whisper settings
├── data/                     # Primary datasets and generated embeddings
│   ├── classifier_features.parquet
│   ├── clews_embeddings.parquet
│   ├── evaluation_master_pairs.csv
│   └── wealy_embeddings.parquet
├── logs/                     # Execution logs for each pipeline stage
├── models/                   # Serialized production models
│   └── final_plagiarism_detector.pkl
├── notebooks/                # Exploratory notebooks and data preparation
├── plots/                    # Output figures (PDF) grouped by phase
│   ├── attribution/
│   ├── classification/
│   ├── explainability/
│   ├── fusion/
│   ├── negative_tiers/
│   ├── robustness/
│   ├── stem_analysis/
│   ├── threshold/
│   └── umap/
├── results/                  # Exported metrics (CSV/Parquet) grouped by experiment
│   ├── attribution/
│   ├── binary_classification/
│   ├── classification/
│   ├── distances/
│   ├── explainability/
│   ├── fusion/
│   ├── pairs/
│   ├── robustness/
│   ├── stem_analysis/
│   ├── threshold/
│   └── vocal_detection/
└── src/                      # Source code
    ├── classification/       # Supervised learning scripts
    │   ├── ablation.py
    │   ├── classification.py
    │   ├── hybrid_experiments.py
    │   ├── selected_model_evaluation.py
    │   ├── binary_supervised_classification.py
    │   └── train_final_model.py
    ├── evaluation/           # Pipeline evaluation scripts
    │   ├── build_pairs.py
    │   └── analysis/
    │       ├── binary_classification.py
    │       ├── explainability.py
    │       ├── fusion_optimization.py
    │       ├── metrics.py
    │       ├── musical_attribution.py
    │       ├── optimal_threshold.py
    │       ├── plot_negative_tiers.py
    │       ├── robustness_analysis.py
    │       ├── stem_analysis.py
    │       └── umap_analysis.py
    ├── inference/            # Feature extraction and prediction endpoints
    │   ├── extract_clews.py
    │   ├── extract_wealy.py
    │   ├── predict_pair.py
    │   └── vocal_detection.py
    └── utils/                # Shared utilities and core helpers
        ├── categorization.py
        ├── classifier_features.py
        ├── clews_lib.py
        ├── constants.py
        ├── dataset_builder.py
        ├── logging_util.py
        ├── vocal_metadata.py
        └── wealy_lib.py
```

---

## [Execution Order & Pipeline Workflow](#-execution-order--pipeline-workflow)

To guarantee full **reproducibility** of the results, scripts must be executed in the exact sequential order defined below. Each stage generates input dependencies for the subsequent stages.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PIPELINE EXECUTION FLOW                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│ [STAGE 0: PREPROCESSING & FEATURE EXTRACTION]                                          │
│   1. src/inference/vocal_detection.py    ──> Estimates source-level vocal validity     │
│   2. src/inference/extract_clews.py      ──> Extracts 1024D CLEWS embeddings           │
│   3. src/inference/extract_wealy.py      ──> Extracts 512D WEALY embeddings            │
│                                   │                                                    │
│                                   ▼                                                    │
│ [STAGE 1: EVALUATION PAIR BUILDING & DATASET ANALYSIS]                                 │
│   4. src/evaluation/build_pairs.py       ──> Constructs master evaluation pairs        │
│   5. src/evaluation/analysis/dataset_analysis.py ──> Descriptive dataset breakdown     │
│                                   │                                                    │
│                                   ▼                                                    │
│ [STAGE 2: DISTANCE COMPUTATION & THRESHOLD BASELINES]                                  │
│   6. src/evaluation/analysis/metrics.py  ──> Computes distance metrics                 │ 
│   7. src/evaluation/analysis/fusion_optimization.py ──> Score-level CLEWS+WEALY fusion │
│   8. src/evaluation/analysis/optimal_threshold.py ──> Threshold optimization (F0.5)    │
│   9. src/evaluation/analysis/binary_classification.py ──> Baseline distance metrics    │
│                                   │                                                    │
│                                   ▼                                                    │
│ [STAGE 3: FEATURE TABLE & SUPERVISED MACHINE LEARNING]                                 │
│  10. src/utils/classifier_features.py   ──> Assembles unified feature parquet          │
│  11. src/classification/ablation.py      ──> Feature ablation study (Phases 1-3)       │
│  12. src/classification/hybrid_experiments.py ──> Engineered + Raw Top-K experiments   │
│  13. src/classification/selected_model_evaluation.py ──> Deep diagnostic evaluation    │
│  14. src/classification/binary_supervised_classification.py ──> Final supervised table │
│                                   │                                                    │
│                                   ▼                                                    │
│ [STAGE 4: DIAGNOSTICS, ATTRIBUTION & XAI]                                              │
│  15. src/evaluation/analysis/explainability.py ──> Dimensional XAI & latent shift      │
│  16. src/evaluation/analysis/robustness_analysis.py ──> DSP stress testing             │
│  17. src/evaluation/analysis/musical_attribution.py ──> 4-Way source identification    │
│  18. src/evaluation/analysis/stem_analysis.py ──> Stem-level inpainting analysis       │
│  19. src/evaluation/analysis/umap_analysis.py ──> Latent space drift visualization     │
│  20. src/evaluation/analysis/plot_negative_tiers.py ──> Negative mining verification   │
│                                   │                                                    │
│                                   ▼                                                    │
│ [STAGE 5: PRODUCTION MODEL TRAINING & INFERENCE]                                       │
│  21. src/classification/train_final_model.py ──> Exports production .pkl artifact      │
│  22. src/inference/predict_pair.py      ──> Single-pair real-time prediction CLI       │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## [File Breakdown & Responsibilities](#-file-breakdown--responsibilities)
### Stage 0: Preprocessing & Feature Extraction
* `src/inference/vocal_detection.py`
  - **Function**: Performs VAD and energy-band heuristic checks on Demucs-separated vocal stems.
  - **Output**: `results/vocal_detection/vocal_ratios_source.csv`
* `src/inference/extract_clews.py`
  - **Function**: Extracts 1024D acoustic embeddings from audio waveforms using CQT + ResNet50.
  - **Output**: `data/clews_embeddings.parquet`
* `src/inference/extract_wealy.py`
  - **Function**: Extracts 512D semantic embeddings from Whisper-Turbo decoder latents.
  - **Output**: `data/wealy_embeddings.parquet`

---

### Stage 1: Dataset Construction & Descriptive Statistics
* `src/evaluation/build_pairs.py`
  - **Function**: Constructs the master evaluation set (`evaluation_master_pairs.csv`) containing ground-truth positive pairs and mined negative pairs across three difficulty tiers (*Random*, *Intra-Category Nearest*, *Global Hard Nearest*).
  - **Inputs**: `clews_embeddings.parquet`, SMP metadata.
  - **Outputs**: `data/evaluation_master_pairs.csv`, `evaluation_master_pairs_summary.csv`
* `src/evaluation/analysis/dataset_analysis.py`
  - **Function**: Generates descriptive dataset statistics, segment inventories, and 10 publication-quality summary plots.
  - **Outputs**: CSV summaries in `results/` and PDF plots in `plots/`.

---

### Stage 2: Distance Computation, Threshold Calibration & Fusion Baselines
* `src/evaluation/analysis/metrics.py`
  - **Function**: Computes 4 distance metrics (Cosine, Euclidean, Manhattan, Pearson) on the unified pair benchmark.
  - **Outputs**: `results/distances/{clews,wealy}_distances.csv`
* `src/evaluation/analysis/fusion_optimization.py`
  - **Function**: Performs exhaustive grid search (336 configs) for late score-level fusion ($d = lpha \cdot d_{	ext{CLEWS}} + (1-lpha) \cdot d_{	ext{WEALY}}$) with a vocal-aware fallback policy.
  - **Outputs**: `results/fusion/optimal_fused_distances.csv`, heatmaps, alpha curves.
* `src/evaluation/analysis/optimal_threshold.py`
  - **Function**: Evaluates distance metrics using 5-Fold Stratified CV, optimizing decision thresholds for $F_{0.5}$-score.
  - **Outputs**: `results/threshold/threshold_analysis_summary.csv`, PR curves, KDE distribution plots.
* `src/evaluation/analysis/binary_classification.py`
  - **Function**: Evaluates deterministic threshold-based classification and conducts triple-tier error analysis.
  - **Outputs**: Broad, detailed, and FP tier breakdown CSVs in `results/binary_classification/`.

---

### Stage 3: Supervised Machine Learning Pipeline
* `src/utils/classifier_features.py`
  - **Function**: Constructs the master feature table (`classifier_features.parquet`), unifying CLEWS/WEALY distances, 22 delta summary stats and vocal flags without data leakage.
  - **Output**: `data/classifier_features.parquet`
* `src/classification/classification.py`
  - **Function**: Core parameterized execution engine for supervised XGBoost experiments using 5-Fold StratifiedGroupKFold CV (grouped by `filename_ori`), dynamic `scale_pos_weight`, out-of-fold (OOF) tracking, and multi-seed statistical evaluation.
* `src/classification/ablation.py`
  - **Function**: Runs 3 ablation phases: (1) Engineered feature families, (2) Raw full embedding deltas, (3) Top-K dimension convergence curve vs. compute trade-offs.
  - **Outputs**: `ablation_results.csv`, `topk_convergence_curve.pdf`, etc.
* `src/classification/hybrid_experiments.py`
  - **Function**: Evaluates hybrid combinations of 24 base engineered features + Top-$K$ ($K \in \{256, 512, 1024\}$) raw CLEWS dimensions ranked by mean positive shift.
  - **Outputs**: `hybrid_results.csv`, `hybrid_f05_comparison.pdf`
* `src/classification/selected_model_evaluation.py`
  - **Function**: Runs deep diagnostic analysis (triple-tier error analysis + permutation feature importance) on selected candidates.
  - **Outputs**: Granular metric breakdowns and feature importance plots in `results/classification/` and `plots/classification/`.
* `src/classification/binary_supervised_classification.py`
  - **Function**: Formats and exports final supervised classification results for direct comparison with unsupervised baselines.
  - **Outputs**: `results/binary_supervised_classification/`

---

### Stage 4: Diagnostics, Attribution & Latent Space Analysis
* `src/evaluation/analysis/explainability.py`
  - **Function**: Conducts latent space analysis: delta vectors, Top-30 affected dimensions, directional shift heatmaps, Cohen's $d$ discrimination (Human vs. AI), and stable-core preservation.
  - **Outputs**: Extensive PDF plots and CSV dimension rankings in `explainability/`.
* `src/evaluation/analysis/robustness_analysis.py`
  - **Function**: Stress-tests distance stability under continuous Pitch/Tempo DSP shifts and extreme modifications.
  - **Outputs**: Distance trendline plots and extreme stress test boxplots.
* `src/evaluation/analysis/musical_attribution.py`
  - **Function**: Evaluates 4-Way Forced Choice Retrieval (Positive vs. Random, Intra-Category, Global Hard Negatives) measuring Top-1 Accuracy, MRR, and Mean Rank.
  - **Outputs**: Attribution CSVs and rank distribution plots.
* `src/evaluation/analysis/stem_analysis.py`
  - **Function**: Evaluates inherent embedding distances of MGE-LDM stem-guided generations (bass, drums, other).
  - **Outputs**: Stem base generation comparison plots with threshold reference lines.
* `src/evaluation/analysis/umap_analysis.py`
  - **Function**: Visualizes 2D centered latent space trajectories using UMAP projection with cosine metric.
  - **Outputs**: `plots/umap/{clews,wealy}_umap_plot.pdf`
* `src/evaluation/analysis/plot_negative_tiers.py`
  - **Function**: Visualizes distance distributions across negative mining difficulty tiers.
  - **Outputs**: `plots/negative_tiers/*.pdf`

---

### Stage 5: Production Deployment & Inference
* `src/classification/train_final_model.py`
  - **Function**: Reconstructs the winning feature configuration (`hybrid_top512`), calibrates decision thresholds on hold-out split, retrains XGBoost on 100% of data, and packages training reference statistics into a single artifact.
  - **Output**: `models/final_plagiarism_detector.pkl`
* `src/inference/predict_pair.py`
  - **Function**: Single-pair CLI interface that extracts embeddings live, computes distances and delta summaries using training reference stats, and outputs probability + binary prediction.
  - **Usage**:
    ```bash
    python src/inference/predict_pair.py --ori path/to/original.wav --mod path/to/modified.wav
    ```

---

## [Reproducibility Guide](#-reproducibility-guide)

To reproduce all results and generated artifacts from scratch:

1. **Environment Setup**:
   Ensure PyTorch, XGBoost, WeasyPrint, OpenPyXL, and Audio Processing packages (nnAudio, librosa, whisper) are installed.

2. **Sequential Pipeline Execution**:
   Run the pipeline scripts in the exact sequence specified in the [Execution Order & Pipeline Workflow](#-execution-order--pipeline-workflow) section:

   ```bash
   # 1. Feature Extraction & Dataset Preparation
   python src/inference/vocal_detection.py
   python src/inference/extract_clews.py
   python src/inference/extract_wealy.py
   python src/evaluation/build_pairs.py
   python src/evaluation/analysis/dataset_analysis.py

   # 2. Distance Computation & Baseline Thresholding
   python src/evaluation/analysis/metrics.py
   python src/evaluation/analysis/fusion_optimization.py
   python src/evaluation/analysis/optimal_threshold.py
   python src/evaluation/analysis/binary_classification.py

   # 3. Feature Assembly & Supervised Machine Learning
   python src/utils/classifier_features.py
   python src/classification/ablation.py
   python src/classification/hybrid_experiments.py
   python src/classification/selected_model_evaluation.py
   python src/classification/binary_supervised_classification.py

   # 4. Diagnostic & Explainability Analyses
   python src/evaluation/analysis/explainability.py
   python src/evaluation/analysis/robustness_analysis.py
   python src/evaluation/analysis/musical_attribution.py
   python src/evaluation/analysis/stem_analysis.py
   python src/evaluation/analysis/umap_analysis.py
   python src/evaluation/analysis/plot_negative_tiers.py

   # 5. Production Artifact Generation
   python src/classification/train_final_model.py
   ```

3. **Inference**:
   To test any arbitrary pair of audio files against the final trained model:
   ```bash
   python src/inference/predict_pair.py --ori sample1.wav --mod sample2.wav
   ```
