"""
Centralized Dataset Building Utility Module.
Encapsulates the logic for constructing positive pairs from raw embeddings and metadata.
Eliminates duplicate code between metrics.py and vector_shift_analysis.py.

This module handles:
- Reading parquet files with embeddings
- Reading CSV metadata for human plagiarism mappings
- Parsing filename schemas for AI models and DSP modifications
- Merging human plagiarism base pairs (ori vs comp)
- Merging AI/DSP derivative pairs back to their unaltered bases
- Merging comp+DSP derivatives against ori base (cross-type SMP pairs)
"""

import ast
import numpy as np
import pandas as pd
from constants import MGELDM_STEMS


def build_positive_pairs(parquet_path: str, smp_metadata_path: str) -> pd.DataFrame:
    """
    Build positive pairs dataset from embeddings and metadata.
    
    This function replicates the exact pairing logic used across metrics and analysis scripts,
    ensuring consistency in how positive pairs are constructed.
    
    Args:
        parquet_path (str): Path to parquet file containing embeddings and filenames.
        smp_metadata_path (str): Path to CSV file with human plagiarism pair metadata.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - pair_id: Identifier for the song pair
            - time: Timestamp of the sample
            - final_mod_type: Type of modification (SMP_, AI model, DSP, or combinations)
            - filename_mod: Modified/derivative filename
            - filename_ori: Original filename
            - embedding_mod: Embedding vector for modified sample
            - embedding_ori: Embedding vector for original sample
    
    Process:
        1. Load embeddings from parquet and remove NaN/duplicates
        2. Parse human plagiarism metadata from CSV
        3. Parse filename schema for AI models and DSP modifications
        4. Merge 1: Human plagiarism pairs (ori_base vs comp_base)
        5. Merge 2: AI/DSP self-comparisons (any_base vs any+DSP, same ori_comp)
                   Label: none_<dsp> or <ai_model>_<dsp>
        5b. Merge 3: Cross-type SMP+DSP pairs (ori_base vs comp+DSP)
                    Label: smp_<dsp>
        6. Concatenate all positive pairs
    """
    
    # LOAD EMBEDDINGS
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=['embedding']).copy()
    df = df.drop_duplicates(subset=['filename'], keep='last').reset_index(drop=True)
    
    # LOAD AND PARSE HUMAN PLAGIARISM METADATA
    df_meta = pd.read_csv(smp_metadata_path)
    
    # Build human mapping from metadata
    mapping_records = []
    for _, row in df_meta.iterrows():
        pair_id = int(row['pair_number'])
        relation = str(row['relation'])
        ori_times = ast.literal_eval(row['ori_times']) if pd.notnull(row['ori_times']) else []
        comp_times = ast.literal_eval(row['comp_times']) if pd.notnull(row['comp_times']) else []

        for o_time in ori_times:
            for c_time in comp_times:
                mapping_records.append({
                    'pair_id': pair_id,
                    'ori_time': int(o_time),
                    'comp_time': int(c_time),
                    'relation': relation,
                })

    df_human_mapping = pd.DataFrame(mapping_records)
    
    # PARSE FILENAME SCHEMA
    # Handles AI-generated (incl. MGE-LDM multi-stem), real song DSP modifications
    def parse_filename(filename):
        """
        Parse filename schema:
        Format: pair_{pair_id}_{ori|comp}_{time}s[_{ai_model}[_{stem}]][_{dsp_mod}]
        
        For MGE-LDM, the stem name (bass/drums/other) immediately follows the
        model token and is treated as part of the ai_model identifier so that
        each stem variant is paired back to the correct base.
        
        Examples:
            pair_1_ori_10s                              -> ai_model='none',          dsp_mod='none'
            pair_1_ori_10s_musicgen                     -> ai_model='musicgen',      dsp_mod='none'
            pair_1_ori_10s_musicgen_pitchU4             -> ai_model='musicgen',      dsp_mod='pitchU4'
            pair_1_ori_10s_mgeldm_bass                  -> ai_model='mgeldm_bass',   dsp_mod='none'
            pair_1_ori_10s_mgeldm_bass_pitchU4          -> ai_model='mgeldm_bass',   dsp_mod='pitchU4'
            pair_1_ori_10s_mgeldm_drums_pitchD4_tempo090-> ai_model='mgeldm_drums',  dsp_mod='pitchD4_tempo090'
            pair_1_ori_10s_pitchD4_tempo090             -> ai_model='none',          dsp_mod='pitchD4_tempo090'
        """
        try:
            clean = filename.replace('.wav', '')
            parts = clean.split('_')

            if len(parts) < 4:
                raise ValueError(f"Unexpected filename format: {filename}")

            pair_id = int(parts[1])
            ori_comp = parts[2]
            time = int(parts[3].replace('s', ''))

            ai_models = ['musicgen', 'audioldm2', 'mgeldm']
            ai_model = 'none'
            dsp_mod = 'none'

            if len(parts) > 4:
                rest = parts[4:]            # everything after the time token

                if rest[0] in ai_models:
                    model_token = rest[0]
                    rest = rest[1:]         # consume model token

                    # MGE-LDM: next token may be a stem name
                    if model_token == 'mgeldm' and rest and rest[0] in MGELDM_STEMS:
                        ai_model = f"mgeldm_{rest[0]}"   # e.g. mgeldm_bass
                        rest = rest[1:]                   # consume stem token
                    else:
                        ai_model = model_token

                    # Remaining tokens (if any) are DSP modifiers
                    if rest:
                        dsp_mod = "_".join(rest)
                else:
                    # No AI model token → pure DSP on original/comp
                    dsp_mod = "_".join(rest)

            return pd.Series([pair_id, ori_comp, time, ai_model, dsp_mod])
        except Exception as e:
            print(f"Warning: Could not parse filename '{filename}': {e}")
            return pd.Series([None, None, None, None, None])

    parsed_meta = df['filename'].apply(parse_filename)
    parsed_meta.columns = ['pair_id', 'ori_comp', 'time', 'ai_model', 'dsp_mod']
    df = pd.concat([df, parsed_meta], axis=1)

    # MERGE 1 - HUMAN PLAGIARISM PAIRS
    # Extract base pairs (no AI generation, no DSP modifications)
    df_bases = df[(df['ai_model'] == 'none') & (df['dsp_mod'] == 'none')].copy()
    
    df_smp_comp_base = df_bases[df_bases['ori_comp'] == 'comp'].copy()
    df_pure_ori_base = df_bases[df_bases['ori_comp'] == 'ori'].copy()

    # Merge original files with comparable versions using human metadata
    df_human = pd.merge(
        df_human_mapping,
        df_pure_ori_base,
        left_on=['pair_id', 'ori_time'],
        right_on=['pair_id', 'time'],
        how='inner',
    )
    df_human = pd.merge(
        df_human,
        df_smp_comp_base,
        left_on=['pair_id', 'comp_time'],
        right_on=['pair_id', 'time'],
        how='inner',
        suffixes=('_ori', '_mod'),
    )
    
    # Create modification type label: SMP_{relation}
    df_human['final_mod_type'] = 'SMP_' + df_human['relation'].astype(str)
    df_human['time'] = df_human['time_ori']
    
    # Keep only necessary columns
    df_human = df_human[[
        'pair_id', 'time', 'final_mod_type',
        'filename_mod', 'filename_ori',
        'embedding_mod', 'embedding_ori',
    ]]

    # MERGE 2 - SELF-COMPARISON (AI / DSP)
    # Each derivative is compared to its own unaltered base (same pair_id, time, ori_comp).
    # For MGE-LDM stems (ai_model = 'mgeldm_bass' etc.) the base is the SMP segment
    # (ai_model = 'none', dsp_mod = 'none') because no "mgeldm_bass base" exists
    # without DSP — the stem generation itself IS the derivative.
    #
    # Labels produced:
    #   none_<dsp>          – pure DSP on original/comp segment
    #   musicgen_none       – MusicGen base generation (compared to SMP segment)
    #   musicgen_<dsp>      – MusicGen + DSP
    #   mgeldm_bass_none    – MGE-LDM bass stem (compared to SMP segment)
    #   mgeldm_bass_<dsp>   – MGE-LDM bass stem + DSP
    df_derivatives = df[(df['ai_model'] != 'none') | (df['dsp_mod'] != 'none')].copy()
    
    df_ai_dsp = pd.merge(
        df_derivatives,
        df_bases,
        on=['pair_id', 'time', 'ori_comp'],
        suffixes=('_mod', '_ori'),
        how='inner',
    )
    
    # Pure DSP self-comparisons always get none_<dsp> regardless of ori_comp.
    # AI (with or without DSP) keeps the <ai_model>_<dsp> label.
    df_ai_dsp['final_mod_type'] = df_ai_dsp.apply(
        lambda r: (
            f"none_{r['dsp_mod_mod']}"
            if r['ai_model_mod'] == 'none'
            else f"{r['ai_model_mod']}_{r['dsp_mod_mod']}"
        ),
        axis=1,
    )
    
    df_ai_dsp = df_ai_dsp[[
        'pair_id', 'time', 'final_mod_type',
        'filename_mod', 'filename_ori',
        'embedding_mod', 'embedding_ori',
    ]]

    # MERGE 3 - CROSS-TYPE SMP+DSP PAIRS
    # ori_base vs comp+DSP: the cover has been DSP-altered, we still want to
    # measure how far the original is from the plagiarised-and-processed version.
    # Uses df_human_mapping to bridge the different timestamps between ori and comp.
    # Label: smp_<dsp>  (parallel to SMP_ pairs but with DSP on top)
    df_comp_dsp = df[
        (df['ori_comp'] == 'comp') &
        (df['ai_model'] == 'none') &
        (df['dsp_mod'] != 'none')
    ].copy()

    if not df_comp_dsp.empty and not df_human_mapping.empty:
        # Join comp+DSP derivatives to the human mapping on (pair_id, comp_time==time)
        df_smp_dsp = pd.merge(
            df_human_mapping,
            df_comp_dsp,
            left_on=['pair_id', 'comp_time'],
            right_on=['pair_id', 'time'],
            how='inner',
        )
        # Join the matching ori_base using (pair_id, ori_time)
        df_smp_dsp = pd.merge(
            df_smp_dsp,
            df_pure_ori_base,
            left_on=['pair_id', 'ori_time'],
            right_on=['pair_id', 'time'],
            how='inner',
            suffixes=('_mod', '_ori'),
        )

        # After the second merge with suffixes=('_mod', '_ori'), dsp_mod becomes dsp_mod_mod
        dsp_col = 'dsp_mod_mod' if 'dsp_mod_mod' in df_smp_dsp.columns else 'dsp_mod'
        df_smp_dsp['final_mod_type'] = 'smp_' + df_smp_dsp[dsp_col].astype(str)
        df_smp_dsp['time'] = df_smp_dsp['time_ori']

        df_smp_dsp = df_smp_dsp[[
            'pair_id', 'time', 'final_mod_type',
            'filename_mod', 'filename_ori',
            'embedding_mod', 'embedding_ori',
        ]]
    else:
        df_smp_dsp = pd.DataFrame(columns=[
            'pair_id', 'time', 'final_mod_type',
            'filename_mod', 'filename_ori',
            'embedding_mod', 'embedding_ori',
        ])

    # COMBINE ALL POSITIVE PAIRS
    df_positives = pd.concat(
        [df_human, df_ai_dsp, df_smp_dsp],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    
    return df_positives


def clean_embedding(emb) -> np.ndarray:
    """
    Robustly parse and flatten an embedding value into a 1-D float32 numpy array.

    Handles ALL storage formats encountered across parquet and CSV sources:
      - Already-flat np.ndarray of float32/float64   → direct cast
      - Nested np.ndarray  (array of arrays)          → flatten then cast
      - Python list / nested list                     → recursive flatten + cast
      - String representation of list                 → ast.literal_eval + recurse
      - None / NaN                                    → empty array

    Used by: umap.py (inline copy), vector_shift_analysis.py
    Centralised here to avoid duplication.
    """
    # None / scalar NaN 
    if emb is None:
        return np.array([], dtype=np.float32)
    if isinstance(emb, float) and np.isnan(emb):
        return np.array([], dtype=np.float32)

    # String → parse first
    if isinstance(emb, str):
        try:
            emb = ast.literal_eval(emb)
        except Exception:
            return np.array([], dtype=np.float32)

    # numpy array
    if isinstance(emb, np.ndarray):
        # Already flat and numeric → fast path
        if emb.ndim == 1 and np.issubdtype(emb.dtype, np.number):
            return emb.astype(np.float32)
        # Nested / object dtype → flatten via ravel after converting to object
        try:
            flat = emb.ravel()
            # ravel on an object array gives an array of sub-arrays; recurse
            if flat.dtype == object:
                return clean_embedding(flat.tolist())
            return flat.astype(np.float32)
        except Exception:
            return clean_embedding(emb.tolist())   # fall through to list path

    # list / tuple → recursive flatten
    if isinstance(emb, (list, tuple)):
        def _flatten(item):
            """Recursively yield scalar floats from arbitrarily nested sequences."""
            if isinstance(item, (list, tuple, np.ndarray)):
                for sub in item:
                    yield from _flatten(sub)
            elif item is not None:
                try:
                    v = float(item)
                    if not np.isnan(v):
                        yield v
                except (TypeError, ValueError):
                    pass

        try:
            return np.array(list(_flatten(emb)), dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

    # Scalar fallback
    try:
        return np.array([float(emb)], dtype=np.float32)
    except (TypeError, ValueError):
        return np.array([], dtype=np.float32)


def validate_and_filter_embeddings(df: pd.DataFrame,
                                   emb_cols: list,
                                   clean: bool = True) -> tuple:
    """
    Validate that all embedding columns have consistent shape across rows.
    Optionally cleans/parses embeddings first (handles string/nested formats).

    Args:
        df       : DataFrame containing embedding columns.
        emb_cols : List of column names that contain embeddings,
                   e.g. ['embedding_ori', 'embedding_mod'] or ['embedding'].
        clean    : If True, apply clean_embedding() to each column first.

    Returns:
        (filtered_df, mode_dim) where:
            filtered_df : Rows where ALL emb_cols have the modal dimension.
            mode_dim    : The dominant (most common) embedding dimension found.

    Raises:
        ValueError: If no valid embeddings are found at all.
    """

    df = df.copy()

    if clean:
        for col in emb_cols:
            print(f"Cleaning column '{col}'…")
            df[col] = df[col].apply(clean_embedding)

    # Compute per-row, per-column lengths
    # shape: (N_rows, N_cols)
    length_arrays = np.column_stack(
        [np.array([len(x) for x in df[col].values], dtype=int)
         for col in emb_cols]
    )

    # A row is valid only if ALL columns have the same non-zero length
    row_min = length_arrays.min(axis=1)
    row_max = length_arrays.max(axis=1)
    consistent = (row_min == row_max) & (row_min > 0)

    # Among consistent rows, pick the modal dimension
    consistent_lengths = row_min[consistent]
    if len(consistent_lengths) == 0:
        raise ValueError(
            "validate_and_filter_embeddings: no rows have consistent "
            "non-zero embeddings across all columns."
        )

    values, counts = np.unique(consistent_lengths, return_counts=True)
    mode_dim = int(values[np.argmax(counts)])

    # Final mask: consistent AND equal to mode_dim
    valid_mask = consistent & (row_min == mode_dim)
    n_removed  = int((~valid_mask).sum())

    if n_removed > 0:
        print(
            f"validate_and_filter_embeddings: removed {n_removed} / {len(df)} rows "
            f"with inconsistent or non-modal embedding shape "
            f"(mode_dim={mode_dim})."
        )

    filtered_df = df.iloc[valid_mask].reset_index(drop=True)
    print(
        f"Kept {len(filtered_df)} rows with embedding dim={mode_dim}."
    )
    return filtered_df, mode_dim