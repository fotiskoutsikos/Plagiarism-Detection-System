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
import pandas as pd
import numpy as np


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
    # Handles both AI-generated and real song DSP modifications
    def parse_filename(filename):
        """
        Parse filename schema:
        Format: p_{pair_id}_{ori|comp}_{time}s[_{ai_model}][_{dsp_mod}]
        
        Examples:
            - p_1_ori_10s -> pair_id=1, ori_comp='ori', time=10, ai_model='none', dsp_mod='none'
            - p_1_comp_10s -> pair_id=1, ori_comp='comp', time=10, ai_model='none', dsp_mod='none'
            - p_1_ori_10s_musicgen -> pair_id=1, ori_comp='ori', time=10, ai_model='musicgen', dsp_mod='none'
            - p_1_ori_10s_musicgen_pitchd12 -> pair_id=1, time=10, ai_model='musicgen', dsp_mod='pitchd12'
            - p_1_ori_10s_pitchd12_tempo110 -> pair_id=1, dsp_mod='pitchd12_tempo110'
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

            # Logic to handle both AI generations and pure SMP DSP modifications
            if len(parts) > 4:
                if parts[4] in ai_models:
                    ai_model = parts[4]
                    if len(parts) > 5:
                        dsp_mod = "_".join(parts[5:])
                else:
                    dsp_mod = "_".join(parts[4:])

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
    # This covers: ori+DSP vs ori_base, comp+DSP vs comp_base, AI vs ori_base, AI+DSP vs ori_base.
    # Label: none_<dsp> for pure DSP, <ai_model>_<dsp> for AI (with or without DSP).
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