"""
Vocal metadata helper for plagiarism detection evaluation pipeline.

Provides utilities to:
1. Extract source_key from any filename in the pipeline
2. Load the source-level vocal validity metadata (vocal_ratios.csv)
3. Attach pair-level vocal validity columns to any pairwise DataFrame
4. Filter DataFrames by vocal validity

Design principle:
    Vocal validity is a SOURCE-LEVEL property. It is determined once,
    on the Demucs-separated vocals.wav stem of the original SMP segment,
    and then propagated to ALL derivatives (DSP variants, AI generations, etc.)
    via the source_key.

Usage in analysis scripts:
    from src.evaluation.utils.vocal_metadata import attach_vocal_metadata

    df = pd.read_csv("results/distances/wealy_distances.csv")
    df = attach_vocal_metadata(df)

    # Full evaluation
    evaluate(df)

    # Vocal-valid subset evaluation
    df_vocal = filter_vocal_valid(df)
    evaluate(df_vocal)
"""

import os
import re
import logging
import pandas as pd

# Allow running from project root or from within src/
try:
    from constants import (
        VOCAL_RATIOS_CSV,
        SOURCE_KEY_REGEX,
        PAIR_VOCAL_COLUMNS,
        PAIR_VOCAL_VALIDITY_POLICY,
    )
except ImportError:
    from utils.constants import (
        VOCAL_RATIOS_CSV,
        SOURCE_KEY_REGEX,
        PAIR_VOCAL_COLUMNS,
        PAIR_VOCAL_VALIDITY_POLICY,
    )

logger = logging.getLogger(__name__)


# SOURCE KEY EXTRACTION
# Precompile the regex for performance
_SOURCE_KEY_PATTERN = re.compile(SOURCE_KEY_REGEX)


def extract_source_key(filename: str) -> str:
    """
    Extract the source segment key from any filename in the pipeline.

    Works on all filename formats:
        pair_9_ori_51s.wav                          -> pair_9_ori_51s
        pair_9_comp_51s_musicgen.wav                -> pair_9_comp_51s
        pair_9_comp_51s_pitchU4.wav                 -> pair_9_comp_51s
        pair_9_comp_51s_musicgen_pitchD4_tempo090.wav -> pair_9_comp_51s
        pair_9_comp_51s_audioldm2_pitchU4.wav       -> pair_9_comp_51s

    Args:
        filename: The filename (with or without path, with or without extension).

    Returns:
        The source_key string, or empty string if no match found.
    """
    # Strip path and extension for safety
    basename = os.path.splitext(os.path.basename(filename))[0]

    match = _SOURCE_KEY_PATTERN.search(basename)
    if match:
        return match.group(1)

    logger.warning(f"Could not extract source_key from: {filename}")
    return ""


# 2. LOAD VOCAL METADATA
# Module-level cache to avoid re-reading CSV on every call
_vocal_metadata_cache = None


def load_vocal_metadata(force_reload: bool = False) -> pd.DataFrame:
    """
    Load the source-level vocal validity metadata from vocal_ratios.csv.

    Returns a DataFrame with columns:
        source_key | vocal_ratio | vocal_valid
    (plus any additional columns from the CSV)

    The result is cached in memory after first load.

    Args:
        force_reload: If True, re-read from disk even if cached.

    Returns:
        DataFrame with vocal metadata indexed by source_key.

    Raises:
        FileNotFoundError: If vocal_ratios.csv does not exist.
    """
    global _vocal_metadata_cache

    if _vocal_metadata_cache is not None and not force_reload:
        return _vocal_metadata_cache

    if not os.path.exists(VOCAL_RATIOS_CSV):
        raise FileNotFoundError(
            f"Vocal metadata file not found: {VOCAL_RATIOS_CSV}\n"
            f"Run vocal_ratios.py first to generate it."
        )

    df = pd.read_csv(VOCAL_RATIOS_CSV)

    # Validate required columns
    required_cols = {"source_key", "vocal_ratio", "vocal_valid"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Vocal metadata CSV is missing required columns: {missing}"
        )

    # Ensure types
    df["source_key"] = df["source_key"].astype(str).str.strip()
    df["vocal_valid"] = df["vocal_valid"].astype(bool)
    df["vocal_ratio"] = df["vocal_ratio"].astype(float)

    # Check for duplicates
    dupes = df[df["source_key"].duplicated(keep=False)]
    if len(dupes) > 0:
        logger.warning(
            f"Found {len(dupes)} duplicate source_keys in vocal metadata. "
            f"Keeping last occurrence."
        )
        df = df.drop_duplicates(subset=["source_key"], keep="last")

    _vocal_metadata_cache = df
    print(
        f"Loaded vocal metadata: {len(df)} sources, "
        f"{df['vocal_valid'].sum()} valid, "
        f"{(~df['vocal_valid']).sum()} invalid"
    )

    return df


def clear_vocal_metadata_cache():
    """Clear the cached vocal metadata (useful for testing)."""
    global _vocal_metadata_cache
    _vocal_metadata_cache = None


# ATTACH VOCAL METADATA TO PAIRWISE DATAFRAMES
def attach_vocal_metadata(
    df: pd.DataFrame,
    filename_ori_col: str = "filename_ori",
    filename_mod_col: str = "filename_mod",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Enrich a pairwise distances/results DataFrame with source-level vocal metadata.

    For each row, extracts the source_key from both filename_ori and filename_mod,
    then merges vocal_ratio and vocal_valid from the vocal metadata CSV.

    Added columns:
        source_key_ori    : source key extracted from filename_ori
        source_key_mod    : source key extracted from filename_mod
        vocal_ratio_ori   : vocal ratio of the ori source segment
        vocal_ratio_mod   : vocal ratio of the mod source segment
        vocal_valid_ori   : whether ori source has valid vocals
        vocal_valid_mod   : whether mod source has valid vocals
        pair_vocal_valid  : whether BOTH sides have valid vocals

    Args:
        df: DataFrame with pairwise data (must have filename_ori and filename_mod columns).
        filename_ori_col: Name of the column containing original filenames.
        filename_mod_col: Name of the column containing modified filenames.
        inplace: If True, modify df in place. If False, return a copy.

    Returns:
        The enriched DataFrame.
    """
    if not inplace:
        df = df.copy()

    # Validate input columns exist
    for col in [filename_ori_col, filename_mod_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    # Step 1: Extract source keys
    col_names = PAIR_VOCAL_COLUMNS

    df[col_names["ori_key"]] = df[filename_ori_col].apply(extract_source_key)
    df[col_names["mod_key"]] = df[filename_mod_col].apply(extract_source_key)

    # Log extraction failures
    ori_empty = (df[col_names["ori_key"]] == "").sum()
    mod_empty = (df[col_names["mod_key"]] == "").sum()
    if ori_empty > 0:
        logger.warning(f"{ori_empty} rows could not extract source_key from {filename_ori_col}")
    if mod_empty > 0:
        logger.warning(f"{mod_empty} rows could not extract source_key from {filename_mod_col}")

    # Step 2: Load vocal metadata
    vocal_df = load_vocal_metadata()

    # Prepare slim lookup tables (only what we need)
    vocal_lookup = vocal_df[["source_key", "vocal_ratio", "vocal_valid"]].copy()

    # Step 3: Merge for ORI side
    df = df.merge(
        vocal_lookup.rename(columns={
            "vocal_ratio": col_names["ori_ratio"],
            "vocal_valid": col_names["ori_valid"],
        }),
        left_on=col_names["ori_key"],
        right_on="source_key",
        how="left",
        suffixes=("", "_voc_ori_drop"),
    )
    # Drop the redundant source_key from merge
    drop_cols = [c for c in df.columns if c.endswith("_voc_ori_drop") or c == "source_key"]
    # Be careful not to drop our own source_key_ori / source_key_mod
    drop_cols = [
        c for c in drop_cols
        if c not in [col_names["ori_key"], col_names["mod_key"]]
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Step 4: Merge for MOD side
    df = df.merge(
        vocal_lookup.rename(columns={
            "vocal_ratio": col_names["mod_ratio"],
            "vocal_valid": col_names["mod_valid"],
        }),
        left_on=col_names["mod_key"],
        right_on="source_key",
        how="left",
        suffixes=("", "_voc_mod_drop"),
    )
    drop_cols = [c for c in df.columns if c.endswith("_voc_mod_drop") or c == "source_key"]
    drop_cols = [
        c for c in drop_cols
        if c not in [col_names["ori_key"], col_names["mod_key"]]
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Step 5: Handle NaN from unmatched source_keys
    ori_nan = df[col_names["ori_valid"]].isna().sum()
    mod_nan = df[col_names["mod_valid"]].isna().sum()

    if ori_nan > 0:
        logger.warning(
            f"{ori_nan} rows have no vocal metadata for ORI side. "
            f"Defaulting vocal_valid_ori=False for unmatched."
        )
    if mod_nan > 0:
        logger.warning(
            f"{mod_nan} rows have no vocal metadata for MOD side. "
            f"Defaulting vocal_valid_mod=False for unmatched."
        )

    # Fill NaN with conservative defaults
    df[col_names["ori_ratio"]] = df[col_names["ori_ratio"]].fillna(0.0)
    df[col_names["mod_ratio"]] = df[col_names["mod_ratio"]].fillna(0.0)
    df[col_names["ori_valid"]] = df[col_names["ori_valid"]].fillna(False).astype(bool)
    df[col_names["mod_valid"]] = df[col_names["mod_valid"]].fillna(False).astype(bool)

    # Step 6: Compute pair-level validity
    if PAIR_VOCAL_VALIDITY_POLICY == "both":
        df[col_names["pair_valid"]] = (
            df[col_names["ori_valid"]] & df[col_names["mod_valid"]]
        )
    elif PAIR_VOCAL_VALIDITY_POLICY == "either":
        df[col_names["pair_valid"]] = (
            df[col_names["ori_valid"]] | df[col_names["mod_valid"]]
        )
    elif PAIR_VOCAL_VALIDITY_POLICY == "mod_only":
        df[col_names["pair_valid"]] = df[col_names["mod_valid"]]
    else:
        raise ValueError(
            f"Unknown PAIR_VOCAL_VALIDITY_POLICY: {PAIR_VOCAL_VALIDITY_POLICY}"
        )

    # Summary log
    total = len(df)
    pair_valid = df[col_names["pair_valid"]].sum()
    pair_invalid = total - pair_valid
    print(
        f"Vocal metadata attached: {total} pairs total, "
        f"{pair_valid} pair_vocal_valid=True ({pair_valid/total:.1%}), "
        f"{pair_invalid} pair_vocal_valid=False ({pair_invalid/total:.1%})"
    )

    return df


# FILTERING UTILITIES
def filter_vocal_valid(
    df: pd.DataFrame,
    pair_valid_col: str = None,
) -> pd.DataFrame:
    """
    Filter a DataFrame to keep only rows where pair_vocal_valid is True.

    Args:
        df: DataFrame that has been enriched by attach_vocal_metadata().
        pair_valid_col: Column name for pair validity.
                        Defaults to the value from constants.

    Returns:
        Filtered copy of the DataFrame.
    """
    if pair_valid_col is None:
        pair_valid_col = PAIR_VOCAL_COLUMNS["pair_valid"]

    if pair_valid_col not in df.columns:
        raise ValueError(
            f"Column '{pair_valid_col}' not found. "
            f"Did you run attach_vocal_metadata() first?"
        )

    df_filtered = df[df[pair_valid_col] == True].copy()

    print(
        f"Vocal filter applied: {len(df)} -> {len(df_filtered)} rows "
        f"({len(df) - len(df_filtered)} removed)"
    )

    return df_filtered


def get_vocal_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of vocal validity statistics from an enriched DataFrame.

    Useful for logging and report generation.

    Args:
        df: DataFrame enriched by attach_vocal_metadata().

    Returns:
        Dictionary with summary statistics.
    """
    col_names = PAIR_VOCAL_COLUMNS

    if col_names["pair_valid"] not in df.columns:
        raise ValueError("DataFrame not enriched. Run attach_vocal_metadata() first.")

    total = len(df)
    pair_valid = int(df[col_names["pair_valid"]].sum())
    pair_invalid = total - pair_valid
    ori_valid = int(df[col_names["ori_valid"]].sum())
    mod_valid = int(df[col_names["mod_valid"]].sum())

    return {
        "total_pairs": total,
        "pair_vocal_valid": pair_valid,
        "pair_vocal_invalid": pair_invalid,
        "pair_vocal_valid_pct": round(pair_valid / total * 100, 1) if total > 0 else 0,
        "ori_vocal_valid": ori_valid,
        "mod_vocal_valid": mod_valid,
        "ori_vocal_valid_pct": round(ori_valid / total * 100, 1) if total > 0 else 0,
        "mod_vocal_valid_pct": round(mod_valid / total * 100, 1) if total > 0 else 0,
    }


# QUICK DIAGNOSTIC (standalone execution)
if __name__ == "__main__":
    """
    Quick diagnostic: test the helper on an actual distance CSV.
    Run from project root:
        python -m src.evaluation.utils.vocal_metadata
    or:
        python src/evaluation/utils/vocal_metadata.py
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("VOCAL METADATA HELPER — DIAGNOSTIC")
    print("=" * 60)

    # Test 1: source_key extraction
    test_filenames = [
        "pair_9_ori_51s.wav",
        "pair_9_comp_51s_musicgen.wav",
        "pair_9_comp_51s_pitchU4.wav",
        "pair_9_comp_51s_musicgen_pitchD4_tempo090.wav",
        "pair_55_comp_163s_audioldm2_pitchU4.wav",
        "pair_60_ori_92s_pitchD2.wav",
        "pair_27_comp_84s_audioldm2.wav",
    ]

    print("\nTest 1: Source Key Extraction")
    for fn in test_filenames:
        key = extract_source_key(fn)
        print(f"  {fn:<55s} -> {key}")

    # Test 2: Load vocal metadata
    print("\nTest 2: Load Vocal Metadata")
    try:
        vdf = load_vocal_metadata()
        print(f"  Loaded {len(vdf)} source entries")
        print(f"  Valid: {vdf['vocal_valid'].sum()}")
        print(f"  Invalid: {(~vdf['vocal_valid']).sum()}")
        print(f"\n  Sample rows:")
        print(vdf.head(5).to_string(index=False))
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        sys.exit(0)

    # Test 3: Attach to a real distance CSV
    print("\nTest 3: Attach to Distance CSV")

    # Try to find any distance CSV
    test_csvs = [
        "results/distances/clews_distances.csv",
        "results/distances/wealy_distances.csv",
    ]

    for csv_path in test_csvs:
        if os.path.exists(csv_path):
            print(f"\n  Testing with: {csv_path}")
            dist_df = pd.read_csv(csv_path)
            print(f"  Rows before: {len(dist_df)}")

            enriched = attach_vocal_metadata(dist_df)
            summary = get_vocal_summary(enriched)

            print(f"Rows after:  {len(enriched)}")
            print(f"Summary:     {summary}")

            # Show a few enriched rows
            vocal_cols = list(PAIR_VOCAL_COLUMNS.values())
            show_cols = ["filename_ori", "filename_mod"] + vocal_cols
            show_cols = [c for c in show_cols if c in enriched.columns]
            print(f"\n  Sample enriched rows:")
            print(enriched[show_cols].head(5).to_string(index=False))

            # Test filter
            filtered = filter_vocal_valid(enriched)
            print(f"\n  After vocal filter: {len(filtered)} rows")

            break
    else:
        print("[SKIP] No distance CSV found for testing.")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)