"""
Centralized categorization logic for modification type analysis.
Unifies feature extraction and categorization across all evaluation scripts.
Handles both positive and negative (Negative_ prefix) samples.
"""

import re
import pandas as pd
import numpy as np


def get_ground_truth_label(mod_type: str) -> int:
    """
    Extracts ground truth label from modification type.
    
    Args:
        mod_type: Modification type string (may have 'Negative_' prefix).
    
    Returns:
        int: 1 if positive (plagiarism/modification), 0 if negative pair.
    """
    if pd.isna(mod_type):
        return 0
    
    is_positive = not str(mod_type).startswith('Negative_')
    return 1 if is_positive else 0


def clean_mod_type(mod_type: str) -> str:
    """
    Removes the 'Negative_' prefix from modification type for categorization.
    
    Args:
        mod_type: Modification type string (may have 'Negative_' prefix).
    
    Returns:
        str: Cleaned modification type without 'Negative_' prefix.
    """
    if pd.isna(mod_type):
        return 'Unknown'
    
    mod_str = str(mod_type)
    if mod_str.startswith('Negative_'):
        return mod_str.replace('Negative_', '')
    return mod_str


def extract_dsp_and_source_features(mod_type: str) -> dict:
    """
    Extracts DSP modifications and source information from modification type.
    Uses regex to parse pitch, tempo, and audio source (Original, MusicGen, AudioLDM2, MGE-LDM).
    
    Args:
        mod_type: Modification type string (assumes 'Negative_' prefix already removed).
    
    Returns:
        dict: {
            'source': str (Original, MusicGen, AudioLDM2, MGE-LDM),
            'pitch_intensity': float (signed: negative for down, positive for up),
            'tempo_intensity': float (ratio: 1.0 = base, 0.9 = 90%, 1.1 = 110%),
            'is_extreme': bool (True if both pitch and tempo present),
            'dsp_category': str (Base Generation, Pure Modification, Extreme Up, Extreme Down, Mixed Extreme)
        }
    """
    if not isinstance(mod_type, str):
        return {
            'source': 'Original',
            'pitch_intensity': 0.0,
            'tempo_intensity': 1.0,
            'is_extreme': False,
            'dsp_category': 'Ignore'
        }
    
    mod_lower = mod_type.lower()
    
    # SOURCE EXTRACTION
    if mod_lower.startswith('smp_'):
        source = 'Cover'
    elif mod_lower.startswith('none_'):
        source = 'Original'
    elif 'musicgen' in mod_lower:
        source = 'MusicGen'
    elif 'audioldm2' in mod_lower:
        source = 'AudioLDM2'
    elif 'mgeldm' in mod_lower or 'mge-ldm' in mod_lower:
        source = 'MGE-LDM'
    else:
        source = 'Original'
    
    # PITCH EXTRACTION
    pitch_intensity = 0.0
    pitch_match = re.search(r'pitch([ud])(\d+)', mod_lower)
    if pitch_match:
        direction = pitch_match.group(1)
        value = float(pitch_match.group(2))
        pitch_intensity = value if direction == 'u' else -value
    
    # TEMPO EXTRACTION
    tempo_intensity = 1.0
    tempo_match = re.search(r'tempo(\d+)', mod_lower)
    if tempo_match:
        tempo_intensity = float(tempo_match.group(1)) / 100.0
    
    # DSP CATEGORY DETERMINATION
    is_extreme = bool(pitch_match and tempo_match)
    
    if pitch_intensity == 0.0 and tempo_intensity == 1.0:
        dsp_category = "Base Generation"
    elif is_extreme:
        if pitch_intensity > 0 and tempo_intensity > 1.0:
            dsp_category = "Extreme Up"
        elif pitch_intensity < 0 and tempo_intensity < 1.0:
            dsp_category = "Extreme Down"
        else:
            dsp_category = "Mixed Extreme"
    else:
        dsp_category = "Pure Modification"
    
    return {
        'source': source,
        'pitch_intensity': pitch_intensity,
        'tempo_intensity': tempo_intensity,
        'is_extreme': is_extreme,
        'dsp_category': dsp_category
    }


def get_broad_category(mod_type: str) -> str:
    """
    Assigns modification type to one of five broad categories:
    1a. Human Plagiarism (Base)
    1b. Human Plagiarism + DSP
    2. Original + DSP
    3. AI Generation (Base)
    4. AI + DSP
    """
    if pd.isna(mod_type):
        return 'Other'
    
    mod_lower = str(mod_type).lower()
    
    # Check for DSP
    has_dsp = bool(re.search(r'pitch[ud]\d+', mod_lower) or re.search(r'tempo\d+', mod_lower) or 'extreme' in mod_lower)
    
    # HUMAN PLAGIARISM
    if mod_lower.startswith('smp_'):
        if has_dsp:
            return '1b. Human Plagiarism + DSP'
        else:
            return '1a. Human Plagiarism (Base)'
    
    # ORIGINAL + DSP
    if mod_lower.startswith('none_'):
        return '2. Original + DSP'
    
    # AI DETECTION
    is_ai = any(ai in mod_lower for ai in ['musicgen', 'audioldm2', 'mgeldm', 'mge-ldm'])
    
    if is_ai:
        if has_dsp:
            return '4. AI + DSP'
        else:
            return '3. AI Generation (Base)'
    
    return 'Other'


def categorize_modification(mod_type: str) -> str:
    """
    Legacy wrapper for backward compatibility.
    Categorizes the modification type into granular groups.
    
    Args:
        mod_type: Modification type string (may have 'Negative_' prefix).
    
    Returns:
        str: Broad category name.
    """
    clean = clean_mod_type(mod_type)
    return get_broad_category(clean)


def extract_features(mod_type: str) -> pd.Series:
    """
    Legacy wrapper for backward compatibility.
    Extracts features from modification type as a pandas Series.
    Designed to be used with df.apply() for batch processing.
    
    Returns:
        pd.Series: [source, pitch_intensity, tempo_intensity, is_extreme, dsp_category]
    """
    if not isinstance(mod_type, str):
        return pd.Series(['Ignore', 0.0, 1.0, False, 'Ignore'])
    
    # Remove 'Negative_' prefix to group negatives with their respective positives
    clean = clean_mod_type(mod_type)
    
    # Extract features
    features = extract_dsp_and_source_features(clean)
    
    return pd.Series([
        features['source'],
        features['pitch_intensity'],
        features['tempo_intensity'],
        features['is_extreme'],
        features['dsp_category']
    ])
