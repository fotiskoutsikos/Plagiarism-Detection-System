#!/usr/bin/env python3
"""
Analyze how distance changes with modification intensity.
Focus on pitch shift (semitones) and tempo change (factor).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re

def extract_pitch_semitones(mod_type):
    """Extract pitch shift in semitones from modification type."""
    if not isinstance(mod_type, str):
        return np.nan
    
    # Match patterns like pitchU2, pitchD4, pitchU2_tempo0.9
    match = re.search(r'pitch([UD])(\d+)', mod_type)
    if match:
        direction = match.group(1)
        semitones = int(match.group(2))
        return semitones if direction == 'U' else -semitones
    return np.nan

def extract_tempo_factor(mod_type):
    """Extract tempo change factor from modification type."""
    if not isinstance(mod_type, str):
        return np.nan
    
    # Match patterns like tempo090, tempo110
    match = re.search(r'tempo(\d+)', mod_type)
    if match:
        return float(match.group(1)) / 100.0
    return np.nan

def analyze_intensity_correlation(df, distance_col='cosine_distance'):
    """Compute correlation between distance and modification intensity."""
    df = df.copy()
    
    # Extract intensity features
    df['pitch_semitones'] = df['final_mod_type'].apply(extract_pitch_semitones)
    df['tempo_factor'] = df['final_mod_type'].apply(extract_tempo_factor)
    df['pitch_abs'] = df['pitch_semitones'].abs()
    df['tempo_deviation'] = (df['tempo_factor'] - 1.0).abs()
    
    results = {}
    
    # Pitch correlation (absolute value)
    pitch_data = df[df['pitch_semitones'].notna()]
    if len(pitch_data) > 10:
        corr_pitch, p_pitch = stats.pearsonr(pitch_data['pitch_abs'], 
                                              pitch_data[distance_col])
        results['pitch'] = {
            'correlation': corr_pitch,
            'p_value': p_pitch,
            'n_samples': len(pitch_data)
        }
    
    # Tempo correlation
    tempo_data = df[df['tempo_factor'].notna()]
    if len(tempo_data) > 10:
        corr_tempo, p_tempo = stats.pearsonr(tempo_data['tempo_deviation'],
                                              tempo_data[distance_col])
        results['tempo'] = {
            'correlation': corr_tempo,
            'p_value': p_tempo,
            'n_samples': len(tempo_data)
        }
    
    return results, df

def plot_intensity_scatter(df, output_path, model_name):
    """Create scatter plots of distance vs intensity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pitch scatter
    pitch_data = df[df['pitch_semitones'].notna()].copy()
    if len(pitch_data) > 5:
        axes[0].scatter(pitch_data['pitch_semitones'], 
                       pitch_data['cosine_distance'],
                       alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
        
        # Add regression line
        z = np.polyfit(pitch_data['pitch_semitones'].abs(), 
                      pitch_data['cosine_distance'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, pitch_data['pitch_semitones'].abs().max(), 100)
        axes[0].plot(x_line, p(x_line), "r--", linewidth=2, 
                    label=f'Trend (R²={np.corrcoef(pitch_data["pitch_semitones"].abs(), pitch_data["cosine_distance"])[0,1]**2:.3f})')
        
        axes[0].set_xlabel('Pitch Shift (semitones)', fontsize=11)
        axes[0].set_ylabel('Cosine Distance', fontsize=11)
        axes[0].set_title('Distance vs Pitch Shift', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Tempo scatter
    tempo_data = df[df['tempo_factor'].notna()].copy()
    if len(tempo_data) > 5:
        axes[1].scatter(tempo_data['tempo_deviation'],
                       tempo_data['cosine_distance'],
                       alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
        
        # Add regression line
        z = np.polyfit(tempo_data['tempo_deviation'],
                      tempo_data['cosine_distance'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, tempo_data['tempo_deviation'].max(), 100)
        axes[1].plot(x_line, p(x_line), "r--", linewidth=2,
                    label=f'Trend (R²={np.corrcoef(tempo_data["tempo_deviation"], tempo_data["cosine_distance"])[0,1]**2:.3f})')
        
        axes[1].set_xlabel('Tempo Deviation from 1.0', fontsize=11)
        axes[1].set_ylabel('Cosine Distance', fontsize=11)
        axes[1].set_title('Distance vs Tempo Change', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.suptitle(f'{model_name} - Distance vs Modification Intensity', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Intensity plot saved to: {output_path}")

def main():
    print("=" * 70)
    print("MODIFICATION INTENSITY ANALYSIS")
    print("=" * 70)
    
    # Analyze CLEWS
    clews_path = "data/clews_distances.csv"
    if os.path.exists(clews_path):
        print("\n[1/2] CLEWS Intensity Analysis")
        df_clews = pd.read_csv(clews_path)
        
        df_clews = df_clews[~df_clews['final_mod_type'].str.startswith('Negative')]
        
        results_clews, df_clews_processed = analyze_intensity_correlation(df_clews, 'cosine_distance')
        
        print("\nCLEWS Correlation Results:")
        for mod_type, stats_dict in results_clews.items():
            print(f"  {mod_type.upper()}:")
            print(f"    Correlation: {stats_dict['correlation']:.4f}")
            print(f"    P-value: {stats_dict['p_value']:.6f}")
            print(f"    N samples: {stats_dict['n_samples']}")
            significance = "✓ Significant" if stats_dict['p_value'] < 0.05 else "✗ Not significant"
            print(f"    Significance: {significance}")
        
        plot_intensity_scatter(df_clews_processed, 
                              'plots/clews_intensity_analysis.png',
                              'CLEWS')
        
        # Save results
        pd.DataFrame([
            {'model': 'CLEWS', 'modification': k, **v}
            for k, v in results_clews.items()
        ]).to_csv('results/clews_intensity_correlation.csv', index=False)
    else:
        print(f"Warning: {clews_path} not found.")
    
    # Analyze WEALY
    wealy_path = "data/wealy_distances.csv"
    if os.path.exists(wealy_path):
        print("\n[2/2] WEALY Intensity Analysis")
        df_wealy = pd.read_csv(wealy_path)
        df_wealy = df_wealy[df_wealy['final_mod_type'] != 'Negative_Baseline']
        
        df_wealy = df_wealy[~df_wealy['final_mod_type'].str.startswith('Negative')]

        results_wealy, df_wealy_processed = analyze_intensity_correlation(df_wealy)
        
        print("\nWEALY Correlation Results:")
        for mod_type, stats_dict in results_wealy.items():
            print(f"  {mod_type.upper()}:")
            print(f"    Correlation: {stats_dict['correlation']:.4f}")
            print(f"    P-value: {stats_dict['p_value']:.6f}")
            print(f"    N samples: {stats_dict['n_samples']}")
            significance = "✓ Significant" if stats_dict['p_value'] < 0.05 else "✗ Not significant"
            print(f"    Significance: {significance}")
        
        plot_intensity_scatter(df_wealy_processed,
                              'plots/wealy_intensity_analysis.png',
                              'WEALY')
        
        # Save results
        pd.DataFrame([
            {'model': 'WEALY', 'modification': k, **v}
            for k, v in results_wealy.items()
        ]).to_csv('results/wealy_intensity_correlation.csv', index=False)
    else:
        print(f"Warning: {wealy_path} not found.")
    
    print("\n" + "=" * 70)
    print("INTENSITY ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()