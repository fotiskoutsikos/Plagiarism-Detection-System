"""
Analyze how distance changes with modification intensity.
Focus on pitch shift (semitones) and tempo change (factor).
"""

import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re
# Resolve repository root and load logging_util without relying on src package path
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent

logging_util_path = repo_root / "src" / "utils" / "logging_util.py"
spec = importlib.util.spec_from_file_location("logging_util", str(logging_util_path))
if spec is None or spec.loader is None:
    raise FileNotFoundError(f"Could not load logging_util from {logging_util_path}")
logging_util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_util)
setup_logging = logging_util.setup_logging

# Initialize logging for this script (logs/dsp_modifications.txt)
setup_logging(__file__)

def get_winning_metric(model_name, summary_path="results/threshold/threshold_analysis_summary.csv"):
    """Get the winning metric (Cosine or Euclidean)."""
    if os.path.exists(summary_path):
        try:
            df_summary = pd.read_csv(summary_path)
            row = df_summary[df_summary['model'].str.upper() == model_name.upper()]
            if not row.empty:
                metric = row.iloc[0]['metric'].lower()
                if 'euclidean' in metric:
                    return 'euclidean_distance'
                return 'cosine_distance'
        except Exception as e:
            print(f"Warning: Could not read metric for {model_name}. Defaulting to cosine_distance. Error: {e}")
    return 'cosine_distance' # Fallback

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

def compute_failure_rate(df, threshold, distance_col='cosine_distance'):
    """Compute percentage of samples where distance exceeds threshold (failure)."""
    if distance_col not in df.columns or len(df) == 0:
        return np.nan
    failures = (df[distance_col] > threshold).sum()
    return 100.0 * failures / len(df)


def compute_failure_rate_by_modification(df, threshold, distance_col='cosine_distance'):
    """Return a dataframe of failure rates by final_mod_type."""
    if distance_col not in df.columns or len(df) == 0:
        return pd.DataFrame(columns=['final_mod_type', 'distance_col', 'threshold', 'failure_rate'])

    table = []
    for mod_type, group in df.groupby('final_mod_type'):
        fail_rate = compute_failure_rate(group, threshold, distance_col)
        table.append({
            'final_mod_type': mod_type,
            'distance_col': distance_col,
            'threshold': threshold,
            'failure_rate': fail_rate,
            'n_samples': len(group),
        })
    return pd.DataFrame(table)


def compute_advanced_impact(df, threshold, winning_metric):
    """Compute failure rate, safety margin, and robustness index per modification type."""
    if winning_metric not in df.columns:
        raise ValueError(f"Winning metric '{winning_metric}' not found in df columns")

    results = []
    for mod_name, group in df.groupby('final_mod_type'):
        distances = group[winning_metric].values
        if len(distances) == 0:
            continue

        failures = np.sum(distances > threshold)
        failure_rate = (failures / len(distances)) * 100.0

        margins = threshold - distances
        mean_margin = np.mean(margins)

        robustness_score = (1.0 - (failure_rate / 100.0)) * (mean_margin / (threshold + 1e-9))

        results.append({
            'Modification': mod_name,
            'Count': len(distances),
            'Mean_Distance': np.mean(distances),
            'Failure_Rate_%': failure_rate,
            'Mean_Safety_Margin': mean_margin,
            'Robustness_Index': robustness_score,
            'Winning_Metric': winning_metric,
            'Threshold': threshold,
        })

    return pd.DataFrame(results)


def compute_neighborhood_similarity(df, distance_col, k=10):
    """Computes similarity using a strict Non-Parametric Empirical CDF (eCDF)."""
    df = df.copy()
    plag_df = df[~df['final_mod_type'].str.startswith('Negative')].copy()
    baseline = df[df['final_mod_type'].str.startswith('Negative')][distance_col].values
    
    if len(baseline) == 0:
        plag_df['relative_distance'] = 0.0
        plag_df['neighborhood_confidence'] = 0.0
        return plag_df
        
    mean_baseline = np.mean(baseline)
    
    # Sort the baseline distances for efficient ranking
    sorted_baseline = np.sort(baseline)
    n_baseline = len(sorted_baseline)
    
    # Find the rank of each plagiarized sample's distance within the baseline distribution
    indices = np.searchsorted(sorted_baseline, plag_df[distance_col].values)
    
    # Estimate neighborhood confidence as the proportion of baseline samples that are farther than the plagiarized sample
    plag_df['neighborhood_confidence'] = 1.0 - (indices / n_baseline)
    
    # Relative distance can be used as an additional feature (lower means more similar to original)
    plag_df['relative_distance'] = plag_df[distance_col] / (mean_baseline + 1e-9)
    
    return plag_df


def load_threshold_summary(summary_csv='results/threshold/threshold_analysis_summary.csv'):
    """Load model threshold results if available."""
    if not os.path.exists(summary_csv):
        return {}

    df_summary = pd.read_csv(summary_csv)
    mapping = {}
    for _, row in df_summary.iterrows():
        model = str(row.get('model', '')).upper()
        mapping[model] = {
            'threshold': float(row.get('optimal_threshold', np.nan)),
            'metric': row.get('metric', None)
        }
    return mapping


def analyze_intensity_correlation(df, distance_col='cosine_distance'):
    """
    Analyzes how DSP intensity correlates with Neighborhood Confidence (Relative Distance).
    Returns correlations dict and processed dataframe.
    """
    # Calculate neighborhood confidence 
    df_with_confidence = compute_neighborhood_similarity(df, distance_col)
    df_pos = df_with_confidence.copy()

    # Extract features
    df_pos['pitch_semitones'] = df_pos['final_mod_type'].apply(extract_pitch_semitones)
    df_pos['tempo_factor'] = df_pos['final_mod_type'].apply(extract_tempo_factor)
    df_pos['tempo_deviation'] = (df_pos['tempo_factor'] - 1.0).abs()

    correlations = {}

    # Analyse Pitch
    pitch_data = df_pos[df_pos['pitch_semitones'].notna()].copy()
    pitch_data['intensity'] = pitch_data['pitch_semitones'].abs()
    if len(pitch_data) > 5:
        corr, p_val = stats.pearsonr(pitch_data['intensity'], pitch_data['neighborhood_confidence'])
        correlations['pitch'] = {'correlation': corr, 'p_value': p_val, 'n_samples': len(pitch_data)}

    # Analyse Tempo
    tempo_data = df_pos[df_pos['tempo_factor'].notna()].copy()
    tempo_data['intensity'] = tempo_data['tempo_deviation']
    if len(tempo_data) > 5:
        corr, p_val = stats.pearsonr(tempo_data['intensity'], tempo_data['neighborhood_confidence'])
        correlations['tempo'] = {'correlation': corr, 'p_value': p_val, 'n_samples': len(tempo_data)}

    return correlations, df_pos

def plot_intensity_scatter(df, output_path, model_name):
    """Create scatter plots of Neighborhood Confidence vs raw modification magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    y_col = 'neighborhood_confidence'
    y_label = 'Neighborhood Confidence\n(Higher = More Certain)'
    
    # Pitch Scatter (-4, -2, +2, +4)
    pitch_data = df[df['pitch_semitones'].notna()].copy()
    if len(pitch_data) > 5:
        x_val = pitch_data['pitch_semitones']
        axes[0].scatter(x_val, pitch_data[y_col], alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
        
        z = np.polyfit(x_val, pitch_data[y_col], 2)
        p = np.poly1d(z)
        x_line = np.linspace(x_val.min(), x_val.max(), 100)
        
        axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label='Quadratic Trend')
        axes[0].axvline(x=0, color='blue', linestyle=':', linewidth=2, label='Original Pitch (0)')
        
        axes[0].set_xlabel('Pitch Shift (Semitones: -4 to +4)', fontsize=11)
        axes[0].set_ylabel(y_label, fontsize=11)
        axes[0].set_title('Confidence vs Full Pitch Spectrum', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    
    # Tempo Scatter (0.90, 0.95, 1.05, 1.10) 
    tempo_data = df[df['tempo_factor'].notna()].copy()
    if len(tempo_data) > 5:
        x_val = tempo_data['tempo_factor']
        axes[1].scatter(x_val, tempo_data[y_col], alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
        
        z = np.polyfit(x_val, tempo_data[y_col], 2)
        p = np.poly1d(z)
        x_line = np.linspace(x_val.min(), x_val.max(), 100)
        
        axes[1].plot(x_line, p(x_line), "r--", linewidth=2, label='Quadratic Trend')
        axes[1].axvline(x=1.0, color='blue', linestyle=':', linewidth=2, label='Original Tempo (1.0x)')
        
        axes[1].set_xlabel('Tempo Factor (e.g., 0.90x, 1.10x)', fontsize=11)
        axes[1].set_ylabel(y_label, fontsize=11)
        axes[1].set_title('Confidence vs Full Tempo Spectrum', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'{model_name} - Model Confidence Across DSP Spectrum', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Intensity plot saved to: {output_path}")

def main():
    print("=" * 70)
    print("MODIFICATION INTENSITY ANALYSIS")
    print("=" * 70)

    os.makedirs('results/dsp', exist_ok=True)
    os.makedirs('plots/dsp', exist_ok=True)

    threshold_map = load_threshold_summary()
    
    # Analyze CLEWS
    clews_path = "results/distances/clews_distances.csv"
    if os.path.exists(clews_path):
        print("\n[1/2] CLEWS Intensity Analysis")
        df_clews = pd.read_csv(clews_path)
        
        # df_clews = df_clews[~df_clews['final_mod_type'].str.startswith('Negative')]
        
        clews_metric = get_winning_metric("CLEWS")
        results_clews, df_clews_processed = analyze_intensity_correlation(df_clews, clews_metric)
        
        print("\nCLEWS Correlation Results:")
        for mod_type, stats_dict in results_clews.items():
            print(f"  {mod_type.upper()}:")
            print(f"    Correlation: {stats_dict['correlation']:.4f}")
            print(f"    P-value: {stats_dict['p_value']:.6f}")
            print(f"    N samples: {stats_dict['n_samples']}")
            significance = "[✓] Significant" if stats_dict['p_value'] < 0.05 else "[✗] Not significant"
            print(f"    Significance: {significance}")
        
        plot_intensity_scatter(df_clews_processed, 
                              'plots/dsp/clews_intensity_analysis.png',
                              'CLEWS')
        
        # Save results
        pd.DataFrame([
            {'model': 'CLEWS', 'modification': k, **v}
            for k, v in results_clews.items()
        ]).to_csv('results/dsp/clews_intensity_correlation.csv', index=False)

        clews_threshold = threshold_map.get('CLEWS', {}).get('threshold', None)
        clews_metric = threshold_map.get('CLEWS', {}).get('metric', 'cosine_distance')

        if isinstance(clews_metric, str):
            metric_clean = clews_metric.lower().replace(' ', '_')
            if metric_clean == 'cosine':
                metric_clean = 'cosine_distance'
            elif metric_clean in ['euclidean', 'euclidean_distance']:
                metric_clean = 'euclidean_distance'
            else:
                metric_clean = clews_metric
        else:
            metric_clean = 'cosine_distance'

        if clews_threshold is not None and not np.isnan(clews_threshold):
            fail_frames = []
            for col in ['cosine_distance', 'euclidean_distance']:
                if col in df_clews.columns:
                    df_fail = compute_failure_rate_by_modification(df_clews, clews_threshold, col)
                    if not df_fail.empty:
                        fail_frames.append(df_fail)

            if fail_frames:
                df_failures = pd.concat(fail_frames, axis=0, ignore_index=True)
                df_failures.to_csv('results/dsp/clews_failure_rate.csv', index=False)
                print(f"CLEWS failure rate saved to: results/dsp/clews_failure_rate.csv")
            else:
                print("CLEWS failure rate could not be computed; no valid distance columns.")

            # Advanced robustness analysis with safety margin
            try:
                df_advanced = compute_advanced_impact(df_clews, clews_threshold, metric_clean)
                df_advanced.to_csv('results/dsp/clews_failure_advanced.csv', index=False)
                print(f"CLEWS advanced failure analysis saved to: results/dsp/clews_failure_advanced.csv")
            except Exception as e:
                print(f"CLEWS advanced impact computation failed: {e}")

            # Neighborhood similarity
            try:
                df_neighborhood = compute_neighborhood_similarity(df_clews, metric_clean, k=10)
                df_neighborhood.to_csv('results/dsp/clews_neighborhood_similarity.csv', index=False)
                print(f"CLEWS neighborhood similarity saved to: results/dsp/clews_neighborhood_similarity.csv")
            except Exception as e:
                print(f"CLEWS neighborhood similarity failed: {e}")
    else:
        print(f"Warning: {clews_path} not found.")
    
    # Analyze WEALY
    wealy_path = "results/distances/wealy_distances.csv"
    if os.path.exists(wealy_path):
        print("\n[2/2] WEALY Intensity Analysis")
        df_wealy = pd.read_csv(wealy_path)

        # df_wealy = df_wealy[df_wealy['final_mod_type'] != 'Negative_Baseline']
        # df_wealy = df_wealy[~df_wealy['final_mod_type'].str.startswith('Negative')]

        wealy_metric = get_winning_metric("WEALY")
        results_wealy, df_wealy_processed = analyze_intensity_correlation(df_wealy, wealy_metric)
        
        print("\nWEALY Correlation Results:")
        for mod_type, stats_dict in results_wealy.items():
            print(f"  {mod_type.upper()}:")
            print(f"    Correlation: {stats_dict['correlation']:.4f}")
            print(f"    P-value: {stats_dict['p_value']:.6f}")
            print(f"    N samples: {stats_dict['n_samples']}")
            significance = "[✓] Significant" if stats_dict['p_value'] < 0.05 else "[✗] Not significant"
            print(f"    Significance: {significance}")
        
        plot_intensity_scatter(df_wealy_processed,
                              'plots/dsp/wealy_intensity_analysis.png',
                              'WEALY')
        
        # Save results
        pd.DataFrame([
            {'model': 'WEALY', 'modification': k, **v}
            for k, v in results_wealy.items()
        ]).to_csv('results/dsp/wealy_intensity_correlation.csv', index=False)

        wealy_threshold = threshold_map.get('WEALY', {}).get('threshold', None)
        wealy_metric = threshold_map.get('WEALY', {}).get('metric', 'cosine_distance')

        if isinstance(wealy_metric, str):
            metric_clean = wealy_metric.lower().replace(' ', '_')
            if metric_clean == 'cosine':
                metric_clean = 'cosine_distance'
            elif metric_clean in ['euclidean', 'euclidean_distance']:
                metric_clean = 'euclidean_distance'
            else:
                metric_clean = wealy_metric
        else:
            metric_clean = 'cosine_distance'

        if wealy_threshold is not None and not np.isnan(wealy_threshold):
            fail_frames = []
            for col in ['cosine_distance', 'euclidean_distance']:
                if col in df_wealy.columns:
                    df_fail = compute_failure_rate_by_modification(df_wealy, wealy_threshold, col)
                    if not df_fail.empty:
                        fail_frames.append(df_fail)
            if fail_frames:
                df_failures = pd.concat(fail_frames, axis=0, ignore_index=True)
                df_failures.to_csv('results/dsp/wealy_failure_rate.csv', index=False)
                print(f"WEALY failure rate saved to: results/dsp/wealy_failure_rate.csv")
            else:
                print("WEALY failure rate could not be computed; no valid distance columns.")

            # Advanced robustness analysis with safety margin
            try:
                df_advanced = compute_advanced_impact(df_wealy, wealy_threshold, metric_clean)
                df_advanced.to_csv('results/dsp/wealy_failure_advanced.csv', index=False)
                print(f"WEALY advanced failure analysis saved to: results/dsp/wealy_failure_advanced.csv")
            except Exception as e:
                print(f"WEALY advanced impact computation failed: {e}")

            # Neighborhood similarity
            try:
                df_neighborhood = compute_neighborhood_similarity(df_wealy, metric_clean, k=10)
                df_neighborhood.to_csv('results/dsp/wealy_neighborhood_similarity.csv', index=False)
                print(f"WEALY neighborhood similarity saved to: results/dsp/wealy_neighborhood_similarity.csv")
            except Exception as e:
                print(f"WEALY neighborhood similarity failed: {e}")
    else:
        print(f"Warning: {wealy_path} not found.")
    
    print("\n" + "=" * 70)
    print("INTENSITY ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()