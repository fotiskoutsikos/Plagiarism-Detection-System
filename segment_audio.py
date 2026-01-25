import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

INPUT_DIR = "data/raw_smp" 
OUTPUT_DIR = "data/processed_smp"
BEATS_PER_SEGMENT = 16

def segment_track(file_path, output_folder):
    """Takes a wav file, finds the beats and segments' it."""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_samples = librosa.frames_to_samples(beat_frames)
        
        # If not enough beats, skip beat-synced segmentation
        if len(beat_samples) < BEATS_PER_SEGMENT:
            # Fallback
            print(f"Warning: Low beats detected in {file_path}. Skipping beat sync.")
            return

        # Segmenting
        num_segments = 0
        for i in range(0, len(beat_samples) - BEATS_PER_SEGMENT, BEATS_PER_SEGMENT):
            start_sample = beat_samples[i]
            end_sample = beat_samples[i + BEATS_PER_SEGMENT]
            
            segment = y[start_sample:end_sample]
            
            # Skip segments shorter than 2 seconds
            if len(segment) / sr < 2.0:
                continue

            # Save segment
            seg_filename = f"{num_segments}.wav"
            sf.write(os.path.join(output_folder, seg_filename), segment, sr)
            num_segments += 1
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Couldn't find the file {INPUT_DIR}")
        return

    # Retrieve all pairs
    pairs = sorted([p for p in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, p))])
    
    print(f"Starting Segmentation for all {len(pairs)} pairs...")

    for pair in tqdm(pairs):
        pair_input_path = os.path.join(INPUT_DIR, pair)
        pair_output_path = os.path.join(OUTPUT_DIR, pair)
        
        # For each file inside the pairs (original.wav, suspicious.wav)
        for wav_file in os.listdir(pair_input_path):
            if not wav_file.endswith(".wav"):
                continue
                
            # Create Subfile: data/processed_smp/pair_0/original/
            version_name = os.path.splitext(wav_file)[0] 
            version_output_folder = os.path.join(pair_output_path, version_name)
            
            os.makedirs(version_output_folder, exist_ok=True)
            
            # Segment the track
            input_wav_path = os.path.join(pair_input_path, wav_file)
            segment_track(input_wav_path, version_output_folder)

    print(f"\nSegmentation Finished! You can fiind them at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
