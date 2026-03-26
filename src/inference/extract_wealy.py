import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
import whisper
from omegaconf import OmegaConf

from src.utils.wealy_lib import Model as WEALYModel


def load_audio(file_path, target_sr=16000):
    """Load audio file, convert to mono, and resample to target_sr."""
    waveform, sr = torchaudio.load(file_path)

    # Convert to mono by averaging channels
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Return 1D waveform
    return waveform.squeeze()


def extract_wealy_embeddings(data_dir, wealy_checkpoint, wealy_config, output_parquet, device="cuda"):
    """Extract WEALY embeddings from audio files using Whisper feature extraction."""

    # Load WEALY config
    conf = OmegaConf.load(wealy_config)
    working_dim = int(conf.model.working_dim)

    # Load Whisper model on the specified device
    print(f"Loading Whisper model on {device}...")
    whisper_model = whisper.load_model("base", device=device)

    # Initialize WEALY model
    print("Initializing WEALY model...")
    wealy_model = WEALYModel(conf.model)

    # Load WEALY checkpoint
    checkpoint = torch.load(wealy_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    clean_state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}

    wealy_model.load_state_dict(clean_state_dict, strict=False)
    wealy_model = wealy_model.to(device)
    wealy_model.eval()
    whisper_model.eval()

    # Define target folders to search for audio files
    target_folders = [
        "data/segment_smp/audio",               # segments from the original SMP dataset
        "data/generated_audio/musicgen",        # generated audio files
        "data/dsp_variants/musicgen"            # DSP variants
    ]

    audio_files = []
    
    # Search only in the specified folders
    for folder in target_folders:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(".wav"):
                        audio_files.append(os.path.join(root, file))
        else:
            print(f"Warning: The folder {folder} does not exist and will be ignored.")

    # Load existing parquet if it exists to resume from where we left off
    existing_df = None
    processed_files = set()
    if os.path.exists(output_parquet):
        print(f"Loading existing parquet file: {output_parquet}")
        existing_df = pd.read_parquet(output_parquet)
        processed_files = set(existing_df["filename"].tolist())
        print(f"Found {len(processed_files)} already processed files.")

    # Filter out already processed files
    audio_files = [f for f in audio_files if os.path.basename(f) not in processed_files]
    print(f"Remaining files to process: {len(audio_files)}")

    results = []

    # Inference loop with no gradients
    with torch.no_grad():
        for file_path in tqdm(audio_files, desc="Extracting WEALY embeddings"):
            try:
                # Load and prepare audio
                audio_waveform = load_audio(file_path, target_sr=16000)
                audio_waveform = audio_waveform.to(device)

                # Pad or trim to 30 seconds (Whisper requirement)
                audio_padded = whisper.pad_or_trim(audio_waveform.flatten())

                # Convert to log-mel spectrogram
                mel = whisper.log_mel_spectrogram(audio_padded).to(device)

                # Add batch dimension
                mel = mel.unsqueeze(0)  # Shape: (1, n_mels, time_steps)

                # Extract Whisper encoder features
                audio_features = whisper_model.encoder(mel)  # Shape: (1, time_steps, encoder_dim)

                # Pass through WEALY model to get final embedding
                embedding = wealy_model(audio_features)

                # Force shape [1, 512] -> [512]
                z_vector = embedding.squeeze(0).cpu().numpy()

                results.append({
                    "filename": os.path.basename(file_path),
                    "embedding": z_vector.tolist(),
                })

            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                continue

    # Convert results to DataFrame
    if results:
        new_df = pd.DataFrame(results)
        
        # Append to existing DataFrame if it exists
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"Appended {len(new_df)} new embeddings to existing {len(existing_df)} embeddings.")
        else:
            df = new_df
    else:
        # No new results, use existing if available
        if existing_df is not None:
            df = existing_df
            print("No new files to process. Using existing embeddings.")
        else:
            df = pd.DataFrame()
            print("No embeddings found.")

    # Create output directory if needed
    output_dir = os.path.dirname(output_parquet)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save to Parquet
    df.to_parquet(output_parquet, engine="pyarrow")
    print(f"WEALY embedding extraction complete. Saved {len(df)} embeddings to {output_parquet}")


if __name__ == "__main__":
    # Paths relative to project root
    data_dir = "data/segment_smp/audio/"
    wealy_checkpoint = "models/wealy/checkpoint.pt"
    wealy_config = "configs/extraction/wealy.yaml"
    output_parquet = "data/wealy_embeddings.parquet"

    extract_wealy_embeddings(data_dir, wealy_checkpoint, wealy_config, output_parquet, device="cuda")
