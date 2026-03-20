import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import whisper
from omegaconf import OmegaConf

# Placeholder for the WEALY model import (commented out for now)
# from src.utils.wealy_lib import Model as WEALYModel


def load_audio_numpy(file_path, target_sr=16000):
    """Load audio from file, convert to mono, resample, and return 1D numpy array."""
    waveform, sr = torchaudio.load(file_path)

    # Convert stereo to mono by averaging channels if necessary
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Return as 1D numpy float32 array (squeeze to remove channel dim)
    return waveform.squeeze().numpy().astype('float32')


def extract_wealy_embeddings(data_dir, wealy_checkpoint, wealy_config, output_parquet, device="cuda"):
    """Extract WEALY embeddings from audio files using Whisper hidden states."""

    # Load WEALY config
    conf = OmegaConf.load(wealy_config)

    # Load Whisper model
    whisper_model = whisper.load_model("turbo", device=device)

    # Initialize WEALY model (commented out for now)
    # wealy_model = WEALYModel(conf.model)
    # checkpoint = torch.load(wealy_checkpoint, map_location='cpu')
    # wealy_model.load_state_dict(checkpoint['state_dict'], strict=False)
    # wealy_model = wealy_model.to(device)
    # wealy_model.eval()

    # Collect WAV files in data_dir recursively
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))

    results = []

    # Inference loop
    for file_path in tqdm(audio_files, desc="Extracting WEALY embeddings"):
        try:
            # Load audio as numpy array
            audio_np = load_audio_numpy(file_path, target_sr=16000)

            # Transcribe with Whisper to get hidden states
            result = whisper_model.transcribe(audio_np, language="en")

            # Extract and flatten hidden states
            all_states = [state for chunk in result["frames_last_hidden_states"] for state in chunk]
            whisper_features = torch.cat(all_states, dim=0)  # Concatenate along time dim
            whisper_features = whisper_features.unsqueeze(0).to(device)  # Add batch dim: (1, Time, 1280)

            # Pass to WEALY model (commented out for now)
            # z, _ = wealy_model.embed(whisper_features)
            # z_vector = z.squeeze().cpu().numpy()

            # Dummy vector for now (512-dimensional)
            z_vector = [0.0] * 512

            results.append({
                "filename": os.path.basename(file_path),
                "embedding": z_vector,
            })

        except Exception as e:
            print(f"Warning: failed to process {file_path}: {e}")
            continue

    # Save results in a single Parquet file
    df = pd.DataFrame(results)

    output_dir = os.path.dirname(output_parquet)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(output_parquet, engine='pyarrow')
    print(f"WEALY embeddings extraction complete. Saved to {output_parquet}")


if __name__ == "__main__":
    # Example usage with dummy paths
    data_dir = "../../data/segments/"
    wealy_checkpoint = "../../models/wealy/checkpoint.pt"
    wealy_config = "../../configs/wealy.yaml"
    output_parquet = "../../data/wealy_embeddings.parquet"

    extract_wealy_embeddings(data_dir, wealy_checkpoint, wealy_config, output_parquet, device="cuda")