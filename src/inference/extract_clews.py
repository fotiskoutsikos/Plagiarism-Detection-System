import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

# Placeholder for the CLEWS model import (commented out for now)
# from src.utils.clews_lib import Model as CLEWSModel


def load_audio(file_path, target_sr=16000):
    """Load audio from file, convert to mono, and resample if needed."""
    waveform, sr = torchaudio.load(file_path)

    # Convert stereo to mono by averaging channels if necessary
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform


def extract_embeddings(data_dir, checkpoint_path, config_path, output_parquet, device="cuda"):
    """Extract embeddings from audio files using the CLEWS model and save to Parquet."""

    # Load config and extract required settings
    conf = OmegaConf.load(config_path)
    target_sr = int(conf.data.samplerate)
    zdim = int(conf.model.zdim)

    # Instantiate and load model (commented out for now)
    # model = CLEWSModel(conf.model, sr=target_sr)
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    # model = model.to(device)
    # model.eval()

    # Collect WAV files in data_dir recursively
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))

    results = []

    # Inference mode ensures no gradients are tracked
    with torch.inference_mode():
        for file_path in tqdm(audio_files, desc="Extracting CLEWS embeddings"):
            try:
                waveform = load_audio(file_path, target_sr=target_sr)
                waveform = waveform.to(device)

                # Perform model forward pass (commented out for now)
                # z = model(waveform, shingle_hop=20.0, shingle_len=20.0)
                # z = z.squeeze().cpu().numpy()

                # Dummy vector for now so script can run without actual model
                dummy_vector = [0.0] * zdim

                results.append({
                    "filename": os.path.basename(file_path),
                    "embedding": dummy_vector,
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
    print(f"✅ Embeddings extraction complete. Saved to {output_parquet}")


if __name__ == "__main__":
    # Example usage with dummy paths
    data_dir = "../../data/segments/"
    checkpoint_path = "../../models/clews/checkpoint.pt"
    config_path = "../../configs/clews.yaml"
    output_parquet = "../../data/clews_embeddings.parquet"

    extract_embeddings(data_dir, checkpoint_path, config_path, output_parquet, device="cuda")
