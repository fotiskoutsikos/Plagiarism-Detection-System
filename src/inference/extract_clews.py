import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

from src.utils.clews_lib import Model as CLEWSModel


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

    # Initialize CLEWS model
    model = CLEWSModel(conf.model)
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            clean_state_dict[k.replace("model.", "", 1)] = v
        else:
            clean_state_dict[k] = v

    # Load model
    model.load_state_dict(clean_state_dict, strict=False)
    model = model.to(device)
    model.eval()

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
    checkpoint_every = 200 

    with torch.no_grad():
        for i, file_path in enumerate(tqdm(audio_files, desc="Extracting CLEWS embeddings")):
            try:
                waveform = load_audio(file_path, target_sr=target_sr)
                waveform = waveform.to(device)

                z = model(waveform, shingle_hop=20.0, shingle_len=20.0)
                z_vector = z.squeeze(0).cpu().numpy()

                results.append({
                    "filename": os.path.basename(file_path),
                    "embedding": z_vector.tolist(),
                })

                # CHECKPOINTING
                if (i + 1) % checkpoint_every == 0:
                    temp_new_df = pd.DataFrame(results)
                    if existing_df is not None:
                        checkpoint_df = pd.concat([existing_df, temp_new_df], ignore_index=True)
                    else:
                        checkpoint_df = temp_new_df
                    
                    checkpoint_df.to_parquet(output_parquet, engine="pyarrow")
                    print(f"\n[Checkpoint] Saved {len(checkpoint_df)} embeddings so far...")

            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                continue

    # Save results to Parquet file
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

    output_dir = os.path.dirname(output_parquet)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(output_parquet, engine="pyarrow")
    print(f"CLEWS embedding extraction complete. Saved {len(df)} embeddings to {output_parquet}")


if __name__ == "__main__":
    # Paths relative to project root
    data_dir = "data/segment_smp/audio/"
    checkpoint_path = "models/clews/checkpoint.pt"
    config_path = "configs/extraction/clews.yaml"
    output_parquet = "data/clews_embeddings.parquet"

    extract_embeddings(data_dir, checkpoint_path, config_path, output_parquet, device="cuda")

