import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
import whisper
from omegaconf import OmegaConf
from src.utils.wealy_lib import Model as WEALYModel

decoder_hidden_states = []

def hook_fn(module, input, output):
    """Hook to capture ONLY the last decoder hidden states."""
    if isinstance(output, tuple):
        decoder_hidden_states.append(output[0].detach().cpu())
    else:
        decoder_hidden_states.append(output.detach().cpu())

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
    """Extract WEALY embeddings from audio files using Whisper DECODER latents."""
    # Load WEALY config
    conf = OmegaConf.load(wealy_config)

    # Load Whisper model on the specified device
    print(f"Loading Whisper model on {device}...")
    whisper_model = whisper.load_model("turbo", device=device)

    # Register hook for decoder hidden states
    decoder_hidden_states.clear()
    hook = whisper_model.decoder.ln.register_forward_hook(hook_fn)
    # Initialize WEALY model
    print("Initializing WEALY model...")
    wealy_model = WEALYModel(conf.model)

    # Load WEALY checkpoint
    checkpoint = torch.load(wealy_checkpoint, map_location="cpu", weights_only=False)

    # Extract model state dict from checkpoint structure
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean DDP prefixes if they exist
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k.replace("module.", "", 1)] = v
        else:
            clean_state_dict[k] = v

    # Load model, with fallback for missing buffers
    try:
        wealy_model.load_state_dict(clean_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: strict load_state_dict failed, retrying with strict=False: {e}")
        missing_buffers = [k for k in wealy_model.state_dict().keys() if k not in clean_state_dict]
        if missing_buffers:
            print(f"Filling missing keys from model init: {missing_buffers}")
            for k in missing_buffers:
                clean_state_dict[k] = wealy_model.state_dict()[k]
        wealy_model.load_state_dict(clean_state_dict, strict=False)

    wealy_model = wealy_model.to(device)
    wealy_model.eval()
    whisper_model.eval()

    # Define target folders to search for audio files
    target_folders = [
        "data/segment_smp/audio",                # segments from the original SMP dataset
        "data/dsp_variants/audio",               # DSP variants (smp)
        "data/generated_audio/musicgen",         # generated audio files (musicgen)
        "data/dsp_variants/musicgen",            # DSP variants (musicgen)
        "data/generated_audio/audioldm2",        # generated audio files (audioldm2)
        "data/dsp_variants/audioldm2"            # DSP variants (audioldm2)
        # "data/generated_audio/mgeldm",          # generated audio files (mgeldm)
        # "data/dsp_variants/mgeldm"              # DSP variants (mgeldm)
    ]

    audio_files = []
    for folder in target_folders:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(".wav"):
                        audio_files.append(os.path.join(root, file))
        else:
            print(f"Warning: The folder {folder} does not exist.")

    # Load existing parquet to resume
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

    # Decoding Options for Whisper
    options = whisper.DecodingOptions(
        task="transcribe",
        language=None, # Το αφήνουμε None για auto-detect
        without_timestamps=True,
        fp16=False
    )

    with torch.no_grad():
        for i, file_path in enumerate(tqdm(audio_files, desc="Extracting WEALY embeddings")):
            try:
                # Load and prepare audio
                audio_waveform = load_audio(file_path, target_sr=16000)
                audio_waveform = audio_waveform.to(device)

                # Pad or trim to 30 seconds (Whisper requirement)
                audio_padded = whisper.pad_or_trim(audio_waveform.flatten())
                mel = whisper.log_mel_spectrogram(audio_padded, n_mels=whisper_model.dims.n_mels).to(device)
                mel = mel.unsqueeze(0)

                decoder_hidden_states.clear()

                result = whisper.decode(whisper_model, mel, options)
                
                if len(decoder_hidden_states) == 0:
                    print(f"\nWarning: No hidden states captured for {file_path}, SKIPPING")
                    continue

                decoder_latents = decoder_hidden_states[-1].clone()
                decoder_latents = decoder_latents.to(device)

                if len(decoder_latents.shape) == 2:
                    decoder_latents = decoder_latents.unsqueeze(0)

                # Pass decoder latents through WEALY
                embedding = wealy_model(decoder_latents)  # [1, 512]
                z_vector = embedding.squeeze(0).cpu().numpy()

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
                    
                    output_dir = os.path.dirname(output_parquet)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                        
                    checkpoint_df.to_parquet(output_parquet, engine="pyarrow")
                    print(f"\n[Checkpoint] Saved {len(checkpoint_df)} embeddings so far...")

            except Exception as e:
                print(f"\nWarning: Failed to process {file_path}: {e}")
                continue

    # Remove hook
    hook.remove()

    # Final save to capture everything
    if results:
        new_df = pd.DataFrame(results)
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df
        
        output_dir = os.path.dirname(output_parquet)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        df.to_parquet(output_parquet, engine="pyarrow")
        print(f"WEALY extraction complete. Total: {len(df)} embeddings in {output_parquet}")
    elif existing_df is not None:
        print("No new files processed. Existing parquet remains unchanged.")

if __name__ == "__main__":
    # Paths relative to project root
    data_dir = "data/segment_smp/audio/"
    wealy_checkpoint = "models/wealy/checkpoint.pt"
    wealy_config = "configs/extraction/wealy.yaml"
    output_parquet = "data/wealy_embeddings.parquet"
    extract_wealy_embeddings(data_dir, wealy_checkpoint, wealy_config, output_parquet, device="cuda")
