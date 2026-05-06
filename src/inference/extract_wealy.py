import os
import torch
import torchaudio
import pandas as pd
import pyarrow.parquet as pq
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

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze()


def atomic_save_parquet(df, output_parquet):
    """Safely save parquet using temp file + atomic replace."""
    output_dir = os.path.dirname(output_parquet)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = output_parquet + ".tmp"
    df.to_parquet(tmp_path, engine="pyarrow", index=False)
    os.replace(tmp_path, output_parquet)


def parquet_num_rows(path):
    """Get row count from parquet metadata without fully loading it."""
    return pq.ParquetFile(path).metadata.num_rows


def collect_audio_files(target_folders):
    """Collect and report all .wav files from target folders."""
    audio_files = []

    for folder in target_folders:
        if os.path.exists(folder):
            folder_files = []
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(".wav"):
                        folder_files.append(os.path.join(root, file))
            folder_files.sort()
            print(f"{folder}: {len(folder_files)} wav files")
            audio_files.extend(folder_files)
        else:
            print(f"Warning: The folder {folder} does not exist.")

    audio_files.sort()
    print(f"TOTAL wav files discovered: {len(audio_files)}")
    return audio_files


def flush_results(results, existing_df, output_parquet, tag="WEALY"):
    """Append buffered results to existing dataframe and save atomically."""
    if not results:
        return existing_df

    temp_new_df = pd.DataFrame(results)

    if existing_df is not None and not existing_df.empty:
        checkpoint_df = pd.concat([existing_df, temp_new_df], ignore_index=True)
    else:
        checkpoint_df = temp_new_df

    atomic_save_parquet(checkpoint_df, output_parquet)
    saved_rows = parquet_num_rows(output_parquet)

    tqdm.write(
        f"[Checkpoint:{tag}] Saved {saved_rows} rows -> {os.path.abspath(output_parquet)}"
    )

    return checkpoint_df


def extract_wealy_embeddings(data_dir, wealy_checkpoint, wealy_config, output_parquet, device="cuda"):
    """Extract WEALY embeddings from audio files using Whisper decoder latents."""

    print(f"Output parquet: {os.path.abspath(output_parquet)}")

    conf = OmegaConf.load(wealy_config)

    print(f"Loading Whisper model on {device}...")
    whisper_model = whisper.load_model("turbo", device=device)

    decoder_hidden_states.clear()
    hook = whisper_model.decoder.ln.register_forward_hook(hook_fn)

    print("Initializing WEALY model...")
    wealy_model = WEALYModel(conf.model)

    checkpoint = torch.load(wealy_checkpoint, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k.replace("module.", "", 1)] = v
        else:
            clean_state_dict[k] = v

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
        # "data/segment_smp/audio",                # segments from the original SMP dataset
        # "data/dsp_variants/audio",               # DSP variants (smp)
        # "data/generated_audio/musicgen",         # generated audio files (musicgen)
        # "data/dsp_variants/musicgen",            # DSP variants (musicgen)
        # "data/generated_audio/audioldm2",        # generated audio files (audioldm2)
        # "data/dsp_variants/audioldm2",           # DSP variants (audioldm2)
        "data/generated_audio/mgeldm",           # generated audio files (mgeldm)
        "data/dsp_variants/mgeldm"               # DSP variants (mgeldm)
    ]

    audio_files = collect_audio_files(target_folders)
    if not audio_files:
        hook.remove()
        print("No audio files found. Exiting.")
        return

    existing_df = None
    processed_files = set()

    if os.path.exists(output_parquet):
        print(f"Loading existing parquet file: {output_parquet}")
        existing_df = pd.read_parquet(output_parquet)

        if "filename" not in existing_df.columns:
            hook.remove()
            raise ValueError(f"'filename' column not found in {output_parquet}")

        processed_files = set(existing_df["filename"].astype(str).tolist())
        print(f"Found {len(processed_files)} already processed files.")
        print(f"Existing parquet rows on disk: {parquet_num_rows(output_parquet)}")

    audio_files = [f for f in audio_files if os.path.basename(f) not in processed_files]
    print(f"Remaining files to process: {len(audio_files)}")

    results = []
    checkpoint_every = 200

    options = whisper.DecodingOptions(
        task="transcribe",
        language=None,
        without_timestamps=True,
        fp16=False,
    )

    try:
        with torch.no_grad():
            for file_path in tqdm(audio_files, desc="Extracting WEALY embeddings"):
                try:
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                except OSError:
                    tqdm.write(f"[Audio Error] Couldn't find: {file_path}")
                    continue

                if file_size_mb > 3.0 or file_size_mb == 0:
                    continue

                try:
                    audio_waveform = load_audio(file_path, target_sr=16000).to(device)

                    audio_padded = whisper.pad_or_trim(audio_waveform.flatten())
                    mel = whisper.log_mel_spectrogram(
                        audio_padded, n_mels=whisper_model.dims.n_mels
                    ).to(device)
                    mel = mel.unsqueeze(0)

                    decoder_hidden_states.clear()
                    _ = whisper.decode(whisper_model, mel, options)

                    if len(decoder_hidden_states) == 0:
                        continue

                    decoder_latents = decoder_hidden_states[-1].clone().to(device)

                    if len(decoder_latents.shape) == 2:
                        decoder_latents = decoder_latents.unsqueeze(0)

                    embedding = wealy_model(decoder_latents)
                    z_vector = embedding.squeeze(0).cpu().numpy()

                    results.append({
                        "filename": os.path.basename(file_path),
                        "embedding": z_vector.tolist(),
                    })

                except Exception as e:
                    tqdm.write(f"[Audio Error] Failed to process {os.path.basename(file_path)}: {e}")
                    continue

                if len(results) >= checkpoint_every:
                    try:
                        existing_df = flush_results(
                            results=results,
                            existing_df=existing_df,
                            output_parquet=output_parquet,
                            tag="WEALY"
                        )
                        results = []
                    except Exception as e:
                        raise RuntimeError(f"[CRITICAL I/O ERROR] Failed to save WEALY checkpoint: {e}")

    finally:
        hook.remove()

    if results:
        try:
            existing_df = flush_results(
                results=results,
                existing_df=existing_df,
                output_parquet=output_parquet,
                tag="WEALY-FINAL"
            )
            print(
                f"WEALY extraction complete. Total rows in {output_parquet}: "
                f"{parquet_num_rows(output_parquet)}"
            )
        except Exception as e:
            raise RuntimeError(f"[CRITICAL I/O ERROR] Final WEALY save failed: {e}")

    elif existing_df is not None:
        print("No new files processed. Existing parquet remains unchanged.")
    else:
        print("No embeddings were created.")


if __name__ == "__main__":
    data_dir = "data/segment_smp/audio/"  # kept only for compatibility
    wealy_checkpoint = "models/wealy/checkpoint.pt"
    wealy_config = "configs/extraction/wealy.yaml"
    output_parquet = "data/wealy_mgeldm_embeddings.parquet"

    extract_wealy_embeddings(
        data_dir=data_dir,
        wealy_checkpoint=wealy_checkpoint,
        wealy_config=wealy_config,
        output_parquet=output_parquet,
        device="cuda",
    )