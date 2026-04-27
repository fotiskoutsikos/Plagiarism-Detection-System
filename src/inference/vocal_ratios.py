"""
Standalone VAD script for already-processed audio files.
Reads wav files directly — does NOT re-run Whisper or WEALY.
Outputs a CSV: filename | vocal_ratio | vocal_valid
"""

import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm


# VAD

def vad_vocal_ratio(
    waveform: torch.Tensor,
    sr: int = 16_000,
    frame_ms: int = 20,
    energy_threshold_db: float = -40.0,
    freq_low: int = 80,
    freq_high: int = 3_400,
) -> float:
    """
    Lightweight energy + spectral VAD. No external model needed.
    Returns fraction of active frames whose energy is in the speech band.
    """
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    wav = waveform.float().cpu()

    frame_len = int(sr * frame_ms / 1000)
    n_frames  = len(wav) // frame_len
    if n_frames == 0:
        return 0.0

    frames  = wav[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms     = frames.pow(2).mean(dim=1).sqrt()
    rms_db  = 20 * torch.log10(rms.clamp(min=1e-9))
    active  = rms_db > energy_threshold_db

    total_active = int(active.sum().item())
    if total_active == 0:
        return 0.0

    window = torch.hann_window(frame_len)
    freqs  = torch.fft.rfftfreq(frame_len, d=1.0 / sr)
    speech_mask = (freqs >= freq_low) & (freqs <= freq_high)

    speech_frames = 0
    for idx in range(n_frames):
        if not active[idx]:
            continue
        spec           = torch.fft.rfft(frames[idx] * window).abs()
        speech_energy  = spec[speech_mask].pow(2).sum()
        total_energy   = spec.pow(2).sum().clamp(min=1e-9)
        if (speech_energy / total_energy).item() >= 0.50:
            speech_frames += 1

    return speech_frames / total_active


# MAIN

VOCAL_RATIO_THRESHOLD = 0.30

TARGET_FOLDERS = [
    "data/segment_smp/audio",
    "data/dsp_variants/audio",
    "data/generated_audio/musicgen",
    "data/dsp_variants/musicgen",
    "data/generated_audio/audioldm2",
    "data/dsp_variants/audioldm2",
]

OUTPUT_CSV      = "data/vocal_ratios.csv"
CHECKPOINT_EVERY = 500


def main():
    # Collect all wav files
    audio_files = []
    for folder in TARGET_FOLDERS:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(".wav"):
                        audio_files.append(os.path.join(root, f))
        else:
            print(f"[WARN] Folder not found, skipping: {folder}")

    print(f"Found {len(audio_files)} wav files total.")

    # Resume support — skip already processed files
    existing_df     = None
    processed_files = set()
    if os.path.exists(OUTPUT_CSV):
        existing_df     = pd.read_csv(OUTPUT_CSV)
        processed_files = set(existing_df["filename"].tolist())
        print(f"Resuming — {len(processed_files)} files already processed.")

    audio_files = [f for f in audio_files if os.path.basename(f) not in processed_files]
    print(f"Remaining: {len(audio_files)} files to process.\n")

    results = []

    for i, file_path in enumerate(tqdm(audio_files, desc="VAD")):
        try:
            waveform, sr = torchaudio.load(file_path)
            if sr != 16_000:
                waveform = torchaudio.functional.resample(waveform, sr, 16_000)

            ratio = vad_vocal_ratio(waveform, sr=16_000)
            results.append({
                "filename":    os.path.basename(file_path),
                "vocal_ratio": round(ratio, 4),
                "vocal_valid": ratio >= VOCAL_RATIO_THRESHOLD,
            })

        except Exception as e:
            tqdm.write(f"[ERROR] {os.path.basename(file_path)}: {e}")
            continue

        # Checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            _save(results, existing_df)
            print(f"[Checkpoint] {len(processed_files) + len(results)} files saved.")

    _save(results, existing_df)
    print(f"\nDone. Results saved to: {OUTPUT_CSV}")

    # Quick summary
    df = pd.read_csv(OUTPUT_CSV)
    valid   = df["vocal_valid"].sum()
    invalid = (~df["vocal_valid"]).sum()
    print(f"\nSummary:")
    print(f"  vocal_valid=True  : {valid}  ({valid/len(df):.1%})")
    print(f"  vocal_valid=False : {invalid} ({invalid/len(df):.1%})")


def _save(results, existing_df):
    if not results:
        return
    new_df = pd.DataFrame(results)
    final  = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    final.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()