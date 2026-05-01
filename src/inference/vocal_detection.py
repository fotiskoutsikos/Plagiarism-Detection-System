"""
Source-level vocal validity estimation using Demucs vocal stems.

Scans ONLY:
    data/separated_segment_smp/mdx_extra_q/**/vocals.wav

Outputs one row per original SMP segment:
    source_key | source_filename | stem_relpath | vocal_ratio | active_ratio | stem_rms_db | vocal_valid

Why this version:
- We detect vocals on Demucs-separated vocals stems, not on full mixes.
- We run detection once per original source segment.
- The resulting CSV can later be propagated to all descendants
  (audio, DSP, MusicGen, AudioLDM2, MGE-LDM) via source_key.

Typical downstream use:
- If vocal_valid == False, skip/flag WEALY extraction for that source and its descendants.
"""

import os
import re
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm


# CONFIG
STEMS_ROOT = "data/separated_segment_smp/mdx_extra_q"
OUTPUT_CSV = "data/vocal_ratios_source.csv"

TARGET_SR = 16_000
FRAME_MS = 20

# VAD / spectral heuristic
ENERGY_THRESHOLD_DB = -40.0
FREQ_LOW = 80
FREQ_HIGH = 3_400
SPEECH_BAND_RATIO_THRESHOLD = 0.50

# Validity thresholds
VOCAL_RATIO_THRESHOLD = 0.30   # fraction of active frames that look vocal-like
ACTIVE_RATIO_THRESHOLD = 0.05  # fraction of all frames that are active at all
STEM_RMS_DB_THRESHOLD = -45.0  # reject extremely weak / near-silent stems

CHECKPOINT_EVERY = 200

# Optional pattern for extracting source key from path
SOURCE_KEY_REGEX = re.compile(r"(pair_\d+_(?:ori|comp)_\d+s)")


# AUDIO HELPERS
def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert [C, T] or [T] tensor to mono [T]."""
    if waveform.dim() == 1:
        return waveform.float().cpu()
    return waveform.mean(dim=0).float().cpu()


def rms_db(waveform: torch.Tensor) -> float:
    """Global RMS in dBFS-like scale."""
    wav = to_mono(waveform)
    if wav.numel() == 0:
        return -120.0
    rms = wav.pow(2).mean().sqrt().clamp(min=1e-9)
    return float(20 * torch.log10(rms))


# VAD / VOCAL HEURISTIC
def vad_vocal_stats(
    waveform: torch.Tensor,
    sr: int = 16_000,
    frame_ms: int = 20,
    energy_threshold_db: float = -40.0,
    freq_low: int = 80,
    freq_high: int = 3_400,
    band_ratio_threshold: float = 0.50,
):
    """
    Lightweight energy + spectral VAD on a vocal stem.

    Returns:
        vocal_ratio:
            speech/vocal-like active frames / total active frames
        active_ratio:
            total active frames / total frames
        n_frames:
            total number of frames
        active_frames:
            number of active frames
        vocal_like_frames:
            active frames whose spectral energy is concentrated
            in the vocal/speech band
    """
    wav = to_mono(waveform)

    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(wav) // frame_len
    if n_frames == 0:
        return {
            "vocal_ratio": 0.0,
            "active_ratio": 0.0,
            "n_frames": 0,
            "active_frames": 0,
            "vocal_like_frames": 0,
        }

    frames = wav[: n_frames * frame_len].reshape(n_frames, frame_len)

    rms_vals = frames.pow(2).mean(dim=1).sqrt()
    rms_db_vals = 20 * torch.log10(rms_vals.clamp(min=1e-9))
    active_mask = rms_db_vals > energy_threshold_db

    active_frames = int(active_mask.sum().item())
    active_ratio = active_frames / n_frames

    if active_frames == 0:
        return {
            "vocal_ratio": 0.0,
            "active_ratio": active_ratio,
            "n_frames": n_frames,
            "active_frames": 0,
            "vocal_like_frames": 0,
        }

    window = torch.hann_window(frame_len)
    freqs = torch.fft.rfftfreq(frame_len, d=1.0 / sr)
    speech_mask = (freqs >= freq_low) & (freqs <= freq_high)

    vocal_like_frames = 0
    for i in range(n_frames):
        if not active_mask[i]:
            continue

        spec = torch.fft.rfft(frames[i] * window).abs()
        speech_energy = spec[speech_mask].pow(2).sum()
        total_energy = spec.pow(2).sum().clamp(min=1e-9)

        if (speech_energy / total_energy).item() >= band_ratio_threshold:
            vocal_like_frames += 1

    vocal_ratio = vocal_like_frames / active_frames

    return {
        "vocal_ratio": vocal_ratio,
        "active_ratio": active_ratio,
        "n_frames": n_frames,
        "active_frames": active_frames,
        "vocal_like_frames": vocal_like_frames,
    }


# PATH / KEY HELPERS
def derive_source_key(stem_path: str, stems_root: str) -> tuple[str, str, str]:
    """
    Derive:
        source_key       e.g. pair_9_comp_51s
        source_filename  e.g. pair_9_comp_51s.wav
        stem_relpath     relative path to vocals.wav
    """
    relpath = os.path.relpath(stem_path, stems_root).replace("\\", "/")

    # Try regex first from full relative path
    match = SOURCE_KEY_REGEX.search(relpath)
    if match:
        source_key = match.group(1)
    else:
        # Fallback: use parent directory name
        parent = os.path.basename(os.path.dirname(stem_path))
        if parent.lower().endswith(".wav"):
            source_key = os.path.splitext(parent)[0]
        else:
            source_key = parent

    source_filename = f"{source_key}.wav"
    return source_key, source_filename, relpath


def collect_vocal_stems(stems_root: str) -> list[str]:
    """Collect all vocals.wav files under the Demucs output root."""
    stem_files = []
    if not os.path.exists(stems_root):
        print(f"[ERROR] STEMS_ROOT not found: {stems_root}")
        return stem_files

    for root, _, files in os.walk(stems_root):
        for f in files:
            if f.lower() == "vocals.wav":
                stem_files.append(os.path.join(root, f))

    stem_files = sorted(stem_files)
    return stem_files


# SAVE / RESUME
def load_existing(csv_path: str):
    """Load existing CSV safely for resume."""
    if not os.path.exists(csv_path):
        return None, set()

    existing_df = pd.read_csv(csv_path)

    required_cols = {"source_key"}
    if not required_cols.issubset(existing_df.columns):
        raise ValueError(
            f"Existing CSV at {csv_path} does not match the new schema. "
            f"Please remove or rename it and rerun."
        )

    processed_keys = set(existing_df["source_key"].astype(str).tolist())
    return existing_df, processed_keys


def save_results(results: list[dict], existing_df: pd.DataFrame | None, output_csv: str):
    """Append-like save with deduplication by source_key."""
    if not results:
        return

    new_df = pd.DataFrame(results)

    if existing_df is not None:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df = final_df.drop_duplicates(subset=["source_key"], keep="last")
    final_df = final_df.sort_values("source_key").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    final_df.to_csv(output_csv, index=False)



def main():
    print(f"[INFO] Scanning Demucs stems under: {STEMS_ROOT}")
    stem_files = collect_vocal_stems(STEMS_ROOT)
    print(f"[INFO] Found {len(stem_files)} vocals.wav stems total.")

    existing_df, processed_keys = load_existing(OUTPUT_CSV)
    if existing_df is not None:
        print(f"[INFO] Resuming: {len(processed_keys)} source segments already processed.")

    remaining_stems = []
    for stem_path in stem_files:
        source_key, _, _ = derive_source_key(stem_path, STEMS_ROOT)
        if source_key not in processed_keys:
            remaining_stems.append(stem_path)

    print(f"[INFO] Remaining: {len(remaining_stems)} stems to process.\n")

    results = []

    for i, stem_path in enumerate(tqdm(remaining_stems, desc="Vocal detection")):
        source_key, source_filename, stem_relpath = derive_source_key(stem_path, STEMS_ROOT)

        try:
            waveform, sr = torchaudio.load(stem_path)

            if sr != TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
                sr = TARGET_SR

            stats = vad_vocal_stats(
                waveform=waveform,
                sr=sr,
                frame_ms=FRAME_MS,
                energy_threshold_db=ENERGY_THRESHOLD_DB,
                freq_low=FREQ_LOW,
                freq_high=FREQ_HIGH,
                band_ratio_threshold=SPEECH_BAND_RATIO_THRESHOLD,
            )

            stem_level_rms_db = rms_db(waveform)

            vocal_valid = (
                stats["vocal_ratio"] >= VOCAL_RATIO_THRESHOLD
                and stats["active_ratio"] >= ACTIVE_RATIO_THRESHOLD
                and stem_level_rms_db >= STEM_RMS_DB_THRESHOLD
            )

            results.append({
                "source_key": source_key,
                "source_filename": source_filename,
                "stem_relpath": stem_relpath,
                "vocal_ratio": round(stats["vocal_ratio"], 4),
                "active_ratio": round(stats["active_ratio"], 4),
                "stem_rms_db": round(stem_level_rms_db, 2),
                "n_frames": stats["n_frames"],
                "active_frames": stats["active_frames"],
                "vocal_like_frames": stats["vocal_like_frames"],
                "vocal_valid": bool(vocal_valid),
            })

        except Exception as e:
            tqdm.write(f"[ERROR] {source_key} | {stem_relpath}: {e}")
            continue

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_results(results, existing_df, OUTPUT_CSV)
            print(f"[Checkpoint] Saved {len(processed_keys) + len(results)} rows.")

    save_results(results, existing_df, OUTPUT_CSV)

    print(f"\n[INFO] Done. Results saved to: {OUTPUT_CSV}")

    df = pd.read_csv(OUTPUT_CSV)
    valid = int(df["vocal_valid"].sum())
    invalid = int((~df["vocal_valid"]).sum())

    print("\nSummary:")
    print(f"  Total sources       : {len(df)}")
    print(f"  vocal_valid=True    : {valid} ({valid/len(df):.1%})")
    print(f"  vocal_valid=False   : {invalid} ({invalid/len(df):.1%})")

    print("\nThresholds used:")
    print(f"  VOCAL_RATIO_THRESHOLD  : {VOCAL_RATIO_THRESHOLD}")
    print(f"  ACTIVE_RATIO_THRESHOLD : {ACTIVE_RATIO_THRESHOLD}")
    print(f"  STEM_RMS_DB_THRESHOLD  : {STEM_RMS_DB_THRESHOLD} dB")


if __name__ == "__main__":
    main()