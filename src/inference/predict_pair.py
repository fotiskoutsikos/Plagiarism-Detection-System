"""
End-to-End Plagiarism Detection Inference.

Takes two raw audio files and produces a binary plagiarism decision
using the trained production classifier artifact.

Pipeline
--------
1. Load CLEWS and WEALY models from checkpoints.
2. Extract embeddings for both audio files (identical logic to batch extractors).
3. Compute pairwise distances using the canonical implementation from metrics.py.
4. Compute delta summary features using the canonical implementation from
   classifier_features.py with training-derived reference statistics loaded
   from the artifact (exact train/inference consistency).
5. Extract top-K CLEWS raw delta dimensions using stored indices from artifact.
6. Assemble the full hybrid feature vector in the exact stored column order.
7. Load trained XGBoost classifier and apply calibrated threshold.
8. Produce probability score and thresholded binary decision.

Centralization
--------------
All feature computation is delegated to canonical shared utilities:
    - Distances : evaluation/analysis/metrics.py::_compute_all_distances
    - Delta summaries : utils/classifier_features.py::compute_delta_summary_features
    - Reference stats : loaded from artifact (computed during training by
                        classifier_features.py::_compute_reference_from_positives)
    - Top-K indices : loaded from artifact (no re-ranking at inference time)

Usage
-----
    python src/inference/predict_pair.py --ori path/to/songA.wav --mod path/to/songB.wav

    Optional:
        --device cuda|cpu          (default: cuda if available)
        --model  path/to/model.pkl (default: models/final_plagiarism_detector.pkl)

Requirements
------------
    - CLEWS checkpoint : models/clews/checkpoint.pt
    - CLEWS config     : configs/extraction/clews.yaml
    - WEALY checkpoint : models/wealy/checkpoint.pt
    - WEALY config     : configs/extraction/wealy.yaml
    - Trained artifact : models/final_plagiarism_detector.pkl
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torchaudio
import whisper
from omegaconf import OmegaConf

# Resolve repository root 
repo_root = Path(__file__).resolve()
for _ in range(6):
    if (repo_root / "src").exists():
        break
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root / "src"))

from utils.clews_lib import Model as CLEWSModel
from utils.wealy_lib import Model as WEALYModel

# Canonical feature computation utilities
from utils.classifier_features import compute_delta_summary_features

# Canonical distance computation
# _compute_all_distances works on torch tensors; we wrap it for numpy input
from evaluation.analysis.metrics import _compute_all_distances

# Default paths 
DEFAULT_CLEWS_CHECKPOINT = "models/clews/checkpoint.pt"
DEFAULT_CLEWS_CONFIG     = "configs/extraction/clews.yaml"
DEFAULT_WEALY_CHECKPOINT = "models/wealy/checkpoint.pt"
DEFAULT_WEALY_CONFIG     = "configs/extraction/wealy.yaml"
DEFAULT_ARTIFACT_PATH    = "models/final_plagiarism_detector.pkl"


# AUDIO LOADING  (identical to extract_clews.py / extract_wealy.py)
def load_audio_mono(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load audio, convert to mono, resample to target_sr.
    Logic is identical to load_audio() in extract_clews.py / extract_wealy.py.
    Returns a 1D waveform tensor (T,).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform  = resampler(waveform)

    return waveform.squeeze(0)   # (T,)


# CLEWS  (identical loading + forward pass to extract_clews.py)
def load_clews_model(
    config_path:     str = DEFAULT_CLEWS_CONFIG,
    checkpoint_path: str = DEFAULT_CLEWS_CHECKPOINT,
    device:          str = "cuda",
) -> tuple:
    """Load CLEWS model. Identical to extract_clews.py::extract_embeddings."""
    conf  = OmegaConf.load(config_path)
    model = CLEWSModel(conf.model)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    clean_state_dict = {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v
        for k, v in state_dict.items()
    }

    model.load_state_dict(clean_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, conf, device


def extract_clews_embedding(
    audio_path: str,
    model:      CLEWSModel,
    conf,
    device:     str,
) -> np.ndarray:
    """
    Single-file CLEWS embedding. Identical forward-pass logic to
    extract_clews.py::extract_embeddings (inner loop body).
    Returns 1-D float32 numpy array.
    """
    target_sr   = int(conf.data.samplerate)
    waveform    = load_audio_mono(audio_path, target_sr=target_sr)
    waveform    = waveform.unsqueeze(0).to(device)          # (1, T)
    shingle_hop = float(conf.model.shingling.hop)
    shingle_len = float(conf.model.shingling.len)

    with torch.no_grad():
        z = model(waveform, shingle_hop=shingle_hop, shingle_len=shingle_len)

    return z.squeeze(0).cpu().numpy().astype(np.float32)


# WEALY  (identical loading + forward pass to extract_wealy.py)
_decoder_hidden_states: list[torch.Tensor] = []


def _hook_fn(module, input, output) -> None:
    """Capture last Whisper decoder hidden states. Same as extract_wealy.py."""
    if isinstance(output, tuple):
        _decoder_hidden_states.append(output[0].detach().cpu())
    else:
        _decoder_hidden_states.append(output.detach().cpu())


def load_wealy_model(
    config_path:     str = DEFAULT_WEALY_CONFIG,
    checkpoint_path: str = DEFAULT_WEALY_CHECKPOINT,
    device:          str = "cuda",
) -> tuple:
    """Load WEALY + Whisper. Identical to extract_wealy.py::extract_wealy_embeddings."""
    conf          = OmegaConf.load(config_path)
    whisper_model = whisper.load_model("turbo", device=device)

    _decoder_hidden_states.clear()
    hook = whisper_model.decoder.ln.register_forward_hook(_hook_fn)

    wealy_model = WEALYModel(conf.model)
    checkpoint  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    try:
        wealy_model.load_state_dict(clean_state_dict, strict=True)
    except RuntimeError:
        for k in wealy_model.state_dict():
            if k not in clean_state_dict:
                clean_state_dict[k] = wealy_model.state_dict()[k]
        wealy_model.load_state_dict(clean_state_dict, strict=False)

    wealy_model   = wealy_model.to(device)
    wealy_model.eval()
    whisper_model.eval()
    return wealy_model, whisper_model, hook, conf, device


def extract_wealy_embedding(
    audio_path:   str,
    wealy_model:  WEALYModel,
    whisper_model,
    device:       str,
) -> np.ndarray:
    """
    Single-file WEALY embedding. Identical forward-pass logic to
    extract_wealy.py::extract_wealy_embeddings (inner loop body).
    Returns 1-D float32 numpy array.
    """
    waveform     = load_audio_mono(audio_path, target_sr=16000).to(device)
    audio_padded = whisper.pad_or_trim(waveform.flatten())
    mel          = whisper.log_mel_spectrogram(
        audio_padded, n_mels=whisper_model.dims.n_mels
    ).to(device).unsqueeze(0)

    _decoder_hidden_states.clear()
    options = whisper.DecodingOptions(
        task="transcribe", language=None, without_timestamps=True, fp16=False
    )
    with torch.no_grad():
        _ = whisper.decode(whisper_model, mel, options)

    if not _decoder_hidden_states:
        raise RuntimeError(
            f"Whisper decoder produced no hidden states for {audio_path}"
        )

    decoder_latents = _decoder_hidden_states[-1].clone().to(device)
    if decoder_latents.ndim == 2:
        decoder_latents = decoder_latents.unsqueeze(0)

    with torch.no_grad():
        embedding = wealy_model(decoder_latents)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


# FEATURE CONSTRUCTION  (fully centralized — no local reimplementation)
def _distances_from_numpy(
    emb_ori: np.ndarray,
    emb_mod: np.ndarray,
) -> dict[str, float]:
    """
    Wrap metrics.py::_compute_all_distances for numpy input.

    _compute_all_distances operates on torch tensors, so we convert here.
    The canonical formulas remain in metrics.py — nothing is reimplemented.
    """
    t_ori = torch.tensor(emb_ori, dtype=torch.float32)
    t_mod = torch.tensor(emb_mod, dtype=torch.float32)
    return _compute_all_distances(t_ori, t_mod)


def _delta_summaries_from_artifact(
    delta:     np.ndarray,
    ref:       dict,
) -> dict[str, float]:
    """
    Compute delta summary features using the canonical function from
    classifier_features.py and the training-derived reference statistics
    stored in the artifact.

    This guarantees that the 11 summary features produced here are
    mathematically identical to those computed during training.

    Args:
        delta : 1-D absolute delta vector for this pair.
        ref   : artifact["clews_reference"] or artifact["wealy_reference"],
                containing global_q75, stable_dims, volatile_dims.

    Returns:
        Dict with 11 feature values (same keys as classifier_features.py).
    """
    stable_dims   = np.array(ref["stable_dims"],   dtype=np.int64)
    volatile_dims = np.array(ref["volatile_dims"],  dtype=np.int64)
    global_q75    = float(ref["global_q75"])

    # Reshape to (1, D) as expected by compute_delta_summary_features
    delta_2d = delta.reshape(1, -1).astype(np.float32)

    df_feat = compute_delta_summary_features(
        delta_matrix  = delta_2d,
        stable_dims   = stable_dims,
        volatile_dims = volatile_dims,
        global_q75    = global_q75,
    )

    # Return as plain dict (single row)
    return df_feat.iloc[0].to_dict()


def build_feature_vector(
    clews_ori: np.ndarray,
    clews_mod: np.ndarray,
    wealy_ori: np.ndarray,
    wealy_mod: np.ndarray,
    artifact:  dict,
) -> np.ndarray:
    """
    Assemble the full hybrid feature vector for a single pair.

    All computation delegates to canonical utilities:
        - Distances     → metrics.py::_compute_all_distances (via wrapper)
        - Summaries     → classifier_features.py::compute_delta_summary_features
        - Reference     → artifact["clews_reference"] / artifact["wealy_reference"]
        - Top-K indices → artifact["clews_top_k_indices"]
        - Column order  → artifact["engineered_feature_columns"]

    Returns:
        1-D float32 array of shape (n_features,) matching the training schema.
    """
    engineered_cols     = artifact["engineered_feature_columns"]
    clews_top_k_indices = artifact.get("clews_top_k_indices", [])
    clews_ref           = artifact["clews_reference"]
    wealy_ref           = artifact["wealy_reference"]

    # Pairwise deltas 
    min_clews = min(len(clews_ori), len(clews_mod))
    min_wealy = min(len(wealy_ori), len(wealy_mod))
    clews_delta = np.abs(clews_mod[:min_clews] - clews_ori[:min_clews])
    wealy_delta = np.abs(wealy_mod[:min_wealy] - wealy_ori[:min_wealy])

    # Distances (canonical: metrics.py) 
    clews_dists = _distances_from_numpy(clews_ori, clews_mod)
    wealy_dists = _distances_from_numpy(wealy_ori, wealy_mod)

    # Delta summaries (canonical: classifier_features.py) 
    clews_sums = _delta_summaries_from_artifact(clews_delta, clews_ref)
    wealy_sums = _delta_summaries_from_artifact(wealy_delta, wealy_ref)

    # Build prefixed feature dict 
    feature_dict: dict[str, float] = {}
    for k, v in clews_dists.items():
        feature_dict[f"clews_{k}"] = v
    for k, v in wealy_dists.items():
        feature_dict[f"wealy_{k}"] = v
    for k, v in clews_sums.items():
        feature_dict[f"clews_{k}"] = v
    for k, v in wealy_sums.items():
        feature_dict[f"wealy_{k}"] = v

    # Assemble in the EXACT stored column order 
    engineered_vector = np.array(
        [feature_dict.get(col, 0.0) for col in engineered_cols],
        dtype=np.float32,
    )

    # Append top-K CLEWS raw delta dimensions 
    if clews_top_k_indices:
        valid_idx  = [i for i in clews_top_k_indices if i < len(clews_delta)]
        topk_vals  = clews_delta[valid_idx].astype(np.float32)

        # Zero-pad if some stored indices exceed embedding dimensionality
        if len(topk_vals) < len(clews_top_k_indices):
            topk_vals = np.concatenate([
                topk_vals,
                np.zeros(len(clews_top_k_indices) - len(topk_vals), dtype=np.float32),
            ])

        feature_vector = np.concatenate([engineered_vector, topk_vals])
    else:
        feature_vector = engineered_vector

    return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)


# MAIN INFERENCE
def predict_pair(
    ori_path:      str,
    mod_path:      str,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    device:        str = "cuda",
) -> dict:
    """
    Full end-to-end inference for a single audio pair.

    Args:
        ori_path:      Path to original audio file.
        mod_path:      Path to modified / suspected audio file.
        artifact_path: Path to trained .pkl artifact.
        device:        "cuda" or "cpu".

    Returns:
        Dictionary with prediction results.
    """
    for p in (ori_path, mod_path, artifact_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"File not found: {p}")

    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Load artifact 
    print("Loading trained classifier artifact...")
    artifact    = joblib.load(artifact_path)
    clf         = artifact["classifier"]
    threshold   = artifact["optimal_threshold"]
    n_expected  = artifact["n_features"]
    config_name = artifact.get("selected_config", "unknown")

    print(f"  Config     : {config_name}")
    print(f"  Features   : {n_expected}")
    print(f"  Threshold  : {threshold:.4f}")
    print(f"  CLEWS q75  : {artifact['clews_reference']['global_q75']:.6f}")
    print(f"  WEALY q75  : {artifact['wealy_reference']['global_q75']:.6f}")

    # Load embedding models 
    print("\nLoading CLEWS model...")
    clews_model, clews_conf, _ = load_clews_model(device=device)
    print("  Done.")

    print("Loading WEALY model...")
    wealy_model, whisper_model, hook, _, _ = load_wealy_model(device=device)
    print("  Done.")

    # Extract embeddings 
    print(f"\nExtracting embeddings for pair:")
    print(f"  Original : {Path(ori_path).name}")
    print(f"  Modified : {Path(mod_path).name}")

    try:
        clews_ori = extract_clews_embedding(ori_path, clews_model, clews_conf, device)
        clews_mod = extract_clews_embedding(mod_path, clews_model, clews_conf, device)
        print(f"  CLEWS    : {clews_ori.shape[0]}D")
    except Exception:
        hook.remove()
        raise

    try:
        wealy_ori = extract_wealy_embedding(ori_path, wealy_model, whisper_model, device)
        wealy_mod = extract_wealy_embedding(mod_path, wealy_model, whisper_model, device)
        print(f"  WEALY    : {wealy_ori.shape[0]}D")
    finally:
        hook.remove()

    # Build feature vector ─
    print("\nBuilding feature vector...")
    feature_vector = build_feature_vector(
        clews_ori, clews_mod,
        wealy_ori, wealy_mod,
        artifact,
    )
    print(f"  Shape : {feature_vector.shape[0]}D")

    if feature_vector.shape[0] != n_expected:
        print(
            f"  [WARNING] Expected {n_expected} features, got "
            f"{feature_vector.shape[0]}. Prediction may be unreliable."
        )

    # Predict 
    prob          = float(clf.predict_proba(feature_vector.reshape(1, -1))[0, 1])
    is_plagiarism = prob >= threshold

    return {
        "original":    Path(ori_path).name,
        "modified":    Path(mod_path).name,
        "probability": prob,
        "threshold":   threshold,
        "decision":    "PLAGIARISM" if is_plagiarism else "NOT PLAGIARISM",
        "config":      config_name,
    }


def print_result(result: dict) -> None:
    """Print a formatted prediction result."""
    indicator = "🔴" if result["decision"] == "PLAGIARISM" else "🟢"
    print(f"\n{'═' * 60}")
    print(f"  PLAGIARISM DETECTION RESULT")
    print(f"{'═' * 60}")
    print(f"  Original  : {result['original']}")
    print(f"  Modified  : {result['modified']}")
    print(f"{'─' * 60}")
    print(f"  Score     : {result['probability'] * 100:.1f}%")
    print(f"  Threshold : {result['threshold'] * 100:.1f}%")
    print(f"  Decision  : {indicator}  {result['decision']}")
    print(f"{'─' * 60}")
    print(f"  Config    : {result['config']}")
    print(f"{'═' * 60}")


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end plagiarism detection for a pair of audio files."
    )
    parser.add_argument("--ori",    required=True,
                        help="Path to original audio (.wav)")
    parser.add_argument("--mod",    required=True,
                        help="Path to modified / suspected audio (.wav)")
    parser.add_argument("--model",  default=DEFAULT_ARTIFACT_PATH,
                        help=f"Trained artifact path (default: {DEFAULT_ARTIFACT_PATH})")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Inference device (default: cuda)")
    args = parser.parse_args()

    result = predict_pair(
        ori_path=args.ori,
        mod_path=args.mod,
        artifact_path=args.model,
        device=args.device,
    )
    print_result(result)


if __name__ == "__main__":
    main()