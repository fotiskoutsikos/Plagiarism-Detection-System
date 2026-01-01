import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any

# ==========================================
# PART 1: PREPROCESSING DATASET (WAV -> MERT)
# Χρησιμοποιείται από το extract_mert.py
# ==========================================

class AudioDataset(Dataset):
    def __init__(self, tracks_dir: str, audio_processor=None):
        """
        Φορτώνει τα αρχεία ήχου (.wav) για να περάσουν από το MERT.
        """
        self.tracks_dir = tracks_dir
        # Βρίσκουμε τους φακέλους των τραγουδιών (αγνοούμε αρχεία συστήματος)
        self.tracklist = sorted([
            t for t in os.listdir(tracks_dir) 
            if os.path.isdir(os.path.join(tracks_dir, t)) and not t.startswith('.')
        ])
        
        self.audio_processor = audio_processor
        self.sample_rate = None

    def __len__(self) -> int:
        return len(self.tracklist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audios = {}
        track_name = self.tracklist[idx]
        track_path = os.path.join(self.tracks_dir, track_name)
        
        # Λίστα με τα versions (π.χ. original, musicgen_cover)
        versions = sorted([
            v for v in os.listdir(track_path) 
            if os.path.isdir(os.path.join(track_path, v)) and not v.startswith('.')
        ])

        for version in versions:
            version_path = os.path.join(track_path, version)
            
            # Βρίσκουμε τα wav αρχεία μέσα στο version folder
            # Υποθέτουμε ότι είναι ήδη segmented (π.χ. 0.wav, 1.wav...) από το beat tracking
            files = sorted([
                f for f in os.listdir(version_path) 
                if f.endswith('.wav')
            ])
            
            if not files:
                continue

            audios[version] = []
            for file in files:
                file_path = os.path.join(version_path, file)
                # Φόρτωση ήχου
                waveform, sr = torchaudio.load(file_path)
                self.sample_rate = sr
                audios[version].append(waveform)

            # Pre-processing για το MERT (Resampling)
            if self.audio_processor:
                target_sr = self.audio_processor.sampling_rate
                processed_audios = []
                for waveform in audios[version]:
                    
                    # ΔΙΟΡΘΩΣΗ ΕΔΩ:
                    # Ελέγχουμε αν υπάρχει sample_rate και το κάνουμε int()
                    if self.sample_rate is not None and self.sample_rate != target_sr:
                        waveform = F.resample(waveform, int(self.sample_rate), target_sr)
                    
                    # Προετοιμασία tensors μέσω του processor
                    inputs = self.audio_processor(
                        waveform.squeeze().numpy(), 
                        sampling_rate=target_sr, 
                        return_tensors="pt"
                    )["input_values"].squeeze()
                    
                    processed_audios.append(inputs)
                
                audios[version] = processed_audios
        
        # Κόβουμε όλα τα versions στο ίδιο μήκος (αν κάποιο έχει λιγότερα segments)
        if audios:
            min_frames = min([len(audios[v]) for v in audios.keys()])
            for v in audios.keys():
                audios[v] = audios[v][:min_frames]
        
        return {
            'track': track_name,
            'audios': audios
        }

def audio_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ενώνει τα batch για το AudioDataset, κάνοντας padding στο χρόνο.
    """
    batch_dict = {}
    
    # 1. Βρίσκουμε το μέγιστο μήκος (samples) σε όλο το batch
    max_len = 0
    for item in batch:
        for version in item["audios"]:
            for segment in item["audios"][version]:
                if segment.shape[-1] > max_len:
                    max_len = segment.shape[-1]

    # 2. Κάνουμε Padding
    for item in batch:
        track_name = item["track"]
        batch_dict[track_name] = []
        
        # Για κάθε version του τραγουδιού
        for version in item["audios"]:
            padded_segments = []
            for segment in item["audios"][version]:
                pad_amount = max_len - segment.shape[-1]
                padded_seg = torch.nn.functional.pad(segment, (0, pad_amount))
                padded_segments.append(padded_seg)
            
            # Stack segments: [Num_Segments, Samples]
            # Αποθηκεύουμε τα αποτελέσματα σε λίστα (θα τα διαχειριστεί το extract_mert)
            # Εδώ επιστρέφουμε λίστα από Tensors, όχι dict, για να είναι πιο εύκολο το batching
            batch_dict[track_name].append(torch.stack(padded_segments))

    return batch_dict

def create_audio_dataloader(
    tracks_dir: str, 
    batch_size: int, 
    num_workers: int, 
    audio_processor=None,
) -> DataLoader:
    dataset = AudioDataset(tracks_dir, audio_processor=audio_processor)
    print(f"Audio Dataset created with {len(dataset)} tracks.")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,
        collate_fn=audio_collate_fn
    )
    return dataloader


class TripletDataset(Dataset):
    def __init__(
        self, 
        tracks_dir: str, 
        store_dict: bool = True, 
        length_mult: int = 1, 
    ):
        super().__init__()
        # Φιλτράρουμε μόνο τα .npy αρχεία
        self.source = sorted([
            s for s in os.listdir(tracks_dir) 
            if s.endswith('.npy') and not s.startswith('.')
        ])
        
        self.tracks_dir = tracks_dir
        self.store_dict = store_dict
        
        if self.store_dict:
            self.track_dict = self._create_track_dict()
            
        print(f"[Dataset] Loaded {len(self.source)} embedding tracks.")
        self.length_mult = length_mult

    def _create_track_dict(self) -> Dict[str, np.ndarray]:
        track_dict = {}
        print("Loading embeddings into RAM...")
        for track in self.source:
            if track not in track_dict:
                track_dict[track] = np.load(os.path.join(self.tracks_dir, track))
        return track_dict

    def __len__(self):
        return len(self.source) * self.length_mult

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(idx // self.length_mult)
        idx = int(idx % len(self.source))
        
        anchor_name = self.source[idx]
        
        if self.store_dict:
            anchor_data = self.track_dict[anchor_name]
        else:
            anchor_data = np.load(os.path.join(self.tracks_dir, anchor_name))

        # Triplet Logic
        # Anchor
        anchor_ver_idx = int(torch.randint(high=anchor_data.shape[0], size=()).item())
        if anchor_data.shape[1] > 0:
            seg_idx = int(torch.randint(high=anchor_data.shape[1], size=()).item())
        else:
            seg_idx = 0
        anchor_sample = anchor_data[anchor_ver_idx][seg_idx]

        # Positive (Hard)
        pos_choices = [i for i in range(anchor_data.shape[0]) if i != anchor_ver_idx]
        if pos_choices:
            pos_ver_idx = random.choice(pos_choices)
        else:
            pos_ver_idx = anchor_ver_idx
        positive_sample = anchor_data[pos_ver_idx][seg_idx]

        # Negative
        neg_idx = random.randint(0, len(self.source) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.source) - 1)
        
        neg_name = self.source[neg_idx]
        if self.store_dict:
            neg_data = self.track_dict[neg_name]
        else:
            neg_data = np.load(os.path.join(self.tracks_dir, neg_name))
            
        neg_ver_idx = int(torch.randint(high=neg_data.shape[0], size=()).item())
        if neg_data.shape[1] > 0:
            neg_seg_idx = int(torch.randint(high=neg_data.shape[1], size=()).item())
        else:
            neg_seg_idx = 0
        negative_sample = neg_data[neg_ver_idx][neg_seg_idx]

        # Prepare Tensors
        def prepare(x):
            t = torch.from_numpy(x)
            if t.ndim == 3: t = t.permute(0, 2, 1) # (Layers, Seq, Emb) -> (Layers, Emb, Seq)
            if t.ndim == 2: t = t.unsqueeze(-1).permute(0, 2, 1)
            return torch.cat([h for h in t], dim=0) # Concat layers

        return prepare(anchor_sample), prepare(positive_sample), prepare(negative_sample)

def triplet_collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    
    def pad(t_list):
        max_l = max([t.shape[-1] for t in t_list])
        return torch.stack([torch.nn.functional.pad(t, (0, max_l - t.shape[-1])) for t in t_list])
        
    return pad(anchors), pad(positives), pad(negatives)

def create_triplet_dataloader(tracks_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = TripletDataset(tracks_dir)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=triplet_collate_fn)