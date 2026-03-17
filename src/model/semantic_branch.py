import torch
import torch.nn as nn
import whisper
import contextlib
from model.attention_pooling import AttentionPooling

class SemanticBranch(nn.Module):
    def __init__(self, model_size="turbo", z_dim=512):
        """
        model_size: "tiny", "base", "small", "medium", "large-v2", or "turbo"
        """
        super().__init__()
        
        print(f"Loading Whisper '{model_size}'...")
        whisper_model = whisper.load_model(model_size, device="cpu")
        
        # Keep only the encoder part of Whisper
        self.encoder = whisper_model.encoder
        self.embed_dim = whisper_model.dims.n_audio_state 
        
        # Freeze weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        print(f"Whisper Encoder frozen successfully. Output dimension: {self.embed_dim}")
            
        # Attention Pooling
        self.pool = AttentionPooling(dim=self.embed_dim, num_heads=1)
        
        # Final projection to Z_sem
        self.proj = nn.Linear(self.embed_dim, z_dim)

    def forward(self, audio_waveform):
        """
        Input raw audio (Batch, Time) 16000Hz
        Returns Z_sem (Batch, Dim)
        """
        use_autocast = audio_waveform.device.type == "cuda"
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()

        # Frozen part
        with torch.inference_mode():
            mel = whisper.log_mel_spectrogram(audio_waveform)
            mel = whisper.pad_or_trim(mel)
            
            # mixed precision
            with autocast_ctx:
                output_features = self.encoder(mel)   # shape: (Batch, Time, Embed_Dim)
                output_features = output_features.float()
            
        # Training part 
        pooled_features = self.pool(output_features)  # (Batch, Embed_Dim)
        
        # Final projection to Z_sem
        z_sem = self.proj(pooled_features)  # (Batch, z_dim)
        
        return z_sem