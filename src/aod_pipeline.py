import torch
import torch.nn as nn
import torch.nn.functional as F

from model.acoustic_branch import CLEWSAcousticBranch
from model.semantic_branch import SemanticBranch

class AODPipeline(nn.Module):
    def __init__(self, acoustic_conf, semantic_model_size="turbo", z_aco_dim=1024, z_sem_dim=512, num_heads=4):
        super().__init__()
        # Load the acoustic branch and semantic branch
        self.acoustic_branch = CLEWSAcousticBranch(conf=acoustic_conf)
        self.semantic_branch = SemanticBranch(model_size=semantic_model_size, z_dim=z_sem_dim)
        # Projection layer to align acoustic features to semantic space
        self.aco_proj = nn.Linear(z_aco_dim, z_sem_dim)
        # Late Cross-Attention Fusion
        # Q = Lyrics (Z_sem), K = V = Audio (Z_aco)
        self.cross_attention = nn.MultiheadAttention(embed_dim=z_sem_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(z_sem_dim)

    def calculate_orthogonal_loss(self, z_sem, z_aco):
        '''
         - Calculate the orthogonal loss between semantic and acoustic features.
         - Loss = cosine_similarity(z_sem, z_aco)^2
         - Encourages the model to learn complementary features in both branches.
        '''
        cos_sim = F.cosine_similarity(z_sem, z_aco, dim=-1)
        return torch.mean(cos_sim ** 2)

    def forward(self, audio_waveform):
        '''
         - input: raw 20s audio segment (B, T) at 16kHz
         - output: fused representation (B, z_sem_dim) and orthogonal loss
        '''
        z_sem = self.semantic_branch(audio_waveform)
        z_aco_raw = self.acoustic_branch(audio_waveform)
        z_aco_segment = z_aco_raw.mean(dim=1)
        
        # Project acoustic features to semantic space
        z_aco_proj = self.aco_proj(z_aco_segment)
        # Calculate orthogonal loss
        loss_ortho = self.calculate_orthogonal_loss(z_sem, z_aco_proj)

        # Cross-Attention Fusion
        q = z_sem.unsqueeze(1)
        k = v = z_aco_proj.unsqueeze(1)
        attn_output, _ = self.cross_attention(query=q, key=k, value=v)
        z_fused = self.norm(attn_output.squeeze(1) + z_sem)

        return z_fused, loss_ortho