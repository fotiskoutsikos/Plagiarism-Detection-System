import torch
import torch.nn as nn
from einops import rearrange
from nnAudio import features
from lib import layers
from lib import tensor_ops as tops

class CLEWSAcousticBranch(nn.Module):
    def __init__(self, conf, sr=16000):
        super().__init__()
        self.sr = sr
        
        # Shingling
        self.shingling_len = conf.shingling.len
        self.shingling_hop = conf.shingling.hop
        self.minlen = self.shingling_len  # set minlen to training shinglen
        
        # CQT 
        self.cqt = nn.Sequential(
            features.CQT1992v2(
                sr=self.sr,
                hop_length=int(conf.cqt.hoplen * sr),
                n_bins=conf.cqt.noctaves * conf.cqt.nbinsoct,
                bins_per_octave=conf.cqt.nbinsoct,
                filter_scale=conf.cqt.fscale,
                trainable=False,
                verbose=False,
            ),
            nn.AvgPool1d(conf.cqt.pool, stride=conf.cqt.pool),
        )
        
        # Frontend
        ncha0, ncha = conf.frontend.channels
        self.frontend = nn.Sequential(
            layers.CQTPrepare(pow=conf.frontend.cqtpow),
            nn.Conv2d(1, ncha0, (12, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(ncha0),
            nn.ReLU(inplace=True),
            nn.Conv2d(ncha0, ncha, (12, 3), stride=2, bias=False),
        )
        
        # ResNet Backbone
        aux = []
        for nb, nc, st in zip(
            conf.backbone.blocks, conf.backbone.channels, conf.backbone.down
        ):
            aux += [layers.MyIBNResBlock(ncha, nc, stride=st)]
            for _ in range(nb - 1):
                aux += [layers.MyIBNResBlock(nc, nc)]
            ncha = nc
        self.backbone = nn.Sequential(*aux)
        
        # Pooling & Projection (Z_aco)
        self.pool = layers.GeMPool()
        self.proj = nn.Sequential(
            nn.BatchNorm1d(ncha),
            nn.Linear(ncha, conf.zdim, bias=False),
        )

    def prepare(self, h, shingle_len=None, shingle_hop=None):
        """Prepare raw audio in overlapping frames for CQT extraction."""
        assert h.ndim == 2
        slen = self.shingling_len if shingle_len is None else shingle_len
        shop = self.shingling_hop if shingle_hop is None else shingle_hop
        
        # Shingling
        h = tops.get_frames(
            h, int(self.sr * slen), int(self.sr * shop), pad_mode="repeat"
        )
        h = tops.force_length(
            h, int(self.sr * self.minlen), dim=-1, pad_mode="repeat", allow_longer=True
        )
        
        # CQT
        s = h.size(1)
        h = rearrange(h, "b s t -> (b s) t")
        h = self.cqt(h)
        h = rearrange(h, "(b s) c t -> b s c t", s=s)
        return h  # (B,S,C,T)

    def embed(self, h):
        """Pass CQT through the CNN for embedding generation."""
        assert h.ndim == 4
        s = h.size(1)
        h = rearrange(h, "b s c t -> (b s) 1 c t")
        
        h = self.frontend(h)
        h = self.backbone(h)
        h = self.pool(h)
        z = self.proj(h)
        
        z = rearrange(z, "(b s) c -> b s c", s=s)
        return z

    def forward(self, h, shingle_len=None, shingle_hop=None):
        with torch.inference_mode():
            h = self.prepare(h, shingle_len=shingle_len, shingle_hop=shingle_hop)
        
        z = self.embed(h)
        return z 

    def load_and_freeze(self, checkpoint_path):
        """Load the pre-trained weights and freeze the branch."""
        print(f"Loading pre-trained weights from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        
        self.load_state_dict(clean_state_dict, strict=False)
        
        print("Freezing parameters of the Acoustic Branch...")
        for param in self.parameters():
            param.requires_grad = False
        print("The Acoustic Branch is ready!")