import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        # learnable positional encodings (seq_len, d_model)
        self.pe = nn.Parameter(torch.zeros(seq_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        if x.size(1) != self.seq_len:
            # if sequence length differs, interpolate position embeddings
            pe = nn.functional.interpolate(
                self.pe.unsqueeze(0).permute(0,2,1),
                size=x.size(1),
                mode='linear',
                align_corners=False
            ).permute(0,2,1).squeeze(0)
        else:
            pe = self.pe
        return x + pe.unsqueeze(0).to(x.dtype).to(x.device)

class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",
        n_classes: int = 2
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = LearnablePositionalEncoding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, n_classes)
        )

        # weight init
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)     # (batch, seq_len, d_model)

        if self.pooling == "mean":
            rep = x.mean(dim=1)   # (batch, d_model)
        elif self.pooling == "max":
            rep, _ = x.max(dim=1)
        elif self.pooling == "cls":
            # assume first token is CLS
            rep = x[:, 0, :]
        else:
            rep = x.mean(dim=1)

        logits = self.classifier(rep)  # (batch, n_classes)
        return logits

def build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    seq_len = int(cfg.get("seq_len"))
    n_features = int(cfg.get("n_features"))
    d_model = int(cfg.get("d_model", 128))
    n_heads = int(cfg.get("n_heads", 4))
    num_layers = int(cfg.get("num_layers", 4))
    dim_feedforward = int(cfg.get("dim_feedforward", 256))
    dropout = float(cfg.get("dropout", 0.1))
    pooling = cfg.get("pooling", "mean")
    n_classes = int(cfg.get("n_classes", 2))

    model = TransformerEncoderModel(
        seq_len=seq_len,
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pooling=pooling,
        n_classes=n_classes
    )
    return model

if __name__ == "__main__":
    # simple smoke test
    m = TransformerEncoderModel(seq_len=60, n_features=10)
    x = torch.randn(2, 60, 10)
    y = m(x)
    print("output shape:", y.shape)
