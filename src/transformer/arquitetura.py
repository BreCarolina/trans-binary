"""
model_transformer.py

Arquitetura Transformer encoder leve pronta para classificação binária/ternária.
Arquivo pensado para `src/transformer/model.py`.

Características:
- Projeção de entrada (n_features -> d_model)
- Positional embeddings aprendíveis com interpolação quando seq_len variar
- Encoder Transformer padrão (batch_first=True)
- Pooling (mean / max / cls)
- Classificador com LayerNorm, MLP e regularização
- Inicialização Xavier para pesos lineares
- Função auxiliar build_model_from_config(cfg)

Smoke-test no final do arquivo.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    """Positional encoding aprendível com suporte a interpolação de comprimento.

    Args:
        seq_len: comprimento de posição originalmente treinado
        d_model: dimensão do embedding
    """

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        # parâmetros (seq_len, d_model)
        pe = torch.zeros(self.seq_len, self.d_model)
        nn.init.normal_(pe, mean=0.0, std=0.02)
        self.pe = nn.Parameter(pe)  # (seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soma as positional encodings ao tensor de entrada.

        Se a sequência de entrada tiver comprimento diferente de self.seq_len,
        realiza interpolação linear nas posições.

        x: (batch, seq_len_in, d_model)
        returns: (batch, seq_len_in, d_model)
        """
        seq_in = x.size(1)
        if seq_in == self.seq_len:
            pe = self.pe.to(x.device).to(x.dtype)  # (seq_len, d_model)
            return x + pe.unsqueeze(0)

        # Interpola ao longo da dimensão temporal (seq_len)
        # Permutar para (1, d_model, seq_len) para usar F.interpolate
        pe = self.pe.unsqueeze(0).permute(0, 2, 1)  # (1, d_model, seq_len)
        pe = F.interpolate(pe, size=seq_in, mode="linear", align_corners=False)
        pe = pe.permute(0, 2, 1).squeeze(0)  # (seq_in, d_model)
        pe = pe.to(x.device).to(x.dtype)
        return x + pe.unsqueeze(0)


class TransformerEncoderModel(nn.Module):
    """Transformer encoder para classificação sobre séries temporais.

    Args:
        seq_len: comprimento de sequência esperado (usado especialmente pelas pos. encodings)
        n_features: número de features de entrada
        d_model, n_heads, num_layers, dim_feedforward, dropout: parâmetros do encoder
        pooling: 'mean' | 'max' | 'cls' — estratégia de agregação temporal
        n_classes: número de classes de saída
    """

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
        n_classes: int = 2,
        use_layernorm_input: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.pooling = pooling

        # Entrada -> projeção para d_model
        self.input_proj = nn.Linear(self.n_features, self.d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.use_layernorm_input = use_layernorm_input
        if self.use_layernorm_input:
            self.input_ln = nn.LayerNorm(self.d_model)

        # Positional encodings aprendíveis
        self.pos_enc = LearnablePositionalEncoding(self.seq_len, self.d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classificador
        mid = max(8, self.d_model // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, n_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Inicialização Xavier para tensores com dimensão >1
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, return_proba: bool = False) -> torch.Tensor:
        """Forward pass.

        x: (batch, seq_len_in, n_features)
        return: logits (batch, n_classes) ou (probs, logits) se return_proba True
        """
        # projeção
        x = self.input_proj(x)  # -> (batch, seq_len_in, d_model)
        if self.use_layernorm_input:
            x = self.input_ln(x)
        x = self.input_dropout(x)

        # add positional encodings (interpolará se necessário)
        x = self.pos_enc(x)

        # encoder
        x = self.encoder(x)  # (batch, seq_len_in, d_model)

        # pooling
        if self.pooling == "mean":
            rep = x.mean(dim=1)
        elif self.pooling == "max":
            rep, _ = x.max(dim=1)
        elif self.pooling == "cls":
            rep = x[:, 0, :]
        else:
            rep = x.mean(dim=1)

        logits = self.classifier(rep)
        if return_proba:
            probs = F.softmax(logits, dim=-1)
            return probs, logits
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
        n_classes=n_classes,
    )
    return model


# Smoke test
if __name__ == "__main__":
    m = TransformerEncoderModel(seq_len=60, n_features=10)
    x = torch.randn(2, 60, 10)
    y = m(x)
    print("output shape:", y.shape)
