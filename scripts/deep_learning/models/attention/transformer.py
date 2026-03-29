import torch
import torch.nn as nn
from typing import Optional

from ..base.base_model import BaseDeepModel


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


class Transformer(BaseDeepModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        super().__init__(input_dim, output_dim)

        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.pooling = nn.Linear(d_model, 1) if d_model > 1 else nn.Identity()

        self.output_head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_projection(x)

        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        for block in self.transformer_blocks:
            x = block(x)

        x = x.mean(dim=1)

        x = self.output_head(x)

        if self.output_dim == 1:
            x = x.squeeze(-1)

        return x
