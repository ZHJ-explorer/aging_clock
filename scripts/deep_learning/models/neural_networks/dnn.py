import torch
import torch.nn as nn
from typing import List, Optional

from ..base.base_model import BaseDeepModel


class DeepMLP(BaseDeepModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        output_dim: int = 1,
        activation: str = 'relu',
        output_activation: Optional[str] = None
    ):
        super().__init__(input_dim, output_dim)

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU(0.2)
        elif activation == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            self.activation_fn = nn.ReLU()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))

            layers.append(self.activation_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers)

        self.output_layer = nn.Linear(prev_dim, output_dim)

        if output_activation == 'relu':
            self.output_activation_fn = nn.ReLU()
        elif output_activation == 'sigmoid':
            self.output_activation_fn = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation_fn = nn.Tanh()
        else:
            self.output_activation_fn = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)

        x = self.output_layer(x)

        if self.output_activation_fn is not None:
            x = self.output_activation_fn(x)

        if self.output_dim == 1:
            x = x.squeeze(-1)

        return x


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3, use_batchnorm: bool = True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batchnorm else nn.Identity()
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResNetMLP(BaseDeepModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_res_blocks: int = 4,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        output_dim: int = 1
    ):
        super().__init__(input_dim, output_dim)

        self.hidden_dim = hidden_dim
        self.n_res_blocks = n_res_blocks

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(n_res_blocks):
            layers.append(ResBlock(hidden_dim, dropout, use_batchnorm))

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        if self.output_dim == 1:
            x = x.squeeze(-1)
        return x
