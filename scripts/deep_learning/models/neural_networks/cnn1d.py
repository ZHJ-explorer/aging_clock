import torch
import torch.nn as nn
from typing import List, Optional

from ..base.base_model import BaseDeepModel


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualConv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels or stride != 1 else None

        self.final_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.final_activation(out)

        return out


class CNN1D(BaseDeepModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        use_batchnorm: bool = True
    ):
        super().__init__(input_dim, output_dim)

        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 128]

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        layers = []
        in_channels = 1

        for i, out_channels in enumerate(hidden_channels):
            layers.append(
                Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)

        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc_layers(x)

        if self.output_dim == 1:
            x = x.squeeze(-1)

        return x


class ResCNN1D(BaseDeepModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_channels: List[int] = None,
        n_res_blocks: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super().__init__(input_dim, output_dim)

        if hidden_channels is None:
            hidden_channels = [64, 128, 256]

        self.hidden_channels = hidden_channels

        self.input_conv = nn.Conv1d(1, hidden_channels[0], kernel_size=7, stride=2, padding=3)
        self.input_bn = nn.BatchNorm1d(hidden_channels[0])
        self.input_activation = nn.ReLU()
        self.input_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv_blocks = nn.ModuleList()
        in_ch = hidden_channels[0]
        for ch in hidden_channels:
            blocks = []
            for _ in range(n_res_blocks):
                blocks.append(
                    ResidualConv1DBlock(
                        in_channels=in_ch,
                        out_channels=ch,
                        kernel_size=kernel_size,
                        stride=1,
                        dropout=dropout
                    )
                )
                in_ch = ch
            self.conv_blocks.append(nn.ModuleList(blocks))

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.input_activation(x)
        x = self.input_pool(x)

        for block_group in self.conv_blocks:
            for block in block_group:
                x = block(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        if self.output_dim == 1:
            x = x.squeeze(-1)

        return x
