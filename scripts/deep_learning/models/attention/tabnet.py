import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from ..base.base_model import BaseDeepModel


class TabNet(BaseDeepModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.3,
        epsilon: float = 1e-15,
        momentum: float = 0.02
    ):
        super().__init__(input_dim, output_dim)

        self.n_d = n_d
        self.n_a = n_a if n_a >= input_dim else input_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon

        self.initial_mapping = nn.Linear(input_dim, n_d + self.n_a, bias=False)

        self.batch_norm = nn.BatchNorm1d(input_dim, momentum=momentum)

        self.fc_transformers = nn.ModuleList()
        for step in range(n_steps):
            self.fc_transformers.append(
                nn.Sequential(
                    nn.Linear(n_d, n_d * 2, bias=False),
                    nn.BatchNorm1d(n_d * 2, momentum=momentum),
                    nn.ReLU()
                )
            )

        self.final_mapping = nn.Linear(n_d, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.batch_norm(x)

        prior_scales = torch.ones(batch_size, self.input_dim, device=x.device)

        processed_x = self.initial_mapping(x)
        d = processed_x[:, :self.n_d]
        a = processed_x[:, self.n_d:]
        scale_agg = d

        for step in range(self.n_steps):
            if step > 0:
                a_truncated = a[:, :self.input_dim]
                mask_values = torch.softmax(a_truncated, dim=-1)
                prior_scales = prior_scales * self.gamma * (1 - mask_values)

                masked_x = x * prior_scales
                processed_x = self.initial_mapping(masked_x)
                d = processed_x[:, :self.n_d]
                a = processed_x[:, self.n_d:]

            d = self.fc_transformers[step](d)
            scale_agg = scale_agg + d[:, :self.n_d]

        scale_agg = scale_agg / self.n_steps

        output = self.final_mapping(scale_agg)

        if self.output_dim == 1:
            output = output.squeeze(-1)

        return output
