import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseDeepModel(nn.Module, ABC):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_tensor = torch.FloatTensor(x)
            else:
                x_tensor = x
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
            return self.forward(x_tensor).cpu().numpy()

    def predict_with_uncertainty(self, x: np.ndarray, n_samples: int = 10) -> tuple:
        self.train()
        predictions = []
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
            for _ in range(n_samples):
                predictions.append(self.forward(x_tensor).cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        return predictions.mean(axis=0), predictions.std(axis=0)

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'BaseDeepModel':
        checkpoint = torch.load(path, map_location='cpu')
        model_kwargs = {k: v for k, v in kwargs.items() if k not in ['input_dim', 'output_dim']}
        model = cls(input_dim=checkpoint['input_dim'],
                    output_dim=checkpoint['output_dim'],
                    **model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
