import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from ..base.base_model import BaseDeepModel
from ..neural_networks.dnn import DeepMLP


class DLEnsemble(BaseDeepModel):
    def __init__(
        self,
        models: List[BaseDeepModel],
        weights: Optional[List[float]] = None
    ):
        input_dim = models[0].input_dim
        output_dim = models[0].output_dim

        super().__init__(input_dim, output_dim)

        self.models = nn.ModuleList(models)
        self.model_names = [type(m).__name__ for m in models]

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []

        for model in self.models:
            output = model(x)
            outputs.append(output)

        weighted_outputs = []
        for output, weight in zip(outputs, self.weights):
            weighted_outputs.append(output * weight)

        ensemble_output = sum(weighted_outputs)

        return ensemble_output

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_tensor = torch.FloatTensor(x)
            else:
                x_tensor = x
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)

            predictions = []
            for model in self.models:
                pred = model(x_tensor).cpu().numpy()
                predictions.append(pred)

            predictions = np.array(predictions)
            weighted_predictions = np.sum(predictions * np.array(self.weights).reshape(-1, 1), axis=0)

        return weighted_predictions

    def predict_with_uncertainty(self, x: np.ndarray) -> tuple:
        self.eval()

        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_tensor = torch.FloatTensor(x)
            else:
                x_tensor = x
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)

            predictions = []
            for model in self.models:
                pred = model(x_tensor).cpu().numpy()
                predictions.append(pred)

            predictions = np.array(predictions)

            mean_pred = np.sum(predictions * np.array(self.weights).reshape(-1, 1), axis=0)
            std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred

    def add_model(self, model: BaseDeepModel, weight: float = 1.0) -> None:
        assert model.input_dim == self.input_dim, "Model input dimension must match"
        assert model.output_dim == self.output_dim, "Model output dimension must match"

        self.models.append(model)
        self.weights.append(weight)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def set_weights(self, weights: List[float]) -> None:
        assert len(weights) == len(self.models), "Number of weights must match number of models"
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def get_model_weights(self) -> dict:
        return {name: weight for name, weight in zip(self.model_names, self.weights)}
