import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..models.base.base_model import BaseDeepModel


class Evaluator:
    def __init__(self, model: BaseDeepModel, device: Optional[str] = None):
        self.model = model

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if X_tensor.dim() == 1:
                X_tensor = X_tensor.unsqueeze(0)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """使用MC Dropout进行不确定性估计

        Args:
            X: 输入数据
            n_samples: MC采样次数

        Returns:
            (mean_predictions, std_predictions): 预测均值和标准差
        """
        self.model.train()

        X_tensor = torch.FloatTensor(X).to(self.device)
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)

        all_predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions = self.model(X_tensor).cpu().numpy()
                all_predictions.append(predictions)

        self.model.eval()

        all_predictions = np.array(all_predictions)
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        return mean_predictions, std_predictions

    def evaluate_with_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 50
    ) -> Dict[str, float]:
        """使用MC Dropout评估模型性能及不确定性

        Args:
            X: 输入数据
            y: 目标值
            n_samples: MC采样次数

        Returns:
            包含评估指标和不确定性统计的字典
        """
        mean_preds, std_preds = self.predict_with_uncertainty(X, n_samples)

        mae = mean_absolute_error(y, mean_preds)
        mse = mean_squared_error(y, mean_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, mean_preds)

        coverage_95 = np.mean(np.abs(y - mean_preds) <= 1.96 * std_preds) * 100

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mean_std': np.mean(std_preds),
            'coverage_95': coverage_95,
            'predictions': mean_preds,
            'uncertainties': std_preds
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        predictions = self.predict(X)

        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

    def evaluate_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        n_repeats: int = 3,
        random_state: int = 42
    ) -> List[Dict[str, float]]:
        from sklearn.model_selection import RepeatedKFold

        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        results = []

        for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X)):
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]

            fold_metrics = self.evaluate(X_test_fold, y_test_fold)
            results.append(fold_metrics)

        return results

    def get_predictions_and_targets(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        predictions = self.predict(X)
        return predictions, y

    def compute_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        predictions = self.predict(X)
        residuals = y - predictions
        return residuals

    def compute_per_sample_errors(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        predictions = self.predict(X)
        errors = np.abs(predictions - y)

        return {
            'predictions': predictions,
            'targets': y,
            'errors': errors,
            'squared_errors': errors ** 2
        }
