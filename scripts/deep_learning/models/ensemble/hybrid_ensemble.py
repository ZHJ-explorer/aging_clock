import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

logger = logging.getLogger(__name__)


class HybridEnsemble:
    def __init__(
        self,
        traditional_models: List[Tuple[str, any]],
        deep_learning_models: List[Tuple[str, any]],
        device: str = 'cuda'
    ):
        self.traditional_models = traditional_models
        self.deep_learning_models = deep_learning_models
        self.device = device

        self.weights = None
        self.best_score = -float('inf')

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []

        for name, model in self.traditional_models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred)

        import torch
        for name, model in self.deep_learning_models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                if X_tensor.dim() == 1:
                    X_tensor = X_tensor.unsqueeze(0)
                pred = model(X_tensor).cpu().numpy()
                predictions.append(pred)

        predictions = np.array(predictions)

        if self.weights is None:
            return predictions.mean(axis=0)
        else:
            return np.sum(predictions * np.array(self.weights).reshape(-1, 1), axis=0)

    def tune_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        n_repeats: int = 3
    ):
        logger.info("开始权重调优...")

        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

        all_predictions = []

        for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X)):
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]

            fold_predictions = []

            for name, model in self.traditional_models:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_test_fold)
                    fold_predictions.append(pred)

            import torch
            for name, model in self.deep_learning_models:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test_fold).to(self.device)
                    if X_tensor.dim() == 1:
                        X_tensor = X_tensor.unsqueeze(0)
                    pred = model(X_tensor).cpu().numpy()
                    fold_predictions.append(pred)

            all_predictions.append((np.array(fold_predictions), y_test_fold))

        best_score = -float('inf')
        best_weights = None

        n_models = len(self.traditional_models) + len(self.deep_learning_models)

        xgb_weight_range = np.arange(0.5, 0.96, 0.05)

        for xgb_weight in xgb_weight_range:
            remaining_weight = 1 - xgb_weight
            other_weights = remaining_weight / (n_models - 1) if n_models > 1 else 0

            weights = []
            for name, model in self.traditional_models:
                if 'xgb' in name.lower() or 'xgboost' in name.lower():
                    weights.append(xgb_weight)
                else:
                    weights.append(other_weights)

            for name, model in self.deep_learning_models:
                weights.append(other_weights)

            weights = np.array(weights) / sum(weights)

            total_r2 = 0
            total_mae = 0

            for preds, targets in all_predictions:
                weighted_pred = np.sum(preds * weights.reshape(-1, 1), axis=0)
                total_r2 += r2_score(targets, weighted_pred)
                total_mae += mean_absolute_error(targets, weighted_pred)

            avg_r2 = total_r2 / len(all_predictions)
            avg_mae = total_mae / len(all_predictions)

            score = avg_r2 - 0.1 * avg_mae

            if score > best_score:
                best_score = score
                best_weights = weights

        self.weights = best_weights
        self.best_score = best_score

        logger.info(f"最优权重: {best_weights}")
        logger.info(f"最优综合评分: {best_score:.4f}")

        return best_weights

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
