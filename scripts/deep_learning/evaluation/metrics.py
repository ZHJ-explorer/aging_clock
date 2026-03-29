import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def compute_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mape


def compute_symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    smape = np.mean(
        np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8)
    ) * 100
    return smape


def compute_r2_adjusted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int
) -> float:
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adjusted_r2


def compute_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    return np.corrcoef(y_true, y_pred)[0, 1]


def compute_mean_bias_deviation(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    return np.mean(y_pred - y_true)


def compute_median_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    return np.median(np.abs(y_true - y_pred))


def compute_quantile_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: List[float] = None
) -> Dict[str, float]:
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]

    errors = np.abs(y_true - y_pred)

    result = {}
    for q in quantiles:
        result[f'q{int(q*100)}'] = np.quantile(errors, q)

    return result
