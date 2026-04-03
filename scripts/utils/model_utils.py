import numpy as np
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_ml_training_history(history: dict, model_name: str, save_path: str = None):
    if not history or not history.get('train_loss'):
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if history.get('val_loss'):
        plt.plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'{model_name}: Training Loss Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_training_loss.png'), dpi=150)
        plt.close()


def plot_ml_prediction_vs_actual(predictions: np.ndarray, targets: np.ndarray,
                                   model_name: str, metrics: dict = None, save_path: str = None):
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, predictions, alpha=0.5, edgecolors='none')
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual Age', fontsize=12)
    plt.ylabel('Predicted Age', fontsize=12)
    plt.title(f'{model_name}: Prediction vs Actual', fontsize=14)

    if metrics:
        metrics_text = f"MAE: {metrics.get('mae', 0):.4f}\nRMSE: {metrics.get('rmse', 0):.4f}\nR²: {metrics.get('r2', 0):.4f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_prediction_vs_actual.png'), dpi=150)
        plt.close()


def plot_ml_residuals(predictions: np.ndarray, targets: np.ndarray,
                        model_name: str, save_path: str = None):
    residuals = targets - predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.5, edgecolors='none')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Age', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title(f'{model_name}: Residual Plot', fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_residuals.png'), dpi=150)
        plt.close()


def plot_ml_error_distribution(predictions: np.ndarray, targets: np.ndarray,
                                  model_name: str, save_path: str = None):
    errors = np.abs(predictions - targets)
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    plt.axvline(x=np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
    plt.xlabel('Absolute Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name}: Error Distribution', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_error_distribution.png'), dpi=150)
        plt.close()


def train_models(X_train, y_train, X_val, y_val):
    """训练多个模型"""
    logger.info("训练模型...")
    from sklearn.linear_model import Lasso

    xgb_history = {'train_loss': [], 'val_loss': []}
    lgb_history = {'train_loss': [], 'val_loss': []}

    class XGBHistoryCallback:
        def __init__(self):
            self.train_losses = []
            self.val_losses = []

        def after_iteration(self, model, epoch, evals):
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            train_loss = mean_squared_error(y_train, train_pred)
            val_loss = mean_squared_error(y_val, val_pred)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            return False

    class LGBHistoryCallback:
        def __init__(self):
            self.train_losses = []
            self.val_losses = []

        def __call__(self, env):
            train_loss = env.evaluation_result_list[0][2]
            val_loss = env.evaluation_result_list[1][2]
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    xgb_callback = XGBHistoryCallback()
    lgb_callback = LGBHistoryCallback()

    logger.info("使用默认参数...")
    base_models = [
        ('xgboost', xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            callbacks=[xgb.callback.TrainingCallback(
                callback=xgb_callback.after_iteration
            )] if hasattr(xgb, 'callback') else None
        )),
        ('lightgbm', lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
            callbacks=[lgb_callback] if hasattr(lgb, 'callback') else None
        )),
        ('ridge', Ridge(alpha=1.0, random_state=42)),
        ('lasso', Lasso(alpha=0.01, random_state=42, max_iter=10000)),
        ('elasticnet', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)),
        ('svr', SVR(kernel='rbf', C=100.0, gamma='scale', cache_size=1000)),
        ('random_forest', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ]

    from joblib import Parallel, delayed

    def train_model(name, model):
        logger.info(f"训练{name}模型...")
        if name == 'lightgbm':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae')
        elif name == 'xgboost':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train)
        return (name, model)

    trained_base_models = Parallel(n_jobs=-1)(
        delayed(train_model)(name, model) for name, model in base_models
    )

    xgb_history['train_loss'] = xgb_callback.train_losses
    xgb_history['val_loss'] = xgb_callback.val_losses
    lgb_history['train_loss'] = lgb_callback.train_losses
    lgb_history['val_loss'] = lgb_callback.val_losses

    if trained_base_models[0][0] == 'xgboost':
        trained_base_models[0][1].set_attr(xgb_history=xgb_history)
    if trained_base_models[1][0] == 'lightgbm':
        trained_base_models[1][1].set_attr(lgb_history=lgb_history)

    logger.info("训练Stacking模型...")
    stacking_model = StackingRegressor(
        estimators=trained_base_models,
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=10, scoring='r2'),
        cv=10
    )
    stacking_model.fit(X_train, y_train)

    model_dict = dict(trained_base_models)
    xgb_model = model_dict.get('xgboost')
    lgb_model = model_dict.get('lightgbm')
    ridge_model = model_dict.get('ridge')
    lasso_model = model_dict.get('lasso')
    en_model = model_dict.get('elasticnet')
    svr_model = model_dict.get('svr')

    model_histories = {
        'xgboost': xgb_history,
        'lightgbm': lgb_history
    }

    return ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, model_histories


def evaluate_model(model, X_test, y_test, model_name, output_file='test_result.txt'):
    """评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
        model_name: 模型名称
        output_file: 输出文件路径，默认为 'test_result.txt'
    
    Returns:
        tuple: (MAE, RMSE, R², 预测值)
    """
    logger.info(f"评估 {model_name} 模型...")
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"{model_name} 模型性能:")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2: {r2:.4f}")
    
    result_lines = [
        f"\n{model_name} 模型测试结果:",
        f"MAE: {mae:.4f}",
        f"RMSE: {rmse:.4f}",
        f"R2: {r2:.4f}",
        "预测值,实际值"
    ]
    
    for pred, actual in zip(y_pred, y_test):
        result_lines.append(f"{pred:.4f},{actual:.4f}")
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(result_lines) + '\n')

    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
    plot_ml_prediction_vs_actual(y_pred, y_test, model_name, metrics)
    plot_ml_residuals(y_pred, y_test, model_name)
    plot_ml_error_distribution(y_pred, y_test, model_name)

    return mae, rmse, r2, y_pred


def save_models(ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, suffix=""):
    """保存训练好的模型"""
    logger.info("保存模型...")
    
    # 确保模型目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 保存模型，跳过None值
    joblib.dump(ridge_model, os.path.join(MODELS_DIR, f'ridge_model{suffix}.pkl'))
    if lasso_model is not None:
        joblib.dump(lasso_model, os.path.join(MODELS_DIR, f'lasso_model{suffix}.pkl'))
    if en_model is not None:
        joblib.dump(en_model, os.path.join(MODELS_DIR, f'elasticnet_model{suffix}.pkl'))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, f'xgboost_model{suffix}.pkl'))
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, f'lightgbm_model{suffix}.pkl'))
    joblib.dump(svr_model, os.path.join(MODELS_DIR, f'svr_model{suffix}.pkl'))
    joblib.dump(stacking_model, os.path.join(MODELS_DIR, f'stacking_model{suffix}.pkl'))
    
    logger.info("模型保存完成")
