import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scripts.deep_learning.models.neural_networks.dnn import DeepMLP, ResNetMLP
from scripts.deep_learning.models.attention.tabnet import TabNet
from scripts.deep_learning.models.base.trainer import DatasetWrapper
from scripts.deep_learning.training.trainer import Trainer
from scripts.deep_learning.training.optimizer import build_optimizer, build_scheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PREPROCESSED_DIR = 'preprocessed_data'
SELECTED_FEATURES_DIR = 'selected_features'
OPTUNA_DIR = 'optuna_results'
ENSEMBLE_DIR = 'ensemble_results'

def load_data():
    merged_csv = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    mask_path = os.path.join(SELECTED_FEATURES_DIR, 'feature_mask.npy')

    df = pd.read_csv(merged_csv, index_col=0)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if 'age' not in numeric_columns:
        numeric_columns = list(numeric_columns) + ['age']
    df = df[numeric_columns]
    df = df.dropna(axis=1, how='any')
    df = df.dropna()
    df = df[df['age'] >= 20]

    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        X = df.drop('age', axis=1).values[:, mask]
    else:
        X = df.drop('age', axis=1).values
    y = df['age'].values
    return X, y

def build_model(model_name, input_dim, best_params, train_kwargs, device):
    if model_name == 'DeepMLP':
        hidden_dims = []
        base_dim = best_params['base_dim']
        for i in range(best_params['n_layers']):
            hidden_dims.append(base_dim // (2 ** i))
        model = DeepMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=best_params['dropout'],
            use_batchnorm=best_params['use_batchnorm'],
            activation=best_params['activation'],
            output_dim=1
        )
    elif model_name == 'ResNetMLP':
        model = ResNetMLP(
            input_dim=input_dim,
            hidden_dim=best_params['hidden_dim'],
            n_res_blocks=best_params['n_res_blocks'],
            dropout=best_params['dropout'],
            use_batchnorm=best_params['use_batchnorm'],
            output_dim=1
        )
    elif model_name == 'TabNet':
        model = TabNet(
            input_dim=input_dim,
            n_d=best_params['n_d'],
            n_a=best_params['n_a'],
            n_steps=best_params['n_steps'],
            output_dim=1
        )

    optimizer = build_optimizer(model, 'adamw', learning_rate=train_kwargs['lr'], weight_decay=train_kwargs['weight_decay'])
    scheduler = build_scheduler(optimizer, 'cosine', epochs=200)
    criterion = nn.MSELoss()
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device,
                      scheduler=scheduler, early_stopping_patience=30, l1_reg=0.0, l2_reg=1e-4)
    return trainer

def train_optimized_models(X_train, y_train, X_val, y_val, device):
    logger.info("加载最优超参数...")
    with open(os.path.join(OPTUNA_DIR, 'optuna_summary.json'), 'r') as f:
        optuna_results = json.load(f)

    results = {}

    for model_name in ['DeepMLP', 'ResNetMLP', 'TabNet']:
        logger.info(f"\n训练优化后的 {model_name}...")
        best_params = optuna_results['best_params'][model_name.lower()]
        train_kwargs = {
            'lr': best_params.pop('lr'),
            'weight_decay': best_params.pop('weight_decay'),
            'batch_size': best_params.pop('batch_size')
        }

        trainer = build_model(model_name, X_train.shape[1], best_params, train_kwargs, device)

        train_dataset = DatasetWrapper(X_train, y_train)
        val_dataset = DatasetWrapper(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=int(train_kwargs['batch_size']), shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=int(train_kwargs['batch_size']), shuffle=False, num_workers=0)

        trainer.fit(train_loader, val_loader, epochs=200)
        val_results = trainer.evaluate(val_loader)

        results[model_name] = {
            'trainer': trainer,
            'val_predictions': val_results['predictions'],
            'val_targets': val_results['targets'],
            'mae': val_results['mae'],
            'r2': val_results['r2']
        }
        logger.info(f"{model_name} 验证集 MAE: {val_results['mae']:.4f}, R²: {val_results['r2']:.4f}")

    return results

def weighted_average_ensemble(predictions, weights=None):
    if weights is None:
        weights = np.ones(len(predictions)) / len(predictions)
    else:
        weights = np.array(weights) / np.sum(weights)
    return np.average(predictions, axis=0, weights=weights)

def optimize_weights(results, y_val):
    best_mae = float('inf')
    best_weights = None

    for w1 in np.arange(0.1, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.8 - w1, 0.1):
            w3 = 1 - w1 - w2
            if w3 < 0.1:
                continue
            weights = [w1, w2, w3]
            preds = weighted_average_ensemble([results['DeepMLP']['val_predictions'],
                                              results['ResNetMLP']['val_predictions'],
                                              results['TabNet']['val_predictions']], weights)
            mae = mean_absolute_error(y_val, preds)
            if mae < best_mae:
                best_mae = mae
                best_weights = weights

    return best_weights, best_mae

def train_stacking_ensemble(X_train, y_train, X_val, y_val, dl_predictions_train, device):
    logger.info("\n训练Stacking集成模型...")

    dl_train_preds = np.column_stack([
        dl_predictions_train['DeepMLP'],
        dl_predictions_train['ResNetMLP'],
        dl_predictions_train['TabNet']
    ])
    dl_val_preds = np.column_stack([
        dl_predictions_train['DeepMLP_val'],
        dl_predictions_train['ResNetMLP_val'],
        dl_predictions_train['TabNet_val']
    ])

    xgb_meta = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb_meta.fit(dl_train_preds, y_train)

    stacking_val_preds = xgb_meta.predict(dl_val_preds)
    stacking_mae = mean_absolute_error(y_val, stacking_val_preds)
    stacking_r2 = r2_score(y_val, stacking_val_preds)

    logger.info(f"Stacking集成 验证集 MAE: {stacking_mae:.4f}, R²: {stacking_r2:.4f}")
    return xgb_meta, stacking_val_preds, stacking_mae, stacking_r2

def main():
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    X, y = load_data()
    logger.info(f"数据加载完成: {X.shape[0]}样本, {X.shape[1]}特征")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

    results = train_optimized_models(X_train, y_train, X_val, y_val, device)

    logger.info("\n=== 优化后单模型结果 ===")
    for model_name, result in results.items():
        logger.info(f"{model_name}: MAE={result['mae']:.4f}, R²={result['r2']:.4f}")

    best_weights, weighted_mae = optimize_weights(results, y_val)
    logger.info(f"\n=== 加权平均集成 ===")
    logger.info(f"最优权重: DeepMLP={best_weights[0]:.2f}, ResNetMLP={best_weights[1]:.2f}, TabNet={best_weights[2]:.2f}")
    logger.info(f"加权平均集成 MAE: {weighted_mae:.4f}")

    dl_predictions_train = {}
    for model_name in ['DeepMLP', 'ResNetMLP', 'TabNet']:
        trainer = results[model_name]['trainer']
        train_dataset = DatasetWrapper(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        train_results = trainer.evaluate(train_loader)
        dl_predictions_train[model_name] = train_results['predictions']
        dl_predictions_train[f'{model_name}_val'] = results[model_name]['val_predictions']

    stacking_model, stacking_val_preds, stacking_mae, stacking_r2 = train_stacking_ensemble(
        X_train, y_train, X_val, y_val, dl_predictions_train, device)

    logger.info("\n=== 测试集最终评估 ===")
    final_results = {}

    for model_name, result in results.items():
        trainer = result['trainer']
        test_dataset = DatasetWrapper(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_results = trainer.evaluate(test_loader)
        final_results[model_name] = {
            'mae': test_results['mae'],
            'r2': test_results['r2'],
            'predictions': test_results['predictions']
        }
        logger.info(f"{model_name}: MAE={test_results['mae']:.4f}, R²={test_results['r2']:.4f}")

    ensemble_preds = weighted_average_ensemble([final_results['DeepMLP']['predictions'],
                                               final_results['ResNetMLP']['predictions'],
                                               final_results['TabNet']['predictions']], best_weights)
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    ensemble_r2 = r2_score(y_test, ensemble_preds)
    logger.info(f"\n加权平均集成: MAE={ensemble_mae:.4f}, R²={ensemble_r2:.4f}")

    dl_test_preds = np.column_stack([final_results['DeepMLP']['predictions'],
                                      final_results['ResNetMLP']['predictions'],
                                      final_results['TabNet']['predictions']])
    stacking_test_preds = stacking_model.predict(dl_test_preds)
    stacking_test_mae = mean_absolute_error(y_test, stacking_test_preds)
    stacking_test_r2 = r2_score(y_test, stacking_test_preds)
    logger.info(f"Stacking集成: MAE={stacking_test_mae:.4f}, R²={stacking_test_r2:.4f}")

    xgb_baseline = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    xgb_baseline.fit(X_train, y_train)
    xgb_preds = xgb_baseline.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)
    logger.info(f"\nXGBoost基准: MAE={xgb_mae:.4f}, R²={xgb_r2:.4f}")

    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    summary = {
        'single_models': {name: {'mae': float(r['mae']), 'r2': float(r['r2'])} for name, r in final_results.items()},
        'weighted_ensemble': {'weights': [float(w) for w in best_weights], 'mae': float(ensemble_mae), 'r2': float(ensemble_r2)},
        'stacking_ensemble': {'mae': float(stacking_test_mae), 'r2': float(stacking_test_r2)},
        'xgboost_baseline': {'mae': float(xgb_mae), 'r2': float(xgb_r2)}
    }
    with open(os.path.join(ENSEMBLE_DIR, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    end_time = time.time()
    logger.info(f"\n总耗时: {(end_time - start_time)/60:.1f} 分钟")

if __name__ == "__main__":
    main()