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
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scripts.deep_learning.models.neural_networks.dnn import DeepMLP, ResNetMLP
from scripts.deep_learning.models.attention.tabnet import TabNet
from scripts.deep_learning.models.base.trainer import DatasetWrapper
from scripts.deep_learning.training.trainer import Trainer
from scripts.deep_learning.training.optimizer import build_optimizer, build_scheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PREPROCESSED_DIR = 'preprocessed_data'
OPTUNA_DIR = 'optuna_results'
SELECTED_FEATURES_DIR = 'selected_features'

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
        feature_names = df.drop('age', axis=1).columns[mask].tolist()
    else:
        X = df.drop('age', axis=1).values
        feature_names = df.drop('age', axis=1).columns.tolist()

    y = df['age'].values
    logger.info(f"数据加载完成: {X.shape[0]}样本, {X.shape[1]}特征")
    return X, y, feature_names

def train_and_evaluate(model_class, model_kwargs, train_kwargs, X_train, y_train, X_val, y_val, device, epochs=50):
    model = model_class(**model_kwargs, output_dim=1)
    optimizer = build_optimizer(model, 'adamw', learning_rate=train_kwargs.get('lr', 0.001), weight_decay=train_kwargs.get('weight_decay', 1e-4))
    scheduler = build_scheduler(optimizer, 'cosine', epochs=epochs)
    criterion = nn.MSELoss()

    train_dataset = DatasetWrapper(X_train, y_train)
    val_dataset = DatasetWrapper(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=int(train_kwargs.get('batch_size', 32)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(train_kwargs.get('batch_size', 32)), shuffle=False, num_workers=0)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device,
                      scheduler=scheduler, early_stopping_patience=15, l1_reg=0.0, l2_reg=1e-4)

    trainer.fit(train_loader, val_loader, epochs=epochs)
    results = trainer.evaluate(val_loader)
    return results['mae']

def objective_deepmlp(trial, X_train, y_train, X_val, y_val, device):
    hidden_dims = []
    n_layers = trial.suggest_int('n_layers', 2, 5)
    base_dim = trial.suggest_categorical('base_dim', [128, 256, 512])
    for i in range(n_layers):
        hidden_dims.append(base_dim // (2 ** i))

    model_kwargs = {
        'input_dim': X_train.shape[1],
        'hidden_dims': hidden_dims,
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'use_batchnorm': trial.suggest_categorical('use_batchnorm', [True, False]),
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu'])
    }
    train_kwargs = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }
    return train_and_evaluate(DeepMLP, model_kwargs, train_kwargs, X_train, y_train, X_val, y_val, device, epochs=50)

def objective_resnetmlp(trial, X_train, y_train, X_val, y_val, device):
    model_kwargs = {
        'input_dim': X_train.shape[1],
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
        'n_res_blocks': trial.suggest_int('n_res_blocks', 2, 6),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'use_batchnorm': trial.suggest_categorical('use_batchnorm', [True, False])
    }
    train_kwargs = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }
    return train_and_evaluate(ResNetMLP, model_kwargs, train_kwargs, X_train, y_train, X_val, y_val, device, epochs=50)

def objective_tabnet(trial, X_train, y_train, X_val, y_val, device):
    model_kwargs = {
        'input_dim': X_train.shape[1],
        'n_d': trial.suggest_categorical('n_d', [32, 64, 128]),
        'n_a': trial.suggest_categorical('n_a', [32, 64, 128]),
        'n_steps': trial.suggest_int('n_steps', 2, 5)
    }
    train_kwargs = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }
    return train_and_evaluate(TabNet, model_kwargs, train_kwargs, X_train, y_train, X_val, y_val, device, epochs=50)

def run_optuna(model_name, objective_fn, X_train, y_train, X_val, y_val, device, n_trials=50):
    logger.info(f"\n{'='*50}")
    logger.info(f"Optuna优化: {model_name} ({n_trials} trials)")
    logger.info(f"{'='*50}")

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective_fn(trial, X_train, y_train, X_val, y_val, device), n_trials=n_trials, show_progress_bar=False)

    logger.info(f"最优MAE: {study.best_value:.4f}")
    logger.info(f"最优参数: {study.best_params}")

    os.makedirs(os.path.join(OPTUNA_DIR, model_name.lower()), exist_ok=True)
    with open(os.path.join(OPTUNA_DIR, model_name.lower(), 'best_params.json'), 'w') as f:
        json.dump(study.best_params, f, indent=2)

    return study.best_params, study.best_value

def main():
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    X, y, feature_names = load_data()

    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

    best_params = {}
    best_values = {}

    best_params['deepmlp'], best_values['deepmlp'] = run_optuna('DeepMLP', objective_deepmlp, X_train, y_train, X_val, y_val, device, n_trials=50)
    best_params['resnetmlp'], best_values['resnetmlp'] = run_optuna('ResNetMLP', objective_resnetmlp, X_train, y_train, X_val, y_val, device, n_trials=50)
    best_params['tabnet'], best_values['tabnet'] = run_optuna('TabNet', objective_tabnet, X_train, y_train, X_val, y_val, device, n_trials=50)

    summary = {
        'best_params': best_params,
        'best_validation_mae': best_values,
        'feature_count': X_train.shape[1]
    }
    with open(os.path.join(OPTUNA_DIR, 'optuna_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    end_time = time.time()
    logger.info(f"\nOptuna优化完成! 总耗时: {(end_time - start_time)/60:.1f} 分钟")

if __name__ == "__main__":
    main()