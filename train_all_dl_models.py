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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.deep_learning.models.neural_networks.dnn import DeepMLP, ResNetMLP
from scripts.deep_learning.models.neural_networks.cnn1d import CNN1D, ResCNN1D
from scripts.deep_learning.models.attention.transformer import Transformer
from scripts.deep_learning.models.attention.tabnet import TabNet
from scripts.deep_learning.models.base.trainer import DatasetWrapper
from scripts.deep_learning.training.trainer import Trainer
from scripts.deep_learning.training.optimizer import build_optimizer, build_scheduler
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PREPROCESSED_DIR = 'preprocessed_data'
MODELS_DIR = 'models/deep_learning'

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    if 'age' not in df.columns:
        raise ValueError("DataFrame must contain 'age' column")
    X = df.drop('age', axis=1)
    y = df['age']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=random_state)
    val_adjusted_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_adjusted_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_and_preprocess_data():
    merged_csv = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    if not os.path.exists(merged_csv):
        raise FileNotFoundError(f"数据文件不存在: {merged_csv}")

    df = pd.read_csv(merged_csv, index_col=0)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if 'age' not in numeric_columns:
        numeric_columns = list(numeric_columns) + ['age']
    df = df[numeric_columns]
    df = df.dropna(axis=1, how='any')
    df = df.dropna()
    df = df[df['age'] >= 20]
    return df

def get_model_configs(input_dim):
    return [
        ('DeepMLP', DeepMLP, {
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128, 64],
            'dropout': 0.3,
            'use_batchnorm': True,
            'activation': 'relu'
        }),
        ('ResNetMLP', ResNetMLP, {
            'input_dim': input_dim,
            'hidden_dim': 256,
            'n_res_blocks': 4,
            'dropout': 0.3,
            'use_batchnorm': True
        }),
        ('CNN1D', CNN1D, {
            'input_dim': input_dim,
            'hidden_channels': [64, 128, 256],
            'kernel_size': 3,
            'dropout': 0.3
        }),
        ('ResCNN1D', ResCNN1D, {
            'input_dim': input_dim,
            'hidden_channels': [64, 128, 256],
            'n_res_blocks': 2,
            'kernel_size': 3,
            'dropout': 0.3
        }),
        ('Transformer', Transformer, {
            'input_dim': input_dim,
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'ff_dim': 256,
            'dropout': 0.3
        }),
        ('TabNet', TabNet, {
            'input_dim': input_dim,
            'n_d': 64,
            'n_a': 64,
            'n_steps': 3
        }),
    ]

def train_and_evaluate(model_name, model_class, model_kwargs, X_train, y_train, X_val, y_val, X_test, y_test, device):
    logger.info(f"\n{'='*50}")
    logger.info(f"训练模型: {model_name}")
    logger.info(f"{'='*50}")

    model = model_class(**model_kwargs, output_dim=1)
    optimizer = build_optimizer(model, 'adamw', learning_rate=0.001, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, 'cosine', epochs=100)
    criterion = nn.MSELoss()

    train_dataset = DatasetWrapper(X_train, y_train)
    val_dataset = DatasetWrapper(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=20,
        l1_reg=0.0,
        l2_reg=1e-4
    )

    history = trainer.fit(train_loader, val_loader, epochs=100)

    test_dataset = DatasetWrapper(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    results = trainer.evaluate(test_loader)

    logger.info(f"{model_name} 测试结果 - MSE: {results['loss']:.4f}, MAE: {results['mae']:.4f}, R²: {results['r2']:.4f}")

    save_dir = os.path.join(MODELS_DIR, model_name.lower())
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'final_model.pt'))

    return {
        'model_name': model_name,
        'mse': results['loss'],
        'mae': results['mae'],
        'rmse': results['rmse'],
        'r2': results['r2']
    }

def main():
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    logger.info("加载数据...")
    df = load_and_preprocess_data()
    logger.info(f"处理后样本数: {len(df)}, 特征数: {len(df.columns)-1}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test = X_train.values, X_val.values, X_test.values
    y_train, y_val, y_test = y_train.values, y_val.values, y_test.values

    logger.info(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

    all_results = []
    model_configs = get_model_configs(X_train.shape[1])

    for name, model_class, kwargs in model_configs:
        try:
            result = train_and_evaluate(name, model_class, kwargs, X_train, y_train, X_val, y_val, X_test, y_test, device)
            all_results.append(result)
        except Exception as e:
            logger.error(f"训练 {name} 时出错: {e}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('r2', ascending=False)

    logger.info("\n" + "="*60)
    logger.info("所有模型结果对比 (按R²排序):")
    logger.info("="*60)
    logger.info(f"{'Model':<15} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    logger.info("-"*60)
    for _, row in results_df.iterrows():
        logger.info(f"{row['model_name']:<15} {row['mse']:>10.4f} {row['mae']:>10.4f} {row['rmse']:>10.4f} {row['r2']:>10.4f}")

    xgb_baseline = {'model_name': 'XGBoost (Baseline)', 'mse': None, 'mae': 8.5312, 'rmse': None, 'r2': 0.5897}
    logger.info("-"*60)
    logger.info(f"{'XGBoost (Baseline)':<15} {'--':>10} {xgb_baseline['mae']:>10.4f} {'--':>10} {xgb_baseline['r2']:>10.4f}")

    results_df.to_csv('deep_learning_results.csv', index=False)
    logger.info(f"\n结果已保存到: deep_learning_results.csv")

    end_time = time.time()
    logger.info(f"\n总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()