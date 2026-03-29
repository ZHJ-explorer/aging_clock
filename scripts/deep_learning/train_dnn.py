import os
import sys
import time
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.deep_learning.configs.dnn_config import DNNConfig
from scripts.deep_learning.models.neural_networks.dnn import DeepMLP
from scripts.deep_learning.models.base.trainer import DatasetWrapper
from scripts.deep_learning.training.trainer import Trainer
from scripts.deep_learning.training.optimizer import build_optimizer, build_scheduler
from scripts.utils.data_utils import split_data


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_dnn.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    PLOTS_DIR = 'plots'
    PREPROCESSED_DIR = 'preprocessed_data'

    N_FEATURES = 350
    RANDOM_STATE = 42


def main():
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("深度学习 DNN 模型训练")
    logger.info("=" * 60)

    merged_csv = os.path.join(Config.PREPROCESSED_DIR, 'merged_scaled.csv')
    if not os.path.exists(merged_csv):
        logger.error(f"数据文件不存在: {merged_csv}")
        logger.error("请先运行 preprocess_and_merge.py 生成数据")
        return

    logger.info(f"从 {merged_csv} 加载数据...")
    merged_df = pd.read_csv(merged_csv, index_col=0)

    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    if 'age' not in numeric_columns:
        numeric_columns = list(numeric_columns) + ['age']
    merged_df = merged_df[numeric_columns]

    merged_df = merged_df.dropna(axis=1, how='any')
    merged_df = merged_df.dropna()
    merged_df = merged_df[merged_df['age'] >= 20]

    logger.info(f"处理后样本数: {len(merged_df)}, 特征数: {len(merged_df.columns) - 1}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df)

    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    logger.info(f"训练集: {X_train.shape[0]} 样本")
    logger.info(f"验证集: {X_val.shape[0]} 样本")
    logger.info(f"测试集: {X_test.shape[0]} 样本")

    dnn_config = DNNConfig(
        input_dim=X_train.shape[1],
        hidden_dims=[512, 256, 128, 64],
        dropout=0.3,
        use_batchnorm=True,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=32,
        epochs=200,
        early_stopping_patience=20
    )

    logger.info("\nDNN 配置:")
    logger.info(f"  隐藏层: {dnn_config.hidden_dims}")
    logger.info(f"  Dropout: {dnn_config.dropout}")
    logger.info(f"  BatchNorm: {dnn_config.use_batchnorm}")
    logger.info(f"  学习率: {dnn_config.learning_rate}")
    logger.info(f"  Batch Size: {dnn_config.batch_size}")
    logger.info(f"  Epochs: {dnn_config.epochs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    model = DeepMLP(
        input_dim=dnn_config.input_dim,
        hidden_dims=dnn_config.hidden_dims,
        dropout=dnn_config.dropout,
        use_batchnorm=dnn_config.use_batchnorm,
        activation='relu'
    )

    logger.info(f"模型参数量: {model.get_num_trainable_params()}")

    optimizer = build_optimizer(
        model,
        optimizer_name='adamw',
        learning_rate=dnn_config.learning_rate,
        weight_decay=dnn_config.weight_decay
    )

    scheduler = build_scheduler(
        optimizer,
        scheduler_name='cosine',
        epochs=dnn_config.epochs
    )

    criterion = nn.MSELoss()

    train_dataset = DatasetWrapper(X_train, y_train)
    val_dataset = DatasetWrapper(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=dnn_config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dnn_config.batch_size,
        shuffle=False,
        num_workers=0
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=dnn_config.early_stopping_patience,
        l1_reg=dnn_config.l1_reg,
        l2_reg=dnn_config.l2_reg
    )

    save_dir = os.path.join(Config.MODELS_DIR, 'deep_learning', 'dnn')
    os.makedirs(save_dir, exist_ok=True)

    logger.info("\n开始训练...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=dnn_config.epochs,
        save_dir=save_dir
    )

    logger.info("\n在测试集上评估...")
    test_dataset = DatasetWrapper(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=dnn_config.batch_size, shuffle=False)

    test_results = trainer.evaluate(test_loader)

    logger.info("\n" + "=" * 60)
    logger.info("测试集结果:")
    logger.info(f"  MSE: {test_results['loss']:.4f}")
    logger.info(f"  MAE: {test_results['mae']:.4f}")
    logger.info(f"  RMSE: {test_results['rmse']:.4f}")
    logger.info(f"  R²: {test_results['r2']:.4f}")
    logger.info("=" * 60)

    model.save(os.path.join(save_dir, 'final_model.pt'))

    config_dict = dnn_config.to_dict()
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    results_df = pd.DataFrame({
        'prediction': test_results['predictions'],
        'actual': test_results['targets']
    })
    results_df.to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)

    end_time = time.time()
    logger.info(f"\n训练完成! 总耗时: {end_time - start_time:.2f} 秒")

    logger.info("\n结果保存位置:")
    logger.info(f"  模型: {os.path.join(save_dir, 'final_model.pt')}")
    logger.info(f"  配置: {os.path.join(save_dir, 'config.json')}")
    logger.info(f"  预测结果: {os.path.join(save_dir, 'test_predictions.csv')}")


if __name__ == "__main__":
    main()
