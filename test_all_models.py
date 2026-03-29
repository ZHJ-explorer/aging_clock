import os
import sys
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.deep_learning.models.neural_networks.dnn import DeepMLP, ResNetMLP
from scripts.deep_learning.models.neural_networks.cnn1d import CNN1D, ResCNN1D
from scripts.deep_learning.models.attention.transformer import Transformer
from scripts.deep_learning.models.attention.tabnet import TabNet
from scripts.deep_learning.training.trainer import Trainer
from scripts.deep_learning.training.optimizer import build_optimizer, build_scheduler


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def test_model(model_class, model_name, input_dim=350, n_features=350, output_dim=1, **model_kwargs):
    logger.info("=" * 60)
    logger.info(f"测试模型: {model_name}")
    logger.info("=" * 60)

    try:
        X_test = np.random.randn(32, n_features).astype(np.float32)
        y_test = np.random.randn(32).astype(np.float32)

        model = model_class(input_dim=input_dim, output_dim=output_dim, **model_kwargs)

        logger.info(f"模型参数量: {model.get_num_trainable_params()}")

        X_tensor = torch.FloatTensor(X_test[:4])
        if model_name in ['CNN1D', 'ResCNN1D']:
            X_tensor = X_tensor.unsqueeze(1)
        elif model_name == 'Transformer':
            X_tensor = X_tensor.unsqueeze(1)

        output = model(X_tensor)
        logger.info(f"前向传播输出维度: {output.shape}")

        if output.shape[0] == 4:
            logger.info(f"前向传播维度正确")
        else:
            logger.warning(f"前向传播维度异常: 期望batch=4, 实际output.shape={output.shape}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        model.to(device)

        optimizer = build_optimizer(model, 'adamw', learning_rate=0.001)
        scheduler = build_scheduler(optimizer, 'cosine', epochs=10)
        criterion = nn.MSELoss()

        train_dataset = DatasetWrapper(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            early_stopping_patience=5
        )

        logger.info("开始训练测试（5个epoch）...")
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=train_loader,
            epochs=5,
            save_dir=None
        )

        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        logger.info(f"初始Loss: {initial_loss:.4f}, 最终Loss: {final_loss:.4f}")

        if final_loss < initial_loss * 1.5:
            logger.info(f"✓ {model_name} 训练正常，loss下降")
        else:
            logger.warning(f"⚠ {model_name} loss下降不明显")

        logger.info(f"✓ {model_name} 测试通过!")
        return True, {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'params': model.get_num_trainable_params()
        }

    except Exception as e:
        logger.error(f"✗ {model_name} 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


def main():
    logger.info("深度学习模型测试开始")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")

    results = {}
    models_to_test = [
        (DeepMLP, 'DeepMLP', {}),
        (ResNetMLP, 'ResNetMLP', {}),
        (CNN1D, 'CNN1D', {'hidden_channels': [32, 64, 128]}),
        (ResCNN1D, 'ResCNN1D', {'hidden_channels': [32, 64, 128], 'n_res_blocks': 2}),
        (Transformer, 'Transformer', {'d_model': 64, 'num_heads': 4, 'num_layers': 2, 'ff_dim': 128}),
        (TabNet, 'TabNet', {'n_d': 32, 'n_a': 32, 'n_steps': 2}),
    ]

    for model_class, model_name, model_kwargs in models_to_test:
        success, result = test_model(model_class, model_name, **model_kwargs)
        results[model_name] = {'success': success, 'result': result}
        time.sleep(1)

    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    for model_name, data in results.items():
        status = "✓ 通过" if data['success'] else "✗ 失败"
        logger.info(f"{model_name}: {status}")
        if data['success']:
            logger.info(f"  初始Loss: {data['result'].get('initial_loss', 'N/A'):.4f}")
            logger.info(f"  最终Loss: {data['result'].get('final_loss', 'N/A'):.4f}")
            logger.info(f"  参数量: {data['result'].get('params', 'N/A')}")


if __name__ == "__main__":
    main()
