import os
import random
import numpy as np
import torch
from typing import Dict, Any


DATA_DIR = 'data'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'
PREPROCESSED_DIR = 'preprocessed_data'
RESULTS_DIR = 'results'
OPTUNA_RESULTS_DIR = 'optuna_results'
ENSEMBLE_RESULTS_DIR = 'ensemble_results'
SELECTED_FEATURES_DIR = 'selected_features'

RANDOM_SEED = 42

DEFAULT_N_FEATURES = 350
TEST_SIZE = 0.2
VAL_SIZE = 0.1

OPTUNA_N_TRIALS = 70
OPTUNA_CV = 5


class Config:
    """项目统一配置类"""

    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    PLOTS_DIR = 'plots'
    PREPROCESSED_DIR = 'preprocessed_data'
    RESULTS_DIR = 'results'
    OPTUNA_RESULTS_DIR = 'optuna_results'
    ENSEMBLE_RESULTS_DIR = 'ensemble_results'
    SELECTED_FEATURES_DIR = 'selected_features'

    RANDOM_SEED = 42

    DEFAULT_N_FEATURES = 350
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    OPTUNA_N_TRIALS = 70
    OPTUNA_CV = 5

    @classmethod
    def ensure_directories_exist(cls) -> None:
        """确保所有配置目录存在"""
        directories = [
            cls.DATA_DIR, cls.MODELS_DIR, cls.PLOTS_DIR,
            cls.PREPROCESSED_DIR, cls.RESULTS_DIR,
            cls.OPTUNA_RESULTS_DIR, cls.ENSEMBLE_RESULTS_DIR,
            cls.SELECTED_FEATURES_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """返回所有配置项的字典"""
        return {
            'DATA_DIR': cls.DATA_DIR,
            'MODELS_DIR': cls.MODELS_DIR,
            'PLOTS_DIR': cls.PLOTS_DIR,
            'PREPROCESSED_DIR': cls.PREPROCESSED_DIR,
            'RESULTS_DIR': cls.RESULTS_DIR,
            'OPTUNA_RESULTS_DIR': cls.OPTUNA_RESULTS_DIR,
            'ENSEMBLE_RESULTS_DIR': cls.ENSEMBLE_RESULTS_DIR,
            'SELECTED_FEATURES_DIR': cls.SELECTED_FEATURES_DIR,
            'RANDOM_SEED': cls.RANDOM_SEED,
            'DEFAULT_N_FEATURES': cls.DEFAULT_N_FEATURES,
            'TEST_SIZE': cls.TEST_SIZE,
            'VAL_SIZE': cls.VAL_SIZE,
            'OPTUNA_N_TRIALS': cls.OPTUNA_N_TRIALS,
            'OPTUNA_CV': cls.OPTUNA_CV
        }

    @classmethod
    def set_random_seed(cls, seed: int = None) -> None:
        """统一设置所有随机种子，确保结果可复现"""
        if seed is None:
            seed = cls.RANDOM_SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """统一设置所有随机种子，确保结果可复现"""
    Config.set_random_seed(seed)


def get_config() -> dict:
    """返回所有配置项的字典"""
    return Config.get_config()