import random
import numpy as np
import torch


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


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """统一设置所有随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_config() -> dict:
    """返回所有配置项的字典"""
    return {
        'DATA_DIR': DATA_DIR,
        'MODELS_DIR': MODELS_DIR,
        'PLOTS_DIR': PLOTS_DIR,
        'PREPROCESSED_DIR': PREPROCESSED_DIR,
        'RESULTS_DIR': RESULTS_DIR,
        'OPTUNA_RESULTS_DIR': OPTUNA_RESULTS_DIR,
        'ENSEMBLE_RESULTS_DIR': ENSEMBLE_RESULTS_DIR,
        'SELECTED_FEATURES_DIR': SELECTED_FEATURES_DIR,
        'RANDOM_SEED': RANDOM_SEED,
        'DEFAULT_N_FEATURES': DEFAULT_N_FEATURES,
        'TEST_SIZE': TEST_SIZE,
        'VAL_SIZE': VAL_SIZE,
        'OPTUNA_N_TRIALS': OPTUNA_N_TRIALS,
        'OPTUNA_CV': OPTUNA_CV
    }
