import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PREPROCESSED_DIR = 'preprocessed_data'


def load_and_preprocess_merged_data(min_age=20):
    """加载并预处理合并后的数据

    Args:
        min_age: 最小年龄阈值，默认为20

    Returns:
        tuple: (merged_df, selected_features, X_train_selected, X_val_selected, X_test_selected, y_train, y_val, y_test)
            如果加载失败返回 (None, None, None, None, None, None, None, None)
    """
    from scripts.utils.data_utils import split_data

    merged_csv = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    if os.path.exists(merged_csv):
        logger.info(f"从 {merged_csv} 加载已处理的数据集...")
        merged_df = pd.read_csv(merged_csv, index_col=0)
        logger.info(f"加载完成，样本数: {len(merged_df)}")
    else:
        logger.error("merged_scaled.csv 不存在，请先运行主流程生成数据")
        return None, None, None, None, None, None, None, None

    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    if 'age' not in numeric_columns:
        numeric_columns = list(numeric_columns) + ['age']
    merged_df = merged_df[numeric_columns]
    logger.info(f"保留了 {len(merged_df.columns) - 1} 个数值型特征")

    logger.info("处理NaN值...")
    merged_df = merged_df.dropna(axis=1, how='any')
    logger.info(f"删除NaN列后，特征数: {len(merged_df.columns) - 1}")

    if len(merged_df.columns) <= 1:
        logger.error("错误：没有特征可用")
        return None, None, None, None, None, None, None, None

    merged_df = merged_df.dropna()
    logger.info(f"删除NaN行后，样本数: {len(merged_df)}")

    if min_age is not None:
        merged_df = merged_df[merged_df['age'] >= min_age]
        logger.info(f"过滤后（年龄>={min_age}）样本数: {len(merged_df)}")

    if len(merged_df) == 0:
        logger.error("错误：没有样本可用")
        return None, None, None, None, None, None, None, None

    return merged_df, None, None, None, None, None, None, None


def prepare_data_for_training(merged_df, selected_features=None, n_features=350):
    """准备训练数据

    Args:
        merged_df: 合并后的DataFrame
        selected_features: 预选的特征列表，如果为None则进行特征选择
        n_features: 特征数量（当selected_features为None时使用）

    Returns:
        tuple: (selected_features, X_train_selected, X_val_selected, X_test_selected, y_train_values, y_val_values, y_test_values)
    """
    from scripts.utils.data_utils import split_data

    X = merged_df.drop('age', axis=1)
    y = merged_df['age']

    if selected_features is None:
        from scripts.traditional_ml.optimization.optuna_tuning import select_features_xgboost
        logger.info(f"使用XGBoost特征重要性选择Top {n_features}个特征...")
        selected_features = select_features_xgboost(X, y, n_features=n_features)

    X_selected = X[selected_features]
    logger.info(f"特征选择后维度: {X_selected.shape[1]}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df)
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    X_train_values = X_train_selected.values
    X_val_values = X_val_selected.values
    X_test_values = X_test_selected.values
    y_train_values = y_train.values
    y_val_values = y_val.values
    y_test_values = y_test.values

    return selected_features, X_train_values, X_val_values, X_test_values, y_train_values, y_val_values, y_test_values


def preprocess_merged_data(merged_df=None, min_age=20):
    """预处理合并数据的完整流程

    Args:
        merged_df: 合并后的DataFrame，如果为None则从文件加载
        min_age: 最小年龄阈值，默认为20

    Returns:
        dict: 包含预处理后数据的字典
    """
    if merged_df is None:
        merged_df, _, _, _, _, _, _, _ = load_and_preprocess_merged_data(min_age)
        if merged_df is None:
            return None

    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    if 'age' not in numeric_columns:
        numeric_columns = list(numeric_columns) + ['age']
    merged_df = merged_df[numeric_columns]

    merged_df = merged_df.dropna(axis=1, how='any')

    if len(merged_df.columns) <= 1:
        return None

    merged_df = merged_df.dropna()

    if min_age is not None:
        merged_df = merged_df[merged_df['age'] >= min_age]

    if len(merged_df) == 0:
        return None

    return merged_df
