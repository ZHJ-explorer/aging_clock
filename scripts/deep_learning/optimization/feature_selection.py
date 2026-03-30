import os
import sys
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PREPROCESSED_DIR = 'preprocessed_data'
SELECTED_FEATURES_DIR = 'selected_features'

def load_and_preprocess_data():
    merged_csv = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    df = pd.read_csv(merged_csv, index_col=0)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if 'age' not in numeric_columns:
        numeric_columns = list(numeric_columns) + ['age']
    df = df[numeric_columns]
    df = df.dropna(axis=1, how='any')
    df = df.dropna()
    df = df[df['age'] >= 20]
    return df

def variance_threshold_selection(X, y, threshold=0.1):
    logger.info(f"方差阈值过滤 (threshold={threshold})...")
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    selected_features = selector.get_support()
    logger.info(f"方差过滤后特征数: {X_selected.shape[1]}")
    return X_selected, selected_features

def correlation_selection(X, y, threshold=0.1):
    logger.info(f"相关性分析 (threshold={threshold})...")
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(abs(corr) if not np.isnan(corr) else 0)
    correlations = np.array(correlations)
    selected = correlations > threshold
    logger.info(f"相关性过滤后特征数: {selected.sum()}")
    return selected

def xgb_importance_selection(X, y, top_n=1000):
    logger.info(f"XGBoost特征重要性筛选 (top_n={top_n})...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X, y)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_n:]
    selected = np.zeros(X.shape[1], dtype=bool)
    selected[top_indices] = True
    logger.info(f"XGBoost重要性筛选后特征数: {selected.sum()}")
    return selected

def select_features(X, y, feature_names, method='combined'):
    logger.info(f"\n=== 特征选择 ({method}) ===")
    logger.info(f"原始特征数: {X.shape[1]}")

    if method == 'variance':
        X_selected, mask = variance_threshold_selection(X, y)
        return X_selected, mask
    elif method == 'correlation':
        mask = correlation_selection(X, y)
        return X[:, mask], mask
    elif method == 'xgb':
        mask = xgb_importance_selection(X, y)
        return X[:, mask], mask
    elif method == 'combined':
        _, var_mask = variance_threshold_selection(X, y, threshold=0.05)
        corr_mask = correlation_selection(X, y, threshold=0.05)
        xgb_mask = xgb_importance_selection(X, y, top_n=1500)
        combined_mask = var_mask & corr_mask & xgb_mask
        logger.info(f"组合筛选后特征数: {combined_mask.sum()}")
        if combined_mask.sum() < 100:
            logger.warning("组合筛选特征太少，使用XGB top 1000")
            combined_mask = xgb_mask
        return X[:, combined_mask], combined_mask

def main():
    os.makedirs(SELECTED_FEATURES_DIR, exist_ok=True)

    logger.info("加载数据...")
    df = load_and_preprocess_data()
    logger.info(f"样本数: {len(df)}, 特征数: {len(df.columns)-1}")

    X = df.drop('age', axis=1).values
    y = df['age'].values
    feature_names = df.drop('age', axis=1).columns.tolist()

    logger.info("\n执行组合特征选择...")
    X_selected, mask = select_features(X, y, feature_names, method='combined')

    selected_indices = np.where(mask)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    selected_df = pd.DataFrame({
        'feature_name': selected_features,
        'original_index': selected_indices
    })
    selected_df.to_csv(os.path.join(SELECTED_FEATURES_DIR, 'selected_features.csv'), index=False)
    logger.info(f"选中的特征已保存到 {SELECTED_FEATURES_DIR}/selected_features.csv")

    np.save(os.path.join(SELECTED_FEATURES_DIR, 'feature_mask.npy'), mask)
    logger.info(f"特征掩码已保存")

    logger.info(f"\n最终选择特征数: {len(selected_features)}")
    logger.info(f"特征减少比例: {(1 - len(selected_features)/len(feature_names))*100:.1f}%")

    return X_selected, y, selected_features

if __name__ == "__main__":
    X_selected, y, selected_features = main()
    print(f"\n特征选择完成！选择特征数: {len(selected_features)}")