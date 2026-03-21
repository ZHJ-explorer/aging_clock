import os
import sys
import time
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import logging
import joblib
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from scripts.utils.data_utils import split_data
from optuna_tuning import tune_xgboost_optuna, select_features_xgboost
from scripts.analysis.plot_results import convert_test_result_to_image


class Config:
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    PLOTS_DIR = 'plots'
    PREPROCESSED_DIR = 'preprocessed_data'
    
    N_FEATURES_XGBOOST = 350
    OPTUNA_N_TRIALS = 70
    OPTUNA_CV = 5


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_xgboost.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs(Config.MODELS_DIR, exist_ok=True)
os.makedirs(Config.PLOTS_DIR, exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name):
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
    
    with open('test_result_xgboost.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(result_lines) + '\n')
    
    return mae, rmse, r2


def main():
    with open('training_xgboost.log', 'w', encoding='utf-8') as f:
        f.write("XGBoost单模型优化训练日志\n")
        f.write("=" * 50 + "\n")
    
    logger.info("开始XGBoost单模型优化训练流程...")
    start_time = time.time()
    
    with open('test_result_xgboost.txt', 'w', encoding='utf-8') as f:
        f.write("XGBoost单模型测试结果\n")
        f.write("=" * 50 + "\n")
    
    merged_csv = os.path.join(Config.PREPROCESSED_DIR, 'merged_scaled.csv')
    if os.path.exists(merged_csv):
        logger.info(f"从 {merged_csv} 加载已处理的数据集...")
        merged_df = pd.read_csv(merged_csv, index_col=0)
        logger.info(f"加载完成，样本数: {len(merged_df)}")
    else:
        logger.error("merged_scaled.csv 不存在，请先运行主流程生成数据")
        return
    
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
        exit(1)
    
    merged_df = merged_df.dropna()
    logger.info(f"删除NaN行后，样本数: {len(merged_df)}")
    
    # 过滤掉年龄小于20的样本，只使用20及以上的样本
    merged_df = merged_df[merged_df['age'] >= 20]
    logger.info(f"过滤后（年龄>=20）样本数: {len(merged_df)}")
    
    if len(merged_df) == 0:
        logger.error("错误：没有样本可用")
        exit(1)
    
    X = merged_df.drop('age', axis=1)
    y = merged_df['age']
    
    logger.info("\n=== 步骤1: 基于XGBoost特征重要性选择特征 ===")
    selected_features = select_features_xgboost(X, y, n_features=Config.N_FEATURES_XGBOOST)
    X_selected = X[selected_features]
    logger.info(f"特征选择后维度: {X_selected.shape[1]}")
    
    logger.info("\n=== 步骤2: 使用Optuna调优XGBoost超参数 ===")
    xgb_model, best_params, best_cv_score = tune_xgboost_optuna(
        X_selected.values, y.values,
        n_trials=Config.OPTUNA_N_TRIALS,
        cv=Config.OPTUNA_CV
    )
    
    logger.info("\n=== 步骤3: 重复交叉验证评估模型稳定性 ===")
    # 5折交叉验证，重复3次
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_selected)):
        X_train_fold, X_test_fold = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train_fold, y_train_fold)
        
        y_pred = model.predict(X_test_fold)
        r2 = r2_score(y_test_fold, y_pred)
        mae = mean_absolute_error(y_test_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
        
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        
        logger.info(f"  重复交叉验证第{fold_idx+1}次: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    logger.info("\n=== 重复交叉验证结果汇总 ===")
    logger.info(f"平均R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    logger.info(f"平均MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    logger.info(f"平均RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    
    # 标准的训练集、验证集、测试集拆分
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df)
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    # 重新训练模型用于最终评估
    final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train_selected, y_train)
    
    logger.info("\n=== 步骤4: 评估优化后的XGBoost模型 ===")
    evaluate_model(final_model, X_test_selected, y_test, "XGBoost_Optimized")
    
    logger.info("\n=== 步骤5: SHAP分析，识别核心衰老基因 ===")
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(final_model)
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_test_selected)
    
    # 计算特征重要性（基于SHAP值的绝对值）
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'SHAP Importance': feature_importance
    })
    
    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values('SHAP Importance', ascending=False)
    
    # 保存前50个重要特征
    top_50_features = feature_importance_df.head(50)
    top_50_features.to_csv('shap_feature_importance_xgboost.csv', index=False)
    
    logger.info("\n=== SHAP特征重要性Top 20 ===")
    for i, (index, row) in enumerate(top_50_features.head(20).iterrows()):
        logger.info(f"  {i+1}. {row['Feature']}: {row['SHAP Importance']:.4f}")
    
    logger.info("\n核心衰老基因已保存到 shap_feature_importance_xgboost.csv")
    
    logger.info("\n=== 保存模型 ===")
    joblib.dump(final_model, os.path.join(Config.MODELS_DIR, 'xgboost_optimized.pkl'))
    joblib.dump(selected_features, os.path.join(Config.MODELS_DIR, 'selected_features_xgboost.pkl'))
    joblib.dump(best_params, os.path.join(Config.MODELS_DIR, 'best_params_xgboost.pkl'))
    logger.info("模型保存完成")
    
    end_time = time.time()
    logger.info(f"\n训练流程完成，耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"参与训练的数据条数: {len(merged_df)}")
    
    logger.info("\n运行绘图函数...")
    convert_test_result_to_image(test_file='test_result_xgboost.txt')


if __name__ == "__main__":
    main()
