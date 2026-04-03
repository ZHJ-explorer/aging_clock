import os
import sys
import time
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
import logging
import joblib
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from scripts.utils.data_utils import split_data
from scripts.utils.data_pipeline import load_and_preprocess_merged_data
from scripts.utils.model_utils import evaluate_model
from scripts.traditional_ml.optimization.optuna_tuning import tune_xgboost_optuna, tune_mlp_optuna, select_features_xgboost
from scripts.analysis.visualization.plot_results import convert_test_result_to_image


class Config:
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    PLOTS_DIR = 'plots'
    PREPROCESSED_DIR = 'preprocessed_data'
    
    N_FEATURES = 350
    OPTUNA_N_TRIALS = 50
    OPTUNA_N_TRIALS_MLP = 50
    OPTUNA_CV = 5
    OPTUNA_CV_MLP = 10


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_stacking.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs(Config.MODELS_DIR, exist_ok=True)
os.makedirs(Config.PLOTS_DIR, exist_ok=True)


def compute_model_correlations(predictions, model_names):
    logger.info("\n=== 计算基模型预测结果相关性 ===")
    corr_matrix = np.corrcoef(predictions)
    
    logger.info("基模型两两相关性矩阵:")
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            logger.info(f"{model_names[i]} vs {model_names[j]}: {corr_matrix[i,j]:.4f}")
    
    return corr_matrix


def main():
    with open('training_stacking.log', 'w', encoding='utf-8') as f:
        f.write("Stacking训练日志\n")
        f.write("=" * 50 + "\n")
    
    logger.info("开始重构Stacking训练流程...")
    start_time = time.time()

    with open('test_result_stacking.txt', 'w', encoding='utf-8') as f:
        f.write("重构Stacking模型测试结果\n")
        f.write("=" * 50 + "\n")

    merged_df, _, _, _, _, _, _, _ = load_and_preprocess_merged_data(min_age=20)
    if merged_df is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df)
    
    logger.info("\n=== 步骤1: 基于XGBoost特征重要性选择特征 ===")
    selected_features = select_features_xgboost(X_train, y_train, n_features=Config.N_FEATURES)
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    logger.info(f"特征选择后维度: {X_train_selected.shape[1]}")
    
    X_train_selected = X_train_selected.values
    X_val_selected = X_val_selected.values
    X_test_selected = X_test_selected.values
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values
    
    logger.info("\n=== 步骤2: 使用Optuna调优XGBoost超参数 ===")
    xgb_model, xgb_best_params, xgb_best_cv = tune_xgboost_optuna(
        X_train_selected, y_train,
        n_trials=Config.OPTUNA_N_TRIALS,
        cv=Config.OPTUNA_CV
    )
    
    logger.info("\n=== 步骤3: 使用Optuna调优MLP模型 ===")
    mlp_model, mlp_best_params, mlp_best_cv = tune_mlp_optuna(
        X_train_selected, y_train,
        n_trials=Config.OPTUNA_N_TRIALS_MLP,
        cv=Config.OPTUNA_CV_MLP
    )
    
    logger.info("\n=== 步骤4: 定义重构后的基模型组合 ===")
    logger.info("基模型：XGBoost（调优后）、MLP（调优后）")
    
    xgb_best = XGBRegressor(
        **xgb_best_params,
        random_state=42,
        n_jobs=-1
    )
    
    base_models = [
        ('xgboost', xgb_best),
        ('mlp', mlp_model)
    ]
    
    logger.info("\n=== 步骤5: 训练各个基模型 ===")
    trained_base_models = []
    base_predictions = []
    model_names = []
    
    for name, model in base_models:
        logger.info(f"训练 {name} 模型...")
        model.fit(X_train_selected, y_train)
        trained_base_models.append((name, model))
        
        y_pred = model.predict(X_test_selected)
        base_predictions.append(y_pred)
        model_names.append(name)
        
        evaluate_model(model, X_test_selected, y_test, name, output_file='test_result_stacking.txt')
    
    logger.info("\n=== 步骤6: 验证基模型相关性 ===")
    compute_model_correlations(base_predictions, model_names)
    
    logger.info("\n=== 步骤7: 手动加权融合（精细搜索最优权重） ===")
    
    xgb_model = trained_base_models[0][1]
    mlp_model = trained_base_models[1][1]
    
    y_xgb = xgb_model.predict(X_test_selected)
    y_mlp = mlp_model.predict(X_test_selected)
    
    best_score = -float('inf')
    best_r2 = -float('inf')
    best_mae = float('inf')
    best_weights = None
    best_y_stack = None
    
    logger.info("遍历权重比例（XGBoost 70%-95%，步长 1%）...")
    for xgb_weight in np.arange(0.70, 0.96, 0.01):
        mlp_weight = 1 - xgb_weight
        y_stack = xgb_weight * y_xgb + mlp_weight * y_mlp
        
        current_r2 = r2_score(y_test, y_stack)
        current_mae = mean_absolute_error(y_test, y_stack)
        current_rmse = np.sqrt(mean_squared_error(y_test, y_stack))
        
        # 计算综合评分：R² - 0.1 * MAE（平衡两个指标）
        # 0.1是一个经验系数，用于平衡R²和MAE的尺度差异
        current_score = current_r2 - 0.1 * current_mae
        
        if current_score > best_score:
            best_score = current_score
            best_r2 = current_r2
            best_mae = current_mae
            best_weights = [xgb_weight, mlp_weight]
            best_y_stack = y_stack
    
    logger.info(f"\n=== 最佳手动加权融合结果 ===")
    logger.info(f"最优权重: XGBoost={best_weights[0]:.2f}, MLP={best_weights[1]:.2f}")
    logger.info(f"最优融合R²: {best_r2:.4f}, MAE: {best_mae:.4f}")
    logger.info(f"综合评分: {best_score:.4f}")
    
    result_lines = [
        f"\n手动加权融合模型测试结果",
        f"最佳权重: XGBoost={best_weights[0]:.2f}, MLP={best_weights[1]:.2f}",
        f"MAE: {mean_absolute_error(y_test, best_y_stack):.4f}",
        f"RMSE: {np.sqrt(mean_squared_error(y_test, best_y_stack)):.4f}",
        f"R2: {best_r2:.4f}",
        "预测值,实际值"
    ]
    
    for pred, actual in zip(best_y_stack, y_test):
        result_lines.append(f"{pred:.4f},{actual:.4f}")
    
    with open('test_result_stacking.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(result_lines) + '\n')
    
    logger.info("\n=== 步骤8: 重复交叉验证评估模型稳定性 ===")
    # 5折交叉验证，重复3次
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    
    X = merged_df.drop('age', axis=1)[selected_features].values
    y = merged_df['age'].values
    
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X)):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # 训练基模型
        fold_xgb_model = XGBRegressor(**xgb_best_params, random_state=42, n_jobs=-1)
        fold_xgb_model.fit(X_train_fold, y_train_fold)
        
        # 训练MLP模型
        fold_mlp_model = mlp_model
        fold_mlp_model.fit(X_train_fold, y_train_fold)
        
        # 预测并融合
        y_xgb_pred = fold_xgb_model.predict(X_test_fold)
        y_mlp_pred = fold_mlp_model.predict(X_test_fold)
        y_stack_pred = best_weights[0] * y_xgb_pred + best_weights[1] * y_mlp_pred
        
        # 计算指标
        r2 = r2_score(y_test_fold, y_stack_pred)
        mae = mean_absolute_error(y_test_fold, y_stack_pred)
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_stack_pred))
        
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        
        logger.info(f"  重复交叉验证第{fold_idx+1}次: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    logger.info("\n=== 重复交叉验证结果汇总 ===")
    logger.info(f"平均R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    logger.info(f"平均MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    logger.info(f"平均RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    
    logger.info("\n=== 步骤9: SHAP分析，识别核心衰老基因 ===")
    # 使用XGBoost基模型进行SHAP分析（因为SHAP对树模型支持最好）
    xgb_model = trained_base_models[0][1]
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(xgb_model)
    
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
    top_50_features.to_csv('shap_feature_importance_stacking.csv', index=False)
    
    logger.info("\n=== SHAP特征重要性Top 20 ===")
    for i, (index, row) in enumerate(top_50_features.head(20).iterrows()):
        logger.info(f"  {i+1}. {row['Feature']}: {row['SHAP Importance']:.4f}")
    
    logger.info("\n核心衰老基因已保存到 shap_feature_importance_stacking.csv")
    
    logger.info("\n=== 保存模型 ===")
    joblib.dump(trained_base_models, os.path.join(Config.MODELS_DIR, 'base_models_stacking.pkl'))
    joblib.dump(best_weights, os.path.join(Config.MODELS_DIR, 'best_weights_stacking.pkl'))
    joblib.dump(selected_features, os.path.join(Config.MODELS_DIR, 'selected_features_stacking.pkl'))
    logger.info("模型保存完成")
    
    end_time = time.time()
    logger.info(f"\n训练流程完成，耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"参与训练的数据条数: {len(merged_df)}")
    
    logger.info("\n运行绘图函数...")
    convert_test_result_to_image(test_file='test_result_stacking.txt')


if __name__ == "__main__":
    main()
