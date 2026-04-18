import os
import sys
import time
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
import logging
import joblib
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from scripts.utils.data_utils import split_data
from scripts.utils.data_pipeline import load_and_preprocess_merged_data, prepare_data_for_training
from scripts.utils.model_utils import evaluate_model
from scripts.traditional_ml.optimization.optuna_tuning import tune_xgboost_optuna, select_features_xgboost
from scripts.analysis.visualization.plot_results import convert_test_result_to_image
from scripts.config import MODELS_DIR, PLOTS_DIR, DEFAULT_N_FEATURES, OPTUNA_N_TRIALS, OPTUNA_CV, Config


N_FEATURES_XGBOOST = DEFAULT_N_FEATURES


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_xgboost.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Config.ensure_directories_exist()


def main():
    with open('training_xgboost.log', 'w', encoding='utf-8') as f:
        f.write("XGBoost单模型优化训练日志\n")
        f.write("=" * 50 + "\n")
    
    logger.info("开始XGBoost单模型优化训练流程...")
    start_time = time.time()

    with open('test_result_xgboost.txt', 'w', encoding='utf-8') as f:
        f.write("XGBoost单模型测试结果\n")
        f.write("=" * 50 + "\n")

    merged_df, _, _, _, _, _, _, _ = load_and_preprocess_merged_data(min_age=20)
    if merged_df is None:
        return

    X = merged_df.drop('age', axis=1)
    y = merged_df['age']

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df)

    logger.info("\n=== 步骤1: 只在训练集上进行基于XGBoost特征重要性选择特征 ===")
    logger.info(f"特征选择只在 {len(X_train)} 个训练样本上进行，避免数据泄漏")
    selected_features = select_features_xgboost(X_train, y_train, n_features=Config.N_FEATURES_XGBOOST)
    logger.info(f"特征选择后维度: {len(selected_features)}")

    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    logger.info("\n=== 步骤2: 使用Optuna调优XGBoost超参数（只在训练集上） ===")
    xgb_model, best_params, best_cv_score = tune_xgboost_optuna(
        X_train_selected.values, y_train.values,
        n_trials=OPTUNA_N_TRIALS,
        cv=OPTUNA_CV
    )
    
    logger.info("\n=== 步骤3: 重复交叉验证评估模型稳定性（只在训练集上进行） ===")
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    r2_scores = []
    mae_scores = []
    rmse_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_train_selected)):
        X_train_fold, X_test_fold = X_train_selected.iloc[train_idx], X_train_selected.iloc[test_idx]
        y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

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

    logger.info("\n=== 步骤4: 在训练集上重新训练模型用于最终评估 ===")
    final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train_selected, y_train)
    
    logger.info("\n=== 步骤4: 评估优化后的XGBoost模型 ===")
    evaluate_model(final_model, X_test_selected, y_test, "XGBoost_Optimized", output_file='test_result_xgboost.txt')
    
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
    joblib.dump(final_model, os.path.join(MODELS_DIR, 'xgboost_optimized.pkl'))
    joblib.dump(selected_features, os.path.join(MODELS_DIR, 'selected_features_xgboost.pkl'))
    joblib.dump(best_params, os.path.join(MODELS_DIR, 'best_params_xgboost.pkl'))
    logger.info("模型保存完成")
    
    end_time = time.time()
    logger.info(f"\n训练流程完成，耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"参与训练的数据条数: {len(merged_df)}")
    
    logger.info("\n运行绘图函数...")
    convert_test_result_to_image(test_file='test_result_xgboost.txt')


if __name__ == "__main__":
    main()
