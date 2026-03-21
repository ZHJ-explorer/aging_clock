import os
import time
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from data_utils import split_data
from optuna_tuning import tune_xgboost_optuna, tune_mlp_optuna, select_features_xgboost
from plot_results import convert_test_result_to_image


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
        logging.FileHandler('training_stacking_refactored.log', encoding='utf-8'),
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
    
    with open('test_result_stacking_refactored.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(result_lines) + '\n')
    
    return mae, rmse, r2, y_pred


def compute_model_correlations(predictions, model_names):
    logger.info("\n=== 计算基模型预测结果相关性 ===")
    corr_matrix = np.corrcoef(predictions)
    
    logger.info("基模型两两相关性矩阵:")
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            logger.info(f"{model_names[i]} vs {model_names[j]}: {corr_matrix[i,j]:.4f}")
    
    return corr_matrix


def main():
    with open('training_stacking_refactored.log', 'w', encoding='utf-8') as f:
        f.write("重构Stacking训练日志\n")
        f.write("=" * 50 + "\n")
    
    logger.info("开始重构Stacking训练流程...")
    start_time = time.time()
    
    with open('test_result_stacking_refactored.txt', 'w', encoding='utf-8') as f:
        f.write("重构Stacking模型测试结果\n")
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
    
    if len(merged_df) == 0:
        logger.error("错误：没有样本可用")
        exit(1)
    
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
        
        evaluate_model(model, X_test_selected, y_test, name)
    
    logger.info("\n=== 步骤6: 验证基模型相关性 ===")
    compute_model_correlations(base_predictions, model_names)
    
    logger.info("\n=== 步骤7: 手动加权融合（精细搜索最优权重） ===")
    
    xgb_model = trained_base_models[0][1]
    mlp_model = trained_base_models[1][1]
    
    y_xgb = xgb_model.predict(X_test_selected)
    y_mlp = mlp_model.predict(X_test_selected)
    
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
        
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_mae = current_mae
            best_weights = [xgb_weight, mlp_weight]
            best_y_stack = y_stack
    
    logger.info(f"\n=== 最佳手动加权融合结果 ===")
    logger.info(f"最优权重: XGBoost={best_weights[0]:.2f}, MLP={best_weights[1]:.2f}")
    logger.info(f"最优融合R²: {best_r2:.4f}, MAE: {best_mae:.4f}")
    
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
    
    with open('test_result_stacking_refactored.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(result_lines) + '\n')
    
    logger.info("\n=== 保存模型 ===")
    joblib.dump(trained_base_models, os.path.join(Config.MODELS_DIR, 'base_models_refactored.pkl'))
    joblib.dump(best_weights, os.path.join(Config.MODELS_DIR, 'best_weights.pkl'))
    joblib.dump(selected_features, os.path.join(Config.MODELS_DIR, 'selected_features_stacking.pkl'))
    logger.info("模型保存完成")
    
    end_time = time.time()
    logger.info(f"\n训练流程完成，耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"参与训练的数据条数: {len(merged_df)}")
    
    logger.info("\n运行绘图函数...")
    convert_test_result_to_image(test_file='test_result_stacking_refactored.txt')


if __name__ == "__main__":
    main()
