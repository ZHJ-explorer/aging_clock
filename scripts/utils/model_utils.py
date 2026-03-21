import numpy as np
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import logging
from hyperparameter_tuning import tune_all_models

# 配置日志
logger = logging.getLogger(__name__)

# 模型目录
MODELS_DIR = 'models'


import xgboost as xgb

def train_models(X_train, y_train, X_val, y_val, use_hyperparameter_tuning=False):
    """训练多个模型"""
    logger.info("训练模型...")
    
    # 定义基模型，优化参数
    logger.info("初始化基模型...")
    from sklearn.linear_model import Lasso
    
    if use_hyperparameter_tuning:
        logger.info("使用超参数调优...")
        # 调优树模型和SVR
        xgb_model, lgb_model, rf_model, svr_model = tune_all_models(X_train, y_train, X_val, y_val)
        
        # 线性模型使用默认参数
        ridge_model = Ridge(alpha=1.0, random_state=42)
        lasso_model = Lasso(alpha=0.01, random_state=42, max_iter=10000)
        en_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
        
        # 训练线性模型
        logger.info("训练Ridge模型...")
        ridge_model.fit(X_train, y_train)
        
        logger.info("训练LASSO模型...")
        lasso_model.fit(X_train, y_train)
        
        logger.info("训练ElasticNet模型...")
        en_model.fit(X_train, y_train)
        
        # 构建训练好的模型列表
        trained_base_models = [
            ('xgboost', xgb_model),
            ('lightgbm', lgb_model),
            ('random_forest', rf_model),
            ('svr', svr_model),
            ('ridge', ridge_model),
            ('lasso', lasso_model),
            ('elasticnet', en_model)
        ]
    else:
        logger.info("使用默认参数...")
        base_models = [
            ('xgboost', xgb.XGBRegressor(
                n_estimators=500, 
                learning_rate=0.01, 
                max_depth=6, 
                min_child_weight=1, 
                subsample=0.8, 
                colsample_bytree=0.8, 
                random_state=42, 
                n_jobs=-1
            )),
            ('lightgbm', lgb.LGBMRegressor(
                n_estimators=500, 
                learning_rate=0.01, 
                max_depth=6, 
                num_leaves=31, 
                subsample=0.8, 
                colsample_bytree=0.8, 
                random_state=42, 
                verbose=-1, 
                n_jobs=-1
            )),
            ('ridge', Ridge(alpha=1.0, random_state=42)),
            ('lasso', Lasso(alpha=0.01, random_state=42, max_iter=10000)),
            ('elasticnet', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)),
            ('svr', SVR(kernel='rbf', C=100.0, gamma='scale', cache_size=1000)),
            ('random_forest', RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42, 
                n_jobs=-1
            ))
        ]
        
        # 使用joblib并行训练基模型
        from joblib import Parallel, delayed
        
        def train_model(name, model):
            logger.info(f"训练{name}模型...")
            if name == 'lightgbm':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae')
            elif name == 'xgboost':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            else:
                model.fit(X_train, y_train)
            return (name, model)
        
        trained_base_models = Parallel(n_jobs=-1)(
            delayed(train_model)(name, model) for name, model in base_models
        )
    
    # 训练Stacking模型，使用10折交叉验证
    logger.info("训练Stacking模型...")
    stacking_model = StackingRegressor(
        estimators=trained_base_models, 
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=10, scoring='r2'),
        cv=10  # 使用10折交叉验证
    )
    stacking_model.fit(X_train, y_train)
    
    # 提取训练好的基模型
    model_dict = dict(trained_base_models)
    xgb_model = model_dict.get('xgboost')
    lgb_model = model_dict.get('lightgbm')
    ridge_model = model_dict.get('ridge')
    lasso_model = model_dict.get('lasso')
    en_model = model_dict.get('elasticnet')
    svr_model = model_dict.get('svr')
    
    return ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model


def evaluate_model(model, X_test, y_test, model_name):
    """评估模型性能"""
    logger.info(f"评估 {model_name} 模型...")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"{model_name} 模型性能:")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2: {r2:.4f}")
    
    # 构建结果字符串，批量写入
    result_lines = [
        f"\n{model_name} 模型测试结果:",
        f"MAE: {mae:.4f}",
        f"RMSE: {rmse:.4f}",
        f"R2: {r2:.4f}",
        "预测值,实际值"
    ]
    
    for pred, actual in zip(y_pred, y_test):
        result_lines.append(f"{pred:.4f},{actual:.4f}")
    
    # 批量写入文件
    with open('test_result.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(result_lines) + '\n')
    
    return mae, rmse, r2


def save_models(ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, suffix=""):
    """保存训练好的模型"""
    logger.info("保存模型...")
    
    # 确保模型目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 保存模型，跳过None值
    joblib.dump(ridge_model, os.path.join(MODELS_DIR, f'ridge_model{suffix}.pkl'))
    if lasso_model is not None:
        joblib.dump(lasso_model, os.path.join(MODELS_DIR, f'lasso_model{suffix}.pkl'))
    if en_model is not None:
        joblib.dump(en_model, os.path.join(MODELS_DIR, f'elasticnet_model{suffix}.pkl'))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, f'xgboost_model{suffix}.pkl'))
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, f'lightgbm_model{suffix}.pkl'))
    joblib.dump(svr_model, os.path.join(MODELS_DIR, f'svr_model{suffix}.pkl'))
    joblib.dump(stacking_model, os.path.join(MODELS_DIR, f'stacking_model{suffix}.pkl'))
    
    logger.info("模型保存完成")
