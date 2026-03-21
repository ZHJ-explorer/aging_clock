import numpy as np
import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# 配置日志
logger = logging.getLogger(__name__)

def tune_xgboost(X_train, y_train, X_val, y_val, n_iter=30, cv=5):
    """调优XGBoost模型"""
    logger.info("开始调优XGBoost模型...")
    
    # 参数空间
    param_dist = {
        'n_estimators': [200],  # 先固定为200
        'learning_rate': np.logspace(-3, -1, 10),
        'max_depth': [3, 4, 5, 6, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': np.linspace(0.6, 1.0, 5),
        'colsample_bytree': np.linspace(0.6, 1.0, 5),
        'gamma': np.logspace(-3, 0, 10)
    }
    
    # 创建模型
    model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # 随机搜索
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合
    random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # 最佳参数
    logger.info(f"XGBoost最佳参数: {random_search.best_params_}")
    logger.info(f"XGBoost最佳R2得分: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_lightgbm(X_train, y_train, X_val, y_val, n_iter=30, cv=5):
    """调优LightGBM模型"""
    logger.info("开始调优LightGBM模型...")
    
    # 参数空间
    param_dist = {
        'n_estimators': [200],  # 先固定为200
        'learning_rate': np.logspace(-3, -1, 10),
        'max_depth': [3, 4, 5, 6, 7],
        'num_leaves': [15, 31, 63, 127],
        'subsample': np.linspace(0.6, 1.0, 5),
        'colsample_bytree': np.linspace(0.6, 1.0, 5),
        'min_child_samples': [5, 10, 20]
    }
    
    # 创建模型
    model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    
    # 随机搜索
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合
    random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # 最佳参数
    logger.info(f"LightGBM最佳参数: {random_search.best_params_}")
    logger.info(f"LightGBM最佳R2得分: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_random_forest(X_train, y_train, X_val, y_val, n_iter=30, cv=5):
    """调优RandomForest模型"""
    logger.info("开始调优RandomForest模型...")
    
    # 参数空间
    param_dist = {
        'n_estimators': [200],  # 先固定为200
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    # 创建模型
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # 随机搜索
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合
    random_search.fit(X_train, y_train)
    
    # 最佳参数
    logger.info(f"RandomForest最佳参数: {random_search.best_params_}")
    logger.info(f"RandomForest最佳R2得分: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_svr(X_train, y_train, X_val, y_val, n_iter=30, cv=5):
    """调优SVR模型"""
    logger.info("开始调优SVR模型...")
    
    # 参数空间
    param_dist = {
        'C': np.logspace(0, 3, 10),
        'gamma': np.logspace(-4, 0, 10),
        'kernel': ['rbf']
    }
    
    # 创建模型
    model = SVR(cache_size=1000)
    
    # 随机搜索
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合
    random_search.fit(X_train, y_train)
    
    # 最佳参数
    logger.info(f"SVR最佳参数: {random_search.best_params_}")
    logger.info(f"SVR最佳R2得分: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_all_models(X_train, y_train, X_val, y_val):
    """调优所有模型"""
    logger.info("开始调优所有模型...")
    
    # 调优XGBoost
    xgb_model = tune_xgboost(X_train, y_train, X_val, y_val)
    
    # 调优LightGBM
    lgb_model = tune_lightgbm(X_train, y_train, X_val, y_val)
    
    # 调优RandomForest
    rf_model = tune_random_forest(X_train, y_train, X_val, y_val)
    
    # 调优SVR
    svr_model = tune_svr(X_train, y_train, X_val, y_val)
    
    return xgb_model, lgb_model, rf_model, svr_model
