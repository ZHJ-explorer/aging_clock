import numpy as np
import logging
import optuna
import xgboost as xgb
import os
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)


def objective_xgboost(trial, X_train, y_train, cv=5):
    """Optuna目标函数：XGBoost超参数优化"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 100.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    return np.mean(scores)


def tune_xgboost_optuna(X_train, y_train, n_trials=50, cv=5):
    """使用Optuna调优XGBoost模型"""
    logger.info("开始使用Optuna调优XGBoost模型...")
    logger.info(f"试验次数: {n_trials}, 交叉验证折数: {cv}")
    
    # 固定随机种子，确保结果可复现
    # 使用Optuna推荐的方式设置随机种子
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    
    study = optuna.create_study(direction='maximize', study_name='XGBoost_Tuning')
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, cv=cv),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"最佳R²得分: {study.best_value:.4f}")
    logger.info(f"最佳参数: {study.best_params}")
    
    # 保存调优记录
    os.makedirs('optuna_logs', exist_ok=True)
    study.trials_dataframe().to_csv('optuna_logs/xgboost_tuning_logs.csv', index=False)
    logger.info("调优记录已保存到 optuna_logs/xgboost_tuning_logs.csv")
    
    best_model = xgb.XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    return best_model, study.best_params, study.best_value


def select_features_xgboost(X_train, y_train, n_features=250):
    """使用5折交叉验证平均特征重要性选择Top N特征"""
    logger.info(f"使用5折交叉验证平均特征重要性选择Top {n_features}个特征...")
    
    if hasattr(X_train, 'columns'):
        features = X_train.columns
    else:
        features = np.arange(X_train.shape[1])
    
    # 初始化特征重要性累加器
    feature_importance_sum = np.zeros(len(features))
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        logger.info(f"  第{fold+1}折交叉验证...")
        
        X_fold_train, y_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx], y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_fold_train, y_fold_train)
        
        # 累加特征重要性
        feature_importance_sum += model.feature_importances_
    
    # 计算平均特征重要性
    feature_importance_avg = feature_importance_sum / 5
    
    feature_importance_dict = dict(zip(features, feature_importance_avg))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    top_features = [f for f, _ in sorted_features[:n_features]]
    
    logger.info(f"5折交叉验证平均特征重要性Top 10:")
    for i, (feature, importance) in enumerate(sorted_features[:10]):
        logger.info(f"  {i+1}. {feature}: {importance:.4f}")
    
    logger.info(f"选择了 {len(top_features)} 个特征")
    
    return top_features


def objective_knn(trial, X_train, y_train, cv=5):
    """Optuna目标函数：KNN回归超参数优化"""
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 50)
    }
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(**params, n_jobs=-1))
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    return np.mean(scores)


def tune_knn_optuna(X_train, y_train, n_trials=50, cv=10):
    """使用Optuna调优KNN回归模型"""
    logger.info("开始使用Optuna调优KNN回归模型...")
    logger.info(f"试验次数: {n_trials}, 交叉验证折数: {cv}")
    
    # 固定随机种子，确保结果可复现
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    
    study = optuna.create_study(direction='maximize', study_name='KNN_Tuning')
    study.optimize(
        lambda trial: objective_knn(trial, X_train, y_train, cv=cv),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"KNN最佳R²得分: {study.best_value:.4f}")
    logger.info(f"KNN最佳参数: {study.best_params}")
    
    # 保存调优记录
    os.makedirs('optuna_logs', exist_ok=True)
    study.trials_dataframe().to_csv('optuna_logs/knn_tuning_logs.csv', index=False)
    logger.info("调优记录已保存到 optuna_logs/knn_tuning_logs.csv")
    
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(**study.best_params, n_jobs=-1))
    ])
    
    best_pipeline.fit(X_train, y_train)
    
    return best_pipeline, study.best_params, study.best_value


def build_gpr_pipeline(n_components=50, random_state=42):
    """构建GPR Pipeline（归一化 + PCA降维 + GPR）"""
    gpr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("gpr", GaussianProcessRegressor(
            kernel=Matern(nu=1.5) + WhiteKernel(noise_level=1.0),
            alpha=1e-5,
            n_restarts_optimizer=10,
            random_state=random_state
        ))
    ])
    return gpr_pipeline


def build_mlp_pipeline(random_state=42):
    """构建MLP Pipeline（归一化 + MLP）"""
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=10.0,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=0
    )
    
    mlp_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp)
    ])
    return mlp_pipeline


def objective_gpr(trial, X_train, y_train, cv=5):
    """Optuna目标函数：GPR超参数优化"""
    n_components = trial.suggest_int('n_components', 30, 100)
    noise_level = trial.suggest_float('noise_level', 0.1, 10.0, log=True)
    nu = trial.suggest_categorical('nu', [0.5, 1.5, 2.5])
    
    kernel = Matern(nu=nu) + WhiteKernel(noise_level=noise_level)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('gpr', GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-5,
            n_restarts_optimizer=5,
            random_state=42
        ))
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    return np.mean(scores)


def tune_gpr_optuna(X_train, y_train, n_trials=30, cv=5):
    """使用Optuna调优GPR模型"""
    logger.info("开始使用Optuna调优GPR模型...")
    logger.info(f"试验次数: {n_trials}, 交叉验证折数: {cv}")
    
    # 固定随机种子，确保结果可复现
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    
    study = optuna.create_study(direction='maximize', study_name='GPR_Tuning')
    study.optimize(
        lambda trial: objective_gpr(trial, X_train, y_train, cv=cv),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"GPR最佳R²得分: {study.best_value:.4f}")
    logger.info(f"GPR最佳参数: {study.best_params}")
    
    # 保存调优记录
    os.makedirs('optuna_logs', exist_ok=True)
    study.trials_dataframe().to_csv('optuna_logs/gpr_tuning_logs.csv', index=False)
    logger.info("调优记录已保存到 optuna_logs/gpr_tuning_logs.csv")
    
    best_params = study.best_params
    kernel = Matern(nu=best_params['nu']) + WhiteKernel(noise_level=best_params['noise_level'])
    
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=best_params['n_components'], random_state=42)),
        ('gpr', GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-5,
            n_restarts_optimizer=10,
            random_state=42
        ))
    ])
    
    best_pipeline.fit(X_train, y_train)
    
    return best_pipeline, study.best_params, study.best_value


def objective_mlp(trial, X_train, y_train, cv=5):
    """Optuna目标函数：MLP超参数优化（精细调优版本）"""
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(32, 16), (64, 32), (128, 64, 32)])
    alpha = trial.suggest_float('alpha', 1.0, 50.0, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver='adam',
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=0
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', mlp)
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    return np.mean(scores)


def tune_mlp_optuna(X_train, y_train, n_trials=50, cv=10):
    """使用Optuna调优MLP模型（精细调优版本）"""
    logger.info("开始使用Optuna调优MLP模型（精细调优版本）...")
    logger.info(f"试验次数: {n_trials}, 交叉验证折数: {cv}")
    
    # 固定随机种子，确保结果可复现
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    
    study = optuna.create_study(direction='maximize', study_name='MLP_Fine_Tuning')
    study.optimize(
        lambda trial: objective_mlp(trial, X_train, y_train, cv=cv),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"MLP最佳交叉验证R²: {study.best_value:.4f}")
    logger.info(f"MLP最佳参数: {study.best_params}")
    
    # 保存调优记录
    os.makedirs('optuna_logs', exist_ok=True)
    study.trials_dataframe().to_csv('optuna_logs/mlp_tuning_logs.csv', index=False)
    logger.info("调优记录已保存到 optuna_logs/mlp_tuning_logs.csv")
    
    best_params = study.best_params
    mlp = MLPRegressor(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation=best_params['activation'],
        solver='adam',
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        batch_size=best_params['batch_size'],
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=0
    )
    
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', mlp)
    ])
    
    best_pipeline.fit(X_train, y_train)
    
    return best_pipeline, study.best_params, study.best_value
