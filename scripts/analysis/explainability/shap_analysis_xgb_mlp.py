import os
import numpy as np
import pandas as pd
import joblib
import logging
import shap
import matplotlib.pyplot as plt
from data_utils import split_data


class Config:
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    PLOTS_DIR = 'plots'
    PREPROCESSED_DIR = 'preprocessed_data'
    
    N_TOP_FEATURES = 20


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shap_analysis_xgb_mlp.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs(Config.PLOTS_DIR, exist_ok=True)


def load_data_and_models():
    logger.info("加载数据和模型...")
    
    merged_csv = os.path.join(Config.PREPROCESSED_DIR, 'merged_scaled.csv')
    if not os.path.exists(merged_csv):
        logger.error("merged_scaled.csv 不存在")
        return None, None, None, None, None
    
    merged_df = pd.read_csv(merged_csv, index_col=0)
    
    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    if 'age' not in numeric_columns:
        numeric_columns = list(numeric_columns) + ['age']
    merged_df = merged_df[numeric_columns]
    merged_df = merged_df.dropna(axis=1, how='any')
    merged_df = merged_df.dropna()
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df)
    
    selected_features = joblib.load(os.path.join(Config.MODELS_DIR, 'selected_features_stacking.pkl'))
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    gene_names = selected_features
    
    base_models = joblib.load(os.path.join(Config.MODELS_DIR, 'base_models_refactored.pkl'))
    xgb_model = None
    mlp_model = None
    
    for name, model in base_models:
        if name == 'xgboost':
            xgb_model = model
        elif name == 'mlp':
            mlp_model = model
    
    if xgb_model is None or mlp_model is None:
        logger.error("模型加载失败")
        return None, None, None, None, None, None
    
    logger.info(f"数据加载完成，测试集样本数: {len(X_test_selected)}")
    logger.info(f"特征数: {len(gene_names)}")
    
    return X_train_selected, X_test_selected, y_test, gene_names, xgb_model, mlp_model


def calculate_xgb_shap_values(model, X):
    logger.info("计算XGBoost SHAP值...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    logger.info("XGBoost SHAP值计算完成")
    return explainer, shap_values


def calculate_mlp_shap_values(model, X_train, X_test):
    logger.info("计算MLP SHAP值...")
    
    X_train_sample = shap.sample(X_train, 100)
    explainer = shap.KernelExplainer(model.predict, X_train_sample)
    shap_values = explainer.shap_values(X_test)
    
    logger.info("MLP SHAP值计算完成")
    return explainer, shap_values


def plot_shap_summary(shap_values, X, gene_names, n_top=20, prefix="xgb"):
    logger.info(f"绘制{prefix} SHAP summary plot...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=gene_names, 
        plot_type="bar", 
        max_display=n_top,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, f'shap_summary_bar_{prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=gene_names, 
        max_display=n_top,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, f'shap_summary_beeswarm_{prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"{prefix} SHAP summary plots保存完成")


def get_top_genes(shap_values, gene_names, n_top=20):
    abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    
    top_indices = np.argsort(abs_shap)[::-1][:n_top]
    
    results = []
    for idx in top_indices:
        results.append({
            'gene': gene_names[idx],
            'abs_shap_mean': abs_shap[idx],
            'shap_mean': mean_shap[idx],
            'rank': len(results) + 1
        })
    
    return pd.DataFrame(results)


def find_common_genes(xgb_top_genes, mlp_top_genes):
    logger.info("\n寻找两个模型共同识别的核心基因...")
    
    xgb_gene_set = set(xgb_top_genes['gene'])
    mlp_gene_set = set(mlp_top_genes['gene'])
    
    common_genes = xgb_gene_set.intersection(mlp_gene_set)
    
    logger.info(f"XGBoost Top 20 基因: {list(xgb_gene_set)}")
    logger.info(f"MLP Top 20 基因: {list(mlp_gene_set)}")
    logger.info(f"共同核心基因 ({len(common_genes)} 个): {list(common_genes)}")
    
    return common_genes


def main():
    with open('shap_analysis_xgb_mlp.log', 'w', encoding='utf-8') as f:
        f.write("XGBoost + MLP 模型SHAP分析日志\n")
        f.write("=" * 50 + "\n")
    
    logger.info("开始XGBoost + MLP模型SHAP分析...")
    
    X_train, X_test, y_test, gene_names, xgb_model, mlp_model = load_data_and_models()
    
    if X_test is None:
        logger.error("数据加载失败")
        return
    
    logger.info("\n=== XGBoost SHAP分析 ===")
    xgb_explainer, xgb_shap_values = calculate_xgb_shap_values(xgb_model, X_test)
    plot_shap_summary(xgb_shap_values, X_test, gene_names, n_top=Config.N_TOP_FEATURES, prefix="xgb")
    xgb_top_genes = get_top_genes(xgb_shap_values, gene_names, n_top=Config.N_TOP_FEATURES)
    xgb_top_genes.to_csv(os.path.join(Config.PLOTS_DIR, 'top_genes_shap_xgb.csv'), index=False, encoding='utf-8')
    
    logger.info("\nXGBoost Top 20 基因:")
    for _, row in xgb_top_genes.iterrows():
        logger.info(f"  {row['rank']}. {row['gene']}: |SHAP|={row['abs_shap_mean']:.4f}, SHAP均值={row['shap_mean']:.4f}")
    
    logger.info("\n=== MLP SHAP分析 ===")
    mlp_explainer, mlp_shap_values = calculate_mlp_shap_values(mlp_model, X_train, X_test)
    plot_shap_summary(mlp_shap_values, X_test, gene_names, n_top=Config.N_TOP_FEATURES, prefix="mlp")
    mlp_top_genes = get_top_genes(mlp_shap_values, gene_names, n_top=Config.N_TOP_FEATURES)
    mlp_top_genes.to_csv(os.path.join(Config.PLOTS_DIR, 'top_genes_shap_mlp.csv'), index=False, encoding='utf-8')
    
    logger.info("\nMLP Top 20 基因:")
    for _, row in mlp_top_genes.iterrows():
        logger.info(f"  {row['rank']}. {row['gene']}: |SHAP|={row['abs_shap_mean']:.4f}, SHAP均值={row['shap_mean']:.4f}")
    
    common_genes = find_common_genes(xgb_top_genes, mlp_top_genes)
    
    common_genes_df = pd.DataFrame({
        'gene': list(common_genes)
    })
    common_genes_df.to_csv(os.path.join(Config.PLOTS_DIR, 'common_core_genes.csv'), index=False, encoding='utf-8')
    
    logger.info("\nSHAP分析完成！")
    logger.info("生成的文件:")
    logger.info("  - plots/shap_summary_bar_xgb.png: XGBoost Top基因SHAP重要性柱状图")
    logger.info("  - plots/shap_summary_beeswarm_xgb.png: XGBoost Top基因SHAP beeswarm图")
    logger.info("  - plots/shap_summary_bar_mlp.png: MLP Top基因SHAP重要性柱状图")
    logger.info("  - plots/shap_summary_beeswarm_mlp.png: MLP Top基因SHAP beeswarm图")
    logger.info("  - plots/top_genes_shap_xgb.csv: XGBoost Top基因详细分析结果")
    logger.info("  - plots/top_genes_shap_mlp.csv: MLP Top基因详细分析结果")
    logger.info("  - plots/common_core_genes.csv: 两个模型共同识别的核心基因")


if __name__ == "__main__":
    main()
