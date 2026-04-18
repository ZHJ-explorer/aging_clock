import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.utils.data_utils import split_data
from scripts.config import MODELS_DIR, PLOTS_DIR, PREPROCESSED_DIR, Config


N_TOP_FEATURES = 20


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shap_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Config.ensure_directories_exist()


def load_data_and_models():
    logger.info("加载数据和模型...")
    
    merged_csv = os.path.join(Config.PREPROCESSED_DIR, 'merged_scaled.csv')
    if not os.path.exists(merged_csv):
        logger.error("merged_scaled.csv 不存在")
        return None, None, None, None
    
    merged_df = pd.read_csv(merged_csv, index_col=0)
    
    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    if 'age' not in numeric_columns:
        numeric_columns = list(numeric_columns) + ['age']
    merged_df = merged_df[numeric_columns]
    merged_df = merged_df.dropna(axis=1, how='any')
    merged_df = merged_df.dropna()
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df)
    
    selected_features = joblib.load(os.path.join(MODELS_DIR, 'selected_features_xgboost.pkl'))
    
    X_test_selected = X_test[selected_features]
    gene_names = selected_features
    
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_optimized.pkl'))
    
    logger.info(f"数据加载完成，测试集样本数: {len(X_test_selected)}")
    logger.info(f"特征数: {len(gene_names)}")
    
    return X_test_selected, y_test, gene_names, xgb_model


def calculate_shap_values(model, X):
    logger.info("计算SHAP值...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    logger.info("SHAP值计算完成")
    return explainer, shap_values


def plot_shap_summary(shap_values, X, gene_names, n_top=20):
    logger.info("绘制SHAP summary plot...")
    
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
    plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("SHAP summary plots保存完成")


def plot_shap_dependence(shap_values, X, gene_names, top_n_genes=5):
    logger.info("绘制SHAP dependence plots...")
    
    abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(abs_shap)[::-1][:top_n_genes]
    
    for idx in top_indices:
        gene = gene_names[idx]
        logger.info(f"绘制 {gene} 的dependence plot...")
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            idx, 
            shap_values, 
            X, 
            feature_names=gene_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'shap_dependence_{gene}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("SHAP dependence plots保存完成")


def analyze_top_genes(shap_values, gene_names, n_top=20):
    logger.info("分析Top基因...")
    
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
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(PLOTS_DIR, 'top_genes_shap.csv'), index=False, encoding='utf-8')
    
    logger.info(f"Top {n_top} 基因分析完成")
    logger.info("\nTop 20 基因:")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['rank']}. {row['gene']}: |SHAP|={row['abs_shap_mean']:.4f}, SHAP均值={row['shap_mean']:.4f}")
    
    return results_df


def main():
    with open('shap_analysis.log', 'w', encoding='utf-8') as f:
        f.write("XGBoost模型SHAP分析日志\n")
        f.write("=" * 50 + "\n")
    
    logger.info("开始XGBoost模型SHAP分析...")
    
    X_test, y_test, gene_names, xgb_model = load_data_and_models()
    
    if X_test is None:
        logger.error("数据加载失败")
        return
    
    explainer, shap_values = calculate_shap_values(xgb_model, X_test)
    
    plot_shap_summary(shap_values, X_test, gene_names, n_top=N_TOP_FEATURES)
    
    plot_shap_dependence(shap_values, X_test, gene_names, top_n_genes=5)
    
    top_genes_df = analyze_top_genes(shap_values, gene_names, n_top=Config.N_TOP_FEATURES)
    
    logger.info("\nSHAP分析完成！")
    logger.info("生成的文件:")
    logger.info("  - plots/shap_summary_bar.png: Top基因SHAP重要性柱状图")
    logger.info("  - plots/shap_summary_beeswarm.png: Top基因SHAP beeswarm图")
    logger.info("  - plots/shap_dependence_*.png: 重要基因的SHAP依赖图")
    logger.info("  - plots/top_genes_shap.csv: Top基因详细分析结果")


if __name__ == "__main__":
    main()
