import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.config import PLOTS_DIR, PREPROCESSED_DIR, MODELS_DIR, Config

Config.ensure_directories_exist()

NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#7E6148',
              '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148',
              '#B09C85', '#5F559B', '#A20056', '#1E2D5C', '#B21F2D']

NPG_CMAP = sns.diverging_palette(220, 20, as_cmap=True)

N_CORE_GENES = 50
CORRELATION_THRESHOLD = 0.3
NETWORK_THRESHOLD = 0.5


def load_expression_data():
    """加载表达量数据"""
    merged_csv = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    if os.path.exists(merged_csv):
        df = pd.read_csv(merged_csv, index_col=0)
        return df
    return None


def load_core_genes(n_top=50):
    """加载核心基因列表"""
    try:
        selected_features = joblib.load(os.path.join(MODELS_DIR, 'selected_features_xgboost.pkl'))
        if len(selected_features) > n_top:
            return list(selected_features[:n_top])
        return list(selected_features)
    except:
        pass

    try:
        selected_features = joblib.load(os.path.join(MODELS_DIR, 'selected_features_stacking.pkl'))
        if len(selected_features) > n_top:
            return list(selected_features[:n_top])
        return list(selected_features)
    except:
        pass

    return None


def compute_gene_correlation_matrix(df, gene_names, method='spearman'):
    """计算基因间相关性矩阵"""
    X = df[gene_names].values

    if method == 'spearman':
        corr_matrix, _ = spearmanr(X, axis=0)
    else:
        corr_matrix = np.corrcoef(X, rowvar=False)

    corr_df = pd.DataFrame(corr_matrix, index=gene_names, columns=gene_names)
    return corr_df


def plot_correlation_heatmap(corr_matrix, output_path):
    """绘制核心基因相关性热图（NPG配色）"""
    n_genes = len(corr_matrix)

    g = sns.clustermap(corr_matrix,
                       method='average',
                       metric='euclidean',
                       cmap=NPG_CMAP,
                       center=0,
                       vmin=-1, vmax=1,
                       figsize=(14, 12),
                       dendrogram_ratio=(0.1, 0.1),
                       cbar_pos=(-0.06, 0.2, 0.03, 0.6),
                       linewidths=0.3,
                       xticklabels=True,
                       yticklabels=True)

    g.ax_heatmap.set_xlabel('Genes', fontsize=20)
    g.ax_heatmap.set_ylabel('Genes', fontsize=20)
    g.fig.suptitle(f'Core Gene Co-expression Network (Spearman Correlation)\nTop {n_genes} Genes',
                   fontsize=20, y=1.05)

    g.ax_cbar.set_ylabel('')

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"热图已保存到: {output_path}")


def plot_correlation_heatmap_simple(corr_matrix, output_path):
    """绘制简洁版相关性热图"""
    n_genes = len(corr_matrix)

    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, True)

    g = sns.heatmap(corr_matrix,
                    mask=mask,
                    cmap=NPG_CMAP,
                    center=0,
                    vmin=-1, vmax=1,
                    square=True,
                    linewidths=0.3,
                    cbar_kws={'shrink': 0.6},
                    xticklabels=True,
                    yticklabels=True,
                    ax=ax)

    ax.set_xlabel('Genes', fontsize=20)
    ax.set_ylabel('Genes', fontsize=20)
    ax.set_title(f'Core Gene Co-expression Heatmap (Spearman)\nTop {n_genes} Genes', fontsize=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"简洁热图已保存到: {output_path}")


def main():
    print("=" * 60)
    print("核心基因共表达网络分析")
    print("=" * 60)

    df = load_expression_data()
    if df is None:
        print("错误: 无法加载数据，请确保 merged_scaled.csv 存在")
        return

    print(f"加载数据: {df.shape[0]} 样本, {len(df.columns)-1} 基因")

    gene_names = load_core_genes(n_top=N_CORE_GENES)
    if gene_names is None:
        print("错误: 无法加载核心基因列表")
        return

    available_genes = [g for g in gene_names if g in df.columns]
    if len(available_genes) < len(gene_names):
        print(f"警告: 只找到 {len(available_genes)}/{len(gene_names)} 个基因在数据中")
        gene_names = available_genes

    if len(gene_names) < 2:
        print("错误: 可用基因数不足")
        return

    print(f"\n计算 {len(gene_names)} 个核心基因的相关性矩阵...")

    corr_matrix = compute_gene_correlation_matrix(df, gene_names, method='spearman')

    print("\n绘制聚类热图...")
    heatmap_path = os.path.join(PLOTS_DIR, 'core_gene_correlation_heatmap.png')
    plot_correlation_heatmap(corr_matrix, heatmap_path)

    print("\n绘制简洁热图...")
    simple_heatmap_path = os.path.join(PLOTS_DIR, 'core_gene_correlation_heatmap_simple.png')
    plot_correlation_heatmap_simple(corr_matrix, simple_heatmap_path)

    print("\n" + "=" * 60)
    print("核心基因共表达网络分析完成!")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - {heatmap_path}")
    print(f"  - {simple_heatmap_path}")


if __name__ == "__main__":
    main()
