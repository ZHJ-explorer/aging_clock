import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, dendrogram

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.config import PLOTS_DIR, PREPROCESSED_DIR, Config

Config.ensure_directories_exist()

NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#7E6148', '#B09C85', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']

def load_expression_data():
    """加载表达量数据"""
    merged_csv = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    if os.path.exists(merged_csv):
        df = pd.read_csv(merged_csv, index_col=0)
        return df
    return None

def filter_low_expression_genes(df, threshold=0.1, min_samples_ratio=0.5):
    """过滤低表达基因

    Args:
        df: 表达量DataFrame
        threshold: 表达阈值
        min_samples_ratio: 低于阈值的样本比例阈值
    """
    gene_cols = [c for c in df.columns if c != 'age']
    low_exp_genes = []

    for gene in gene_cols:
        expr_values = df[gene].values
        ratio_below = np.sum(np.abs(expr_values) < threshold) / len(expr_values)
        if ratio_below >= min_samples_ratio:
            low_exp_genes.append(gene)

    high_exp_genes = [g for g in gene_cols if g not in low_exp_genes]
    return high_exp_genes, low_exp_genes

def plot_expression_density(df, high_exp_genes, output_path):
    """绘制基因表达量密度图（过滤前后对比）"""
    gene_cols = [c for c in df.columns if c != 'age']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.rcParams['font.size'] = 20

    ax1 = axes[0]
    for i, gene in enumerate(gene_cols[:20]):
        color = NPG_COLORS[i % len(NPG_COLORS)]
        sns.kdeplot(df[gene].values, ax=ax1, color=color, alpha=0.6, linewidth=1.5)
    ax1.set_xlabel('Expression Level (Z-score)', fontsize=20)
    ax1.set_ylabel('Density', fontsize=20)
    ax1.set_title(f'Before Filtering\n(All {len(gene_cols)} genes)', fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)

    ax2 = axes[1]
    for i, gene in enumerate(high_exp_genes[:20]):
        color = NPG_COLORS[i % len(NPG_COLORS)]
        sns.kdeplot(df[gene].values, ax=ax2, color=color, alpha=0.6, linewidth=1.5)
    ax2.set_xlabel('Expression Level (Z-score)', fontsize=20)
    ax2.set_ylabel('Density', fontsize=20)
    ax2.set_title(f'After Filtering\n({len(high_exp_genes)} high-expression genes)', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"密度图已保存到: {output_path}")

def compute_sample_correlations(df, method='spearman'):
    """计算样本间相关性矩阵"""
    gene_cols = [c for c in df.columns if c != 'age']
    X = df[gene_cols].values

    if method == 'spearman':
        corr_matrix, _ = spearmanr(X, axis=1)
    else:
        corr_matrix = np.corrcoef(X, rowvar=True)

    return corr_matrix

def plot_correlation_heatmap(df, output_path, method='spearman'):
    """绘制样本相关性热图（带批次/年龄注释）"""
    gene_cols = [c for c in df.columns if c != 'age']
    X = df[gene_cols].values

    n_samples = X.shape[0]
    sample_names = [f'S{i}' for i in range(n_samples)]

    if method == 'spearman':
        corr_matrix = compute_sample_correlations(df, method='spearman')
    else:
        corr_matrix = compute_sample_correlations(df, method='pearson')

    corr_df = pd.DataFrame(corr_matrix, index=sample_names, columns=sample_names)

    age_bins = pd.cut(df['age'].values, bins=5, labels=['20-30', '30-40', '40-50', '50-60', '60+'])
    row_colors = pd.Series(age_bins, index=sample_names, name='Age')

    age_color_map = {
        '20-30': '#E64B35',
        '30-40': '#4DBBD5',
        '40-50': '#00A087',
        '50-60': '#3C5488',
        '60+': '#7E6148'
    }
    row_colors_mapped = row_colors.map(age_color_map)

    from matplotlib.colors import to_rgba
    row_colors_transparent = [to_rgba(c, alpha=0.3) for c in row_colors_mapped.values]

    g = sns.clustermap(corr_df,
                       row_cluster=True,
                       col_cluster=True,
                       row_colors=row_colors_transparent,
                       col_colors=row_colors_transparent,
                       cmap='RdBu_r',
                       center=0,
                       vmin=-1, vmax=1,
                       figsize=(12, 10),
                       dendrogram_ratio=(0.15, 0.15),
                       cbar_pos=(0.02, 0.8, 0.03, 0.15),
                       colors_ratio=(0.03, 0.03),
                       xticklabels=False,
                       yticklabels=False)

    g.ax_heatmap.set_xlabel('Samples', fontsize=20)
    g.ax_heatmap.set_ylabel('Samples', fontsize=20)
    g.fig.suptitle(f'Sample Correlation Heatmap ({method.capitalize()})', fontsize=20, y=1.02)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label) for label, color in age_color_map.items()]
    g.ax_heatmap.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.12, 1), title='Age Group', fontsize=14)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"相关性热图已保存到: {output_path}")

def main():
    print("=" * 60)
    print("基因表达量QC分析")
    print("=" * 60)

    df = load_expression_data()
    if df is None:
        print("错误: 无法加载数据，请确保merged_scaled.csv存在")
        return

    print(f"加载数据: {df.shape[0]} 样本, {len(df.columns)-1} 基因")

    print("\n过滤低表达基因...")
    high_exp_genes, low_exp_genes = filter_low_expression_genes(df)
    print(f"保留 {len(high_exp_genes)} 个高表达基因, 过滤 {len(low_exp_genes)} 个低表达基因")

    print("\n绘制表达量密度图...")
    density_path = os.path.join(PLOTS_DIR, 'gene_expression_density.png')
    plot_expression_density(df, high_exp_genes, density_path)

    print("\n绘制Spearman相关性热图...")
    spearman_path = os.path.join(PLOTS_DIR, 'sample_correlation_heatmap_spearman.png')
    plot_correlation_heatmap(df, spearman_path, method='spearman')

    print("\n绘制Pearson相关性热图...")
    pearson_path = os.path.join(PLOTS_DIR, 'sample_correlation_heatmap_pearson.png')
    plot_correlation_heatmap(df, pearson_path, method='pearson')

    print("\n" + "=" * 60)
    print("QC分析完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
