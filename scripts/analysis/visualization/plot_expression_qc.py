import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.config import PLOTS_DIR, PREPROCESSED_DIR, Config

Config.ensure_directories_exist()

def load_expression_data():
    """加载表达量数据"""
    merged_csv = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    if os.path.exists(merged_csv):
        df = pd.read_csv(merged_csv, index_col=0)
        return df
    return None

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
