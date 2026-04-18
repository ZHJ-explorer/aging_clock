import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scripts.config import PLOTS_DIR, PREPROCESSED_DIR, Config


Config.ensure_directories_exist()

NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

def load_merged_data():
    """加载合并后的数据"""
    print("加载合并后的数据...")

    # 加载标准化后的数据
    scaled_file = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    if os.path.exists(scaled_file):
        df = pd.read_csv(scaled_file, index_col=0)
        print(f"成功加载数据，样本数: {len(df)}, 基因数: {len(df.columns)-1}")
        return df
    else:
        print("错误：找不到合并后的数据文件")
        return None

def perform_pca(df, n_components=2):
    """执行PCA分析"""
    print("执行PCA分析...")

    # 分离特征和目标变量
    X = df.drop('age', axis=1)

    # 执行PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    # 计算解释方差比
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA解释方差比: PC1={explained_variance[0]:.4f}, PC2={explained_variance[1]:.4f}")

    return principal_components, explained_variance

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12

def create_pca_plot(principal_components, dataset_labels, explained_variance):
    """创建NPG学术风格的PCA散点图"""
    print("创建PCA散点图...")

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Dataset'] = dataset_labels

    fig, ax = plt.subplots(figsize=(8, 6))

    unique_datasets = np.unique(dataset_labels)
    color_map = dict(zip(unique_datasets, NPG_COLORS[:len(unique_datasets)]))

    for dataset in unique_datasets:
        subset = pca_df[pca_df['Dataset'] == dataset]
        x = subset['PC1'].values
        y = subset['PC2'].values
        color = color_map[dataset]

        ax.scatter(x, y, label=dataset, alpha=0.3, s=15, c=color, edgecolors='none')

    for dataset in unique_datasets:
        subset = pca_df[pca_df['Dataset'] == dataset]
        x = subset['PC1'].values
        y = subset['PC2'].values
        color = color_map[dataset]

        if len(x) > 4:
            confidence_ellipse(x, y, ax, n_std=2.0, edgecolor=color, linewidth=1.5, linestyle='--', facecolor='none')

    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%})', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%})', fontsize=14, fontweight='bold')
    ax.tick_params(direction='in', length=4, width=1.0, labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', fontsize=11,
              markerscale=1.2, borderpad=0.8, labelspacing=0.5, shadow=False)

    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_file = os.path.join(PLOTS_DIR, 'pca_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png', facecolor='white')
    print(f"PCA图已保存到: {output_file}")
    plt.show()


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """绘制置信椭圆"""
    if len(x) < 3:
        return

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(np.mean(x), np.mean(y))

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_dataset_labels():
    """获取每个样本所属的数据集标签"""
    # 数据集大小（从预处理流程的输出中获取）
    dataset_sizes = [66, 60, 68, 121, 17, 455, 30, 803]
    dataset_names = [
        'GSE123696',
        'GSE123697',
        'GSE123698',
        'GSE164191',
        'GSE213516',
        'GSE231409',
        'GSE293163',
        'GTEx'
    ]

    # 创建标签
    labels = []
    for name, size in zip(dataset_names, dataset_sizes):
        labels.extend([name] * size)

    return labels

def main():
    """主函数"""
    print("开始PCA分析...")

    # 加载数据
    df = load_merged_data()
    if df is None:
        return

    # 执行PCA
    principal_components, explained_variance = perform_pca(df)

    # 获取数据集标签
    dataset_labels = get_dataset_labels()

    # 确保标签长度与样本数一致
    if len(dataset_labels) != len(df):
        print(f"警告：标签长度({len(dataset_labels)})与样本数({len(df)})不一致")
        return

    # 创建PCA图
    create_pca_plot(principal_components, dataset_labels, explained_variance)

    print("PCA分析完成！")

if __name__ == "__main__":
    main()