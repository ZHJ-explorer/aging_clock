import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 数据目录
PREPROCESSED_DIR = 'preprocessed_data'
PLOTS_DIR = 'plots'

# 确保目录存在
os.makedirs(PLOTS_DIR, exist_ok=True)

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

def create_pca_plot(principal_components, dataset_labels, explained_variance):
    """创建PCA散点图"""
    print("创建PCA散点图...")
    
    # 创建DataFrame
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Dataset'] = dataset_labels
    
    # 绘制散点图
    plt.figure(figsize=(12, 8))
    
    # 定义颜色映射
    unique_datasets = np.unique(dataset_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_datasets)))
    color_map = dict(zip(unique_datasets, colors))
    
    # 绘制每个数据集的点
    for dataset in unique_datasets:
        subset = pca_df[pca_df['Dataset'] == dataset]
        plt.scatter(subset['PC1'], subset['PC2'], label=dataset, alpha=0.7, s=50, marker='.')
    
    # 添加标题和标签
    plt.title('PCA Analysis of Aging Clock Datasets', fontsize=16)
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize=12)
    
    # 添加图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加网格
    plt.grid(alpha=0.3)
    
    # 保存图像
    output_file = os.path.join(PLOTS_DIR, 'pca_analysis.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"PCA图已保存到: {output_file}")
    
    # 显示图像
    plt.show()

def get_dataset_labels():
    """获取每个样本所属的数据集标签"""
    # 数据集大小（从preprocess_and_merge.py的输出中获取）
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
