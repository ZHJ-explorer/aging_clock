import os
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 抑制警告信息
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

from scripts.utils.data_utils import standardize_data
from scripts.utils.gene_utils import standardize_gene_name, map_gene_ids, align_genes_across_datasets

# 数据目录
PREPROCESSED_DIR = 'preprocessed_data'

# 确保目录存在
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def collect_all_genes():
    """收集所有数据集的基因ID"""
    print("收集所有基因ID...")
    
    all_genes = []
    
    # 加载微阵列数据集
    microarray_files = [
        'GSE123696_processed.csv',
        'GSE123697_processed.csv',
        'GSE123698_processed.csv',
        'GSE164191_processed.csv'
    ]
    
    for file in microarray_files:
        file_path = os.path.join(PREPROCESSED_DIR, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, nrows=0)
            genes = [col for col in df.columns if col != 'age']
            all_genes.extend(genes)
            print(f"收集 {file} 的基因: {len(genes)} 个")
    
    # 加载RNA-seq数据集
    rnaseq_files = [
        'GSE293163_processed.csv',
        'GSE231409_combined_processed.csv',
        'GSE213516_processed.csv',
        'GTEx_processed.csv'
    ]
    
    for rnaseq_file in rnaseq_files:
        file_path = os.path.join(PREPROCESSED_DIR, rnaseq_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, nrows=0)
            genes = [col for col in df.columns if col not in ['age', 'age_years']]
            all_genes.extend(genes)
            print(f"收集 {rnaseq_file} 的基因: {len(genes)} 个")
    
    print(f"总共收集了 {len(all_genes)} 个基因ID")
    return all_genes

def load_datasets():
    """加载所有数据集"""
    print("加载所有数据集...")
    
    # 先收集所有基因ID并批量标准化
    all_genes = collect_all_genes()
    if all_genes:
        print("批量标准化基因名称...")
        map_gene_ids(all_genes)
    
    all_datasets = []
    
    # 加载所有处理过的数据集文件，排除重复的 GSE231409 数据集
    processed_files = [f for f in os.listdir(PREPROCESSED_DIR) if f.endswith('_processed.csv') and 'merged' not in f and 'GSE231409_rnaseq' not in f]
    
    for file in processed_files:
        file_path = os.path.join(PREPROCESSED_DIR, file)
        if os.path.exists(file_path):
            print(f"正在加载数据集: {file}")
            df = pd.read_csv(file_path, index_col=0)
            print(f"  原始样本数: {len(df)}")
            
            # 数据质量检查
            print(f"  数据质量检查...")
            # 检查是否有基因表达列（非age列）
            gene_columns = [col for col in df.columns if col not in ['age', 'age_years', 'id', 'Unnamed: 0']]
            if len(gene_columns) == 0:
                print(f"  警告：数据集 {file} 没有基因表达列，跳过")
                continue
            
            # 检查数值型基因表达列
            numeric_gene_columns = df[gene_columns].select_dtypes(include=['number']).columns
            if len(numeric_gene_columns) == 0:
                print(f"  警告：数据集 {file} 没有数值型基因表达列，跳过")
                continue
            
            print(f"  找到 {len(numeric_gene_columns)} 个数值型基因表达列")
            
            # 移除age_years列（如果存在）
            if 'age_years' in df.columns:
                df = df.drop(columns=['age_years'])
            
            # 标准化基因名称
            print(f"  正在标准化基因名称...")
            standardized_columns = []
            for col in df.columns:
                if col == 'age':
                    standardized_columns.append('age')
                else:
                    standardized_columns.append(standardize_gene_name(col))
            df.columns = standardized_columns
            
            all_datasets.append((file, df))
            print(f"  成功加载数据集: {file}, 样本数={len(df)}")
    
    if not all_datasets:
        print("没有找到任何数据集")
        return None
    
    print(f"总共加载了 {len(all_datasets)} 个数据集")
    return all_datasets

def merge_datasets():
    """合并所有数据集"""
    print("\n=== 合并数据集 ===")
    
    # 加载所有数据集
    all_datasets = load_datasets()
    if not all_datasets:
        return None
    
    # 对齐所有数据集的基因
    print("对齐所有数据集的基因...")
    aligned_datasets, common_genes = align_genes_across_datasets(all_datasets)
    
    if len(common_genes) == 0:
        print("警告：没有找到共同基因，无法合并训练")
        return None
    
    print(f"找到 {len(common_genes)} 个共同基因")
    
    # 对每个数据集进行标准化，消除批次效应
    print("对每个数据集进行标准化...")
    normalized_datasets = []
    dataset_sizes = []
    for i, df in enumerate(aligned_datasets):
        # 标准化每个数据集
        df_normalized = standardize_data(df)
        # 再次处理NaN值，确保没有NaN值
        print(f"  处理数据集 {i+1} 中的NaN值...")
        # 删除包含NaN值的列
        df_normalized = df_normalized.dropna(axis=1, how='any')
        # 确保age列存在
        if 'age' not in df_normalized.columns:
            df_normalized['age'] = df['age']
        # 确保至少有一个特征列
        if len(df_normalized.columns) > 1:
            normalized_datasets.append(df_normalized)
            dataset_sizes.append(len(df_normalized))
            print(f"  数据集 {i+1} 处理完成，特征数: {len(df_normalized.columns) - 1}")
        else:
            print(f"  数据集 {i+1} 没有有效特征，跳过")
    
    # 合并所有数据集
    if normalized_datasets:
        merged_df = pd.concat(normalized_datasets, ignore_index=True)
    else:
        print("警告：没有有效的数据集可以合并")
        return None
    print(f"合并后总样本数: {len(merged_df)}, 基因数: {len(merged_df.columns)-1}")
    
    # 处理缺失值（KNN填补）
    print("处理缺失值...")
    from sklearn.impute import KNNImputer
    # 分离特征和目标变量
    X = merged_df.drop('age', axis=1)
    y = merged_df['age']
    # 使用KNN填补缺失值
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    # 创建填补后的DataFrame
    merged_df_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=merged_df.index)
    merged_df_imputed['age'] = y
    print("缺失值处理完成")
    
    # 批次效应校正（ComBat）
    print("批次效应校正（ComBat）...")
    # 提取批次信息（每个样本所属的数据集）
    batch = []
    for i, size in enumerate(dataset_sizes):
        batch.extend([i] * size)
    # 确保批次长度与样本数一致
    batch = batch[:len(merged_df_imputed)]
    
    # 使用pyComBat进行批次效应校正
    try:
        from pycombat import Combat
        # 分离特征和目标变量
        X_combat = merged_df_imputed.drop('age', axis=1).values
        # 确保批次信息是正确的形状
        batch_array = np.array(batch)
        # 创建ComBat对象并校正批次效应
        combat = Combat()
        # 正确的API：fit_transform(Y, b)
        X_corrected = combat.fit_transform(X_combat, batch_array)
        # 转换回DataFrame
        merged_df_corrected = pd.DataFrame(X_corrected, columns=merged_df_imputed.drop('age', axis=1).columns, index=merged_df_imputed.index)
        merged_df_corrected['age'] = y
        print("批次效应校正完成（ComBat）")
    except Exception as e:
        print(f"ComBat批次效应校正失败: {e}")
        print("使用未校正的数据继续")
        merged_df_corrected = merged_df_imputed
    
    # 对所有特征进行z-score标准化（用户要求）
    print("对所有特征进行z-score标准化...")
    from sklearn.preprocessing import StandardScaler
    X_final = merged_df_corrected.drop('age', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    merged_df_final = pd.DataFrame(X_scaled, columns=X_final.columns, index=merged_df_corrected.index)
    merged_df_final['age'] = y
    print("z-score标准化完成")
    
    # 保存合并后的数据
    merged_file = os.path.join(PREPROCESSED_DIR, 'merged_processed.csv')
    merged_df.to_csv(merged_file)
    print(f"合并数据已保存到: {merged_file}")
    
    # 保存标准化后的数据
    scaled_file = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    merged_df_final.to_csv(scaled_file)
    print(f"标准化数据已保存到: {scaled_file}")
    
    return merged_df_final

def main():
    """主函数"""
    print("开始预处理和合并数据集...")
    
    # 合并数据集
    merged_df = merge_datasets()
    
    if merged_df is not None:
        print("\n数据集预处理和合并完成！")
        print(f"合并后样本数: {len(merged_df)}")
        print(f"合并后基因数: {len(merged_df.columns)-1}")
    else:
        print("\n数据集预处理和合并失败！")

if __name__ == "__main__":
    main()
