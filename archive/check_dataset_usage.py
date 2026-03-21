import os
import pandas as pd

# 数据集目录
PREPROCESSED_DIR = 'preprocessed_data'

# 所有可能的数据集文件
dataset_files = [
    'GSE123696_processed.csv',
    'GSE123697_processed.csv',
    'GSE123698_processed.csv',
    'GSE213516_processed.csv',
    'GSE231409_combined_processed.csv',
    'GSE231409_processed.csv',
    'GSE231409_processed_years.csv',
    'GSE231409_rnaseq_processed.csv',
    'GSE293163_processed.csv',
    'GSE75337_processed.csv',
    'GSE75337_processed_years.csv'
]

print("=== 数据集使用情况分析 ===")
print("\n1. 数据集文件状态:")

for file in dataset_files:
    file_path = os.path.join(PREPROCESSED_DIR, file)
    if os.path.exists(file_path):
        print(f"✓ {file} - 存在")
    else:
        print(f"✗ {file} - 不存在")

print("\n2. 数据集内容分析:")

for file in dataset_files:
    file_path = os.path.join(PREPROCESSED_DIR, file)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0)
            print(f"\n{file}:")
            print(f"  样本数: {len(df)}")
            print(f"  列数: {len(df.columns)}")
            print(f"  列名: {list(df.columns[:5])}..." if len(df.columns) > 5 else f"  列名: {list(df.columns)}")
            
            # 检查是否有基因表达列
            gene_columns = [col for col in df.columns if col not in ['age', 'age_years', 'id', 'Unnamed: 0']]
            if len(gene_columns) > 0:
                print(f"  基因表达列数: {len(gene_columns)}")
            else:
                print(f"  警告: 没有基因表达列")
                
        except Exception as e:
            print(f"  错误: {e}")

print("\n3. 数据集使用状态:")
print("\n已使用的数据集:")
print("- GSE123696_processed.csv (微阵列数据)")
print("- GSE123697_processed.csv (微阵列数据)")
print("- GSE123698_processed.csv (微阵列数据)")
print("- GSE213516_processed.csv (RNA-seq数据)")
print("- GSE231409_combined_processed.csv (RNA-seq数据，包含年龄信息)")
print("- GSE293163_processed.csv (RNA-seq数据)")

print("\n未使用的数据集:")
print("- GSE231409_processed.csv (只有年龄信息，无基因表达)")
print("- GSE231409_processed_years.csv (可能是年龄单位转换版本)")
print("- GSE231409_rnaseq_processed.csv (已被合并到combined版本)")
print("- GSE75337_processed.csv (只有年龄信息，无基因表达)")
print("- GSE75337_processed_years.csv (可能是年龄单位转换版本)")
