
import os
import gzip
import pandas as pd

DATA_DIR = 'data'

def explore_gct():
    """探索GCT格式的数据集"""
    print("探索gene_tpm_v10_whole_blood.gct.gz数据集...")
    
    gct_file = os.path.join(DATA_DIR, 'gene_tpm_v10_whole_blood.gct.gz')
    
    if not os.path.exists(gct_file):
        print(f"文件 {gct_file} 不存在")
        return
    
    print(f"加载文件: {gct_file}")
    
    # 读取GCT文件
    with gzip.open(gct_file, 'rt') as f:
        # 跳过前两行（GCT格式头）
        line1 = f.readline().strip()  # #1.2
        line2 = f.readline().strip()  # 行数 列数
        print(f"GCT头: {line1}")
        print(f"尺寸行: {line2}")
        
        # 读取数据
        df = pd.read_csv(f, sep='\t', low_memory=False)
        print(f"\n数据形状: {df.shape}")
        print(f"列名: {list(df.columns[:10])}")  # 只显示前10列
        print(f"\n前5行数据:")
        print(df.head())
        
        # 检查是否有年龄信息或样本注释
        print(f"\n第一列: {df.iloc[:, 0].name}")
        print(f"第二列: {df.iloc[:, 1].name}")
        
        # 检查样本数量
        num_samples = df.shape[1] - 2  # 减去前两列（通常是Name和Description）
        print(f"\n样本数量: {num_samples}")
        
        # 查看样本ID
        if num_samples > 0:
            sample_ids = list(df.columns[2:])
            print(f"前10个样本ID: {sample_ids[:10]}")

if __name__ == "__main__":
    explore_gct()
