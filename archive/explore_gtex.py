
import os
import gzip
import pandas as pd

DATA_DIR = 'data'

def explore_gtex():
    """探索GTEx数据集"""
    print("探索GTEx数据集...")
    
    gct_file = os.path.join(DATA_DIR, 'gene_tpm_v11_whole_blood.gct.gz')
    annot_file = os.path.join(DATA_DIR, 'GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt')
    
    if not os.path.exists(gct_file):
        print(f"文件 {gct_file} 不存在")
        return
    
    if not os.path.exists(annot_file):
        print(f"文件 {annot_file} 不存在")
        return
    
    print(f"\n=== 探索表达数据文件: {gct_file} ===")
    with gzip.open(gct_file, 'rt') as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()
        print(f"GCT头: {line1}")
        print(f"尺寸行: {line2}")
        
        df_expr = pd.read_csv(f, sep='\t', low_memory=False, nrows=5)
        print(f"\n前5行数据形状: {df_expr.shape}")
        print(f"列名: {list(df_expr.columns[:10])}")
        print(f"\n前5行数据:")
        print(df_expr.head())
    
    print(f"\n=== 探索样本注释文件: {annot_file} ===")
    df_annot = pd.read_csv(annot_file, sep='\t', low_memory=False)
    print(f"注释数据形状: {df_annot.shape}")
    print(f"列名: {list(df_annot.columns)}")
    print(f"\n前5行数据:")
    print(df_annot.head())
    
    print(f"\n=== 查找年龄相关列 ===")
    age_cols = [col for col in df_annot.columns if 'age' in col.lower()]
    print(f"年龄相关列: {age_cols}")
    
    if age_cols:
        print(f"\n年龄信息统计:")
        print(df_annot[age_cols].describe())
        print(f"\n前10个年龄值:")
        print(df_annot[age_cols].head(10))
    
    print(f"\n=== 样本ID匹配检查 ===")
    with gzip.open(gct_file, 'rt') as f:
        f.readline()
        f.readline()
        df_expr_header = pd.read_csv(f, sep='\t', low_memory=False, nrows=0)
        expr_sample_ids = list(df_expr_header.columns[2:])
    
    print(f"表达数据样本数: {len(expr_sample_ids)}")
    print(f"注释数据样本数: {len(df_annot)}")
    
    common_samples = set(expr_sample_ids) & set(df_annot['SAMPID'])
    print(f"共同样本数: {len(common_samples)}")
    
    if len(common_samples) > 0:
        print(f"\n前10个共同样本ID: {list(common_samples)[:10]}")

if __name__ == "__main__":
    explore_gtex()

