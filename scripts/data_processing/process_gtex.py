
import os
import gzip
import pandas as pd
import numpy as np

DATA_DIR = 'data'
PREPROCESSED_DIR = 'preprocessed_data'

def convert_age_range_to_midpoint(age_range):
    """将年龄范围转换为中间值"""
    if pd.isna(age_range):
        return np.nan
    if '-' not in age_range:
        try:
            return float(age_range)
        except:
            return np.nan
    parts = age_range.split('-')
    if len(parts) == 2:
        try:
            low = float(parts[0])
            high = float(parts[1])
            return (low + high) / 2
        except:
            return np.nan
    return np.nan

def process_gtex():
    """处理GTEx数据集"""
    print("处理GTEx数据集...")
    
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    gct_file = os.path.join(DATA_DIR, 'gene_tpm_v11_whole_blood.gct.gz')
    sample_annot_file = os.path.join(DATA_DIR, 'GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt')
    subject_annot_file = os.path.join(DATA_DIR, 'GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt')
    
    if not os.path.exists(gct_file):
        print(f"文件 {gct_file} 不存在")
        return None
    if not os.path.exists(sample_annot_file):
        print(f"文件 {sample_annot_file} 不存在")
        return None
    if not os.path.exists(subject_annot_file):
        print(f"文件 {subject_annot_file} 不存在")
        return None
    
    print(f"\n=== 加载表达数据 ===")
    with gzip.open(gct_file, 'rt') as f:
        f.readline()
        f.readline()
        df_expr = pd.read_csv(f, sep='\t', low_memory=False)
    
    print(f"表达数据形状: {df_expr.shape}")
    
    print(f"\n=== 加载受试者表型数据 ===")
    df_subject = pd.read_csv(subject_annot_file, sep='\t', low_memory=False)
    print(f"受试者表型数据形状: {df_subject.shape}")
    
    print(f"\n=== 转换年龄信息 ===")
    df_subject['age_numeric'] = df_subject['AGE'].apply(convert_age_range_to_midpoint)
    print(f"年龄信息转换完成")
    print(f"年龄统计:")
    print(df_subject['age_numeric'].describe())
    
    print(f"\n=== 转换表达数据格式 ===")
    gene_names = []
    seen_genes = {}
    for name in df_expr['Name']:
        if name in seen_genes:
            seen_genes[name] += 1
            gene_names.append(f"{name}_{seen_genes[name]}")
        else:
            seen_genes[name] = 0
            gene_names.append(name)
    
    df_expr_transposed = df_expr.drop(['Name', 'Description'], axis=1).T
    df_expr_transposed.columns = gene_names
    df_expr_transposed.index.name = 'SAMPID'
    df_expr_transposed = df_expr_transposed.reset_index()
    
    print(f"转换后表达数据形状: {df_expr_transposed.shape}")
    
    print(f"\n=== 提取Subject ID ===")
    df_expr_transposed['SUBJID'] = df_expr_transposed['SAMPID'].apply(lambda x: '-'.join(x.split('-')[:2]))
    print(f"提取完成")
    
    print(f"\n=== 合并表达数据和年龄信息 ===")
    df_merged = pd.merge(df_expr_transposed, df_subject[['SUBJID', 'age_numeric']], on='SUBJID', how='left')
    print(f"合并后数据形状: {df_merged.shape}")
    
    print(f"\n=== 过滤无年龄信息的样本 ===")
    df_final = df_merged.dropna(subset=['age_numeric'])
    print(f"过滤后样本数: {len(df_final)}")
    
    print(f"\n=== 整理最终数据格式 ===")
    df_final = df_final.drop(['SAMPID', 'SUBJID'], axis=1)
    df_final = df_final.rename(columns={'age_numeric': 'age'})
    
    print(f"最终数据形状: {df_final.shape}")
    print(f"样本数: {len(df_final)}")
    print(f"基因数: {len(df_final.columns) - 1}")
    print(f"年龄统计:")
    print(df_final['age'].describe())
    
    output_file = os.path.join(PREPROCESSED_DIR, 'GTEx_processed.csv')
    df_final.to_csv(output_file, index=False)
    print(f"\n处理后的数据已保存到: {output_file}")
    
    return df_final

if __name__ == "__main__":
    process_gtex()

