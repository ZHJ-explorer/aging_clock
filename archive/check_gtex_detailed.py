
import os
import pandas as pd

DATA_DIR = 'data'

def check_gtex_detailed():
    """详细检查GTEx数据集中的年龄信息"""
    print("详细检查GTEx年龄信息...")
    
    annot_file = os.path.join(DATA_DIR, 'GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt')
    
    if not os.path.exists(annot_file):
        print(f"文件 {annot_file} 不存在")
        return
    
    df_annot = pd.read_csv(annot_file, sep='\t', low_memory=False)
    
    print(f"注释数据形状: {df_annot.shape}")
    
    print(f"\n=== 查找所有包含'age'、'AGE'或'Age'的列 ===")
    for col in df_annot.columns:
        if 'age' in col.lower():
            print(f"  - {col}")
    
    print(f"\n=== 查看所有列的前5个值 ===")
    for col in df_annot.columns[:30]:
        print(f"\n{col}:")
        print(f"  数据类型: {df_annot[col].dtype}")
        print(f"  前5个值: {list(df_annot[col].head())}")
    
    print(f"\n=== 检查是否有SubjectID或类似的列 ===")
    for col in df_annot.columns:
        if 'subject' in col.lower() or 'subjid' in col.lower() or 'donor' in col.lower():
            print(f"  - {col}")
            print(f"    前10个值: {list(df_annot[col].head(10))}")
    
    print(f"\n=== 查看SAMPID的结构 ===")
    print(f"前10个SAMPID: {list(df_annot['SAMPID'].head(10))}")
    
    print(f"\n=== 尝试从SAMPID中提取Subject ID ===")
    df_annot['SubjectID'] = df_annot['SAMPID'].apply(lambda x: '-'.join(x.split('-')[:2]))
    print(f"提取的SubjectID示例: {list(df_annot['SubjectID'].head(10))}")
    print(f"唯一Subject数量: {df_annot['SubjectID'].nunique()}")
    
    print(f"\n=== 查看所有列的数据类型 ===")
    print(df_annot.dtypes.head(50))

if __name__ == "__main__":
    check_gtex_detailed()

