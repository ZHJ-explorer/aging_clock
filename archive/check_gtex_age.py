
import os
import pandas as pd

DATA_DIR = 'data'

def check_gtex_age():
    """详细检查GTEx样本注释文件中的年龄信息"""
    print("详细检查GTEx年龄信息...")
    
    annot_file = os.path.join(DATA_DIR, 'GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt')
    
    if not os.path.exists(annot_file):
        print(f"文件 {annot_file} 不存在")
        return
    
    df_annot = pd.read_csv(annot_file, sep='\t', low_memory=False)
    
    print(f"注释数据形状: {df_annot.shape}")
    print(f"\n所有列名:")
    for i, col in enumerate(df_annot.columns):
        print(f"  {i+1}. {col}")
    
    print(f"\n=== 查看前20行的所有数据 ===")
    print(df_annot.head(20).to_string())
    
    print(f"\n=== 查看包含数字的列 ===")
    for col in df_annot.columns:
        if df_annot[col].dtype in ['int64', 'float64']:
            print(f"\n{col}:")
            print(f"  非空值数量: {df_annot[col].notna().sum()}")
            print(f"  唯一值数量: {df_annot[col].nunique()}")
            if df_annot[col].notna().sum() > 0:
                print(f"  统计:")
                print(df_annot[col].describe())
                print(f"  前10个值: {list(df_annot[col].dropna().head(10))}")

if __name__ == "__main__":
    check_gtex_age()

