
import os
import pandas as pd

DATA_DIR = 'data'

def explore_gtex_subject():
    """探索GTEx受试者表型文件"""
    print("探索GTEx受试者表型文件...")
    
    subject_file = os.path.join(DATA_DIR, 'GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt')
    
    if not os.path.exists(subject_file):
        print(f"文件 {subject_file} 不存在")
        return
    
    df_subject = pd.read_csv(subject_file, sep='\t', low_memory=False)
    print(f"受试者表型数据形状: {df_subject.shape}")
    print(f"\n所有列名:")
    for i, col in enumerate(df_subject.columns):
        print(f"  {i+1}. {col}")
    
    print(f"\n前10行数据:")
    print(df_subject.head(10).to_string())
    
    print(f"\n=== 查找年龄相关列 ===")
    age_cols = [col for col in df_subject.columns if 'age' in col.lower()]
    print(f"年龄相关列: {age_cols}")
    
    if age_cols:
        print(f"\n年龄信息统计:")
        print(df_subject[age_cols].describe())
        print(f"\n前20个年龄值:")
        print(df_subject[age_cols].head(20))
    
    print(f"\n=== 查找Subject ID列 ===")
    subject_id_cols = [col for col in df_subject.columns if 'subject' in col.lower() or 'subjid' in col.lower() or 'donor' in col.lower() or 'dbgap' in col.lower()]
    print(f"Subject ID相关列: {subject_id_cols}")
    
    if subject_id_cols:
        print(f"\n前20个Subject ID:")
        print(df_subject[subject_id_cols].head(20))

if __name__ == "__main__":
    explore_gtex_subject()

