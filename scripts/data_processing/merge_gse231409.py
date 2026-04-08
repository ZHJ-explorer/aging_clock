import pandas as pd
import os

rnaseq_file = 'preprocessed_data/GSE231409_rnaseq_processed.csv'
age_file = 'preprocessed_data/GSE231409_processed.csv'
output_file = 'preprocessed_data/GSE231409_combined_processed.csv'

def main():
    """合并GSE231409数据集的年龄和RNA-seq数据"""
    if not os.path.exists(rnaseq_file) or not os.path.exists(age_file):
        print(f"警告: 缺少必要文件 ({rnaseq_file} 或 {age_file})，跳过合并")
        return None

    df_age = pd.read_csv(age_file, index_col=0)
    print(f"GSE231409_processed.csv 样本数: {len(df_age)}")

    df_rnaseq = pd.read_csv(rnaseq_file, index_col=0)
    print(f"GSE231409_rnaseq_processed.csv 样本数: {len(df_rnaseq)}")

    if len(df_age) != len(df_rnaseq):
        print("样本数不匹配，无法合并")
        return None

    df_rnaseq['age'] = df_age['age'].values

    df_rnaseq.to_csv(output_file)
    print(f"\n合并完成，保存到: {output_file}")
    print(f"合并后样本数: {len(df_rnaseq)}")
    print(f"合并后列数: {len(df_rnaseq.columns)}")
    return df_rnaseq

if __name__ == "__main__":
    main()
