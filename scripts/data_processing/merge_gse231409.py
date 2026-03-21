import pandas as pd
import os

# 加载两个文件
rnaseq_file = 'preprocessed_data/GSE231409_rnaseq_processed.csv'
age_file = 'preprocessed_data/GSE231409_processed.csv'
output_file = 'preprocessed_data/GSE231409_combined_processed.csv'

# 读取年龄文件
df_age = pd.read_csv(age_file, index_col=0)
print(f"GSE231409_processed.csv 样本数: {len(df_age)}")

# 读取RNA-seq文件
df_rnaseq = pd.read_csv(rnaseq_file, index_col=0)
print(f"GSE231409_rnaseq_processed.csv 样本数: {len(df_rnaseq)}")

# 检查样本数是否匹配
if len(df_age) != len(df_rnaseq):
    print("样本数不匹配，无法合并")
    exit(1)

# 通过位置匹配添加年龄信息
df_rnaseq['age'] = df_age['age'].values

# 保存合并后的文件
df_rnaseq.to_csv(output_file)
print(f"\n合并完成，保存到: {output_file}")
print(f"合并后样本数: {len(df_rnaseq)}")
print(f"合并后列数: {len(df_rnaseq.columns)}")
print(f"前5列: {list(df_rnaseq.columns[:5])}")
print(f"最后5列: {list(df_rnaseq.columns[-5:])}")
