import pandas as pd
import os

# 加载两个文件
rnaseq_file = 'preprocessed_data/GSE231409_rnaseq_processed.csv'
age_file = 'preprocessed_data/GSE231409_processed.csv'

# 检查文件是否存在
if not os.path.exists(rnaseq_file):
    print(f"文件不存在: {rnaseq_file}")
    exit(1)

if not os.path.exists(age_file):
    print(f"文件不存在: {age_file}")
    exit(1)

# 读取年龄文件
df_age = pd.read_csv(age_file, index_col=0)
print(f"GSE231409_processed.csv 样本数: {len(df_age)}")
print(f"GSE231409_processed.csv 列: {list(df_age.columns)}")
print(f"前5个样本ID: {list(df_age['id'].head())}")

# 读取RNA-seq文件
df_rnaseq = pd.read_csv(rnaseq_file, index_col=0)
print(f"\nGSE231409_rnaseq_processed.csv 样本数: {len(df_rnaseq)}")
print(f"GSE231409_rnaseq_processed.csv 列数: {len(df_rnaseq.columns)}")
print(f"前5个样本ID: {list(df_rnaseq.index[:5])}")

# 检查是否有共同的样本标识符
rnaseq_ids = set(df_rnaseq.index)
age_ids = set(df_age['id'])
common_ids = rnaseq_ids.intersection(age_ids)

print(f"\n共同样本数: {len(common_ids)}")
if common_ids:
    print(f"共同样本ID: {list(common_ids)}")
else:
    print("没有共同的样本标识符")

# 检查样本数是否匹配
print(f"\n年龄文件样本数: {len(df_age)}")
print(f"RNA-seq文件样本数: {len(df_rnaseq)}")

# 检查是否可以通过位置匹配
if len(df_age) == len(df_rnaseq):
    print("\n样本数相同，可能可以通过位置匹配")
    print("前5个年龄值:")
    print(df_age['age'].head())
else:
    print("\n样本数不同，无法通过位置匹配")
