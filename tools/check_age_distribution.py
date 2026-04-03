import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

merged_df = pd.read_csv('preprocessed_data/merged_scaled.csv', index_col=0)
numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
if 'age' not in numeric_columns:
    numeric_columns = list(numeric_columns) + ['age']
merged_df = merged_df[numeric_columns]
merged_df = merged_df.dropna(axis=1, how='any')
merged_df = merged_df.dropna()

print(f'总样本数: {len(merged_df)}')
print(f'年龄<20的样本数: {len(merged_df[merged_df["age"] < 20])}')
print(f'年龄>=20的样本数: {len(merged_df[merged_df["age"] >= 20])}')

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
merged_df['age_bin'] = pd.cut(merged_df['age'], bins=bins)
print()
print('完整年龄分布:')
print(merged_df['age_bin'].value_counts().sort_index())
