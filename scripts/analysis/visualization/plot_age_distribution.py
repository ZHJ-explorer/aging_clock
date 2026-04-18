import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.utils.data_utils import split_data
from scripts.config import PLOTS_DIR, PREPROCESSED_DIR, Config


plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_age_distribution_histogram():
    print("绘制年龄分布直方图...")

    merged_csv = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    if not os.path.exists(merged_csv):
        print(f"错误：找不到数据文件 {merged_csv}")
        return

    merged_df = pd.read_csv(merged_csv, index_col=0)
    print(f"加载数据完成，样本数: {len(merged_df)}")

    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    if 'age' not in numeric_columns:
        numeric_columns = list(numeric_columns) + ['age']
    merged_df = merged_df[numeric_columns]
    merged_df = merged_df.dropna(axis=1, how='any')
    merged_df = merged_df.dropna()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df)

    df_under20 = merged_df[merged_df['age'] < 20]
    under20_count = len(df_under20)

    y_train_20plus = y_train[y_train >= 20]
    y_val_20plus = y_val[y_val >= 20]
    y_test_20plus = y_test[y_test >= 20]

    print(f"年龄<20样本数（未参与训练）: {under20_count}")
    print(f"年龄>=20样本数: {len(y_train_20plus) + len(y_val_20plus) + len(y_test_20plus)}")
    print(f"  训练集样本数: {len(y_train_20plus)}")
    print(f"  验证集样本数: {len(y_val_20plus)}")
    print(f"  测试集样本数: {len(y_test_20plus)}")

    fig, ax = plt.subplots(figsize=(12, 9))

    age_groups = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    x = np.arange(len(age_groups))
    bar_width = 0.7

    bins = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])
    hist_train, _ = np.histogram(y_train_20plus, bins=bins)
    hist_val, _ = np.histogram(y_val_20plus, bins=bins)
    hist_test, _ = np.histogram(y_test_20plus, bins=bins)

    df_0_10 = merged_df[merged_df['age'] < 10]
    df_10_20 = merged_df[(merged_df['age'] >= 10) & (merged_df['age'] < 20)]
    
    train_counts = [0, 0] + list(hist_train)
    val_counts = [0, 0] + list(hist_val)
    test_counts = [0, 0] + list(hist_test)

    not_used_counts = [len(df_0_10), len(df_10_20)] + [0] * 8

    ax.bar(x, not_used_counts, width=bar_width,
           label=f'Not Used (n={under20_count})', color='#8491B4', edgecolor='white', alpha=0.85)

    ax.bar(x, train_counts, width=bar_width, bottom=not_used_counts,
           label=f'Training (n={len(y_train_20plus)})', color='#26AE99', edgecolor='white', alpha=0.7)

    ax.bar(x, val_counts, width=bar_width, bottom=np.array(not_used_counts) + np.array(train_counts),
           label=f'Validation (n={len(y_val_20plus)})', color='#3C5488', edgecolor='white', alpha=0.85)

    ax.bar(x, test_counts, width=bar_width, bottom=np.array(not_used_counts) + np.array(train_counts) + np.array(val_counts),
           label=f'Test (n={len(y_test_20plus)})', color='#DC0000', edgecolor='white', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(age_groups)
    ax.set_xlabel('Age (years)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Count', fontsize=20, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=20)

    ax.grid(alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.xticks(rotation=15)

    output_file = os.path.join(PLOTS_DIR, 'age_distribution_histogram.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"年龄分布直方图已保存到: {output_file}")

    plt.show()


if __name__ == "__main__":
    plot_age_distribution_histogram()