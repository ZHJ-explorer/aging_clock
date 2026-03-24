import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.utils.data_utils import split_data

PLOTS_DIR = 'plots'
PREPROCESSED_DIR = 'preprocessed_data'

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

    y_under20 = merged_df[merged_df['age'] < 20]['age']

    print(f"训练集样本数: {len(y_train)}")
    print(f"验证集样本数: {len(y_val)}")
    print(f"测试集样本数: {len(y_test)}")
    print(f"年龄<20样本数: {len(y_under20)}")

    fig, ax = plt.subplots(figsize=(14, 6))

    bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    hist_under20, _ = np.histogram(y_under20, bins=bins)
    hist_train, _ = np.histogram(y_train, bins=bins)
    hist_val, _ = np.histogram(y_val, bins=bins)
    hist_test, _ = np.histogram(y_test, bins=bins)

    bar_width = bin_width * 0.8

    ax.bar(bin_centers, hist_under20, width=bar_width, 
           label=f'Age<20 (n={len(y_under20)})', color='#95a5a6', edgecolor='white', alpha=0.8)
    ax.bar(bin_centers, hist_train, width=bar_width, bottom=hist_under20, 
           label=f'Training (n={len(y_train)})', color='#2ecc71', edgecolor='white', alpha=0.8)
    ax.bar(bin_centers, hist_val, width=bar_width, bottom=hist_under20 + hist_train, 
           label=f'Validation (n={len(y_val)})', color='#3498db', edgecolor='white', alpha=0.8)
    ax.bar(bin_centers, hist_test, width=bar_width, bottom=hist_under20 + hist_train + hist_val, 
           label=f'Test (n={len(y_test)})', color='#e74c3c', edgecolor='white', alpha=0.8)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right')
    
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Age Distribution of Train/Validation/Test Sets', fontsize=14)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f'{int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins)-1)])
    ax.grid(alpha=0.3, axis='y')

    output_file = os.path.join(PLOTS_DIR, 'age_distribution_histogram.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"年龄分布直方图已保存到: {output_file}")

    plt.show()

if __name__ == "__main__":
    plot_age_distribution_histogram()
