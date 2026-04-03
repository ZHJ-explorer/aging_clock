import os
import time
import pandas as pd
import numpy as np
import warnings
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.exceptions import ConvergenceWarning

# 抑制警告信息
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

from data_utils import standardize_data, select_features, split_data
from model_utils import train_models, evaluate_model, save_models
from gene_utils import standardize_gene_name, map_gene_ids, align_genes_across_datasets

# 数据目录
PREPROCESSED_DIR = 'preprocessed_data'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'

# 确保目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_processed_data():
    """加载已处理的数据"""
    print("加载已处理的数据...")
    
    # 加载微阵列数据集
    microarray_files = [
        'GSE123696_processed.csv',
        'GSE123697_processed.csv',
        'GSE123698_processed.csv'
    ]
    
    microarray_datasets = []
    for file in microarray_files:
        file_path = os.path.join(PREPROCESSED_DIR, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            microarray_datasets.append(df)
            print(f"加载微阵列数据集: {file}, 样本数={len(df)}")
        else:
            print(f"微阵列数据集文件不存在: {file}")
    
    # 加载RNA-seq数据集
    rnaseq_file = 'GSE293163_processed.csv'
    rnaseq_df = None
    file_path = os.path.join(PREPROCESSED_DIR, rnaseq_file)
    if os.path.exists(file_path):
        rnaseq_df = pd.read_csv(file_path, index_col=0)
        print(f"加载RNA-seq数据集: {rnaseq_file}, 样本数={len(rnaseq_df)}")
    else:
        print(f"RNA-seq数据集文件不存在: {rnaseq_file}")
    
    return microarray_datasets, rnaseq_df

def align_microarray_genes(datasets):
    """对齐微阵列数据集的基因"""
    print("对齐微阵列数据集的基因...")
    
    # 标准化所有数据集的基因名称
    standardized_datasets = []
    for i, df in enumerate(datasets):
        # 标准化基因名称
        standardized_columns = []
        for col in df.columns:
            if col == 'age':
                standardized_columns.append('age')
            else:
                standardized_columns.append(standardize_gene_name(col))
        df_standardized = df.copy()
        df_standardized.columns = standardized_columns
        standardized_datasets.append(df_standardized)
    
    # 获取所有数据集的基因集合
    all_genes = set()
    for df in standardized_datasets:
        genes = set(df.columns)
        genes.remove('age')
        all_genes.update(genes)
    
    # 计算每个基因在数据集中的出现次数
    gene_counts = {}
    for gene in all_genes:
        count = 0
        for df in standardized_datasets:
            if gene in df.columns:
                count += 1
        gene_counts[gene] = count
    
    # 选择在所有数据集中都出现的基因
    common_genes = [gene for gene, count in gene_counts.items() if count == len(standardized_datasets)]
    
    print(f"找到 {len(common_genes)} 个共同基因")
    
    # 对齐所有数据集
    aligned_datasets = []
    for i, df in enumerate(standardized_datasets):
        # 只保留共同基因和年龄列
        aligned_df = df[common_genes + ['age']]
        aligned_datasets.append(aligned_df)
        print(f"微阵列数据集 {i+1} 对齐后样本数: {len(aligned_df)}")
    
    return aligned_datasets, common_genes

def train_microarray_models():
    """训练微阵列数据的模型"""
    print("\n=== 训练微阵列数据模型 ===")
    
    # 加载微阵列数据集
    microarray_files = [
        'GSE123696_processed.csv',
        'GSE123697_processed.csv',
        'GSE123698_processed.csv'
    ]
    
    microarray_datasets = []
    for file in microarray_files:
        file_path = os.path.join(PREPROCESSED_DIR, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            # 标准化数据
            df_scaled = standardize_data(df)
            microarray_datasets.append(df_scaled)
    
    if not microarray_datasets:
        print("没有找到微阵列数据集")
        return None
    
    # 对齐基因
    aligned_datasets, common_genes = align_microarray_genes(microarray_datasets)
    
    # 合并所有微阵列数据集
    merged_df = pd.concat(aligned_datasets, ignore_index=True)
    print(f"合并后微阵列样本数: {len(merged_df)}")
    
    # 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(merged_df)
    
    # 特征选择
    selected_features = select_features(X_train, y_train, method='anova', n_features=2000)
    
    # 应用特征选择
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    print(f"特征选择后维度: {X_train_selected.shape[1]}")
    
    # 转换为numpy数组
    X_train_selected = X_train_selected.values
    X_val_selected = X_val_selected.values
    X_test_selected = X_test_selected.values
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values
    
    # 训练模型
    print("训练微阵列数据模型...")
    ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, model_histories = train_models(X_train_selected, y_train, X_val_selected, y_val)

    print("\n生成训练过程可视化...")
    for model_name, history in model_histories.items():
        if history and history.get('train_loss'):
            epochs = range(1, len(history['train_loss']) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if history.get('val_loss'):
                plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss (MSE)', fontsize=12)
            plt.title(f'{model_name}: Training Loss Curve', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_training_loss.png'), dpi=150)
            plt.close()
            print(f"  {model_name} 训练曲线已保存到 {PLOTS_DIR}/")

    # 评估模型
    print("\n微阵列数据模型评估结果:")
    evaluate_model(ridge_model, X_test_selected, y_test, "Ridge (Microarray)")
    evaluate_model(lasso_model, X_test_selected, y_test, "LASSO (Microarray)")
    evaluate_model(en_model, X_test_selected, y_test, "ElasticNet (Microarray)")
    evaluate_model(xgb_model, X_test_selected, y_test, "XGBoost (Microarray)")
    evaluate_model(lgb_model, X_test_selected, y_test, "LightGBM (Microarray)")
    evaluate_model(svr_model, X_test_selected, y_test, "SVR (Microarray)")
    evaluate_model(stacking_model, X_test_selected, y_test, "Stacking (Microarray)")
    
    # 保存模型
    save_models(ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, suffix="_microarray")
    
    return stacking_model

def align_rnaseq_genes(datasets):
    """对齐RNA-seq数据集的基因（处理基因ID格式不同的问题）"""
    print("对齐RNA-seq数据集的基因...")
    
    # 使用新的基因对齐函数
    aligned_datasets, common_genes = align_genes_across_datasets(datasets)
    
    # 打印对齐结果
    print(f"找到 {len(common_genes)} 个共同基因")
    for i, (name, df) in enumerate(datasets):
        aligned_df = aligned_datasets[i]
        print(f"{name} 对齐后样本数: {len(aligned_df)}, 基因数: {len(aligned_df.columns)-1}")
    
    return aligned_datasets, common_genes

def train_rnaseq_models():
    """训练RNA-seq数据的模型"""
    print("\n=== 训练RNA-seq数据模型 ===")
    
    # 加载RNA-seq数据集
    rnaseq_files = [
        'GSE293163_processed.csv',
        'GSE231409_rnaseq_processed.csv'
    ]
    
    rnaseq_datasets = []
    for rnaseq_file in rnaseq_files:
        file_path = os.path.join(PREPROCESSED_DIR, rnaseq_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            # 移除age_years列（如果存在），只保留age列
            if 'age_years' in df.columns:
                df = df.drop(columns=['age_years'])
            
            # 标准化基因名称
            print(f"标准化 {rnaseq_file} 的基因名称...")
            standardized_columns = []
            for col in df.columns:
                if col == 'age':
                    standardized_columns.append('age')
                else:
                    standardized_columns.append(standardize_gene_name(col))
            df.columns = standardized_columns
            
            rnaseq_datasets.append((rnaseq_file, df))
            print(f"加载RNA-seq数据集: {rnaseq_file}, 样本数={len(df)}, 基因数={len(df.columns)-1}")
        else:
            print(f"RNA-seq数据集文件不存在: {rnaseq_file}")
    
    if not rnaseq_datasets:
        print("没有找到RNA-seq数据集")
        return None
    
    # 如果有多个RNA-seq数据集，需要对齐基因
    if len(rnaseq_datasets) > 1:
        aligned_datasets, common_genes = align_rnaseq_genes(rnaseq_datasets)
        
        if len(common_genes) == 0:
            print("警告：没有找到共同基因，将分别训练每个数据集的模型")
            # 改为分别训练
            for name, df in rnaseq_datasets:
                print(f"\n单独训练 {name}...")
        else:
            # 合并所有RNA-seq数据集
            merged_df = pd.concat(aligned_datasets, ignore_index=True)
            print(f"合并后RNA-seq总样本数: {len(merged_df)}, 基因数: {len(merged_df.columns)-1}")
            
            # 标准化数据
            df_scaled = standardize_data(merged_df)
            
            # 处理NaN值 - 删除包含NaN值的特征
            print("处理NaN值...")
            # 计算每列的NaN值数量
            nan_counts = df_scaled.isna().sum()
            # 删除包含NaN值的列
            df_scaled = df_scaled.dropna(axis=1)
            print(f"删除了 {len(nan_counts) - len(df_scaled.columns)} 个包含NaN值的特征")
            print(f"处理后特征数: {len(df_scaled.columns) - 1}")  # 减去age列
            
            # 分割数据
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_scaled)
            
            # 特征选择
            n_features = min(5000, len(df_scaled.columns) - 1)  # 减去age列
            selected_features = select_features(X_train, y_train, method='anova', n_features=n_features)
            
            # 应用特征选择
            X_train_selected = X_train[selected_features]
            X_val_selected = X_val[selected_features]
            X_test_selected = X_test[selected_features]
            
            print(f"特征选择后维度: {X_train_selected.shape[1]}")
            
            # 转换为numpy数组
            X_train_selected = X_train_selected.values
            X_val_selected = X_val_selected.values
            X_test_selected = X_test_selected.values
            y_train = y_train.values
            y_val = y_val.values
            y_test = y_test.values
            
            # 训练模型
            print("训练RNA-seq数据模型...")
            ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, model_histories = train_models(X_train_selected, y_train, X_val_selected, y_val)

            print("\n生成训练过程可视化...")
            for model_name, history in model_histories.items():
                if history and history.get('train_loss'):
                    epochs = range(1, len(history['train_loss']) + 1)
                    plt.figure(figsize=(10, 6))
                    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
                    if history.get('val_loss'):
                        plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
                    plt.xlabel('Iteration', fontsize=12)
                    plt.ylabel('Loss (MSE)', fontsize=12)
                    plt.title(f'{model_name}: Training Loss Curve', fontsize=14)
                    plt.legend(fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_training_loss.png'), dpi=150)
                    plt.close()
                    print(f"  {model_name} 训练曲线已保存到 {PLOTS_DIR}/")

            # 评估模型
            print("\nRNA-seq数据模型评估结果:")
            evaluate_model(ridge_model, X_test_selected, y_test, "Ridge (RNA-seq)")
            evaluate_model(lasso_model, X_test_selected, y_test, "LASSO (RNA-seq)")
            evaluate_model(en_model, X_test_selected, y_test, "ElasticNet (RNA-seq)")
            evaluate_model(xgb_model, X_test_selected, y_test, "XGBoost (RNA-seq)")
            evaluate_model(lgb_model, X_test_selected, y_test, "LightGBM (RNA-seq)")
            evaluate_model(svr_model, X_test_selected, y_test, "SVR (RNA-seq)")
            evaluate_model(stacking_model, X_test_selected, y_test, "Stacking (RNA-seq)")
            
            # 保存模型
            save_models(ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, suffix="_rnaseq")
            
            return stacking_model
    else:
        # 只有一个数据集
        name, merged_df = rnaseq_datasets[0]
        print(f"使用单个RNA-seq数据集: {name}, 样本数: {len(merged_df)}")
        
        # 标准化数据
        df_scaled = standardize_data(merged_df)
    
    # 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_scaled)
    
    # 特征选择 - RNA-seq数据基因数较多，需要更严格的特征选择
    selected_features = select_features(X_train, y_train, method='anova', n_features=5000)
    
    # 应用特征选择
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    print(f"特征选择后维度: {X_train_selected.shape[1]}")
    
    # 转换为numpy数组
    X_train_selected = X_train_selected.values
    X_val_selected = X_val_selected.values
    X_test_selected = X_test_selected.values
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values
    
    # 训练模型
    print("训练RNA-seq数据模型...")
    ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, model_histories = train_models(X_train_selected, y_train, X_val_selected, y_val)

    print("\n生成训练过程可视化...")
    for model_name, history in model_histories.items():
        if history and history.get('train_loss'):
            epochs = range(1, len(history['train_loss']) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if history.get('val_loss'):
                plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss (MSE)', fontsize=12)
            plt.title(f'{model_name}: Training Loss Curve', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_training_loss.png'), dpi=150)
            plt.close()
            print(f"  {model_name} 训练曲线已保存到 {PLOTS_DIR}/")

    # 评估模型
    print("\nRNA-seq数据模型评估结果:")
    evaluate_model(ridge_model, X_test_selected, y_test, "Ridge (RNA-seq)")
    evaluate_model(lasso_model, X_test_selected, y_test, "LASSO (RNA-seq)")
    evaluate_model(en_model, X_test_selected, y_test, "ElasticNet (RNA-seq)")
    evaluate_model(xgb_model, X_test_selected, y_test, "XGBoost (RNA-seq)")
    evaluate_model(lgb_model, X_test_selected, y_test, "LightGBM (RNA-seq)")
    evaluate_model(svr_model, X_test_selected, y_test, "SVR (RNA-seq)")
    evaluate_model(stacking_model, X_test_selected, y_test, "Stacking (RNA-seq)")
    
    # 保存模型
    save_models(ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, suffix="_rnaseq")
    
    return stacking_model

def train_combined_model():
    """训练合并数据的模型（微阵列 + RNA-seq）"""
    print("\n=== 训练合并数据模型 ===")
    
    # 使用预处理和合并后的数据
    scaled_file = os.path.join(PREPROCESSED_DIR, 'merged_scaled.csv')
    merged_file = os.path.join(PREPROCESSED_DIR, 'merged_processed.csv')
    
    if os.path.exists(scaled_file):
        print(f"使用已有的标准化数据文件: {scaled_file}")
        df_scaled = pd.read_csv(scaled_file, index_col=0)
        print(f"标准化数据样本数: {len(df_scaled)}, 基因数: {len(df_scaled.columns)-1}")
    elif os.path.exists(merged_file):
        print(f"使用已有的合并数据文件: {merged_file}")
        merged_df = pd.read_csv(merged_file, index_col=0)
        print(f"合并数据样本数: {len(merged_df)}, 基因数: {len(merged_df.columns)-1}")
        
        # 标准化数据
        print("标准化数据...")
        df_scaled = standardize_data(merged_df)
    else:
        print("错误：没有找到合并数据文件，请先运行 preprocess_and_merge.py")
        return None
    
    # 处理NaN值
    print("处理NaN值...")
    nan_counts = df_scaled.isna().sum()
    df_scaled = df_scaled.dropna(axis=1)
    print(f"删除了 {len(nan_counts) - len(df_scaled.columns)} 个包含NaN值的特征")
    print(f"处理后特征数: {len(df_scaled.columns) - 1}")
    
    # 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_scaled)
    
    # 特征选择
    n_features = min(5000, len(df_scaled.columns) - 1)
    selected_features = select_features(X_train, y_train, method='anova', n_features=n_features)
    
    # 应用特征选择
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    print(f"特征选择后维度: {X_train_selected.shape[1]}")
    
    # 转换为numpy数组
    X_train_selected = X_train_selected.values
    X_val_selected = X_val_selected.values
    X_test_selected = X_test_selected.values
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values
    
    # 训练模型
    print("训练合并数据模型...")
    ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, model_histories = train_models(X_train_selected, y_train, X_val_selected, y_val)

    print("\n生成训练过程可视化...")
    for model_name, history in model_histories.items():
        if history and history.get('train_loss'):
            epochs = range(1, len(history['train_loss']) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if history.get('val_loss'):
                plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss (MSE)', fontsize=12)
            plt.title(f'{model_name}: Training Loss Curve', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_training_loss.png'), dpi=150)
            plt.close()
            print(f"  {model_name} 训练曲线已保存到 {PLOTS_DIR}/")
    
    # 评估模型
    print("\n合并数据模型评估结果:")
    evaluate_model(ridge_model, X_test_selected, y_test, "Ridge (Combined)")
    evaluate_model(lasso_model, X_test_selected, y_test, "LASSO (Combined)")
    evaluate_model(en_model, X_test_selected, y_test, "ElasticNet (Combined)")
    evaluate_model(xgb_model, X_test_selected, y_test, "XGBoost (Combined)")
    evaluate_model(lgb_model, X_test_selected, y_test, "LightGBM (Combined)")
    evaluate_model(svr_model, X_test_selected, y_test, "SVR (Combined)")
    evaluate_model(stacking_model, X_test_selected, y_test, "Stacking (Combined)")
    
    # 保存模型
    save_models(ridge_model, lasso_model, en_model, xgb_model, lgb_model, svr_model, stacking_model, suffix="_combined")
    
    return stacking_model

def train_with_timeout(result_queue):
    """在超时限制内训练模型"""
    try:
        # 清空test_result.txt文件
        with open('test_result.txt', 'w', encoding='utf-8') as f:
            f.write("衰老时钟模型测试结果\n")
            f.write("=" * 50 + "\n")
        
        # 只训练合并数据模型（微阵列 + RNA-seq）
        combined_model = train_combined_model()
        
        # 运行绘图脚本
        print("运行绘图脚本...")
        import subprocess
        subprocess.run(['python', 'plot_results.py'], check=True)
        
        result_queue.put(True)
    except Exception as e:
        print(f"训练过程出错: {e}")
        result_queue.put(False)

def main():
    """主函数"""
    print("开始重新训练衰老时钟模型...")
    start_time = time.time()
    
    # 设置5分钟超时
    timeout_seconds = 5 * 60  # 5分钟
    
    # 创建队列用于进程间通信
    result_queue = Queue()
    
    # 创建子进程
    p = Process(target=train_with_timeout, args=(result_queue,))
    
    # 启动子进程
    p.start()
    
    # 等待子进程完成或超时
    p.join(timeout_seconds)
    
    # 检查子进程是否仍在运行
    if p.is_alive():
        print(f"\n警告：训练过程超过 {timeout_seconds} 秒，强制终止")
        p.terminate()
        p.join()
        print("训练过程已终止")
    else:
        # 子进程正常完成
        try:
            if not result_queue.empty():
                result = result_queue.get()
                if result:
                    end_time = time.time()
                    print(f"\n重新训练流程完成，耗时: {end_time - start_time:.2f} 秒")
        except Exception as e:
            print(f"获取结果时出错: {e}")

if __name__ == "__main__":
    main()
