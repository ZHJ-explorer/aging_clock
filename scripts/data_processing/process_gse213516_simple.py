import os
import pandas as pd
import numpy as np
import gzip
import logging
from scipy.io import mmread
from data_utils import standardize_data
from gene_utils import standardize_gene_name

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = 'data'
PREPROCESSED_DIR = 'preprocessed_data'

# 确保目录存在
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def process_gse213516():
    """处理GSE213516数据集"""
    logger.info("开始处理GSE213516数据集...")
    
    # 文件路径
    mtx_file = os.path.join(DATA_DIR, 'GSM6588511_F30_matrix.mtx.gz')
    features_file = os.path.join(DATA_DIR, 'GSM6588511_F30_features.tsv.gz')
    soft_file = os.path.join(DATA_DIR, 'GSE213516_family.soft.gz')
    
    # 检查文件是否存在
    for file_path in [mtx_file, features_file, soft_file]:
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
    
    try:
        # 1. 读取MTX文件
        logger.info(f"读取MTX文件: {mtx_file}")
        with gzip.open(mtx_file, 'rb') as f:
            matrix = mmread(f)
        logger.info(f"MTX文件读取完成，形状: {matrix.shape}")
        
        # 2. 读取features.tsv.gz文件获取基因名称
        logger.info(f"读取features文件: {features_file}")
        genes = []
        with gzip.open(features_file, 'rt') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        gene_symbol = parts[1].strip()
                        if gene_symbol and gene_symbol != '---':
                            genes.append(gene_symbol)
                        else:
                            genes.append(parts[0].strip())
                    else:
                        genes.append(parts[0].strip())
        logger.info(f"基因文件读取完成，基因数: {len(genes)}")
        
        # 3. 确保基因数量匹配
        if len(genes) != matrix.shape[0]:
            logger.warning(f"基因数量不匹配: 基因文件 {len(genes)} 个，矩阵 {matrix.shape[0]} 个")
            genes = genes[:matrix.shape[0]]
        
        # 4. 读取SOFT文件获取年龄信息
        logger.info(f"从SOFT文件提取年龄信息: {soft_file}")
        age_dict = {}
        
        # 尝试不同编码
        encodings = ['utf-8', 'latin-1', 'gbk']
        for encoding in encodings:
            try:
                with gzip.open(soft_file, 'rt', encoding=encoding) as f:
                    current_gsm = None
                    for line in f:
                        line = line.strip()
                        
                        # 找到样本ID
                        if line.startswith('^SAMPLE'):
                            current_gsm = line.split('=')[1].strip()
                        
                        # 找到年龄信息
                        elif current_gsm and ('age' in line.lower() or 'characteristics' in line.lower()):
                            if 'age' in line.lower():
                                # 提取年龄值
                                parts = line.split('=')
                                if len(parts) > 1:
                                    age_str = parts[1].strip()
                                    # 尝试解析年龄
                                    try:
                                        import re
                                        age_match = re.search(r'\b(\d+(?:\.\d+)?)\b', age_str)
                                        if age_match:
                                            age_num = float(age_match.group(1))
                                            # 只保留合理的年龄值（0-120岁）
                                            if 0 <= age_num <= 120:
                                                age_dict[current_gsm] = age_num
                                                logger.info(f"样本 {current_gsm} 年龄: {age_num}")
                                    except:
                                        pass
                
                if age_dict:
                    logger.info(f"使用编码 {encoding} 提取完成，找到 {len(age_dict)} 个样本的年龄信息")
                    break
                
            except Exception as e:
                logger.warning(f"使用编码 {encoding} 失败: {e}")
                continue
        
        # 5. 创建DataFrame
        logger.info("创建基因表达DataFrame...")
        # 转换矩阵为DataFrame
        df_expr = pd.DataFrame(matrix.toarray().T, columns=genes)
        
        # 6. 添加年龄列
        logger.info("添加年龄列...")
        # 按顺序分配年龄（假设样本顺序与SOFT文件中的顺序一致）
        age_values = list(age_dict.values())
        num_samples = len(df_expr)
        num_ages = len(age_values)
        
        if num_samples >= num_ages:
            # 只保留有年龄信息的样本
            df_expr = df_expr.iloc[:num_ages]
            df_expr['age'] = age_values
            logger.info(f"保留 {num_ages} 个有年龄信息的样本")
        else:
            # 如果样本数少于年龄数，只使用前num_samples个年龄
            df_expr['age'] = age_values[:num_samples]
            logger.info(f"使用前 {num_samples} 个年龄信息")
        
        # 7. 标准化基因名称
        logger.info("标准化基因名称...")
        standardized_columns = []
        for col in df_expr.columns:
            if col == 'age':
                standardized_columns.append('age')
            else:
                standardized_columns.append(standardize_gene_name(col))
        df_expr.columns = standardized_columns
        
        # 8. 数据质量检查
        logger.info("数据质量检查...")
        # 移除重复列
        df_expr = df_expr.loc[:, ~df_expr.columns.duplicated()]
        
        # 9. 确保所有基因列都是数值类型
        logger.info("确保基因表达值为数值类型...")
        gene_columns = [col for col in df_expr.columns if col != 'age']
        for col in gene_columns:
            df_expr[col] = pd.to_numeric(df_expr[col], errors='coerce')
        
        # 10. 移除包含NaN值的列
        df_expr = df_expr.dropna(axis=1, how='all')
        logger.info(f"移除NaN列后，基因数: {len(df_expr.columns) - 1}")
        
        # 11. 标准化数据
        logger.info("标准化数据...")
        df_scaled = standardize_data(df_expr)
        
        # 12. 保存处理后的数据
        output_file = os.path.join(PREPROCESSED_DIR, 'GSE213516_processed.csv')
        df_scaled.to_csv(output_file)
        logger.info(f"处理后的数据已保存到: {output_file}")
        
        return df_scaled
        
    except Exception as e:
        logger.error(f"处理GSE213516数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    logger.info("=== 处理GSE213516数据集 ===")
    
    result = process_gse213516()
    
    if result is not None:
        logger.info("\nGSE213516数据集处理完成！")
        logger.info(f"样本数: {len(result)}")
        logger.info(f"基因数: {len(result.columns) - 1}")
        logger.info(f"年龄范围: {result['age'].min():.1f} - {result['age'].max():.1f}")
    else:
        logger.error("GSE213516数据集处理失败！")

if __name__ == "__main__":
    main()
