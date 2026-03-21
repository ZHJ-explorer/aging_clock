import os
import pandas as pd
import numpy as np
import gzip
import logging
from scipy.io import mmread
from data_utils import standardize_data
from gene_utils import standardize_gene_name, map_gene_ids

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = 'data'
PREPROCESSED_DIR = 'preprocessed_data'

# 确保目录存在
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def process_mtx_file(mtx_file, genes_file=None, barcodes_file=None):
    """处理MTX格式的RNA-seq数据"""
    logger.info(f"处理MTX文件: {mtx_file}")
    
    try:
        # 读取MTX文件
        with gzip.open(mtx_file, 'rb') as f:
            matrix = mmread(f)
        logger.info(f"MTX文件读取完成，形状: {matrix.shape}")
        
        # 读取基因名称（如果提供）
        if genes_file and os.path.exists(genes_file):
            logger.info(f"读取基因文件: {genes_file}")
            genes = []
            seen_genes = set()
            with gzip.open(genes_file, 'rt') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # features.tsv文件格式：Ensembl ID\t基因符号\t类型
                        # 取第二列作为基因符号
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            gene_symbol = parts[1].strip()
                            if gene_symbol and gene_symbol != '---':
                                # 处理重复基因名称
                                if gene_symbol in seen_genes:
                                    # 添加后缀避免重复
                                    counter = 1
                                    while f"{gene_symbol}_{counter}" in seen_genes:
                                        counter += 1
                                    gene_symbol = f"{gene_symbol}_{counter}"
                                seen_genes.add(gene_symbol)
                                genes.append(gene_symbol)
                            else:
                                # 如果基因符号为空，使用Ensembl ID
                                ensembl_id = parts[0].strip()
                                if ensembl_id in seen_genes:
                                    counter = 1
                                    while f"{ensembl_id}_{counter}" in seen_genes:
                                        counter += 1
                                    ensembl_id = f"{ensembl_id}_{counter}"
                                seen_genes.add(ensembl_id)
                                genes.append(ensembl_id)
                        else:
                            # 如果格式不正确，使用第一列
                            gene_id = parts[0].strip()
                            if gene_id in seen_genes:
                                counter = 1
                                while f"{gene_id}_{counter}" in seen_genes:
                                    counter += 1
                                gene_id = f"{gene_id}_{counter}"
                            seen_genes.add(gene_id)
                            genes.append(gene_id)
            logger.info(f"基因文件读取完成，基因数: {len(genes)}")
        else:
            # 如果没有基因文件，使用默认基因名称
            genes = [f'GENE_{i+1}' for i in range(matrix.shape[0])]
            logger.warning("未找到基因文件，使用默认基因名称")
        
        # 读取样本名称（如果提供）
        if barcodes_file and os.path.exists(barcodes_file):
            logger.info(f"读取样本文件: {barcodes_file}")
            barcodes = []
            with gzip.open(barcodes_file, 'rt') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        barcodes.append(line)
            logger.info(f"样本文件读取完成，样本数: {len(barcodes)}")
        else:
            # 如果没有样本文件，使用默认样本名称
            barcodes = [f'SAMPLE_{i+1}' for i in range(matrix.shape[1])]
            logger.warning("未找到样本文件，使用默认样本名称")
        
        # 确保基因和样本数量匹配
        if len(genes) != matrix.shape[0]:
            logger.warning(f"基因数量不匹配: 基因文件 {len(genes)} 个，矩阵 {matrix.shape[0]} 个")
            genes = genes[:matrix.shape[0]]
        
        if len(barcodes) != matrix.shape[1]:
            logger.warning(f"样本数量不匹配: 样本文件 {len(barcodes)} 个，矩阵 {matrix.shape[1]} 个")
            barcodes = barcodes[:matrix.shape[1]]
        
        # 转换为DataFrame
        df = pd.DataFrame(matrix.toarray().T, index=barcodes, columns=genes)
        logger.info(f"数据转换完成，形状: {df.shape}")
        
        return df
    except Exception as e:
        logger.error(f"处理MTX文件失败: {e}")
        return None

def extract_age_from_soft(soft_file):
    """从SOFT文件中提取样本年龄信息"""
    logger.info(f"从SOFT文件提取年龄信息: {soft_file}")
    
    try:
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
                                        # 提取数字部分（包括小数点）
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
                    return age_dict
                
            except Exception as e:
                logger.warning(f"使用编码 {encoding} 失败: {e}")
                continue
        
        logger.info(f"提取完成，找到 {len(age_dict)} 个样本的年龄信息")
        return age_dict
    except Exception as e:
        logger.error(f"从SOFT文件提取年龄信息失败: {e}")
        return {}

def process_gse213516():
    """处理GSE213516数据集"""
    logger.info("开始处理GSE213516数据集...")
    
    # 文件路径
    mtx_file = os.path.join(DATA_DIR, 'GSM6588511_F30_matrix.mtx.gz')
    soft_file = os.path.join(DATA_DIR, 'GSE213516_family.soft.gz')
    
    # 检查文件是否存在
    if not os.path.exists(mtx_file):
        logger.error(f"MTX文件不存在: {mtx_file}")
        return None
    
    if not os.path.exists(soft_file):
        logger.error(f"SOFT文件不存在: {soft_file}")
        return None
    
    # 处理MTX文件
    features_file = os.path.join(DATA_DIR, 'GSM6588511_F30_features.tsv.gz')
    barcodes_file = os.path.join(DATA_DIR, 'GSM6588511_F30_barcodes.tsv.gz')
    
    df_expr = process_mtx_file(mtx_file, features_file, barcodes_file)
    if df_expr is None:
        logger.error("处理MTX文件失败")
        return None
    
    # 提取年龄信息
    age_dict = extract_age_from_soft(soft_file)
    
    # 添加年龄列
    ages = []
    if age_dict:
        # 如果有年龄信息，按顺序分配
        age_values = list(age_dict.values())
        num_samples = len(df_expr)
        num_ages = len(age_values)
        
        logger.info(f"MTX文件样本数: {num_samples}, SOFT文件年龄信息数: {num_ages}")
        
        if num_samples == num_ages:
            # 样本数匹配，按顺序分配
            ages = age_values
            logger.info("样本数与年龄信息数匹配，按顺序分配年龄")
        else:
            # 样本数不匹配，尝试其他方式
            logger.warning(f"样本数与年龄信息数不匹配: {num_samples} vs {num_ages}")
            # 按顺序分配，多余的样本设为NaN
            for i in range(num_samples):
                if i < num_ages:
                    ages.append(age_values[i])
                else:
                    ages.append(None)
    else:
        # 没有年龄信息
        ages = [None] * len(df_expr)
    
    df_expr['age'] = ages
    
    # 过滤掉年龄为NaN的样本
    df_expr = df_expr.dropna(subset=['age'])
    logger.info(f"过滤后样本数: {len(df_expr)}")
    
    # 标准化基因名称
    logger.info("标准化基因名称...")
    standardized_columns = []
    for col in df_expr.columns:
        if col == 'age':
            standardized_columns.append('age')
        else:
            standardized_columns.append(standardize_gene_name(col))
    df_expr.columns = standardized_columns
    
    # 数据质量检查
    logger.info("数据质量检查...")
    # 检查是否有基因表达列
    gene_columns = [col for col in df_expr.columns if col != 'age']
    if len(gene_columns) == 0:
        logger.error("没有基因表达列")
        return None
    
    # 确保所有基因列都是数值类型
    # 创建一个新的列表来存储有效的基因列
    valid_gene_columns = []
    for col in gene_columns:
        try:
            df_expr[col] = pd.to_numeric(df_expr[col], errors='coerce')
            valid_gene_columns.append(col)
        except Exception as e:
            logger.warning(f"转换列 {col} 失败: {e}")
            # 如果转换失败，删除该列
            if col in df_expr.columns:
                df_expr = df_expr.drop(columns=[col])
    # 更新基因列列表
    gene_columns = valid_gene_columns
    
    # 移除包含NaN值的列
    df_expr = df_expr.dropna(axis=1, how='all')
    logger.info(f"移除NaN列后，基因数: {len(df_expr.columns) - 1}")
    
    # 标准化数据
    logger.info("标准化数据...")
    df_scaled = standardize_data(df_expr)
    
    # 保存处理后的数据
    output_file = os.path.join(PREPROCESSED_DIR, 'GSE213516_processed.csv')
    df_scaled.to_csv(output_file)
    logger.info(f"处理后的数据已保存到: {output_file}")
    
    return df_scaled

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
