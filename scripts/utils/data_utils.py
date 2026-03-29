import os
import numpy as np
import pandas as pd
import GEOparse
import logging
from sklearn.model_selection import train_test_split

logger_geoparse = logging.getLogger('GEOparse')
logger_geoparse.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

DATA_DIR = 'data'


def download_dataset(dataset_id):
    """检查数据集是否存在"""
    logger.info(f"检查数据集 {dataset_id}...")
    data_files = os.listdir(DATA_DIR)
    dataset_files = [f for f in data_files if dataset_id in f]
    if dataset_files:
        logger.info(f"数据集 {dataset_id} 已存在，使用本地文件")
        return True
    else:
        logger.warning(f"数据集 {dataset_id} 不存在，请确保数据文件已正确放置在data目录中")
        return False


def load_and_preprocess_data(dataset_id):
    """加载和预处理基因表达数据"""
    cache_file = os.path.join('preprocessed_data', f'{dataset_id}_processed.parquet')
    if os.path.exists(cache_file):
        logger.info(f"从缓存文件加载数据集 {dataset_id}...")
        return pd.read_parquet(cache_file)

    logger.info(f"处理数据集 {dataset_id}...")

    try:
        data_file = os.path.join(DATA_DIR, f"{dataset_id}_family.soft.gz")

        if not os.path.exists(data_file):
            logger.warning(f"本地文件 {data_file} 不存在")
            return None

        logger.info(f"从本地文件 {data_file} 加载数据...")
        gse = GEOparse.get_GEO(filepath=data_file)

        logger.info("获取基因表达数据...")

        expr_data = []
        sample_ids = []
        ages = []

        for gsm_name, gsm in gse.gsms.items():
            sample_ids.append(gsm_name)

            age = None
            if 'age' in gsm.metadata:
                age = gsm.metadata['age'][0]
            elif 'characteristics_ch1' in gsm.metadata:
                for char in gsm.metadata['characteristics_ch1']:
                    if 'age' in char.lower():
                        age = char.split(':')[-1].strip()
                        break

            if age:
                try:
                    age_num = float(''.join(filter(str.isdigit, age)))
                    ages.append(age_num)
                except:
                    ages.append(np.nan)
            else:
                ages.append(np.nan)

            gene_id_col = None
            for col in ['ID_REF', 'ID', 'PROBE_ID', 'GeneID']:
                if col in gsm.table.columns:
                    gene_id_col = col
                    break

            if gene_id_col:
                expr_data.append(gsm.table.set_index(gene_id_col)['VALUE'].to_dict())
            else:
                logger.warning(f"在样本 {gsm_name} 中未找到基因ID列")
                expr_data.append({})

        df_expr = pd.DataFrame(expr_data, index=sample_ids)

        logger.info("标准化基因名称...")
        try:
            if hasattr(gse, 'gpls') and gse.gpls:
                gpl = list(gse.gpls.values())[0]
                probe_to_gene = {}
                if 'Gene Symbol' in gpl.table.columns:
                    for idx, row in gpl.table.iterrows():
                        probe_id = row['ID']
                        gene_symbol = row['Gene Symbol']
                        if gene_symbol and gene_symbol != '---':
                            probe_to_gene[probe_id] = gene_symbol

                if probe_to_gene:
                    new_columns = []
                    for col in df_expr.columns:
                        if col in probe_to_gene:
                            new_columns.append(probe_to_gene[col])
                        else:
                            new_columns.append(col)
                    df_expr.columns = new_columns
                    logger.info("基因名称标准化完成，使用GEO注释信息")
                else:
                    logger.info("未找到基因注释信息，保持原有的基因ID")
            else:
                logger.info("未找到基因注释信息，保持原有的基因ID")
        except Exception as e:
            logger.error(f"基因名称标准化失败: {e}")
            logger.info("保持原有的基因ID")

        logger.info("确保数据类型正确...")
        try:
            for col in df_expr.columns:
                if col != 'age':
                    df_expr[col] = pd.to_numeric(df_expr[col], errors='coerce')

            df_expr = df_expr.dropna(axis=1, how='all')
            logger.info(f"数据类型转换完成，保留了 {df_expr.shape[1]} 列")
        except Exception as e:
            logger.error(f"数据类型转换失败: {e}")

        df_expr['age'] = ages
        df_expr = df_expr.dropna(subset=['age'])

        logger.info(f"原始数据形状: {df_expr.shape}")

        if df_expr.empty or df_expr.shape[1] <= 1:
            logger.warning("数据集中没有有效的基因表达信息")
            return None

        os.makedirs('preprocessed_data', exist_ok=True)
        df_expr.to_parquet(cache_file)
        logger.info(f"数据已缓存到 {cache_file}")

        return df_expr

    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        return None


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """分割数据集为训练集、验证集和测试集

    Args:
        df: 包含特征的DataFrame，必须包含'age'列
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if 'age' not in df.columns:
        raise ValueError("DataFrame must contain 'age' column")

    X = df.drop('age', axis=1)
    y = df['age']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state
    )

    val_adjusted_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_adjusted_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_gene_list(gene_list_path):
    """加载基因列表"""
    with open(gene_list_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def filter_common_genes(dataframes, top_n=None):
    """找出多个数据集中共同的基因"""
    common_genes = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_genes = common_genes.intersection(set(df.columns))

    common_genes = sorted(list(common_genes))

    if top_n:
        return common_genes[:top_n]
    return common_genes
