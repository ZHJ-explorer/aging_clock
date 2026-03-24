import os
import numpy as np
import pandas as pd
import GEOparse
import logging

# 禁用GEOparse的DEBUG输出
logger_geoparse = logging.getLogger('GEOparse')
logger_geoparse.setLevel(logging.INFO)

# 创建本模块的日志记录器
logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = 'data'


def download_dataset(dataset_id):
    """检查数据集是否存在"""
    logger.info(f"检查数据集 {dataset_id}...")
    
    # 检查是否已存在数据文件
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
    # 检查是否存在预处理后的缓存文件
    cache_file = os.path.join('preprocessed_data', f'{dataset_id}_processed.parquet')
    if os.path.exists(cache_file):
        logger.info(f"从缓存文件加载数据集 {dataset_id}...")
        return pd.read_parquet(cache_file)
    
    logger.info(f"处理数据集 {dataset_id}...")
    
    # 尝试使用GEOparse加载本地数据集文件
    try:
        # 构建本地文件路径
        data_file = os.path.join(DATA_DIR, f"{dataset_id}_family.soft.gz")
        
        if not os.path.exists(data_file):
            logger.warning(f"本地文件 {data_file} 不存在")
            return None
        
        # 加载本地数据集文件
        logger.info(f"从本地文件 {data_file} 加载数据...")
        gse = GEOparse.get_GEO(filepath=data_file)
        
        # 获取基因表达数据
        logger.info("获取基因表达数据...")
        
        # 初始化一个空的DataFrame来存储基因表达数据
        expr_data = []
        sample_ids = []
        ages = []
        
        # 遍历所有样本
        for gsm_name, gsm in gse.gsms.items():
            sample_ids.append(gsm_name)
            
            # 尝试从样本信息中提取年龄
            age = None
            if 'age' in gsm.metadata:
                age = gsm.metadata['age'][0]
            elif 'characteristics_ch1' in gsm.metadata:
                for char in gsm.metadata['characteristics_ch1']:
                    if 'age' in char.lower():
                        age = char.split(':')[-1].strip()
                        break
            
            # 尝试解析年龄值
            if age:
                try:
                    # 提取数字部分
                    age_num = float(''.join(filter(str.isdigit, age)))
                    ages.append(age_num)
                except:
                    ages.append(np.nan)
            else:
                ages.append(np.nan)
            
            # 获取基因表达值
            # 尝试不同的列名来获取基因ID
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
        
        # 创建基因表达DataFrame
        df_expr = pd.DataFrame(expr_data, index=sample_ids)
        
        # 基因名称标准化
        logger.info("标准化基因名称...")
        try:
            # 尝试从GEO对象中获取基因注释信息
            if hasattr(gse, 'gpls') and gse.gpls:
                gpl = list(gse.gpls.values())[0]
                # 创建探针ID到基因名的映射
                probe_to_gene = {}
                if 'Gene Symbol' in gpl.table.columns:
                    for idx, row in gpl.table.iterrows():
                        probe_id = row['ID']
                        gene_symbol = row['Gene Symbol']
                        if gene_symbol and gene_symbol != '---':
                            probe_to_gene[probe_id] = gene_symbol
                
                # 映射基因名称
                if probe_to_gene:
                    # 创建新的列名映射
                    new_columns = []
                    for col in df_expr.columns:
                        if col in probe_to_gene:
                            new_columns.append(probe_to_gene[col])
                        else:
                            new_columns.append(col)
                    
                    # 更新列名
                    df_expr.columns = new_columns
                    logger.info("基因名称标准化完成，使用GEO注释信息")
                else:
                    logger.info("未找到基因注释信息，保持原有的基因名称")
            else:
                logger.info("未找到基因注释信息，保持原有的基因名称")
        except Exception as e:
            logger.error(f"基因名称标准化失败: {e}")
            logger.info("保持原有的基因名称")
        
        # 确保所有列都是数值类型
        logger.info("确保数据类型正确...")
        try:
            # 转换所有列（除了年龄列）为数值类型
            for col in df_expr.columns:
                if col != 'age':
                    df_expr[col] = pd.to_numeric(df_expr[col], errors='coerce')
            
            # 移除包含NaN值的列
            df_expr = df_expr.dropna(axis=1, how='all')
            logger.info(f"数据类型转换完成，保留了{df_expr.shape[1]}列")
        except Exception as e:
            logger.error(f"数据类型转换失败: {e}")
        
        # 添加年龄列
        df_expr['age'] = ages
        
        # 过滤掉年龄为NaN的样本
        df_expr = df_expr.dropna(subset=['age'])
        
        logger.info(f"原始数据形状: {df_expr.shape}")
        
        # 检查数据是否为空或没有基因表达列
        if df_expr.empty or df_expr.shape[1] <= 1:  # 只有age列
            logger.warning("数据集中没有有效的基因表达信息")
            return None
        
        logger.info(f"处理后数据形状: {df_expr.shape}")
        logger.info(f"数据集 {dataset_id} 处理完成，样本数: {len(df_expr)}")
        
        # 保存为缓存文件
        os.makedirs('preprocessed_data', exist_ok=True)
        df_expr.to_parquet(cache_file)
        logger.info(f"数据集 {dataset_id} 已缓存到 {cache_file}")
        
        return df_expr
        
    except Exception as e:
        logger.error(f"加载数据集 {dataset_id} 失败: {e}")
        return None


def standardize_data(df):
    """标准化基因表达数据"""
    logger.info("标准化数据...")
    
    # 分离特征和目标变量
    X = df.drop('age', axis=1)
    y = df['age']
    
    # 只保留数值型列
    X_numeric = X.select_dtypes(include=[np.number])
    logger.info(f"保留了 {X_numeric.shape[1]} 个数值型特征")
    
    # 如果没有数值型特征，返回原始数据集
    if X_numeric.shape[1] == 0:
        logger.warning("没有找到数值型特征，返回原始数据集")
        return df
    
    # 处理NaN值 - 删除包含NaN值的列
    X_clean = X_numeric.dropna(axis=1, how='any')
    logger.info(f"删除包含NaN值的列后，保留了 {X_clean.shape[1]} 个特征")
    
    # 如果没有特征，返回原始数据集
    if X_clean.shape[1] == 0:
        logger.warning("所有特征都包含NaN值，返回原始数据集")
        return df
    
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # 创建标准化后的DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X.index)
    df_scaled['age'] = y
    
    return df_scaled


def align_genes(datasets):
    """对齐不同数据集的基因"""
    logger.info("对齐基因...")
    
    # 获取所有数据集的基因集合
    all_genes = set()
    for df in datasets:
        genes = set(df.columns)
        genes.remove('age')
        all_genes.update(genes)
    
    # 计算每个基因在数据集中的出现次数
    gene_counts = {}
    for gene in all_genes:
        count = 0
        for df in datasets:
            if gene in df.columns:
                count += 1
        gene_counts[gene] = count
    
    # 选择在所有数据集中都出现的基因
    common_genes = [gene for gene, count in gene_counts.items() if count == len(datasets)]
    
    logger.info(f"找到 {len(common_genes)} 个共同基因")
    
    # 对齐所有数据集
    aligned_datasets = []
    for i, df in enumerate(datasets):
        # 只保留共同基因和年龄列
        aligned_df = df[common_genes + ['age']]
        aligned_datasets.append(aligned_df)
        logger.info(f"数据集 {i+1} 对齐后样本数: {len(aligned_df)}")
    
    return aligned_datasets, common_genes


def select_features(X, y, method='combined', n_features=1000):
    """特征选择/降维"""
    logger.info(f"使用{method}方法进行特征选择...")
    
    # 处理NaN值
    logger.info("处理数据中的NaN值...")
    # 删除包含NaN值的列
    X_clean = X.dropna(axis=1, how='any')
    logger.info(f"删除NaN值后，特征数量从 {X.shape[1]} 减少到 {X_clean.shape[1]}")
    
    if X_clean.shape[1] == 0:
        logger.warning("所有特征都包含NaN值，无法进行特征选择")
        return X.columns
    
    # 方差过滤
    logger.info("进行方差过滤...")
    from sklearn.feature_selection import VarianceThreshold
    # 保留方差大于0.1的特征
    var_threshold = VarianceThreshold(threshold=0.1)
    X_var_filtered = var_threshold.fit_transform(X_clean)
    var_selected_features = X_clean.columns[var_threshold.get_support()]
    logger.info(f"方差过滤后，特征数量从 {X_clean.shape[1]} 减少到 {len(var_selected_features)}")
    
    # 如果方差过滤后没有特征，返回原始特征
    if len(var_selected_features) == 0:
        logger.warning("方差过滤后没有特征，返回原始特征")
        return X_clean.columns
    
    X_clean = X_clean[var_selected_features]
    
    if method == 'lasso':
        # 使用LASSO回归进行特征选择
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_clean, y)
        
        # 选择非零系数的特征
        selected_features = X_clean.columns[lasso.coef_ != 0]
        logger.info(f"LASSO选择了{len(selected_features)}个特征")
        
        return selected_features
    
    elif method == 'anova':
        # 使用ANOVA进行特征选择
        from sklearn.feature_selection import SelectKBest, f_regression
        # 确保n_features不超过可用特征数
        n_features = min(n_features, X_clean.shape[1])
        selector = SelectKBest(f_regression, k=n_features)
        selector.fit(X_clean, y)
        
        selected_features = X_clean.columns[selector.get_support()]
        logger.info(f"ANOVA选择了{len(selected_features)}个特征")
        
        return selected_features
    
    elif method == 'combined':
        # 结合LASSO和ANOVA进行特征选择
        logger.info("结合LASSO和ANOVA进行特征选择...")
        
        # LASSO特征选择
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_clean, y)
        lasso_features = X_clean.columns[lasso.coef_ != 0]
        logger.info(f"LASSO选择了{len(lasso_features)}个特征")
        
        # ANOVA特征选择
        from sklearn.feature_selection import SelectKBest, f_regression
        n_features = min(n_features, X_clean.shape[1])
        selector = SelectKBest(f_regression, k=n_features)
        selector.fit(X_clean, y)
        anova_features = X_clean.columns[selector.get_support()]
        logger.info(f"ANOVA选择了{len(anova_features)}个特征")
        
        # 合并特征
        selected_features = list(set(lasso_features) | set(anova_features))
        # 如果特征数量超过n_features，按ANOVA得分排序后选择前n_features个
        if len(selected_features) > n_features:
            # 获取ANOVA得分
            scores = selector.scores_
            feature_scores = dict(zip(X_clean.columns, scores))
            # 按得分排序
            selected_features = sorted(selected_features, key=lambda x: feature_scores.get(x, 0), reverse=True)[:n_features]
        
        logger.info(f"合并后选择了{len(selected_features)}个特征")
        
        return selected_features
    
    elif method == 'pls':
        # 使用PLS进行降维
        from sklearn.cross_decomposition import PLSRegression
        pls = PLSRegression(n_components=min(50, X_clean.shape[1]), random_state=42)
        pls.fit(X_clean, y)
        
        logger.info(f"PLS降维到{pls.n_components}个主成分")
        return pls
    
    else:
        logger.warning("不支持的特征选择方法")
        return X_clean.columns


def split_data(df, test_size=0.1, val_size=0.111):
    """将数据分为训练集、验证集和测试集，只使用年龄>=20的样本，使用分层抽样"""
    logger.info("分割数据...")
    
    # 只保留年龄>=20的样本，完全排除年龄<20的样本
    df_age_ge20 = df[df['age'] >= 20].copy()
    
    logger.info(f"年龄>=20的样本数: {len(df_age_ge20)}")
    logger.info(f"年龄<20的样本数: {len(df) - len(df_age_ge20)} (已排除)")
    
    # 创建年龄分层组
    bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    df_age_ge20['age_stratify'] = pd.cut(df_age_ge20['age'], bins=bins, include_lowest=True)
    
    # 从年龄>=20的样本中分割数据集
    from sklearn.model_selection import train_test_split
    
    X = df_age_ge20.drop(['age', 'age_stratify'], axis=1)
    y = df_age_ge20['age']
    stratify = df_age_ge20['age_stratify']
    
    # 先分割出测试集（使用分层抽样）
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    
    # 为训练验证集创建新的分层组
    df_train_val = df_age_ge20.loc[X_train_val.index].copy()
    stratify_train_val = df_train_val['age_stratify']
    
    # 从训练验证集中分割出验证集（使用分层抽样）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=stratify_train_val
    )
    
    logger.info(f"训练集: {len(X_train)} 样本")
    logger.info(f"验证集: {len(X_val)} 样本")
    logger.info(f"测试集: {len(X_test)} 样本")
    logger.info(f"测试集年龄范围: {y_test.min():.2f} - {y_test.max():.2f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
