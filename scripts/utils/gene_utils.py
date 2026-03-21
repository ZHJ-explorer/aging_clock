import re
import requests
import time
import concurrent.futures
import pickle
import os
from tqdm import tqdm

# 缓存文件路径
CACHE_FILE = 'gene_cache.pkl'

# 缓存，避免重复查询
ensembl_cache = {}

# 加载缓存
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'rb') as f:
            ensembl_cache = pickle.load(f)
        print(f"从缓存文件加载了 {len(ensembl_cache)} 个基因ID映射")
    except Exception as e:
        print(f"加载缓存文件失败: {e}")
        ensembl_cache = {}

# 批量处理的最大并发数
MAX_CONCURRENT = 15

def extract_gene_id(gene_name):
    """提取基因ID的核心部分（去除版本号）"""
    if isinstance(gene_name, str):
        return gene_name.split('.')[0]
    return str(gene_name)

def process_batch(batch_ids):
    """处理单个批次的Ensembl ID转换"""
    # 构建批量请求
    url = "https://rest.ensembl.org/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = {"ids": batch_ids}
    
    results = {}
    try:
        # 发送请求
        response = requests.post(url, json=data, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json()
    except Exception as e:
        # API调用失败，返回空结果
        pass
    
    # 处理结果
    batch_results = {}
    for ensembl_id in batch_ids:
        if ensembl_id in results:
            data = results[ensembl_id]
            if isinstance(data, dict):
                if 'display_name' in data:
                    gene_symbol = data['display_name'].upper()
                    batch_results[ensembl_id] = gene_symbol
                elif 'external_name' in data:
                    gene_symbol = data['external_name'].upper()
                    batch_results[ensembl_id] = gene_symbol
                else:
                    batch_results[ensembl_id] = ensembl_id
            else:
                # API返回错误信息，使用原始ID
                batch_results[ensembl_id] = ensembl_id
        else:
            # ID不在结果中，使用原始ID
            batch_results[ensembl_id] = ensembl_id
    
    return batch_results

def batch_ensembl_ids_to_symbols(ensembl_ids):
    """批量将Ensembl ID转换为基因符号，使用Ensembl REST API"""
    # 去除版本号并过滤已缓存的ID
    unique_ids = []
    for ensembl_id in ensembl_ids:
        ensembl_id = extract_gene_id(ensembl_id)
        if ensembl_id not in ensembl_cache:
            unique_ids.append(ensembl_id)
    
    # 如果所有ID都已缓存，直接返回
    if not unique_ids:
        print("所有Ensembl ID都已缓存，无需查询")
        return
    
    total_ids = len(unique_ids)
    print(f"需要查询 {total_ids} 个Ensembl ID...")
    
    # 增加批量大小，提高处理速度
    batch_size = 200  # 增加到200个ID per batch
    batches = []
    for i in range(0, total_ids, batch_size):
        batch_ids = unique_ids[i:i+batch_size]
        batches.append(batch_ids)
    
    total_batches = len(batches)
    print(f"共分成 {total_batches} 个批次，每批最多 {batch_size} 个ID")
    print(f"使用 {MAX_CONCURRENT} 个并发线程处理")
    
    # 跟踪处理进度
    completed_ids = 0
    
    # 使用tqdm创建进度条
    with tqdm(total=total_ids, desc="转换基因名称", unit="ID") as pbar:
        # 使用并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            # 提交所有批次的任务
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                batch_size = len(batch)
                
                try:
                    batch_results = future.result()
                    # 更新缓存
                    for ensembl_id, gene_symbol in batch_results.items():
                        ensembl_cache[ensembl_id] = gene_symbol
                except Exception as e:
                    print(f"批次处理失败: {e}")
                    # 处理失败的批次，使用原始ID
                    for ensembl_id in batch:
                        ensembl_cache[ensembl_id] = ensembl_id
                
                # 更新进度条
                completed_ids += batch_size
                pbar.update(batch_size)
    
    print(f"\n基因名称转换完成！共处理 {total_ids} 个Ensembl ID")
    
    # 保存缓存到文件
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(ensembl_cache, f)
        print(f"缓存已保存到 {CACHE_FILE}，共 {len(ensembl_cache)} 个基因ID映射")
    except Exception as e:
        print(f"保存缓存文件失败: {e}")
    
    # 减少延迟时间
    time.sleep(0.5)

def ensembl_id_to_symbol(ensembl_id):
    """将Ensembl ID转换为基因符号"""
    # 去除版本号
    ensembl_id = extract_gene_id(ensembl_id)
    
    # 检查缓存
    if ensembl_id in ensembl_cache:
        return ensembl_cache[ensembl_id]
    
    # 单个ID查询
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
    
    try:
        # 发送请求
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'display_name' in data:
                gene_symbol = data['display_name'].upper()
                # 缓存结果
                ensembl_cache[ensembl_id] = gene_symbol
                return gene_symbol
            elif 'external_name' in data:
                gene_symbol = data['external_name'].upper()
                # 缓存结果
                ensembl_cache[ensembl_id] = gene_symbol
                return gene_symbol
    except Exception as e:
        # API调用失败，返回原始ID
        pass
    
    # 如果API调用失败，返回原始ID
    ensembl_cache[ensembl_id] = ensembl_id
    return ensembl_id

def get_gene_biotype(ensembl_id):
    """获取基因的生物类型"""
    # 检查缓存
    cache_key = f"{ensembl_id}_biotype"
    if cache_key in ensembl_cache:
        return ensembl_cache[cache_key]
    
    # 单个ID查询
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
    
    try:
        # 发送请求
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'biotype' in data:
                biotype = data['biotype']
                # 缓存结果
                ensembl_cache[cache_key] = biotype
                return biotype
    except Exception as e:
        # API调用失败，返回None
        pass
    
    # 如果API调用失败，返回None
    ensembl_cache[cache_key] = None
    return None

def standardize_gene_name(gene_name):
    """标准化基因名称，处理不同格式的基因ID"""
    if not isinstance(gene_name, str):
        return str(gene_name)
    
    # 去除版本号
    gene_name = gene_name.split('.')[0]
    
    # 处理Ensembl ID格式 (ENSG00000000003)
    ensembl_pattern = r'^ENS[GTP][0-9]+$'
    if re.match(ensembl_pattern, gene_name):
        # 转换为基因符号
        return ensembl_id_to_symbol(gene_name)
    
    # 处理Entrez ID格式 (10000)
    entrez_pattern = r'^[0-9]+$'
    if re.match(entrez_pattern, gene_name):
        return gene_name
    
    # 处理基因符号格式 (TP53)
    # 基因符号通常由字母和数字组成，可能包含连字符
    gene_symbol_pattern = r'^[A-Za-z0-9\-]+$'
    if re.match(gene_symbol_pattern, gene_name):
        # 转换为大写
        return gene_name.upper()
    
    # 其他格式保持不变
    return gene_name

def prioritize_genes(gene_list):
    """按优先级筛选基因列表"""
    if not gene_list:
        return None
    
    # 优先级1：蛋白编码基因
    protein_coding_genes = []
    for gene in gene_list:
        # 检查是否是Ensembl ID
        ensembl_pattern = r'^ENS[GTP][0-9]+$'
        if re.match(ensembl_pattern, gene):
            biotype = get_gene_biotype(gene)
            if biotype == 'protein_coding':
                protein_coding_genes.append(gene)
    
    if protein_coding_genes:
        # 返回第一个蛋白编码基因
        return protein_coding_genes[0]
    
    # 优先级2：在目标组织中表达的基因（这里简化处理，返回第一个基因）
    # 实际应用中可以根据GTEx数据库或表达量阈值筛选
    if gene_list:
        return gene_list[0]
    
    # 优先级3：返回None
    return None

def map_gene_ids(genes, id_type='auto'):
    """映射基因ID到标准格式"""
    # 检查genes是否为空
    if isinstance(genes, (list, tuple)):
        if not genes:
            return []
    elif hasattr(genes, 'empty'):
        if genes.empty:
            return []
    elif not genes:
        return []
    
    # 收集所有Ensembl ID
    ensembl_ids = []
    for gene in genes:
        gene_name = str(gene).split('.')[0]
        ensembl_pattern = r'^ENS[GTP][0-9]+$'
        if re.match(ensembl_pattern, gene_name):
            ensembl_ids.append(gene_name)
    
    # 批量转换Ensembl ID为基因符号
    if ensembl_ids:
        # 处理所有基因，不限制数量
        print(f"处理 {len(ensembl_ids)} 个基因...")
        batch_ensembl_ids_to_symbols(ensembl_ids)
    
    # 标准化所有基因名称
    standardized_genes = []
    for gene in genes:
        standardized = standardize_gene_name(gene)
        standardized_genes.append(standardized)
    
    return standardized_genes

def align_genes_across_datasets(datasets):
    """对齐多个数据集中的基因，确保使用标准基因名称"""
    # 首先收集所有的基因ID
    all_genes = []
    for name, df in datasets:
        for col in df.columns:
            if col != 'age':
                all_genes.append(col)
    
    # 批量转换所有Ensembl ID为基因符号
    print("批量转换基因名称...")
    map_gene_ids(all_genes)
    
    # 然后标准化所有数据集的基因名称
    standardized_datasets = []
    all_standardized_genes = set()
    
    for name, df in datasets:
        # 标准化基因名称
        standardized_columns = []
        seen_genes = set()
        for col in df.columns:
            if col == 'age':
                standardized_columns.append('age')
            else:
                standardized_gene = standardize_gene_name(col)
                # 确保基因名称唯一
                counter = 1
                original_gene = standardized_gene
                while standardized_gene in seen_genes:
                    standardized_gene = f"{original_gene}_{counter}"
                    counter += 1
                seen_genes.add(standardized_gene)
                standardized_columns.append(standardized_gene)
        
        # 重命名列
        df_standardized = df.copy()
        df_standardized.columns = standardized_columns
        
        # 保存标准化后的数据集
        standardized_datasets.append((name, df_standardized))
        
        # 收集所有标准化后的基因
        genes = set(standardized_columns)
        if 'age' in genes:
            genes.remove('age')
        all_standardized_genes.update(genes)
    
    # 计算每个基因在数据集中的出现次数
    gene_counts = {}
    for gene in all_standardized_genes:
        count = 0
        for name, df in standardized_datasets:
            if gene in df.columns:
                count += 1
        gene_counts[gene] = count
    
    # 选择在所有数据集中出现的基因
    min_count = len(datasets)
    common_genes = [gene for gene, count in gene_counts.items() if count >= min_count]
    
    print(f"找到 {len(common_genes)} 个共同基因（在所有 {min_count} 个数据集中出现）")
    
    # 如果没有共同基因，尝试使用至少在5个数据集中出现的基因
    if len(common_genes) == 0:
        min_count = 5
        common_genes = [gene for gene, count in gene_counts.items() if count >= min_count]
        print(f"没有找到在所有数据集中出现的基因，使用在至少 {min_count} 个数据集中出现的基因: {len(common_genes)} 个")
    
    # 对齐所有数据集
    aligned_datasets = []
    total_samples = 0
    
    for name, df in standardized_datasets:
        # 只保留共同基因和年龄列
        cols_to_keep = [gene for gene in common_genes if gene in df.columns]
        cols_to_keep.append('age')
        
        aligned_df = df[cols_to_keep]
        aligned_datasets.append(aligned_df)
        total_samples += len(aligned_df)
        print(f"数据集 {name} 对齐后样本数: {len(aligned_df)}")
    
    print(f"预计合并后总样本数: {total_samples}")
    return aligned_datasets, common_genes