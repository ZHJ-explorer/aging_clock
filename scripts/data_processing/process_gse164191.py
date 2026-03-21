
import os
import gzip
import pandas as pd
import numpy as np
import GEOparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = 'data'
PREPROCESSED_DIR = 'preprocessed_data'

def process_gse164191():
    """处理GSE164191数据集"""
    print("处理GSE164191数据集...")
    
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    data_file = os.path.join(DATA_DIR, 'GSE164191_family.soft.gz')
    
    if not os.path.exists(data_file):
        print(f"文件 {data_file} 不存在")
        return None
    
    print(f"加载文件: {data_file}")
    gse = GEOparse.get_GEO(filepath=data_file)
    
    print(f"GSE ID: {gse.name}")
    print(f"样本数: {len(gse.gsms)}")
    
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
    
    print("标准化基因名称...")
    probe_to_gene = {}
    if gse.gpls:
        gpl = list(gse.gpls.values())[0]
        if 'Gene Symbol' in gpl.table.columns:
            for idx, row in gpl.table.iterrows():
                probe_id = row['ID']
                gene_symbol = row['Gene Symbol']
                if gene_symbol and gene_symbol != '---':
                    probe_to_gene[probe_id] = gene_symbol
    
    if probe_to_gene:
        new_columns = []
        seen = {}
        for col in df_expr.columns:
            if col in probe_to_gene:
                gene_name = probe_to_gene[col]
                if gene_name in seen:
                    seen[gene_name] += 1
                    new_columns.append(f"{gene_name}_{seen[gene_name]}")
                else:
                    seen[gene_name] = 0
                    new_columns.append(gene_name)
            else:
                if col in seen:
                    seen[col] += 1
                    new_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_columns.append(col)
        
        df_expr.columns = new_columns
        print("基因名称标准化完成")
    
    print("确保数据类型正确...")
    for col in df_expr.columns:
        df_expr[col] = pd.to_numeric(df_expr[col], errors='coerce')
    
    df_expr = df_expr.dropna(axis=1, how='all')
    
    df_expr['age'] = ages
    df_expr = df_expr.dropna(subset=['age'])
    
    print(f"处理后数据形状: {df_expr.shape}")
    print(f"样本数: {len(df_expr)}")
    print(f"基因数: {len(df_expr.columns) - 1}")
    
    output_file = os.path.join(PREPROCESSED_DIR, 'GSE164191_processed.csv')
    df_expr.to_csv(output_file)
    print(f"处理后的数据已保存到: {output_file}")
    
    return df_expr

if __name__ == "__main__":
    process_gse164191()

