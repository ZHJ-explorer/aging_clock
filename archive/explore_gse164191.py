
import os
import GEOparse
import pandas as pd

DATA_DIR = 'data'

def explore_gse164191():
    """探索GSE164191数据集"""
    print("探索GSE164191数据集...")
    
    data_file = os.path.join(DATA_DIR, 'GSE164191_family.soft.gz')
    
    if not os.path.exists(data_file):
        print(f"文件 {data_file} 不存在")
        return
    
    print(f"加载文件: {data_file}")
    gse = GEOparse.get_GEO(filepath=data_file)
    
    print(f"\nGSE ID: {gse.name}")
    print(f"样本数: {len(gse.gsms)}")
    print(f"平台数: {len(gse.gpls)}")
    
    # 查看第一个样本
    print("\n=== 第一个样本信息 ===")
    first_gsm_name = list(gse.gsms.keys())[0]
    first_gsm = gse.gsms[first_gsm_name]
    print(f"样本ID: {first_gsm_name}")
    print(f"Metadata keys: {list(first_gsm.metadata.keys())}")
    
    # 尝试查找年龄信息
    print("\n=== 查找年龄信息 ===")
    ages = []
    for gsm_name, gsm in gse.gsms.items():
        age = None
        if 'age' in gsm.metadata:
            age = gsm.metadata['age'][0]
        elif 'characteristics_ch1' in gsm.metadata:
            for char in gsm.metadata['characteristics_ch1']:
                if 'age' in char.lower():
                    age = char.split(':')[-1].strip()
                    break
        if age:
            ages.append(age)
    
    print(f"找到 {len(ages)} 个年龄信息样本")
    if ages:
        print(f"前5个年龄值: {ages[:5]}")
    
    # 查看样本数据表结构
    print("\n=== 样本数据结构 ===")
    if first_gsm.table is not None:
        print(f"表列名: {list(first_gsm.table.columns)}")
        print(f"表形状: {first_gsm.table.shape}")
        print(f"前几行数据:")
        print(first_gsm.table.head())
    
    # 查看平台信息
    print("\n=== 平台信息 ===")
    if gse.gpls:
        first_gpl_name = list(gse.gpls.keys())[0]
        first_gpl = gse.gpls[first_gpl_name]
        print(f"平台ID: {first_gpl_name}")
        print(f"平台表列名: {list(first_gpl.table.columns)}")
        print(f"平台表形状: {first_gpl.table.shape}")
        print(f"前几行:")
        print(first_gpl.table.head())

if __name__ == "__main__":
    explore_gse164191()
