# 衰老时钟 (Aging Clock)

基于基因表达数据的生物学年龄预测项目，使用机器学习模型预测个体的生物学年龄。

## 项目特点

- 多数据集整合（GEO公共数据库 + GTEx）
- 基因表达数据标准化和批次效应校正（Combat）
- 多种特征选择方法
- XGBoost单模型优化（当前最佳方案）
- Stacking集成学习
- SHAP模型解释性分析（支持生物学解读）
- 完整的可视化结果

## 项目结构

```
aging_clock/
├── data/                          # 原始数据目录
│   ├── GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt
│   ├── GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt
│   └── ensembl_to_symbol_map.csv   # 基因ID映射文件
├── models/                        # 训练好的模型
│   ├── xgboost_optimized.pkl      # 优化后的XGBoost模型（推荐使用）
│   ├── selected_features_xgboost.pkl
│   ├── best_params_xgboost.pkl
│   └── stacking_refactored.pkl
├── plots/                         # 可视化结果
│   ├── shap_summary_bar.png       # SHAP重要性柱状图
│   ├── shap_summary_beeswarm.png  # SHAP beeswarm图
│   ├── shap_dependence_*.png      # 基因依赖图
│   └── top_genes_shap.csv         # Top基因分析结果
├── preprocessed_data/             # 预处理后的数据
│   ├── GSE123696_processed.csv
│   ├── GSE123697_processed.csv
│   ├── GSE123698_processed.csv
│   ├── GSE164191_processed.csv
│   ├── GSE213516_processed.csv
│   ├── GSE231409_processed.csv
│   ├── GSE293163_processed.csv
│   ├── GTEx_processed.csv
│   ├── merged_processed.csv       # 合并后的数据
│   ├── merged_scaled.csv          # 合并并标准化的数据
│   └── integrated_datasets_info.txt
├── scripts/                       # 脚本目录
│   ├── data_processing/           # 数据处理脚本
│   │   ├── process_gse164191.py   # GSE164191数据集处理
│   │   ├── process_gse213516.py   # GSE213516数据集处理
│   │   ├── process_gtex.py        # GTEx数据集处理
│   │   └── merge_gse231409.py     # GSE231409数据合并
│   ├── model_training/            # 模型训练脚本
│   │   ├── main_xgboost_only.py   # XGBoost单模型优化（推荐）
│   │   ├── main_stacking_refactored.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── optuna_tuning.py
│   │   └── retrain_models.py
│   ├── analysis/                  # 分析脚本
│   │   ├── shap_analysis.py       # SHAP模型解释性分析
│   │   ├── shap_analysis_xgb_mlp.py
│   │   └── plot_results.py
│   └── utils/                     # 工具类
│       ├── data_utils.py          # 数据处理工具
│       ├── gene_utils.py          # 基因相关工具
│       └── model_utils.py         # 模型训练工具
├── archive/                       # 归档的临时文件
├── preprocess_and_merge.py        # 主数据预处理脚本
├── requirements.txt                # 依赖包
└── README.md
```

## 快速开始

### 环境要求

- Python 3.7+
- 推荐内存：16GB+

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd aging_clock

# 安装依赖
pip install -r requirements.txt
```

## 使用指南

### 1. 数据预处理

```bash
python preprocess_and_merge.py
```

该脚本会：

- 处理GEO数据集（GSE123696-98, GSE164191, GSE213516, GSE231409, GSE293163）
- 处理GTEx数据集（gene\_tpm\_v11\_whole\_blood.gct.gz）
- 基因ID映射和标准化
- 批次效应校正（Combat）
- Z-score特征标准化
- 合并多个数据集

### 2. 模型训练

#### 方案A：XGBoost单模型优化（**强烈推荐**）

```bash
python scripts/model_training/main_xgboost_only.py
```

**特点**：

- 特征选择：Top 350个基因（基于XGBoost特征重要性）
- 超参数调优：Optuna贝叶斯优化
- 模型性能：交叉验证R²≈0.78，测试集R²≈0.71

#### 方案B：重构Stacking

```bash
python scripts/model_training/main_stacking_refactored.py
```

只保留3个基模型：XGBoost, LightGBM, 线性核SVR

### 3. SHAP模型解释性分析

```bash
python scripts/analysis/shap_analysis.py
```

生成：

- Top 20基因SHAP重要性图
- 基因依赖图（Top 5基因）
- Top基因详细分析结果

## 数据集

| 数据集       | 样本数 | 年龄范围     | 类型      |
| --------- | --- | -------- | ------- |
| GSE123696 | -   | -        | RNA-seq |
| GSE123697 | -   | -        | RNA-seq |
| GSE123698 | -   | -        | RNA-seq |
| GSE164191 | 121 | -        | RNA-seq |
| GSE213516 | -   | -        | 芯片      |
| GSE231409 | 455 | 0.0-83.7 | RNA-seq |
| GSE293163 | 30  | 23-65    | 芯片      |
| GTEx      | 803 | 20-79    | RNA-seq |

**最终合并数据集**：1,620个样本，15,770个共同基因

## 模型性能

| 方案                    | 交叉验证R²    | 测试集R²     | MAE        | RMSE       |
| --------------------- | --------- | --------- | ---------- | ---------- |
| **XGBoost单模型（350特征）** | **≈0.78** | **≈0.71** | **≈10.35** | **≈13.81** |
| Stacking（6个基模型）       | 0.7156    | 0.6833    | 10.49      | 14.87      |
| 重构Stacking（3个基模型）     | -         | 0.6340    | 11.52      | 15.99      |

**最佳方案：XGBoost单模型（350特征）**

## Top 20 衰老相关基因（SHAP分析）

| 排名 | 基因名         | SHAP重要性 | SHAP方向均值 | 功能（推测）    |
| -- | ----------- | ------- | -------- | --------- |
| 1  | **ZNF204P** | 6.9664  | -1.4651  | 转录调控      |
| 2  | **CD248**   | 2.2900  | -0.3667  | 血管生成、肿瘤   |
| 3  | **LRRN3**   | 1.8966  | -0.4832  | 神经元发育     |
| 4  | **IFI30**   | 1.4296  | -0.0330  | 免疫反应      |
| 5  | **FOXL2**   | 1.2323  | 0.5451   | 卵巢发育      |
| 6  | C16ORF92    | 1.1394  | -0.0986  | 功能未知      |
| 7  | WT1         | 1.0763  | 0.0071   | 转录调控、肿瘤抑制 |
| 8  | CR2         | 1.0619  | 0.2249   | B细胞免疫     |
| 9  | HOXB13      | 1.0139  | 0.0358   | 发育调控      |
| 10 | USP50       | 0.9756  | 0.0494   | 蛋白质修饰     |
| 11 | LUM         | 0.8081  | -0.0868  | 细胞外基质     |
| 12 | SYN1        | 0.7886  | -0.1581  | 突触功能      |
| 13 | CNTN6       | 0.7671  | -0.0664  | 神经元黏附     |
| 14 | PMCHL1      | 0.6672  | 0.0645   | 神经肽       |
| 15 | MAPK15      | 0.6302  | -0.1632  | 信号转导      |
| 16 | IL11        | 0.6145  | -0.1120  | 炎症、纤维化    |
| 17 | RBFOX1      | 0.5440  | -0.0881  | 剪接调控      |
| 18 | RASGEF1A    | 0.5313  | 0.0766   | 信号转导      |
| 19 | PRDM9       | 0.5275  | -0.0273  | 减数分裂      |
| 20 | C4ORF36     | 0.5200  | 0.0173   | 功能未知      |

**指标说明**：

- **SHAP重要性**：基因对预测的重要性大小（绝对值均值），用于排序
- **SHAP方向均值**：基因对预测的影响方向，正值=高表达→预测年龄升高，负值=高表达→预测年龄降低

**关键发现**：

- ZNF204P是最重要的基因，贡献远高于其他基因
- Top基因主要参与：转录调控、免疫反应、神经元功能、细胞外基质
- 可用于后续文献验证和实验验证

## 依赖包

```
numpy
pandas
scikit-learn
lightgbm
xgboost
joblib
matplotlib
GEOparse
shap
science_computing_utils  # 包含Combat批次效应校正
```

## 后续改进方向

1. 增加更多数据集提高泛化能力
2. 对Top基因进行GO/KEGG功能富集分析
3. 实验验证关键基因的年龄相关性
4. 探索深度学习模型
5. 开发用户友好的预测界面

## 注意事项

- 首次运行需要处理数据集，耗时较长
- 确保有足够的内存（推荐16GB+）
- 训练日志保存在相关脚本的输出中

## 许可证

MIT License
