# 衰老时钟 (Aging Clock)

基于基因表达数据的生物学年龄预测项目，使用机器学习模型预测个体的生物学年龄。

## 项目特点

- 多数据集整合（GEO公共数据库 + GTEx）
- 基因表达数据标准化和批次效应校正（Combat）
- **5折交叉验证平均特征重要性选择**（提高特征选择稳定性）
- XGBoost单模型优化
- **Stacking集成学习**（XGBoost + MLP）
- **Optuna调优可复现性**（固定随机种子，保存调优记录）
- **权重融合优化**（同时考虑R²和MAE指标）
- **重复交叉验证**（提高验证结果可信度）
- **SHAP模型解释性分析**（支持生物学解读，输出核心衰老基因）
- 完整的可视化结果

## 项目结构

```
aging_clock/
├── data/                          # 原始数据目录
├── models/                        # 训练好的模型
│   ├── xgboost_optimized.pkl      # 优化后的XGBoost模型（推荐使用）
│   ├── selected_features_xgboost.pkl
│   ├── best_params_xgboost.pkl
│   ├── base_models_refactored.pkl # Stacking基模型
│   └── best_weights.pkl           # Stacking最优权重
├── preprocessed_data/             # 预处理后的数据
│   ├── merged_processed.csv       # 合并后的数据
│   └── merged_scaled.csv          # 合并并标准化的数据
├── scripts/                       # 脚本目录
│   ├── data_processing/           # 数据处理脚本
│   │   ├── merge_gse231409.py     # GSE231409数据合并
│   │   ├── process_gse164191.py   # GSE164191数据集处理
│   │   ├── process_gse213516.py   # GSE213516数据集处理
│   │   ├── process_gse213516_simple.py
│   │   └── process_gtex.py        # GTEx数据集处理
│   ├── model_training/            # 模型训练脚本
│   │   ├── hyperparameter_tuning.py
│   │   ├── main_stacking_refactored.py
│   │   ├── main_xgboost_only.py   # XGBoost单模型优化（推荐）
│   │   ├── optuna_tuning.py
│   │   └── retrain_models.py
│   ├── analysis/                  # 分析脚本
│   │   ├── plot_results.py
│   │   ├── shap_analysis.py       # SHAP模型解释性分析
│   │   └── shap_analysis_xgb_mlp.py
│   └── utils/                     # 工具类
│       ├── data_utils.py          # 数据处理工具
│       ├── gene_utils.py          # 基因相关工具
│       └── model_utils.py         # 模型训练工具
├── optuna_logs/                   # Optuna调优记录
├── archive/                       # 归档的临时文件
├── preprocess_and_merge.py        # 主数据预处理脚本
├── requirements.txt                # 依赖包
├── LICENSE.txt                    # MIT许可证
└── README.md
```

## 快速开始

### 环境要求

- Python 3.7+
- 推荐内存：16GB+

### 安装

```bash
# 克隆项目
git clone https://github.com/ZHJ-explorer/aging_clock.git
cd aging_clock

# 安装依赖
pip install -r requirements.txt

# 安装额外依赖（批次效应校正）
pip install pycombat
```

## 使用指南

### 1. 数据预处理

```bash
python preprocess_and_merge.py
```

该脚本会：

- 处理GEO数据集（GSE123696-98, GSE164191, GSE213516, GSE231409, GSE293163）
- 处理GTEx数据集
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

- 特征选择：Top 350个基因（基于5折交叉验证平均特征重要性）
- 超参数调优：Optuna贝叶斯优化（固定随机种子，保存调优记录）
- 模型评估：5折交叉验证重复3次（提高验证可信度）
- SHAP分析：输出核心衰老基因
- 模型性能：交叉验证R²≈0.775，测试集R²≈0.692

#### 方案B：Stacking集成学习

```bash
python scripts/model_training/main_stacking_refactored.py
```

**特点**：

- 基模型：XGBoost（调优后）、MLP（调优后）
- 权重融合：同时考虑R²和MAE指标
- 模型评估：5折交叉验证重复3次
- SHAP分析：输出核心衰老基因

## 数据集

| 数据集       | 样本数 | 年龄范围     | 类型      |
| --------- | --- | -------- | ------- |
| GSE123696 | 66  | -        | RNA-seq |
| GSE123697 | 60  | -        | RNA-seq |
| GSE123698 | 68  | -        | RNA-seq |
| GSE164191 | 121 | -        | RNA-seq |
| GSE213516 | 17  | -        | 芯片      |
| GSE231409 | 455 | 0.0-83.7 | RNA-seq |
| GSE293163 | 30  | 23-65    | 芯片      |
| GTEx      | 803 | 20-79    | RNA-seq |

**最终合并数据集**：1,620个样本，15,624个共同基因
**训练使用**：只使用年龄≥20的样本

## 模型性能

| 方案                    | 交叉验证R²    | 测试集R²     | MAE        | RMSE       |
| --------------------- | --------- | --------- | ---------- | ---------- |
| **XGBoost单模型（350特征）** | **≈0.775** | **≈0.692** | **≈10.94** | **≈14.32** |
| Stacking（XGBoost + MLP）     | -         | -         | -          | -          |

**最佳方案：XGBoost单模型（350特征）**

## 输出文件

- `shap_feature_importance_xgboost.csv`：XGBoost模型的核心衰老基因（按SHAP重要性排序）
- `shap_feature_importance_stacking.csv`：Stacking模型的核心衰老基因
- `optuna_logs/`：Optuna调优记录
- `test_result_xgboost.txt`：XGBoost模型测试结果
- `test_result_stacking.txt`：Stacking模型测试结果

## 依赖包

```
numpy
pandas
scikit-learn
lightgbm
joblib
matplotlib
GEOparse
xgboost
shap
optuna  # 用于超参数调优
pycombat  # 用于批次效应校正
```

## 后续改进方向

1. 增加更多数据集提高泛化能力
2. 对Top基因进行GO/KEGG功能富集分析
3. 实验验证关键基因的年龄相关性
4. 探索深度学习模型
5. 开发用户友好的预测界面
6. 构建更复杂的集成学习模型
7. 优化特征选择方法，结合生物学知识

## 注意事项

- 首次运行需要处理数据集，耗时较长
- 确保有足够的内存（推荐16GB+）
- 训练日志保存在相关脚本的输出中
- 调优记录保存在`optuna_logs/`目录中
- 核心衰老基因分析结果保存在`shap_feature_importance*.csv`文件中

## 许可证

MIT License

## 引用

如果您使用本项目的代码或结果，请引用相关工作。