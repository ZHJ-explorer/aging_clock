# 衰老时钟 (Aging Clock)

基于基因表达数据的生物学年龄预测项目，使用传统机器学习与深度学习模型预测个体的生物学年龄。

## 项目特点

- **多源数据整合**：整合GEO公共数据库（GSE123696-98, GSE164191, GSE213516, GSE231409, GSE293163）与GTEx数据
- **完整数据标准化流程**：基因名称标准化、Z-score变换、ComBat批次效应校正
- **深度学习模型族**：DeepMLP、ResNetMLP、CNN1D、ResCNN1D、Transformer、TabNet
- **GPU加速训练**：支持CUDA加速的PyTorch深度学习模型
- **特征选择优化**：方差阈值 + 相关性分析 + XGBoost重要性筛选（15,624 → 728特征，减少95.3%）
- **Optuna超参数调优**：贝叶斯优化（TPESampler），50轮/模型
- **集成学习**：加权平均集成与Stacking集成
- **SHAP可解释性分析**：输出核心衰老基因

## 项目结构

```
aging_clock/
├── data/                           # 原始数据目录
│   └── raw/                        # 原始数据文件
├── preprocessed_data/              # 预处理后的数据
│   └── merged_scaled.csv           # 合并并标准化的数据（1,620样本，15,624基因）
├── preprocessing/                  # 数据预处理脚本
│   └── preprocess_and_merge.py     # 数据合并与标准化
├── training/                       # 训练脚本目录
│   ├── deep_learning/
│   │   └── train_all_dl_models.py  # 深度学习多模型训练
│   └── traditional_ml/
│       └── test_all_models.py      # 模型测试脚本
├── results/                         # 结果输出目录
│   ├── test_results/               # 测试结果文本
│   │   ├── test_result_xgboost.txt
│   │   ├── test_result_stacking.txt
│   │   └── test_result_stacking_refactored.txt
│   ├── logs/                      # 训练日志
│   │   └── training_dnn.log
│   └── cache/                     # 缓存文件
│       └── gene_cache.pkl
├── models/                         # 保存的模型文件
│   └── deep_learning/             # 深度学习模型
├── scripts/                        # 核心脚本模块
│   ├── data_processing/           # 数据处理脚本
│   │   ├── process_gtex.py        # GTEx数据集处理
│   │   ├── process_gse164191.py   # GSE164191数据集处理
│   │   ├── process_gse213516.py   # GSE213516数据集处理
│   │   ├── process_gse213516_simple.py
│   │   └── merge_gse231409.py     # GSE231409数据合并
│   ├── deep_learning/             # 深度学习模块
│   │   ├── models/                # 模型定义
│   │   │   ├── neural_networks/   # MLP系列模型（DeepMLP, ResNetMLP, CNN1D, ResCNN1D）
│   │   │   ├── attention/         # 注意力模型（Transformer, TabNet）
│   │   │   ├── ensemble/          # 集成模型
│   │   │   └── base/              # 基础组件
│   │   ├── configs/               # 配置文件
│   │   ├── evaluation/            # 评估指标
│   │   ├── optimization/          # 优化模块
│   │   │   ├── feature_selection.py      # 特征选择
│   │   │   ├── hyperparameter_tuning.py  # 超参数调优
│   │   │   └── ensemble_learning.py      # 集成学习
│   │   ├── training/              # 训练组件
│   │   └── train_dnn.py          # DNN训练脚本
│   ├── traditional_ml/            # 传统机器学习模块
│   │   ├── optimization/          # XGBoost/LightGBM调优
│   │   │   ├── hyperparameter_tuning.py
│   │   │   └── optuna_tuning.py
│   │   └── training/              # 模型训练
│   │       ├── retrain_models.py
│   │       ├── train_stacking.py
│   │       └── train_xgboost.py
│   ├── analysis/                   # 分析脚本
│   │   ├── explainability/        # SHAP可解释性分析
│   │   ├── statistics/            # PCA等统计分析
│   │   └── visualization/         # 可视化
│   └── utils/                     # 工具类
│       ├── data_pipeline.py       # 数据流水线
│       ├── data_utils.py          # 数据处理工具
│       ├── gene_utils.py          # 基因相关工具
│       └── model_utils.py         # 模型工具
├── tools/                          # 工具脚本
│   ├── check_age_distribution.py
│   ├── check_cuda.py
│   └── check_data.py
├── optuna_results/                 # Optuna调优结果
│   ├── optuna_summary.json        # 最优超参数汇总
│   ├── deepmlp/best_params.json
│   ├── resnetmlp/best_params.json
│   └── tabnet/best_params.json
├── ensemble_results/               # 集成学习结果
│   └── results_summary.json
├── selected_features/              # 选中的特征
│   └── feature_mask.npy
├── plots/                          # 可视化图表
│   ├── age_distribution_histogram.png
│   ├── pca_analysis.png
│   └── xgboost_optimized_prediction_vs_actual.png
├── archive/                        # 归档的临时文件
├── requirements.txt                # 依赖包
├── LICENSE.txt                     # MIT许可证
└── README.md                       # 项目文档
```

## 快速开始

### 环境要求

- Python 3.8+
- NVIDIA GPU（推荐，用于深度学习训练）
- 推荐内存：16GB+

### 安装

```bash
# 克隆项目
git clone https://github.com/ZHJ-explorer/aging_clock.git
cd aging_clock

# 安装依赖
pip install -r requirements.txt

# 安装PyTorch CUDA版本（使用清华镜像）
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# 安装批次效应校正工具
pip install pycombat
```

### 1. 数据预处理

```bash
python preprocessing/preprocess_and_merge.py
```

该脚本会：
- 处理GEO数据集（GSE123696-98, GSE164191, GSE213516, GSE231409, GSE293163）
- 处理GTEx数据集
- 基因ID映射和标准化
- 批次效应校正（ComBat）
- Z-score特征标准化
- 合并多个数据集

### 2. 深度学习模型训练

```bash
# 训练所有深度学习模型（使用GPU）
python training/deep_learning/train_all_dl_models.py
```

### 3. 特征选择（可选）

```bash
# 运行特征选择优化
python scripts/deep_learning/optimization/feature_selection.py
```

### 4. 超参数调优（可选）

```bash
# 使用Optuna进行超参数优化
python scripts/deep_learning/optimization/hyperparameter_tuning.py
```

### 5. 集成学习（可选）

```bash
# 运行集成学习
python scripts/deep_learning/optimization/ensemble_learning.py
```

### 6. 模型测试

```bash
# 测试所有模型
python training/traditional_ml/test_all_models.py
```

## 数据集信息

| 数据集       | 样本数 | 年龄范围     | 类型      |
| --------- | --- | -------- | ------- |
| GSE123696 | 66  | -        | 微阵列     |
| GSE123697 | 60  | -        | 微阵列     |
| GSE123698 | 68  | -        | 微阵列     |
| GSE164191 | 121 | -        | 微阵列     |
| GSE213516 | 17  | -        | RNA-seq |
| GSE231409 | 455 | 0.0-83.7 | RNA-seq |
| GSE293163 | 30  | 23-65    | RNA-seq |
| GTEx      | 803 | 20-79    | RNA-seq |

**最终合并数据集**：1,620个样本，15,624个共同基因
**训练使用**：只使用年龄≥20的样本（1,240样本）
**测试集年龄范围**：20.20 - 92.00岁

## 模型性能

### 超参数优化后的深度学习模型（728特征）

| 模型 | MAE | R² | 说明 |
| --- | --- | --- | --- |
| ResNetMLP | **9.22** | **0.472** | 最优深度学习模型 |
| DeepMLP | 9.28 | 0.446 | 标准多层感知机 |
| TabNet | 10.18 | 0.370 | 注意力机制模型 |
| 加权平均集成 | 9.20 | 0.465 | DL模型集成 |
| Stacking集成 | 9.29 | 0.443 | Stacking元学习 |

### XGBoost基准模型（对比）

| 模型 | MAE | R² | 说明 |
| --- | --- | --- | --- |
| XGBoost | 9.85 | 0.423 | 单模型基准 |

**关键发现**：优化后的ResNetMLP在MAE指标上超越XGBoost基准约6.4%，表明深度学习模型在捕获基因表达非线性关系方面具有优势。

## 最优超参数

### DeepMLP
```json
{
  "n_layers": 3,
  "base_dim": 128,
  "dropout": 0.163,
  "use_batchnorm": true,
  "activation": "relu",
  "lr": 0.0052,
  "weight_decay": 3.15e-06,
  "batch_size": 16
}
```

### ResNetMLP
```json
{
  "hidden_dim": 256,
  "n_res_blocks": 5,
  "dropout": 0.310,
  "use_batchnorm": true,
  "lr": 0.0077,
  "weight_decay": 2.53e-06,
  "batch_size": 64
}
```

### TabNet
```json
{
  "n_d": 128,
  "n_a": 128,
  "n_steps": 2,
  "lr": 0.0053,
  "weight_decay": 1.08e-05,
  "batch_size": 16
}
```

## 数据标准化流程

1. **基因名称标准化**：统一不同数据集的基因命名规则
2. **数据集内Z-score变换**：消除量纲影响
3. **共同基因对齐**：保留所有数据集共有的基因特征
4. **KNN缺失值填补**：保持数据的局部结构
5. **ComBat批次效应校正**：使用经验贝叶斯框架消除批次效应

   数学模型：$Y_{ij} = \alpha + X_i\beta + \gamma_j + \delta_j Z_{ij} + \epsilon_{ij}$

   其中 $\gamma_j$ 和 $\delta_j$ 分别是批次 $j$ 的加性和乘性批次效应

6. **全局Z-score标准化**：确保所有特征在同一尺度

## 依赖包

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
joblib>=1.3.0
matplotlib>=3.7.0
GEOparse>=1.1.4
xgboost>=2.0.0
shap>=0.42.0
optuna>=3.0.0
pycombat>=0.2.7
torch>=2.0.0
torchvision>=0.15.0
pytorch-tabnet>=0.1.0
```

## 输出文件

| 文件/目录 | 描述 |
| --- | --- |
| `optuna_results/` | Optuna超参数调优结果 |
| `ensemble_results/` | 集成学习结果 |
| `selected_features/feature_mask.npy` | 选中的728个特征掩码 |
| `plots/` | 可视化图表 |
| `results/test_results/` | 测试结果文本文件 |
| `results/logs/` | 训练日志文件 |
| `models/` | 保存的模型文件 |

## 后续改进方向

1. 增加更多数据集提高泛化能力
2. 对Top基因进行GO/KEGG功能富集分析
3. 实验验证关键基因的年龄相关性
4. 探索更先进的深度学习架构
5. 开发用户友好的预测界面
6. 优化特征选择方法，结合生物学知识

## 注意事项

- 首次运行需要处理数据集，耗时较长
- 确保有足够的内存（推荐16GB+）
- 深度学习训练推荐使用GPU以加速
- 训练日志保存在`results/logs/`目录中
- 超参数调优记录保存在`optuna_results/`目录中

## 许可证

MIT License

## 引用

如果您使用本项目的代码或结果，请引用相关工作。
