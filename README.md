# 衰老时钟 (Aging Clock)

基于基因表达数据的生物学年龄预测项目，使用传统机器学习与深度学习模型预测个体的生物学年龄。

## 项目特点

- **多源数据整合**：整合GEO公共数据库（GSE123696-98, GSE164191, GSE213516, GSE231409, GSE293163）与GTEx数据
- **完整数据标准化流程**：基因名称标准化、Z-score变换、ComBat批次效应校正
- **深度学习模型族**：DeepMLP、ResNetMLP、CNN1D、ResCNN1D、Transformer、TabNet、DNN
- **GPU加速训练**：支持CUDA加速的PyTorch深度学习模型
- **特征选择优化**：方差阈值 + 相关性分析 + XGBoost重要性筛选（15,624 → 350特征）
- **数据泄漏防护**：特征选择在训练集上进行，避免验证集/测试集信息泄漏
- **不确定性估计**：MC Dropout蒙特卡洛dropout深度学习不确定性估计
- **学习率预热**：WarmupCosineScheduler学习率预热+余弦退火策略
- **可配置早停**：支持基于loss或MAE的早停策略
- **Optuna超参数调优**：贝叶斯优化（TPESampler），70轮/模型
- **集成学习**：加权平均集成与Stacking集成
- **SHAP可解释性分析**：输出核心衰老基因
- **模块化配置管理**：集中式配置文件（scripts/config.py）

## 项目结构

```
aging_clock/
├── data/                           # 原始数据目录
│   └── raw/                        # 原始数据文件
├── docs/                           # 项目文档
│   └── 技术路线_mermaid.md         # 技术路线图
├── ensemble_results/               # 集成学习结果
│   └── results_summary.json        # 最优超参数汇总
├── optuna_results/                # Optuna超参数调优结果
├── plots/                         # 可视化图表
│   ├── age_distribution_histogram.png
│   ├── pca_analysis.png
│   └── ...
├── preprocessed_data/             # 预处理后的数据
│   └── merged_scaled.csv          # 合并并标准化的数据
├── results/                       # 结果输出目录
│   ├── cache/                     # 缓存文件
│   │   └── gene_cache.pkl
│   ├── logs/                      # 训练日志
│   │   └── training_dnn.log
│   └── test_results/              # 测试结果文本
├── scripts/                       # 核心脚本模块
│   ├── analysis/                  # 分析脚本
│   │   ├── explainability/       # SHAP可解释性分析
│   │   ├── statistics/           # PCA等统计分析
│   │   └── visualization/        # 可视化
│   ├── config.py                  # 集中式配置文件
│   ├── data_processing/           # 数据处理脚本
│   │   ├── run_all_preprocessing.py   # 数据预处理统一入口
│   │   ├── preprocess_and_merge.py    # 数据合并与标准化
│   │   ├── process_gse164191.py       # 处理GSE164191数据集
│   │   ├── process_gse213516.py       # 处理GSE213516数据集
│   │   ├── process_gtex.py            # 处理GTEx数据集
│   │   └── merge_gse231409.py         # 合并GSE231409数据集
│   ├── deep_learning/             # 深度学习模块
│   │   ├── configs/               # 配置文件
│   │   ├── evaluation/           # 评估指标
│   │   ├── optimization/          # 优化模块
│   │   ├── training/              # 训练组件
│   │   └── train_dnn.py           # DNN训练脚本
│   ├── traditional_ml/            # 传统机器学习模块
│   │   ├── optimization/          # XGBoost/LightGBM调优
│   │   └── training/             # 模型训练
│   └── utils/                     # 工具类
├── selected_features/             # 选中的特征
│   └── feature_mask.npy           # 特征掩码
├── training/                      # 训练脚本目录
├── .gitignore
├── LICENSE.txt                    # MIT许可证
├── README.md                      # 项目文档
└── requirements.txt               # 依赖包
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

# 安装PyTorch CUDA版本
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# 安装批次效应校正工具
pip install pycombat
```

### 1. 数据预处理

```bash
python scripts/data_processing/run_all_preprocessing.py
```

该脚本会按顺序执行以下步骤：
1. 处理各个数据集（GSE164191, GSE213516, GTEx等）
2. 合并GSE231409数据集
3. 合并所有数据集并进行标准化、批次效应校正

如果需要单独处理某个数据集，可以直接运行对应的脚本：
```bash
python scripts/data_processing/process_gse164191.py
python scripts/data_processing/process_gse213516.py
python scripts/data_processing/process_gtex.py
```

### 2. 机器学习训练

```bash
# XGBoost单模型训练（带Optuna超参数调优）
python scripts/traditional_ml/training/train_xgboost.py

# Stacking集成模型训练
python scripts/traditional_ml/training/train_stacking.py
```

### 3. 深度学习模型训练

```bash
# 训练单个DNN模型
python scripts/deep_learning/train_dnn.py

# 训练所有深度学习模型（DeepMLP, ResNetMLP, CNN1D, ResCNN1D, Transformer, TabNet）
python training/deep_learning/train_all_dl_models.py
```

### 4. 模型测试

```bash
# 测试所有模型
python training/traditional_ml/test_all_models.py
```

### 5. 可视化（自动生成）

训练完成后自动生成以下图表到 `plots/` 目录：

| 图表类型                               | 说明          |
| ---------------------------------- | ----------- |
| `{model}_training_history.png`     | 训练/验证损失曲线   |
| `{model}_prediction_vs_actual.png` | 实际 vs 预测散点图 |
| `{model}_residuals.png`            | 残差图         |
| `{model}_error_distribution.png`   | 误差分布直方图     |

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

### 深度学习模型（15,624特征）

| 模型          | MAE      | RMSE      | R²         | 说明             |
| ----------- | -------- | --------- | ---------- | -------------- |
| ResNetMLP   | **9.41** | **12.37** | **0.4309** | 最优深度学习模型       |
| DeepMLP     | 10.06    | 12.91     | 0.3700     | 标准多层感知机        |
| Transformer | 11.25    | 14.46     | 0.2223     | Transformer编码器 |
| TabNet      | 12.47    | 16.87     | -0.1812    | 注意力机制模型        |
| ResCNN1D    | 14.86    | 17.70     | -0.1666    | 一维残差CNN        |
| CNN1D       | 16.25    | 18.71     | -0.3027    | 一维CNN          |

### XGBoost机器学习模型（350特征）

| 模型                       | MAE      | RMSE      | R²         | 说明            |
| ------------------------ | -------- | --------- | ---------- | ------------- |
| XGBoost (Optuna调优)       | **8.53** | **10.84** | **0.5897** | 最优单模型         |
| Stacking集成 (XGBoost+MLP) | 8.76     | 11.17     | 0.5646     | 加权融合0.84/0.16 |

**关键发现**：XGBoost在相同特征数量下表现优于深度学习模型，说明传统机器学习更适合这种小样本高维数据。深度学习模型在15,624特征上容易过拟合。

## 最优超参数

### XGBoost（Optuna 70轮调优）

```json
{
  "n_estimators": 296,
  "learning_rate": 0.0466,
  "max_depth": 4,
  "min_child_weight": 9,
  "subsample": 0.83,
  "colsample_bytree": 0.79,
  "reg_alpha": 10.57,
  "reg_lambda": 1.15,
  "gamma": 6.44
}
```

### DeepMLP

```json
{
  "n_layers": 4,
  "hidden_dims": [512, 256, 128, 64],
  "dropout": 0.3,
  "use_batchnorm": true,
  "activation": "relu",
  "lr": 0.001,
  "batch_size": 32,
  "epochs": 200
}
```

### ResNetMLP

```json
{
  "hidden_dim": 256,
  "n_res_blocks": 4,
  "dropout": 0.3,
  "use_batchnorm": true,
  "lr": 0.001,
  "batch_size": 32,
  "epochs": 100
}
```

### TabNet

```json
{
  "n_d": 16,
  "n_a": 16,
  "n_steps": 3,
  "gamma": 1.5,
  "lambda_sparse": 1e-4,
  "momentum": 0.02,
  "clip_value": 2.0
}
```

## 数据标准化流程

1. **基因名称标准化**：统一不同数据集的基因命名规则
2. **数据集内Z-score变换**：消除量纲影响
3. **共同基因对齐**：保留所有数据集共有的基因特征
4. **KNN缺失值填补**：保持数据的局部结构
5. **ComBat批次效应校正**：使用经验贝叶斯框架消除批次效应

   数学模型： $Y_{ij} = \alpha + X_i\beta + \gamma_j + \delta_j Z_{ij} + \epsilon_{ij}$

   其中 $\gamma_j$ 和 $\delta_j$ 分别是批次 $j$ 的加性和乘性批次效应
6. **全局Z-score标准化**：确保所有特征在同一尺度

## 特征选择流程

采用三阶段特征选择策略，将特征从15,624维降至350维：

1. **方差阈值筛选**：移除方差低于阈值的特征
2. **相关性分析**：计算特征与年龄的相关性，移除高度相关的冗余特征
3. **XGBoost重要性排序**：基于增益重要性选择Top 350特征

关键：所有特征选择操作仅在训练集上进行，防止数据泄漏。

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

| 文件/目录                                | 描述            |
| ------------------------------------ | ------------- |
| `optuna_results/`                    | Optuna超参数调优结果 |
| `ensemble_results/`                  | 集成学习结果        |
| `selected_features/feature_mask.npy` | 选中的特征掩码       |
| `plots/`                             | 可视化图表（自动生成）   |
| `results/test_results/`              | 测试结果文本文件      |
| `results/logs/`                      | 训练日志文件        |
| `results/cache/`                     | 基因名称缓存文件     |
| `models/`                            | 保存的模型文件       |
| `preprocessed_data/`                 | 预处理后数据        |
| `docs/技术路线_mermaid.md`           | 技术路线图文档      |

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
- 基因名称缓存保存在`results/cache/gene_cache.pkl`

## 许可证

MIT License

## 引用

如果您使用本项目的代码或结果，请引用相关工作。
