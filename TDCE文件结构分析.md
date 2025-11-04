# TDCE文件结构分析

## 核心内容（必须保留）⭐

### 1. 核心实现代码（必需）
- **`tdce/`** (134KB) - TDCE核心实现
  - `gaussian_multinomial_diffsuion.py` - 扩散模型核心（包含Gumbel-Softmax支持）
  - `gumbel_softmax_utils.py` - Gumbel-Softmax工具函数
  - `classifier_guidance.py` - 分类器梯度引导模块
  - `modules.py` - 模型定义（MLPDiffusion等）
  - `utils.py` - 工具函数
  - `__init__.py` - 模块初始化

### 2. 数据处理库（必需）
- **`lib/`** (31KB) - 数据处理和工具库
  - `data.py` - 数据处理（包含不可变特征掩码功能）
  - `util.py` - 工具函数
  - `metrics.py` - 评估指标
  - `env.py` - 环境配置
  - `deep.py` - 深度学习方法
  - `__init__.py` - 模块初始化

### 3. 核心脚本（必需）
- **`scripts/`** (59KB) - 脚本目录
  - `train_classifier.py` ⭐ - TDCE分类器训练脚本（新增）
  - `sample_counterfactual.py` ⭐ - TDCE反事实生成脚本（新增）
  - `train.py` - 扩散模型训练脚本
  - `sample.py` - 样本生成脚本
  - `utils_train.py` - 训练工具函数
  - `process_dataset.py` - 数据集处理脚本
  - `download_dataset.py` - 数据集下载脚本

### 4. 测试和依赖（必需）
- **`test_gumbel_softmax.py`** - Gumbel-Softmax功能测试
- **`requirements.txt`** - Python依赖包

### 5. 文档（建议保留）
- **`README.md`** - 需要更新为TDCE的README
- **`CONFIG_DESCRIPTION.md`** - 配置说明（有用）
- **`DATA_USAGE_README.md`** - 数据使用说明（有用）
- **`LICENSE.md`** - 许可证

---

## 可选内容（根据需要保留）⚡

### 1. 评估脚本（可选，用于评估）
- `scripts/eval_mlp.py` - MLP评估脚本
- `scripts/eval_catboost.py` - CatBoost评估脚本
- `scripts/eval_seeds.py` - 多种子评估脚本
- `scripts/eval_seeds_simple.py` - 简化评估脚本
- `scripts/eval_simple.py` - 简单评估脚本

### 2. 工具脚本（可选）
- `scripts/pipeline.py` - 管道脚本
- `scripts/tune_ddpm.py` - 超参数调优脚本
- `scripts/tune_evaluation_model.py` - 评估模型调优脚本
- `scripts/resample_privacy.py` - 隐私重采样脚本

---

## 可删除内容（可以清理）🗑️

### 1. 实验配置文件（可删除，但需谨慎）
- **`exp/`** (693KB) ⚠️ - 包含大量实验配置文件
  - 说明：包含多个数据集的实验配置（abalone, adult, buddy等）
  - 建议：
    - 如果不需要这些实验配置，可以删除
    - 如果需要参考或复用，可以保留部分示例配置
  - 操作：可以删除整个文件夹，或只保留1-2个示例配置

### 2. 调优模型参数（可删除）
- **`tuned_models/`** (35KB) - 预调优的评估模型参数
  - 说明：包含CatBoost和MLP的调优参数
  - 建议：如果不需要这些预调优参数，可以删除
  - 操作：可以删除整个文件夹

### 3. 数据集（可删除，如果需要重新下载）
- **`data/`** (11KB) - 数据集目录
  - 说明：如果只有小文件（可能是占位符），可以删除后重新下载
  - 建议：如果是示例数据集，可以删除；如果需要保留，可以保留

### 4. 实现文档（已完成，可删除）
- **`TDCE实现计划.md`** ⚠️ - TDCE实现计划文档（已完成）
- **`tabddpm转TDCE.md`** ⚠️ - TabDDPM转TDCE说明文档（已完成）
  - 说明：这些是开发过程中的文档，核心功能已实现
  - 建议：可以删除或移到单独的docs文件夹归档

### 5. 数据集信息文档（可选）
- **`TDCE用到的数据集信息.md`** - 数据集信息说明
  - 说明：如果不需要数据集信息，可以删除
  - 建议：如果作为文档参考，可以保留

### 6. 结果汇总Notebook（可选）
- **`agg_results.ipynb`** - 结果汇总Jupyter Notebook
  - 说明：用于汇总实验结果
  - 建议：如果不需要查看历史结果，可以删除

### 7. Python缓存文件（可清理）
- **`__pycache__/`** - Python字节码缓存
- **`*.pyc`** - 编译的Python文件
  - 说明：这些是自动生成的缓存文件
  - 操作：可以删除，Python会重新生成

---

## 清理建议

### 最小化核心版本（仅TDCE功能）
```bash
# 删除实验配置（如果不需要）
rm -rf exp/

# 删除调优模型（如果不需要）
rm -rf tuned_models/

# 删除数据集（如果需要重新下载）
rm -rf data/

# 删除实现文档（已完成）
rm TDCE实现计划.md
rm tabddpm转TDCE.md

# 删除Python缓存
find . -type d -name "__pycache__" -exec rm -r {} +
find . -name "*.pyc" -delete
```

### 保留示例版本（推荐用于学习和参考）
- 保留 `exp/` 中的1-2个示例配置
- 保留评估脚本（用于测试）
- 保留数据集信息文档
- 删除实现计划文档（已完成）
- 删除Python缓存

---

## 文件大小统计

- **核心代码**：tdce/ (134KB) + lib/ (31KB) + scripts/ (59KB) = ~224KB
- **可选内容**：exp/ (693KB) + tuned_models/ (35KB) + data/ (11KB) = ~739KB

核心代码仅占约23%，大部分空间被实验配置文件占用。

