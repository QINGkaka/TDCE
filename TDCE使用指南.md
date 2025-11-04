# TDCE使用指南：表格数据反事实生成完整流程

## 📋 执行顺序

### 阶段1：功能测试（验证实现）✅

**优先级：高**

首先验证TDCE的核心功能是否正常工作：

```bash
cd /root/data/gq_antifact/TDCE
python test_gumbel_softmax.py
```

**预期结果**：
- ✅ 所有Gumbel-Softmax工具函数测试通过
- ✅ 扩散模型Gumbel-Softmax集成测试通过
- ✅ p_sample_gumbel_softmax和sample方法测试通过

**如果测试失败**：需要修复代码问题
**如果测试通过**：可以进入下一阶段

---

### 阶段2：准备数据集（必需）📊

**优先级：高**

#### 2.1 下载数据集

TDCE支持的数据集：
- **Adult** - UCI下载（最简单）
- **Lending Club Dataset (LCD)** - Kaggle下载（需要API密钥）
- **Give Me Some Credit (GMC)** - Kaggle下载（需要API密钥）
- **LAW** - OpenML或手动下载（可选）

**方法1：下载Adult数据集（推荐，最简单）**
```bash
cd /root/data/gq_antifact/TDCE
python scripts/download_dataset.py
# 只下载adult数据集，需要修改脚本或只运行adult部分
```

**方法2：手动下载（推荐用于快速测试）**
```bash
# 创建数据目录
mkdir -p data/adult

# 下载Adult数据集（UCI）
wget https://archive.ics.uci.edu/static/public/2/adult.zip -O data/adult/adult.zip
cd data/adult && unzip adult.zip && cd ../..
```

#### 2.2 预处理数据集

```bash
# 处理adult数据集
python scripts/process_dataset.py --dataname adult
```

**输出**：
```
data/adult/
├── X_num_train.npy      # 数值特征训练集
├── X_cat_train.npy      # 分类特征训练集
├── y_train.npy          # 标签训练集
├── X_num_val.npy        # 数值特征验证集
├── X_cat_val.npy        # 分类特征验证集
├── y_val.npy            # 标签验证集
├── X_num_test.npy       # 数值特征测试集
├── X_cat_test.npy       # 分类特征测试集
├── y_test.npy           # 标签测试集
└── info.json            # 数据集元信息
```

---

### 阶段3：训练扩散模型（必需）🎯

**优先级：高**

#### 3.1 准备配置文件

创建一个简单的配置文件 `config_adult.toml`：

```toml
seed = 0
parent_dir = "exp/adult/tdce_test"
real_data_path = "data/adult"
model_type = "mlp"
num_numerical_features = 9
device = "cuda:0"  # 或 "cpu"

[model_params]
num_classes = 2
is_y_cond = true

[model_params.rtdl_params]
d_layers = [256, 512]
dropout = 0.0

[diffusion_params]
num_timesteps = 1000
gaussian_loss_type = "mse"
scheduler = "cosine"
use_gumbel_softmax = true
tau_init = 1.0
tau_final = 0.3
tau_schedule = "anneal"

[train.main]
steps = 5000
lr = 0.0002
weight_decay = 1e-4
batch_size = 1024

[train.T]
seed = 0
normalization = "quantile"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "one-hot"
y_policy = "default"
```

#### 3.2 训练扩散模型

```bash
python scripts/train.py --config config_adult.toml
```

**预期输出**：
- 模型权重保存到 `exp/adult/tdce_test/model.pt`
- 训练日志显示损失值

**训练时间**：根据数据集大小和GPU，可能需要几分钟到几小时

---

### 阶段4：训练分类器（必需）🎯

**优先级：高**

分类器用于梯度引导，是TDCE反事实生成的关键组件。

#### 4.1 训练分类器

```bash
python scripts/train_classifier.py \
    --data_path data/adult \
    --output_path exp/adult/classifier.pt \
    --num_classes 2 \
    --num_epochs 100 \
    --batch_size 1024 \
    --lr 0.001 \
    --device cuda:0
```

**预期输出**：
- 分类器权重保存到 `exp/adult/classifier.pt`
- 训练过程显示训练/验证准确率
- 最佳验证准确率应该>85%（取决于数据集）

---

### 阶段5：生成反事实样本（核心功能）🚀

**优先级：最高**

#### 5.1 准备原始样本

创建包含原始样本的numpy文件：

```python
# create_original_samples.py
import numpy as np
import lib

# 加载测试集
T = lib.Transformations(normalization='quantile', cat_encoding='one-hot', y_policy='default')
dataset = lib.build_dataset('data/adult', T, task_type=lib.TaskType.CLASSIFICATION)

# 选择一些样本（例如标签为0的样本，想翻转成1）
test_indices = np.where(dataset.y['test'] == 0)[0][:10]  # 选择10个样本

# 组合特征
X_num = dataset.X_num['test'][test_indices] if dataset.X_num else None
X_cat = dataset.X_cat['test'][test_indices] if dataset.X_cat else None
y_test = dataset.y['test'][test_indices]

# 组合为完整特征矩阵
if X_num is not None and X_cat is not None:
    X_original = np.concatenate([X_num, X_cat], axis=1)
elif X_num is not None:
    X_original = X_num
else:
    X_original = X_cat

np.save('original_samples.npy', X_original)
print(f"保存了 {len(X_original)} 个原始样本")
```

#### 5.2 生成反事实样本

```bash
python scripts/sample_counterfactual.py \
    --config exp/adult/tdce_test/config.toml \
    --original_data original_samples.npy \
    --classifier_path exp/adult/classifier.pt \
    --output counterfactuals.npy \
    --target_y 1 \
    --lambda_guidance 1.0 \
    --device cuda:0 \
    --start_from_noise
```

**参数说明**：
- `--config`: 扩散模型配置文件
- `--original_data`: 原始样本文件（.npy格式）
- `--classifier_path`: 训练好的分类器路径
- `--output`: 输出反事实样本文件
- `--target_y`: 目标标签（例如：0→1翻转）
- `--lambda_guidance`: 引导权重（1.0是默认值，可调整）
- `--start_from_noise`: 从完全噪声开始生成

**可选参数**：
- `--immutable_indices`: 不可变特征索引列表（例如：`--immutable_indices 0 1 2`）
- `--batch_size`: 批量大小（默认32）

#### 5.3 验证反事实样本

```python
# verify_counterfactuals.py
import numpy as np
import torch
from tdce.classifier_guidance import ClassifierWrapper
from lib.data import prepare_fast_dataloader, make_dataset
import lib

# 加载分类器验证
classifier_path = 'exp/adult/classifier.pt'
counterfactuals = np.load('counterfactuals.npy')

# 加载分类器并验证标签是否翻转
# ... 验证代码 ...
```

---

## 🔄 完整工作流程总结

```
1. 功能测试
   └─ python test_gumbel_softmax.py

2. 准备数据集
   ├─ 下载：python scripts/download_dataset.py
   └─ 预处理：python scripts/process_dataset.py --dataname adult

3. 训练扩散模型
   └─ python scripts/train.py --config config_adult.toml

4. 训练分类器
   └─ python scripts/train_classifier.py --data_path data/adult --output_path classifier.pt

5. 生成反事实样本
   └─ python scripts/sample_counterfactual.py --config ... --original_data ...
```

---

## ⚠️ 注意事项

### 1. 数据集大小
- **Adult**: ~48K样本，9个数值特征，2个分类特征
- **LCD**: ~10K样本（论文使用），5个数值特征，1个分类特征
- **GMC**: ~150K样本，9个数值特征，1个分类特征

### 2. 计算资源
- **训练扩散模型**：需要GPU，训练时间：几小时（取决于数据集）
- **训练分类器**：CPU/GPU均可，训练时间：几分钟到几十分钟
- **生成反事实**：需要GPU，生成时间：几分钟（取决于样本数）

### 3. 参数调优
- **tau_init/tau_final**: Gumbel-Softmax温度参数（默认1.0→0.3）
- **lambda_guidance**: 引导权重（建议从1.0开始，根据效果调整）
- **num_timesteps**: 扩散时间步数（默认1000，可减少到100-500加快速度）

### 4. 常见问题
- **分类器准确率低**：增加训练轮数或调整分类器结构
- **反事实样本无效**：调整lambda_guidance或增加num_timesteps
- **内存不足**：减小batch_size或减少num_timesteps

---

## 📝 快速开始（最小测试）

如果想快速验证TDCE功能，可以：

1. **跳过数据集下载**：使用现有的测试数据或生成模拟数据
2. **使用小模型**：减少隐藏层维度，减少训练步数
3. **使用小数据集**：只使用部分训练数据

**最小测试命令**：
```bash
# 1. 测试功能
python test_gumbel_softmax.py

# 2. 如果有小数据集，快速训练
python scripts/train.py --config config_small.toml  # 使用小配置

# 3. 快速训练分类器（少量轮数）
python scripts/train_classifier.py --data_path data/adult --num_epochs 10 --output_path classifier_test.pt

# 4. 生成少量反事实样本测试
python scripts/sample_counterfactual.py --config ... --batch_size 4 ...
```

