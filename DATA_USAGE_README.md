# TDCE 数据下载和预处理使用说明

## 一、脚本概述

从TABCF项目借鉴的数据下载和预处理脚本，支持TDCE所需的4个数据集中的3个：
- ✅ **Adult** - 支持（UCI下载）
- ✅ **Lending Club Dataset (LCD)** - 支持（Kaggle下载）
- ✅ **Give Me Some Credit (GMC)** - 支持（Kaggle下载）
- ❌ **LAW** - 暂不支持，需手动添加

## 二、依赖检查

### 必需依赖
脚本需要以下Python包（TDCE的requirements.txt中已包含）：
```bash
numpy
pandas
scikit-learn
```

### 额外依赖（下载Kaggle数据集）
如需下载`lending-club`和`gmc`，需要安装：
```bash
pip install kaggle==1.6.14
```

**配置Kaggle API密钥**（首次使用）：
1. 注册Kaggle账号：https://www.kaggle.com
2. 下载API密钥：`https://www.kaggle.com/<username>/account` → 点击"Create New API Token"
3. 将下载的`kaggle.json`放到：`~/.kaggle/kaggle.json`

## 三、使用方法

### 步骤1：下载数据集
```bash
cd /root/data/gq_antifact/TDCE
python scripts/download_dataset.py
```

这将下载：
- `adult` - 从UCI
- `lending-club` - 从Kaggle（需要配置API）
- `gmc` - 从Kaggle（需要配置API）

**注意**：如果只下载单个数据集，需要修改`download_dataset.py`的`__main__`部分。

### 步骤2：预处理数据集
```bash
# 处理单个数据集
python scripts/process_dataset.py --dataname adult

# 处理多个数据集
python scripts/process_dataset.py --dataname adult
python scripts/process_dataset.py --dataname lending-club
python scripts/process_dataset.py --dataname gmc
```

预处理后的数据将保存在：
```
TDCE/data/{dataset_name}/
├── X_num_train.npy      # 数值特征训练集
├── X_cat_train.npy      # 分类特征训练集
├── y_train.npy          # 标签训练集
├── X_num_test.npy       # 数值特征测试集
├── X_cat_test.npy       # 分类特征测试集
├── y_test.npy           # 标签测试集
└── info.json            # 数据集元信息（包含n_classes字段）
```

## 四、输出格式说明

### 文件格式
- **数据文件**：`.npy`格式（NumPy数组），标准表格数据格式
- **元信息**：`info.json`包含以下必需字段：
  - `task_type`: "binclass" / "multiclass" / "regression"
  - `n_classes`: 类别数量（已自动计算）
  - `train_num`: 训练集样本数
  - `test_num`: 测试集样本数
  - 其他字段：特征信息、不可变特征列表等

### 数据兼容性
✅ 完全兼容表格数据的数据加载接口：
```python
from lib.data import Dataset
dataset = Dataset.from_dir('data/adult')  # 可以直接加载
```

## 五、LAW数据集支持

LAW数据集当前未包含在脚本中，有以下选项：

### 选项1：从CARLA库获取（推荐）
LAW数据集可在CARLA库的GitHub仓库获取：
```python
# 使用CARLA库下载
from carla.data.catalog import OnlineCatalog
dataset = OnlineCatalog("law")  # 如果CARLA支持
```

### 选项2：手动添加LAW数据集支持
需要创建`TDCE/data/Info/law.json`配置文件，参考`adult.json`的结构，包含：
- 数据集路径
- 列名和类型
- 不可变特征列表
- 目标列信息

然后运行：
```bash
python scripts/process_dataset.py --dataname law
```

## 六、数据格式验证

### 验证数据集是否正确加载
```python
import numpy as np
from pathlib import Path
import json

dataset_path = Path('data/adult')

# 检查文件是否存在
assert (dataset_path / 'X_num_train.npy').exists()
assert (dataset_path / 'X_cat_train.npy').exists()
assert (dataset_path / 'y_train.npy').exists()
assert (dataset_path / 'info.json').exists()

# 检查info.json格式
with open(dataset_path / 'info.json') as f:
    info = json.load(f)
    assert 'task_type' in info
    assert 'n_classes' in info  # TDCE新增字段
    print(f"Dataset: {info['name']}")
    print(f"Task type: {info['task_type']}")
    print(f"Classes: {info['n_classes']}")
    print(f"Train samples: {info['train_num']}")
    print(f"Test samples: {info['test_num']}")
```

### 使用数据加载验证
```python
from lib.data import Dataset

# 验证TDCE可以正确加载
dataset = Dataset.from_dir('data/adult')
print(f"Features: {dataset.n_num_features} numerical, {dataset.n_cat_features} categorical")
print(f"Task type: {dataset.task_type}")
print(f"Classes: {dataset.n_classes}")
```

## 七、注意事项

### 1. 路径问题
- 脚本使用相对路径，**必须在TDCE根目录执行**
- 数据保存在`TDCE/data/`目录

### 2. Kaggle数据集下载
- 首次使用需要配置Kaggle API密钥
- 如果没有Kaggle账号，可以手动从Kaggle网站下载并解压到相应目录

### 3. 验证集处理
- 脚本只生成train和test集
- TDCE会在需要时从train集中分离出val集（使用`lib.change_val()`函数）

### 4. 不可变特征
- TABCF的配置文件（`data/Info/*.json`）已包含`immutable`字段
- 这个信息会保存到输出的`info.json`中，供TDCE使用

## 八、快速开始示例

```bash
# 1. 安装依赖（如果没有）
pip install pandas numpy scikit-learn kaggle

# 2. 配置Kaggle API（如果下载lending-club和gmc）
# 将kaggle.json放到 ~/.kaggle/kaggle.json

# 3. 下载数据集
cd /root/data/gq_antifact/TDCE
python scripts/download_dataset.py

# 4. 预处理数据集
python scripts/process_dataset.py --dataname adult
python scripts/process_dataset.py --dataname lending-club
python scripts/process_dataset.py --dataname gmc

# 5. 验证数据格式
python -c "from lib.data import Dataset; d = Dataset.from_dir('data/adult'); print(f'Loaded: {d.task_type}, {d.n_classes} classes')"
```

## 九、故障排除

### 问题1：Kaggle认证失败
**解决**：确保`~/.kaggle/kaggle.json`存在且格式正确

### 问题2：找不到配置文件
**解决**：确保`TDCE/data/Info/{dataset}.json`存在

### 问题3：数据格式不兼容
**解决**：检查输出的`info.json`是否包含`n_classes`字段（已在process_dataset.py中自动添加）

### 问题4：路径错误
**解决**：确保在TDCE根目录执行脚本，使用相对路径

