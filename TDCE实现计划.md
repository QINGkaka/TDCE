# TDCE实现计划：从TabDDPM到TDCE的代码改造指南

## 概述
本文档详细说明如何将TabDDPM改造为TDCE（Tabular Diffusion Counterfactual Explanations），核心包括：
1. 将分类特征的多项式扩散替换为Gumbel-softmax扩散
2. 实现分类器梯度引导机制
3. 添加不可变特征处理
4. 实现反事实生成流程

---

## 一、文件结构与修改清单

### 需要新建的文件
```
TDCE/
├── tab_ddpm/
│   ├── gumbel_softmax_utils.py      # 新建：Gumbel-softmax工具函数
│   ├── classifier_guidance.py       # 新建：分类器引导模块
│   └── counterfactual_sampling.py   # 新建：反事实采样逻辑
└── scripts/
    ├── train_classifier.py           # 新建：训练分类器
    └── sample_counterfactual.py      # 新建：反事实生成脚本
```

### 需要修改的核心文件
```
TDCE/
├── tab_ddpm/
│   └── gaussian_multinomial_diffsuion.py  # 核心修改：扩散过程
├── lib/
│   └── data.py                           # 修改：添加不可变特征掩码处理
└── scripts/
    ├── train.py                          # 可选修改：支持分类器训练
    └── sample.py                          # 修改：添加反事实采样接口
```
## 二、详细实施步骤

### 步骤1：实现Gumbel-Softmax工具模块 ⭐⭐⭐ (最高优先级)

**新建文件：`tab_ddpm/gumbel_softmax_utils.py`**

#### 功能1：Gumbel-Softmax重参数化
```python
def gumbel_softmax_relaxation(logits, tau=1.0, hard=True):
    """
    将分类特征的logits转换为Gumbel-Softmax连续向量
    
    Args:
        logits: shape (batch_size, num_classes) - 每个分类特征的logits
        tau: 温度参数
        hard: 是否使用hard模式（可微分的one-hot近似）
    
    Returns:
        gumbel_softmax_sample: shape (batch_size, num_classes) - 连续向量
    """
    import torch.nn.functional as F
    return F.gumbel_softmax(logits, tau=tau, hard=hard)
```

#### 功能2：温度调度器
```python
def temperature_scheduler(timestep, tau_init=1.0, tau_final=0.3, num_timesteps=1000):
    """
    温度退火策略：训练初期用较高tau，后期逐步降低
    
    Args:
        timestep: 当前时间步
        tau_init: 初始温度（默认1.0）
        tau_final: 最终温度（默认0.3，根据数据集调整）
        num_timesteps: 总时间步数
    
    Returns:
        tau: 当前时间步对应的温度值
    """
    progress = timestep / num_timesteps
    tau = tau_init * (tau_final / tau_init) ** progress
    return tau
```
#### 功能3：分类特征前向扩散（Gumbel-Softmax版本）
```python
def gumbel_softmax_q_sample(x_cat_onehot, t, beta_schedule, tau, device):
    """
    Gumbel-Softmax前向扩散：q(x_t | x_{t-1})
    
    公式：q(\\tilde{x}_t | \\tilde{x}_{t-1}) ~ GS(\\overline{\\pi} = (1-\\beta_t)\\tilde{x}_{t-1} + \\beta_t/K)
    
    Args:
        x_cat_onehot: shape (batch_size, num_cat_features, num_classes_per_feat)
                     - one-hot编码的分类特征
        t: 时间步
        beta_schedule: beta调度数组
        tau: 温度参数
        device: 设备
    
    Returns:
        x_t_cat: Gumbel-Softmax连续向量
    """
    # 1. 计算 \\overline{\\pi} = (1-\\beta_t)\\tilde{x}_{t-1} + \\beta_t/K
    beta_t = extract(beta_schedule, t, x_cat_onehot.shape)
    alpha_t = 1.0 - beta_t
    K = x_cat_onehot.shape[-1]  # 类别数
    
    # 2. 如果x_{t-1}已经是one-hot，先转换为logits
    logits_prev = torch.log(x_cat_onehot + 1e-8)
    
    # 3. 计算混合概率 \\overline{\\pi}
    pi_bar = alpha_t.unsqueeze(-1) * x_cat_onehot + (beta_t / K).unsqueeze(-1)
    
    # 4. 转换为logits并应用Gumbel-Softmax
    logits = torch.log(pi_bar + 1e-8)
    x_t_cat = gumbel_softmax_relaxation(logits, tau=tau, hard=False)
    
    return x_t_cat
```

**修改位置：**
- `gaussian_multinomial_diffsuion.py` 中的 `q_sample()` 方法（分类特征部分）
- 替换现有的 `self.q_sample(log_x_start=log_x_cat, t=t)` 调用

---
### 步骤2：修改分类特征扩散为Gumbel-Softmax ⭐⭐⭐

**修改文件：`tab_ddpm/gaussian_multinomial_diffsuion.py`**

#### 修改1：在`__init__`中添加Gumbel-Softmax相关参数
```python
def __init__(
        self,
        num_classes: np.array,
        num_numerical_features: int,
        denoise_fn,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        gaussian_parametrization='eps',
        multinomial_loss_type='vb_stochastic',
        parametrization='x0',
        scheduler='cosine',
        device=torch.device('cpu'),
        # TDCE新增参数
        use_gumbel_softmax=True,  # 是否使用Gumbel-Softmax替代多项式扩散
        tau_init=1.0,             # 初始温度
        tau_final=0.3,            # 最终温度
        tau_schedule='anneal'     # 温度调度策略：'anneal'或'fixed'
    ):
    # ... 原有代码 ...
    
    # TDCE新增：Gumbel-Softmax相关参数
    self.use_gumbel_softmax = use_gumbel_softmax
    self.tau_init = tau_init
    self.tau_final = tau_final
    self.tau_schedule = tau_schedule
```
#### 修改2：替换`q_sample`方法（分类特征部分）
```python
def q_sample_gumbel_softmax(self, x_cat_onehot, t):
    """
    Gumbel-Softmax前向扩散（替换原有的多项式扩散）
    """
    from .gumbel_softmax_utils import gumbel_softmax_q_sample, temperature_scheduler
    
    # 计算当前时间步的温度
    if self.tau_schedule == 'anneal':
        tau = temperature_scheduler(
            t.cpu().numpy() if isinstance(t, torch.Tensor) else t,
            self.tau_init,
            self.tau_final,
            self.num_timesteps
        )
        tau = torch.tensor(tau, device=x_cat_onehot.device).float()
    else:
        tau = torch.tensor(self.tau_final, device=x_cat_onehot.device).float()
    
    # 调用Gumbel-Softmax扩散
    return gumbel_softmax_q_sample(
        x_cat_onehot,
        t,
        self.betas,
        tau,
        x_cat_onehot.device
    )
```
#### 修改3：替换`p_sample`方法（分类特征部分）
```python
def p_sample_gumbel_softmax(self, model_out_cat, x_t_cat, t, out_dict):
    """
    Gumbel-Softmax反向采样（替换原有的多项式采样）
    
    需要结合分类器梯度引导（见步骤3）
    """
    from .gumbel_softmax_utils import gumbel_softmax_relaxation, temperature_scheduler
    
    # 1. 计算温度
    if self.tau_schedule == 'anneal':
        tau = temperature_scheduler(t, self.tau_init, self.tau_final, self.num_timesteps)
        tau = torch.tensor(tau, device=x_t_cat.device).float()
    else:
        tau = torch.tensor(self.tau_final, device=x_t_cat.device).float()
    
    # 2. 从模型输出获取预测的logits（每个分类特征）
    # model_out_cat shape: (batch_size, sum(num_classes))
    # 需要按特征拆分
    
    # 3. 计算Gumbel-Softmax分布的参数
    # 4. 采样下一步
    # ...（具体实现需结合分类器引导）
```
#### 修改4：修改`mixed_loss`方法
```python
def mixed_loss(self, x, out_dict):
    # ... 原有连续特征处理代码 ...
    
    if x_cat.shape[1] > 0:
        if self.use_gumbel_softmax:
            # TDCE：使用Gumbel-Softmax扩散
            x_cat_onehot = index_to_onehot(x_cat.long(), self.num_classes)  # 转为one-hot
            x_cat_t = self.q_sample_gumbel_softmax(x_cat_onehot, t)
            # ... 计算Gumbel-Softmax损失 ...
        else:
            # TabDDPM原有：多项式扩散
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
            # ... 原有损失计算 ...
```
---

### 步骤3：实现分类器梯度引导模块 ⭐⭐⭐

**新建文件：`tab_ddpm/classifier_guidance.py`**

#### 功能1：分类器包装器
```python
class ClassifierWrapper(torch.nn.Module):
    """
    包装分类器，用于计算梯度引导
    
    Args:
        classifier: 可微分分类器（可以是MLP或复用去噪网络）
        num_numerical_features: 数值特征数量
        num_classes: 分类特征类别数数组
    """
    def __init__(self, classifier, num_numerical_features, num_classes):
        super().__init__()
        self.classifier = classifier
        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes
    
    def forward(self, x_num, x_cat_gumbel, return_logits=False):
        """
        前向传播
        
        Args:
            x_num: 数值特征
            x_cat_gumbel: Gumbel-Softmax连续向量
            return_logits: 是否返回logits
        
        Returns:
            logits或概率
        """
        # 将x_cat_gumbel转换为离散索引（用于分类器输入）
        # 或直接使用连续向量（如果分类器支持）
        # ... 实现细节 ...
        pass
```
#### 功能2：计算分类器梯度
```python
def compute_classifier_gradient(classifier, x_num, x_cat_gumbel, target_y, device):
    """
    计算分类器梯度：∇_x log p_φ(y_target | x)
    
    公式：g_classifier = ∇_x log p_φ(y_target | x_t)
    
    Args:
        classifier: 分类器模型
        x_num: 数值特征
        x_cat_gumbel: 分类特征（Gumbel-Softmax连续向量）
        target_y: 目标类别
        device: 设备
    
    Returns:
        g_num: 数值特征梯度
        g_cat: 分类特征梯度
    """
    # 1. 设置requires_grad
    x_num.requires_grad_(True)
    x_cat_gumbel.requires_grad_(True)
    
    # 2. 前向传播
    logits = classifier(x_num, x_cat_gumbel)
    
    # 3. 计算目标类的log概率
    log_probs = F.log_softmax(logits, dim=1)
    target_log_prob = log_probs[:, target_y]
    
    # 4. 反向传播
    grad_num = torch.autograd.grad(
        target_log_prob.sum(),
        x_num,
        create_graph=True,
        retain_graph=True
    )[0]
    
    grad_cat = torch.autograd.grad(
        target_log_prob.sum(),
        x_cat_gumbel,
        create_graph=True,
        retain_graph=True
    )[0]
    
    return grad_num, grad_cat
```
#### 功能3：计算距离约束梯度
```python
def compute_distance_gradient(x_original, x_pred):
    """
    计算距离约束梯度：∇_x ||x_0 - f_dn(x_t)||^2
    
    确保反事实样本与原始样本距离最小
    
    Args:
        x_original: 原始样本
        x_pred: 去噪后的预测样本
    
    Returns:
        g_dist: 距离梯度
    """
    distance = torch.sum((x_original - x_pred) ** 2, dim=1)
    g_dist = torch.autograd.grad(
        distance.sum(),
        x_pred,
        create_graph=True
    )[0]
    return g_dist
```
#### 功能4：归一化组合梯度
```python
def compute_guided_gradient(g_classifier, g_distance, lambda_guidance=1.0):
    """
    归一化组合分类器梯度和距离约束梯度
    
    公式：g_guided = (g_classifier / ||g_classifier||) - λ * (g_distance / ||g_distance||)
    
    Args:
        g_classifier: 分类器梯度
        g_distance: 距离约束梯度
        lambda_guidance: 引导权重
    
    Returns:
        g_guided: 归一化后的引导梯度
    """
    # 归一化
    g_classifier_norm = g_classifier / (torch.norm(g_classifier, dim=1, keepdim=True) + 1e-8)
    g_distance_norm = g_distance / (torch.norm(g_distance, dim=1, keepdim=True) + 1e-8)
    
    # 组合
    g_guided = g_classifier_norm - lambda_guidance * g_distance_norm
    
    return g_guided
```

**修改位置：**
- `gaussian_multinomial_diffsuion.py` 中的 `gaussian_p_sample()` 方法：添加梯度引导
- `gaussian_multinomial_diffsuion.py` 中的 `p_sample_gumbel_softmax()` 方法：添加分类特征梯度引导

---
### 步骤4：实现引导的反向采样 ⭐⭐

**修改文件：`tab_ddpm/gaussian_multinomial_diffsuion.py`**

#### 修改1：增强`gaussian_p_sample`方法
```python
def gaussian_p_sample(
    self,
    model_out,
    x,
    t,
    clip_denoised=False,
    denoised_fn=None,
    model_kwargs=None,
    # TDCE新增参数
    classifier=None,
    target_y=None,
    x_original=None,
    immutable_mask=None,
    lambda_guidance=1.0
):
    """
    引导的高斯反向采样（连续特征）
    
    公式：μ_guided = μ_θ + Σ_θ · ||μ_θ|| · g_guided
    """
    from .classifier_guidance import (
        compute_classifier_gradient,
        compute_distance_gradient,
        compute_guided_gradient
    )
    
    # 原有：计算预测的均值和方差
    out = self.gaussian_p_mean_variance(
        model_out,
        x,
        t,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        model_kwargs=model_kwargs,
    )
    
    mu_theta = out["mean"]
    sigma_theta = out["variance"]
    pred_xstart = out["pred_xstart"]
    # TDCE：如果有分类器引导，添加梯度引导项
    if classifier is not None and target_y is not None:
        # 1. 计算分类器梯度（只对数值特征）
        # 注意：这里需要x_cat_gumbel，需要从model_kwargs获取
        g_classifier_num, _ = compute_classifier_gradient(
            classifier,
            pred_xstart,  # 使用预测的x0
            None,  # x_cat_gumbel在分类特征分支处理
            target_y,
            x.device
        )
        
        # 2. 计算距离约束梯度
        if x_original is not None:
            g_distance = compute_distance_gradient(x_original, pred_xstart)
        else:
            g_distance = torch.zeros_like(g_classifier_num)
        
        # 3. 归一化组合
        g_guided = compute_guided_gradient(
            g_classifier_num,
            g_distance,
            lambda_guidance
        )
        
        # 4. 应用引导到均值
        # 公式：μ_guided = μ_θ + Σ_θ · ||μ_θ|| · g_guided
        mu_guided = mu_theta + torch.sqrt(sigma_theta).unsqueeze(-1) * torch.norm(mu_theta, dim=1, keepdim=True) * g_guided
        
        # 5. 应用不可变特征掩码
        if immutable_mask is not None:
            mu_guided = mu_guided * immutable_mask + mu_theta * (1 - immutable_mask)
    else:
        mu_guided = mu_theta
    
    # 采样
    noise = torch.randn_like(x)
    nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    sample = mu_guided + nonzero_mask * torch.sqrt(sigma_theta).unsqueeze(-1) * noise
    
    return {"sample": sample, "pred_xstart": out["pred_xstart"]}
```
#### 修改2：增强`p_sample_gumbel_softmax`方法（分类特征）
```python
def p_sample_gumbel_softmax(
    self,
    model_out_cat,
    x_t_cat_gumbel,
    t,
    out_dict,
    # TDCE新增参数
    classifier=None,
    target_y=None,
    x_original_cat=None,
    immutable_mask_cat=None,
    lambda_guidance=1.0
):
    """
    引导的Gumbel-Softmax反向采样（分类特征）
    
    使用一阶泰勒展开：log p_φ(y|\\tilde{x}_t) ≈ (\\tilde{x}_t - \\tilde{x}_{t+1})^T g_cat + const
    """
    from .gumbel_softmax_utils import gumbel_softmax_relaxation, temperature_scheduler
    from .classifier_guidance import compute_classifier_gradient