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
├── tdce/
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
├── tdce/
│   └── gaussian_multinomial_diffsuion.py  # 核心修改：扩散过程
├── lib/
│   └── data.py                           # 修改：添加不可变特征掩码处理
└── scripts/
    ├── train.py                          # 可选修改：支持分类器训练
    └── sample.py                          # 修改：添加反事实采样接口
```
## 二、详细实施步骤

### 步骤1：实现Gumbel-Softmax工具模块 ⭐⭐⭐ (最高优先级)

**新建文件：`tdce/gumbel_softmax_utils.py`**

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

**修改文件：`tdce/gaussian_multinomial_diffsuion.py`**

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

**新建文件：`tdce/classifier_guidance.py`**

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

**修改文件：`tdce/gaussian_multinomial_diffsuion.py`**

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
    from .gumbel_softmax_utils import gumbel_softmax_relaxation, temperature_scheduler
    from .classifier_guidance import compute_classifier_gradient
    
    # 1. 计算温度
    if self.tau_schedule == 'anneal':
        tau = temperature_scheduler(t[0].item(), self.tau_init, self.tau_final, self.num_timesteps)
        tau = torch.tensor(tau, device=x_t_cat_gumbel.device).float()
    else:
        tau = torch.tensor(self.tau_final, device=x_t_cat_gumbel.device).float()
    
    # 2. 从模型输出获取预测的logits（每个分类特征）
    # model_out_cat shape: (batch_size, sum(num_classes))
    # 需要按特征拆分为各个分类特征的logits
    
    logits_list = []
    offset = 0
    for i, num_class in enumerate(self.num_classes):
        logits_feat = model_out_cat[:, offset:offset+num_class]
        logits_list.append(logits_feat)
        offset += num_class
    
    # 3. 如果有分类器引导，计算梯度
    if classifier is not None and target_y is not None:
        # 计算分类器梯度（分类特征部分）
        _, g_cat = compute_classifier_gradient(
            classifier,
            None,  # x_num在连续特征分支处理
            x_t_cat_gumbel,
            target_y,
            x_t_cat_gumbel.device
        )
        
        # 一阶泰勒展开：log p_φ(y|\\tilde{x}_t) ≈ (\\tilde{x}_t - \\tilde{x}_{t+1})^T g_cat + const
        # 将梯度融入logits
        g_cat_reshaped = g_cat.view(batch_size, -1)  # 需要reshape以匹配分类特征维度
        # 调整logits：logits_adjusted = logits + lambda_guidance * g_cat
        for i in range(len(logits_list)):
            if g_cat_reshaped.shape[1] >= logits_list[i].shape[1]:
                logits_list[i] = logits_list[i] + lambda_guidance * g_cat_reshaped[:, :logits_list[i].shape[1]]
    
    # 4. 对每个分类特征应用Gumbel-Softmax采样
    x_t_minus_1_cat_list = []
    for logits_feat in logits_list:
        # 转换为概率分布
        pi_t = F.softmax(logits_feat / tau, dim=1)
        
        # 采样下一步（Gumbel-Softmax分布）
        x_t_minus_1_feat = gumbel_softmax_relaxation(logits_feat, tau=tau, hard=False)
        x_t_minus_1_cat_list.append(x_t_minus_1_feat)
    
    # 5. 拼接所有分类特征
    x_t_minus_1_cat = torch.cat(x_t_minus_1_cat_list, dim=1)
    
    # 6. 应用不可变特征掩码
    if immutable_mask_cat is not None:
        x_t_minus_1_cat = x_t_minus_1_cat * immutable_mask_cat + x_t_cat_gumbel * (1 - immutable_mask_cat)
    
    return x_t_minus_1_cat
```

#### 修改3：修改`sample`方法支持反事实生成
```python
@torch.no_grad()
def sample_counterfactual(
    self, 
    x_original,  # 原始样本
    y_original,  # 原始标签
    target_y,  # 目标标签
    classifier,
    immutable_mask=None,
    lambda_guidance=1.0,
    num_steps=None
):
    """
    反事实生成：从原始样本生成满足目标标签的反事实样本
    
    Args:
        x_original: shape (batch_size, total_features) - 原始样本
        y_original: shape (batch_size,) - 原始标签
        target_y: 目标标签（标量或shape (batch_size,)）
        classifier: 分类器模型
        immutable_mask: shape (batch_size, total_features) - 不可变特征掩码
        lambda_guidance: 引导权重
        num_steps: 反向步数（默认使用全部时间步）
    
    Returns:
        x_counterfactual: 反事实样本
    """
    if num_steps is None:
        num_steps = self.num_timesteps
    
    # 1. 拆分特征
    x_num_original = x_original[:, :self.num_numerical_features]
    x_cat_original = x_original[:, self.num_numerical_features:]
    
    # 2. 将原始样本前向扩散到t=T0（可选：部分扩散）
    T0 = num_steps - 1
    t_start = torch.full((x_original.shape[0],), T0, device=x_original.device, dtype=torch.long)
    
    # 前向扩散（加噪）
    if x_num_original.shape[1] > 0:
        noise = torch.randn_like(x_num_original)
        x_num_t = self.gaussian_q_sample(x_num_original, t_start, noise=noise)
    else:
        x_num_t = x_num_original
    
    if x_cat_original.shape[1] > 0:
        if self.use_gumbel_softmax:
            # Gumbel-Softmax前向扩散
            x_cat_onehot = index_to_onehot(x_cat_original.long(), self.num_classes)
            x_cat_t = self.q_sample_gumbel_softmax(x_cat_onehot, t_start)
        else:
            # 多项式扩散
            log_x_cat = index_to_log_onehot(x_cat_original.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_start)
            x_cat_t = log_x_cat_t  # 注意：这里可能需要转换
    else:
        x_cat_t = x_cat_original
    
    # 3. 反向去噪（带分类器引导）
    out_dict = {'y': torch.full((x_original.shape[0],), target_y, device=x_original.device, dtype=torch.long)}
    
    # 拆分不可变掩码
    if immutable_mask is not None:
        immutable_mask_num = immutable_mask[:, :self.num_numerical_features]
        immutable_mask_cat = immutable_mask[:, self.num_numerical_features:]
    else:
        immutable_mask_num = None
        immutable_mask_cat = None
    
    # 反向迭代
    for i in reversed(range(0, num_steps)):
        t = torch.full((x_original.shape[0],), i, device=x_original.device, dtype=torch.long)
        
        # 组合输入
        x_t = torch.cat([x_num_t, x_cat_t], dim=1) if x_cat_t.numel() > 0 else x_num_t
        
        # 去噪网络预测
        model_out = self._denoise_fn(x_t, t, **out_dict)
        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]
        
        # 连续特征反向采样（带引导）
        if x_num_t.shape[1] > 0:
            x_num_t_minus_1 = self.gaussian_p_sample(
                model_out_num,
                x_num_t,
                t,
                clip_denoised=False,
                classifier=classifier,
                target_y=target_y,
                x_original=x_num_original,
                immutable_mask=immutable_mask_num,
                lambda_guidance=lambda_guidance
            )["sample"]
        else:
            x_num_t_minus_1 = x_num_t
# 分类特征反向采样（带引导）
        if x_cat_t.shape[1] > 0:
            if self.use_gumbel_softmax:
                x_cat_t_minus_1 = self.p_sample_gumbel_softmax(
                    model_out_cat,
                    x_cat_t,
                    t,
                    out_dict,
                    classifier=classifier,
                    target_y=target_y,
                    x_original_cat=x_cat_original,
                    immutable_mask_cat=immutable_mask_cat,
                    lambda_guidance=lambda_guidance
                )
            else:
                log_x_cat_t_minus_1 = self.p_sample(model_out_cat, x_cat_t, t, out_dict)
                x_cat_t_minus_1 = log_x_cat_t_minus_1  # 需要转换
        else:
            x_cat_t_minus_1 = x_cat_t
        
        # 更新
        x_num_t = x_num_t_minus_1
        x_cat_t = x_cat_t_minus_1
    
    # 4. 后处理：将分类特征转换回离散形式
    if x_cat_t.shape[1] > 0:
        if self.use_gumbel_softmax:
            # 从Gumbel-Softmax连续向量转换为离散索引
            # 方法：argmax或通过one-hot阈值
            x_cat_final = gumbel_to_categorical(x_cat_t, self.num_classes)
        else:
            # 从log-one-hot转换为索引
            x_cat_final = log_onehot_to_index(x_cat_t, self.num_classes)
    else:
        x_cat_final = x_cat_t
    
    # 5. 组合最终反事实样本
    x_counterfactual = torch.cat([x_num_t, x_cat_final], dim=1) if x_cat_final.numel() > 0 else x_num_t
    
    return x_counterfactual
```

---

### 步骤5：实现不可变特征处理 ⭐⭐

**修改文件：`lib/data.py`**

#### 功能1：创建不可变特征掩码
```python
def create_immutable_mask(immutable_indices, total_features, device='cpu'):
    """
    创建不可变特征掩码
    
    Args:
        immutable_indices: list[int] - 不可变特征的索引列表
        total_features: int - 总特征数（连续+分类）
        device: 设备
    
    Returns:
        mask: shape (total_features,) - 二进制掩码，1表示可变，0表示不可变
    """
    mask = torch.ones(total_features, device=device)
    if immutable_indices:
        mask[immutable_indices] = 0.0
    return mask
    def create_immutable_mask_split(immutable_indices_num, immutable_indices_cat, 
                                num_numerical_features, num_cat_features, device='cpu'):
    """
    创建分离的不可变特征掩码（分别用于连续和分类特征）
    
    Args:
        immutable_indices_num: list[int] - 不可变连续特征索引
        immutable_indices_cat: list[int] - 不可变分类特征索引
        num_numerical_features: int - 连续特征数
        num_cat_features: int - 分类特征数（one-hot展开后）
        device: 设备
    
    Returns:
        mask_num: shape (num_numerical_features,)
        mask_cat: shape (num_cat_features,)
    """
    mask_num = torch.ones(num_numerical_features, device=device)
    mask_cat = torch.ones(num_cat_features, device=device)
    
    if immutable_indices_num:
        mask_num[immutable_indices_num] = 0.0
    if immutable_indices_cat:
        mask_cat[immutable_indices_cat] = 0.0
    
    return mask_num, mask_cat
```

**修改位置：**
- 在数据加载时添加掩码参数
- 在`gaussian_p_sample`和`p_sample_gumbel_softmax`中应用掩码

---

### 步骤6：实现反事实生成脚本 ⭐⭐⭐

**新建文件：`scripts/sample_counterfactual.py`**

```python
import torch
import numpy as np
import argparse
import lib
from tdce import GaussianMultinomialDiffusion
from scripts.utils_train import get_model, make_dataset
from lib.data import create_immutable_mask_split
def load_classifier(classifier_path, model_params, num_numerical_features, num_classes, device):
    """
    加载训练好的分类器
    """
    from tdce.modules import MLPDiffusion
    
    # 构建分类器（可以是MLP）
    d_in = num_numerical_features + sum(num_classes)
    classifier = MLP.make_baseline(
        d_in=d_in,
        d_out=model_params['num_classes'],  # 分类任务的类别数
        **model_params.get('classifier_params', {})
    )
    
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    return classifier

def sample_counterfactual(
    config_path,
    original_data_path,  # 原始样本文件路径
    classifier_path,  # 分类器模型路径
    immutable_indices=None,  # 不可变特征索引
    target_y=1,  # 目标标签
    output_path=None,
    lambda_guidance=1.0,
    device='cuda:0'
):
    """
    生成反事实样本的主函数
    """
    # 1. 加载配置
    config = lib.load_config(config_path)
    
    # 2. 加载数据集（获取特征信息）
    T = lib.Transformations(**config['train']['T'])
    dataset = make_dataset(
        config['real_data_path'],
        T,
        num_classes=config['model_params']['num_classes'],
        is_y_cond=config['model_params']['is_y_cond'],
        change_val=False
    )
    
    # 3. 加载原始样本
    original_data = np.load(original_data_path)  # shape: (n_samples, n_features)
    original_samples = torch.from_numpy(original_data).float().to(device)
     # 4. 加载扩散模型
    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0:
        K = np.array([0])
    
    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    config['model_params']['d_in'] = d_in
    
    model = get_model(
        config['model_type'],
        config['model_params'],
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    
    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        num_timesteps=config['diffusion_params']['num_timesteps'],
        gaussian_loss_type=config['diffusion_params']['gaussian_loss_type'],
        scheduler=config['diffusion_params']['scheduler'],
        device=device,
        # TDCE参数
        use_gumbel_softmax=True,
        tau_init=1.0,
        tau_final=0.3,
        tau_schedule='anneal'
    )
    diffusion.to(device)
    diffusion.eval()
    
    # 5. 加载分类器
    classifier = load_classifier(
        classifier_path,
        config['model_params'],
        num_numerical_features,
        K,
        device
    )
    # 6. 创建不可变特征掩码
    immutable_mask = None
    if immutable_indices:
        total_features = original_samples.shape[1]
        mask_num, mask_cat = create_immutable_mask_split(
            [i for i in immutable_indices if i < num_numerical_features],
            [i - num_numerical_features for i in immutable_indices if i >= num_numerical_features],
            num_numerical_features,
            int(np.sum(K)),
            device
        )
        immutable_mask = torch.cat([mask_num, mask_cat], dim=0)
    
    # 7. 生成反事实样本
    counterfactuals = []
    batch_size = 32  # 批量处理
    
    for i in range(0, len(original_samples), batch_size):
        batch_original = original_samples[i:i+batch_size]
        
        batch_cf = diffusion.sample_counterfactual(
            x_original=batch_original,
            y_original=None,  # 可以从数据中获取
            target_y=target_y,
            classifier=classifier,
            immutable_mask=immutable_mask,
            lambda_guidance=lambda_guidance
        )
        
        counterfactuals.append(batch_cf.cpu().numpy())
    
    counterfactuals = np.concatenate(counterfactuals, axis=0)
    
    # 8. 保存结果
    if output_path:
        np.save(output_path, counterfactuals)
    
    return counterfactuals
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--original_data', type=str, required=True)
    parser.add_argument('--classifier_path', type=str, required=True)
    parser.add_argument('--output', type=str, default='counterfactuals.npy')
    parser.add_argument('--immutable_indices', type=int, nargs='+', default=None)
    parser.add_argument('--target_y', type=int, default=1)
    parser.add_argument('--lambda_guidance', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    counterfactuals = sample_counterfactual(
        args.config,
        args.original_data,
        args.classifier_path,
        args.immutable_indices,
        args.target_y,
        args.output,
        args.lambda_guidance,
        args.device
    )
    
    print(f"Generated {len(counterfactuals)} counterfactual samples")
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
```

---
### 步骤7：实现分类器训练脚本 ⭐⭐

**新建文件：`scripts/train_classifier.py`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import lib
from tdce.modules import MLP
from lib.data import make_dataset
from scripts.utils_train import make_dataset as make_train_dataset

def train_classifier(
    data_path,
    model_params,
    num_numerical_features,
    num_classes,
    num_epochs=100,
    batch_size=1024,
    lr=0.001,
    device='cuda:0',
    output_path='classifier.pt'
):
    """
    训练分类器用于TDCE的梯度引导
    """
    # 1. 加载数据
    T_dict = {
        'normalization': 'quantile',
        'cat_encoding': 'one-hot',  # 分类特征需要one-hot编码
        'y_policy': 'default'
    }
    T = lib.Transformations(**T_dict)
    
    dataset = make_train_dataset(
        data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=False,  # 分类器不需要y条件
        change_val=False
    )
    # 2. 准备数据加载器
    from lib.data import TabDataset, prepare_fast_dataloader
    
    train_loader = prepare_fast_dataloader(dataset, split='train', batch_size=batch_size, shuffle=True)
    val_loader = prepare_fast_dataloader(dataset, split='val', batch_size=batch_size, shuffle=False)
    
    # 3. 构建分类器
    d_in = num_numerical_features + sum(dataset.get_category_sizes('train'))
    d_out = model_params['num_classes']
    
    classifier = MLP.make_baseline(
        d_in=d_in,
        d_out=d_out,
        **model_params.get('classifier_params', {'d_layers': [256, 256], 'dropout': 0.1})
    )
    classifier.to(device)
    
    # 4. 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x, out_dict in train_loader:
            x = x.to(device)
            y = out_dict['y'].to(device)
            
            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # 验证阶段
         classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, out_dict in val_loader:
                x = x.to(device)
                y = out_dict['y'].to(device)
                
                logits = classifier(x)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.2f}%, Val Loss={val_loss/len(val_loader):.4f}, "
              f"Val Acc={val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), output_path)
            print(f"Saved best model with val_acc={val_acc:.2f}%")
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    return classifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='classifier.pt')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
     # 这里需要根据实际情况设置model_params
    model_params = {
        'num_classes': 2,  # 二分类任务
        'classifier_params': {
            'd_layers': [256, 256],
            'dropout': 0.1
        }
    }
    
    # 需要从数据集中获取特征信息
    # 这里简化处理，实际需要根据数据集配置
    train_classifier(
        args.data_path,
        model_params,
        num_numerical_features=10,  # 需要根据实际数据集设置
        num_classes=[],  # 需要根据实际数据集设置
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_path=args.output_path
    )

if __name__ == '__main__':
    main()
```

---
## 三、实施顺序与优先级

### 阶段1：核心功能实现（最高优先级）⭐⭐⭐

1. **步骤1**：实现Gumbel-Softmax工具模块
2. **步骤2**：修改分类特征扩散为Gumbel-Softmax
3. **步骤6**：实现反事实生成脚本（基础版本，不含引导）

**目标**：能够从原始样本生成反事实样本（功能验证）

### 阶段2：引导机制（高优先级）⭐⭐⭐

4. **步骤3**：实现分类器梯度引导模块
5. **步骤4**：实现引导的反向采样

**目标**：生成的反事实样本能够翻转标签

### 阶段3：完善功能（中等优先级）⭐⭐

6. **步骤5**：实现不可变特征处理
7. **步骤7**：实现分类器训练脚本

**目标**：生成符合现实约束的反事实样本

### 阶段4：优化与测试（低优先级）⭐

8. 参数调优（τ温度、λ引导权重）
9. 评估指标实现（有效性、可解释性）
10. 性能优化（批量梯度计算、稀疏梯度）

---

## 四、关键注意事项

### 1. Gumbel-Softmax温度参数

- **初始值**：建议从`tau=1.0`开始
- **最终值**：根据数据集调整（LCD数据集最优为0.3）
- **退火策略**：训练初期用高温度，逐步降低
- **梯度稳定性**：τ过小会导致梯度消失，τ过大会导致JS散度升高

### 2. 分类器梯度计算效率

- **批量计算**：尽量批量处理样本，减少前向传播次数
- **稀疏计算**：可以每隔k步计算一次梯度，中间步骤用插值
- **梯度裁剪**：防止梯度爆炸，建议限制在[-1, 1]范围

### 3. 不可变特征掩码应用时机

- **在梯度计算前**：先将不可变特征维度置零或固定
- **在采样更新后**：通过掩码恢复不可变特征的原始值
- **在距离约束中**：排除不可变特征的距离计算

### 4. 数据格式兼容性

- **分类特征表示**：需要维护两种表示形式
  - 离散索引：用于最终输出
  - One-hot向量：用于Gumbel-Softmax转换
  - Gumbel-Softmax连续向量：用于扩散过程
- **特征拆分**：确保连续特征和分类特征正确拆分和合并

### 5. 反事实生成流程

- **初始状态**：可以从纯噪声开始，也可以从部分加噪的原始样本开始
- **有效性验证**：生成后需要验证`f(x_cf) == target_y`
- **迭代优化**：如果不满足，可以调整参数重新生成

---

## 五、测试验证

### 1. 单元测试

- Gumbel-Softmax重参数化功能
- 温度调度器
- 分类器梯度计算
- 不可变特征掩码

### 2. 集成测试

- 完整反事实生成流程
- 标签翻转有效性验证
- 距离约束检查
- 不可变特征保持验证

### 3. 评估指标

- **有效性（Validity）**：`f(x_cf) == target_y`的比例
- **L2距离**：连续特征的变化量
- **IM1/IM2**：可解释性指标
- **JS散度**：分类特征分布匹配度

---

## 六、常见问题与解决方案

### Q1: Gumbel-Softmax梯度消失

**现象**：温度τ过小时，梯度很小或为0

**解决**：
- 采用温度退火策略
- 最小温度限制（如τ_min=0.1）
- 梯度缩放或梯度累积

### Q2: 分类器梯度计算慢

**现象**：每一步都需要计算梯度，采样速度慢

**解决**：
- 批量并行计算梯度
- 每隔k=10步计算一次，中间用插值
- 使用更小的分类器模型

### Q3: 反事实样本不满足标签翻转

**现象**：生成样本的预测标签仍为原始标签

**解决**：
- 增加λ_guidance权重
- 调整温度τ
- 增加反向迭代步数
- 检查分类器是否训练充分

### Q4: 不可变特征被修改

**现象**：生成的反事实样本修改了不可变特征

**解决**：
- 确保掩码在采样后正确应用
- 检查掩码索引是否正确
- 在梯度计算时排除不可变特征

---

## 七、后续扩展
1. **多目标优化**：支持多个目标类的反事实生成
2. **多样性生成**：生成多个不同的反事实样本
3. **交互式生成**：允许用户指定可修改的特征
4. **可解释性增强**：提供特征重要性分析

---

**完成！** 以上是完整的TDCE实现计划，按照优先级逐步实施即可。