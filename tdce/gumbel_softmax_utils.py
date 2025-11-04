"""
TDCE: Gumbel-Softmax工具函数
用于将分类特征的多项式扩散替换为Gumbel-Softmax扩散
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple


def gumbel_softmax_relaxation(logits: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
    """
    将分类特征的logits转换为Gumbel-Softmax连续向量
    
    Args:
        logits: shape (batch_size, num_classes) - 每个分类特征的logits
        tau: 温度参数，控制Gumbel-Softmax的平滑程度
        hard: 是否使用hard模式（可微分的one-hot近似）
    
    Returns:
        gumbel_softmax_sample: shape (batch_size, num_classes) - 连续向量
    """
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


def temperature_scheduler(
    timestep: Union[int, torch.Tensor],
    tau_init: float = 1.0,
    tau_final: float = 0.3,
    num_timesteps: int = 1000
) -> float:
    """
    温度退火策略：训练初期用较高tau，后期逐步降低
    
    公式：tau = tau_init * (tau_final / tau_init) ^ (t / T)
    
    Args:
        timestep: 当前时间步
        tau_init: 初始温度（默认1.0）
        tau_final: 最终温度（默认0.3，根据数据集调整）
        num_timesteps: 总时间步数
    
    Returns:
        tau: 当前时间步对应的温度值
    """
    if isinstance(timestep, torch.Tensor):
        progress = timestep.float() / num_timesteps
    else:
        progress = float(timestep) / num_timesteps
    
    tau = tau_init * (tau_final / tau_init) ** progress
    return tau


def extract(beta_schedule: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    从beta_schedule中提取对应时间步t的beta值
    
    Args:
        beta_schedule: shape (num_timesteps,) - beta调度数组
        t: shape (batch_size,) - 时间步索引
        shape: 目标形状，用于broadcast
    
    Returns:
        beta_t: shape compatible with input shape - 提取的beta值
    """
    # 对于一维张量，使用索引方式更直接
    out = beta_schedule[t.cpu()]
    return out.reshape(t.shape + (1,) * (len(shape) - len(t.shape))).to(t.device)


def gumbel_softmax_q_sample(
    x_cat_onehot: torch.Tensor,
    t: torch.Tensor,
    beta_schedule: torch.Tensor,
    tau: Union[float, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """
    Gumbel-Softmax前向扩散：q(x_t | x_{t-1})
    
    公式：q(\\tilde{x}_t | \\tilde{x}_{t-1}) ~ GS(\\overline{\\pi} = (1-\\beta_t)\\tilde{x}_{t-1} + \\beta_t/K)
    
    其中：
    - \\overline{\\pi} 是混合概率分布
    - K 是每个分类特征的类别数
    - \\beta_t 是时间步t的扩散系数
    
    Args:
        x_cat_onehot: shape (batch_size, num_cat_features, num_classes_per_feat)
                     - one-hot编码的分类特征
                     - 注意：这里假设所有分类特征有相同数量的类别（需要根据实际情况调整）
        t: shape (batch_size,) - 时间步索引
        beta_schedule: shape (num_timesteps,) - beta调度数组
        tau: 温度参数（可以是标量或tensor）
        device: 设备
    
    Returns:
        x_t_cat: shape (batch_size, num_cat_features, num_classes_per_feat) - Gumbel-Softmax连续向量
    """
    # 1. 提取当前时间步的beta值
    beta_t = extract(beta_schedule, t, x_cat_onehot.shape)
    
    # 2. 计算alpha_t = 1 - beta_t
    alpha_t = 1.0 - beta_t
    
    # 3. 获取类别数K（每个分类特征的类别数）
    K = x_cat_onehot.shape[-1]
    
    # 4. 计算混合概率 \\overline{\\pi} = (1-\\beta_t)\\tilde{x}_{t-1} + \\beta_t/K
    # 这里 \\tilde{x}_{t-1} 就是输入的 x_cat_onehot（如果是前向过程的第一个时间步）
    # alpha_t和beta_t已经是(batch_size, 1, 1)形状，可以直接广播到(batch_size, num_cat_features, num_classes_per_feat)
    pi_bar = alpha_t * x_cat_onehot + (beta_t / K)
    
    # 5. 转换为logits并应用Gumbel-Softmax
    # 为了避免数值不稳定，添加小的epsilon
    logits = torch.log(pi_bar + 1e-8)
    
    # 6. 应用Gumbel-Softmax重参数化
    if isinstance(tau, torch.Tensor):
        tau_value = tau.item() if tau.numel() == 1 else tau
    else:
        tau_value = tau
    
    x_t_cat = gumbel_softmax_relaxation(logits, tau=tau_value, hard=False)
    
    return x_t_cat


def gumbel_softmax_p_sample_logits(
    model_out_cat: torch.Tensor,
    x_t_cat: torch.Tensor,
    t: torch.Tensor,
    num_classes: list,
    tau: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    从模型输出计算Gumbel-Softmax反向采样的logits
    
    公式：logits = model_out_cat（模型预测的logits）
    
    Args:
        model_out_cat: shape (batch_size, sum(num_classes)) - 模型输出的分类特征logits（所有特征拼接）
        x_t_cat: shape (batch_size, num_cat_features, max(num_classes)) - 当前时间步的分类特征
        t: shape (batch_size,) - 时间步索引
        num_classes: list - 每个分类特征的类别数列表
        tau: 温度参数
    
    Returns:
        logits_per_feat: list of tensors - 每个分类特征的logits列表
    """
    batch_size = model_out_cat.shape[0]
    logits_per_feat = []
    
    offset = 0
    for i, num_class in enumerate(num_classes):
        # 从模型输出中提取当前分类特征的logits
        logits_feat = model_out_cat[:, offset:offset + num_class]
        logits_per_feat.append(logits_feat)
        offset += num_class
    
    return logits_per_feat


def index_to_onehot(x: torch.Tensor, num_classes: list) -> torch.Tensor:
    """
    将分类特征的索引编码转换为one-hot向量
    
    Args:
        x: shape (batch_size, num_cat_features) - 分类特征的索引编码
        num_classes: list - 每个分类特征的类别数列表
    
    Returns:
        x_onehot: shape (batch_size, num_cat_features, max(num_classes)) - one-hot编码
    """
    batch_size = x.shape[0]
    num_features = x.shape[1]
    max_classes = max(num_classes)
    
    x_onehot = torch.zeros(batch_size, num_features, max_classes, device=x.device, dtype=x.dtype)
    
    for i in range(num_features):
        num_class = num_classes[i]
        # 确保索引值在有效范围内 [0, num_class)
        x_i = torch.clamp(x[:, i].long(), 0, num_class - 1)
        x_onehot[:, i, :num_class] = F.one_hot(x_i, num_classes=num_class).float()
    
    return x_onehot


def gumbel_softmax_to_index(x_gumbel: torch.Tensor, num_classes: list) -> torch.Tensor:
    """
    将Gumbel-Softmax连续向量转换回分类特征索引
    
    Args:
        x_gumbel: shape (batch_size, num_cat_features, max(num_classes)) - Gumbel-Softmax向量
        num_classes: list - 每个分类特征的类别数列表
    
    Returns:
        x_index: shape (batch_size, num_cat_features) - 分类特征索引
    """
    batch_size = x_gumbel.shape[0]
    num_features = x_gumbel.shape[1]
    
    x_index = torch.zeros(batch_size, num_features, device=x_gumbel.device, dtype=torch.long)
    
    for i in range(num_features):
        num_class = num_classes[i]
        # 取每个特征的最大值索引
        x_index[:, i] = torch.argmax(x_gumbel[:, i, :num_class], dim=-1)
    
    return x_index

