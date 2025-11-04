"""
TDCE: 分类器梯度引导模块
用于在反向采样过程中通过分类器梯度引导生成反事实样本
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List


class ClassifierWrapper(torch.nn.Module):
    """
    包装分类器，用于计算梯度引导
    
    Args:
        classifier: 可微分分类器（可以是MLP或复用去噪网络）
        num_numerical_features: 数值特征数量
        num_classes: 分类特征类别数数组（list）
    """
    
    def __init__(
        self, 
        classifier: torch.nn.Module, 
        num_numerical_features: int, 
        num_classes: List[int]
    ):
        super().__init__()
        self.classifier = classifier
        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes
    
    def forward(
        self, 
        x_num: torch.Tensor, 
        x_cat_gumbel: Optional[torch.Tensor], 
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_num: shape (batch_size, num_numerical_features) - 数值特征
            x_cat_gumbel: shape (batch_size, num_cat_features, max(num_classes)) 
                         - Gumbel-Softmax连续向量（可选，如果使用one-hot编码则为None）
            return_logits: 是否返回logits（False则返回概率）
        
        Returns:
            logits或概率，shape (batch_size, num_output_classes)
        """
        # 如果使用one-hot编码，分类特征已经合并到数值特征中
        if x_cat_gumbel is None or len(self.num_classes) == 0 or (len(self.num_classes) == 1 and self.num_classes[0] == 0):
            # 没有分类特征或分类特征已合并到数值特征中
            # 直接使用数值特征（实际上包含所有特征）
            x = x_num
        else:
            from .gumbel_softmax_utils import gumbel_softmax_to_index, index_to_onehot
            
            batch_size = x_cat_gumbel.shape[0]
            num_cat_features = len(self.num_classes)
            
            # 方法1：将Gumbel-Softmax向量转换为离散索引，再转为one-hot（便于分类器处理）
            # 注意：这会丢失梯度信息，不适合梯度引导
            # x_cat_index = gumbel_softmax_to_index(x_cat_gumbel, self.num_classes)
            # x_cat_onehot = index_to_onehot(x_cat_index, self.num_classes)
            
            # 方法2：直接使用Gumbel-Softmax连续向量（保持梯度）
            # 需要将(batch_size, num_cat_features, max(num_classes))展平为(batch_size, sum(num_classes))
            x_cat_flat = []
            for i, num_class in enumerate(self.num_classes):
                x_cat_flat.append(x_cat_gumbel[:, i, :num_class])
            x_cat_flat = torch.cat(x_cat_flat, dim=1)  # shape: (batch_size, sum(num_classes))
            
            # 组合数值特征和分类特征
            if x_num.shape[1] > 0:
                x = torch.cat([x_num, x_cat_flat], dim=1)
            else:
                x = x_cat_flat
        
        # 分类器前向传播
        logits = self.classifier(x)
        
        if return_logits:
            return logits
        else:
            return F.softmax(logits, dim=1)


def compute_classifier_gradient(
    classifier: torch.nn.Module,
    x_num: Optional[torch.Tensor],
    x_cat_gumbel: torch.Tensor,
    target_y: Union[int, torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算分类器梯度：∇_x log p_φ(y_target | x)
    
    公式：g_classifier = ∇_x log p_φ(y_target | x_t)
    
    Args:
        classifier: 分类器模型（或ClassifierWrapper）
        x_num: shape (batch_size, num_numerical_features) - 数值特征，可选
        x_cat_gumbel: shape (batch_size, num_cat_features, max(num_classes)) 
                     - 分类特征（Gumbel-Softmax连续向量）
        target_y: 目标类别（标量或shape (batch_size,)）
        device: 设备
    
    Returns:
        g_num: shape (batch_size, num_numerical_features) - 数值特征梯度
        g_cat: shape (batch_size, num_cat_features, max(num_classes)) - 分类特征梯度
    """
    # 1. 设置requires_grad
    if x_num is not None:
        x_num = x_num.clone().detach().requires_grad_(True)
    if x_cat_gumbel is not None:
        x_cat_gumbel = x_cat_gumbel.clone().detach().requires_grad_(True)
    
    # 2. 前向传播
    if isinstance(classifier, ClassifierWrapper):
        logits = classifier(x_num, x_cat_gumbel, return_logits=True)
    else:
        # 如果直接传入分类器，需要手动处理输入
        # 注意：如果使用one-hot编码，x_cat_gumbel可能为None
        if x_cat_gumbel is not None:
            # 直接展平Gumbel-Softmax向量（保持梯度）
            # 注意：这里假设分类器可以接受连续向量输入
            # 如果分类器需要离散输入，应该在调用前转换为one-hot
            batch_size = x_cat_gumbel.shape[0]
            num_cat_features = x_cat_gumbel.shape[1]
            
            # 获取num_classes（如果可用）
            if hasattr(classifier, 'num_classes'):
                num_classes = classifier.num_classes
            else:
                # 如果无法获取num_classes，尝试从x_cat_gumbel推断
                # 这里假设最后一个维度是max(num_classes)
                max_classes = x_cat_gumbel.shape[2]
                # 这是一个启发式方法：假设每个特征都有max_classes个类别
                # 实际使用时应明确传入num_classes
                num_classes = [max_classes] * num_cat_features
            
            x_cat_flat = []
            for i, num_class in enumerate(num_classes):
                if i < num_cat_features:
                    x_cat_flat.append(x_cat_gumbel[:, i, :num_class])
            if x_cat_flat:
                x_cat_flat = torch.cat(x_cat_flat, dim=1)
            else:
                # 如果无法推断，直接展平所有维度（保留梯度）
                x_cat_flat = x_cat_gumbel.view(batch_size, -1)
            
            if x_num is not None:
                x = torch.cat([x_num, x_cat_flat], dim=1)
            else:
                x = x_cat_flat
        else:
            # 没有分类特征，只使用数值特征
            x = x_num
        
        logits = classifier(x)
    
    # 3. 计算目标类的log概率
    log_probs = F.log_softmax(logits, dim=1)
    
    # 处理target_y
    if isinstance(target_y, int):
        target_y = torch.full((logits.shape[0],), target_y, device=device, dtype=torch.long)
    else:
        target_y = target_y.long().to(device)
    
    # 提取目标类的log概率
    target_log_prob = log_probs[range(logits.shape[0]), target_y]
    
    # 4. 反向传播计算梯度
    if x_num is not None:
        grad_num = torch.autograd.grad(
            target_log_prob.sum(),
            x_num,
            create_graph=True,
            retain_graph=x_cat_gumbel is not None  # 如果x_cat_gumbel为None，不需要retain_graph
        )[0]
    else:
        # 如果没有数值特征，需要从x_cat_gumbel或logits获取batch_size
        batch_size = logits.shape[0]
        grad_num = torch.zeros((batch_size, 0), device=device)
    
    if x_cat_gumbel is not None:
        grad_cat_flat = torch.autograd.grad(
            target_log_prob.sum(),
            x_cat_gumbel,
            create_graph=True,
            retain_graph=False
        )[0]  # shape: (batch_size, num_cat_features, max(num_classes))
    else:
        # 没有分类特征（已合并到数值特征中）
        batch_size = logits.shape[0]
        grad_cat_flat = torch.zeros((batch_size, 0, 0), device=device)
    
    return grad_num, grad_cat_flat


def compute_distance_gradient(
    x_original: torch.Tensor,
    x_pred: torch.Tensor
) -> torch.Tensor:
    """
    计算距离约束梯度：∇_x ||x_0 - f_dn(x_t)||^2
    
    确保反事实样本与原始样本距离最小
    
    Args:
        x_original: shape (batch_size, num_features) - 原始样本
        x_pred: shape (batch_size, num_features) - 去噪后的预测样本
    
    Returns:
        g_dist: shape (batch_size, num_features) - 距离梯度
    """
    # 计算L2距离
    distance = torch.sum((x_original - x_pred) ** 2, dim=1)  # shape: (batch_size,)
    
    # 计算梯度
    g_dist = torch.autograd.grad(
        distance.sum(),
        x_pred,
        create_graph=True,
        retain_graph=False
    )[0]
    
    return g_dist


def compute_guided_gradient(
    g_classifier: torch.Tensor,
    g_distance: torch.Tensor,
    lambda_guidance: float = 1.0
) -> torch.Tensor:
    """
    归一化组合分类器梯度和距离约束梯度
    
    公式：g_guided = (g_classifier / ||g_classifier||) - λ * (g_distance / ||g_distance||)
    
    Args:
        g_classifier: shape (batch_size, num_features) - 分类器梯度
        g_distance: shape (batch_size, num_features) - 距离约束梯度
        lambda_guidance: 引导权重（控制距离约束的强度）
    
    Returns:
        g_guided: shape (batch_size, num_features) - 归一化后的引导梯度
    """
    # 归一化分类器梯度
    g_classifier_norm = g_classifier / (
        torch.norm(g_classifier, dim=1, keepdim=True) + 1e-8
    )
    
    # 归一化距离梯度
    g_distance_norm = g_distance / (
        torch.norm(g_distance, dim=1, keepdim=True) + 1e-8
    )
    
    # 组合梯度
    # 注意：这里使用减法，因为我们希望减少距离（朝向原始样本），
    # 同时增加目标类概率（朝向目标类）
    g_guided = g_classifier_norm - lambda_guidance * g_distance_norm
    
    return g_guided


def compute_classifier_gradient_split(
    classifier: torch.nn.Module,
    x_num: Optional[torch.Tensor],
    x_cat_gumbel: torch.Tensor,
    target_y: Union[int, torch.Tensor],
    num_numerical_features: int,
    num_classes: List[int],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算分类器梯度（分离数值特征和分类特征）
    
    这是compute_classifier_gradient的变体，专门用于分离处理数值和分类特征的情况
    
    Args:
        classifier: 分类器模型
        x_num: shape (batch_size, num_numerical_features) - 数值特征，可选
        x_cat_gumbel: shape (batch_size, num_cat_features, max(num_classes)) 
                     - 分类特征（Gumbel-Softmax连续向量）
        target_y: 目标类别
        num_numerical_features: 数值特征数量
        num_classes: 分类特征类别数数组
        device: 设备
    
    Returns:
        g_num: shape (batch_size, num_numerical_features) - 数值特征梯度
        g_cat: shape (batch_size, num_cat_features, max(num_classes)) - 分类特征梯度
    """
    # 使用ClassifierWrapper确保正确的输入格式
    if not isinstance(classifier, ClassifierWrapper):
        wrapper = ClassifierWrapper(classifier, num_numerical_features, num_classes)
    else:
        wrapper = classifier
    
    return compute_classifier_gradient(
        wrapper,
        x_num,
        x_cat_gumbel,
        target_y,
        device
    )

