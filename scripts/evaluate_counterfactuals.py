"""
TDCE反事实评估脚本

计算TDCE论文中的评估指标：
1. 连续特征专属指标：L2距离、多样性、不稳定性
2. 分类特征专属指标：JS散度
3. 通用反事实指标：可解释性（IM1、IM2）、有效性
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib
from scripts.utils_train import make_dataset


def l2_distance(original, counterfactual, numerical_indices=None):
    """
    计算L2距离（连续特征）
    
    Args:
        original: 原始样本 (N, D)
        counterfactual: 反事实样本 (N, D)
        numerical_indices: 连续特征的索引列表
    
    Returns:
        float: L2距离均值
    """
    # 确保输入是numpy数组
    # 注意：数据可能包含字符串（分类特征），所以不能直接转换为float
    # 我们只对数值特征部分进行类型转换
    original = np.asarray(original)
    counterfactual = np.asarray(counterfactual)
    
    # 确保numerical_indices是列表
    if numerical_indices is not None:
        if isinstance(numerical_indices, (int, np.integer)):
            numerical_indices = [numerical_indices]
        numerical_indices = list(numerical_indices)
    
    if numerical_indices is not None and len(numerical_indices) > 0:
        # 确保索引是有效的
        numerical_indices = [idx for idx in numerical_indices if idx < original.shape[1] and idx < counterfactual.shape[1]]
        if len(numerical_indices) > 0:
            orig_num = original[:, numerical_indices]
            cf_num = counterfactual[:, numerical_indices]
        else:
            # 没有有效的数值特征索引
            return 0.0
    else:
        orig_num = original
        cf_num = counterfactual
    
    # 确保是二维数组
    if orig_num.ndim == 1:
        orig_num = orig_num.reshape(-1, 1)
    if cf_num.ndim == 1:
        cf_num = cf_num.reshape(-1, 1)
    
    # 确保数据类型是float（只对数值特征部分）
    try:
        orig_num = np.asarray(orig_num, dtype=np.float64)
        cf_num = np.asarray(cf_num, dtype=np.float64)
    except (ValueError, TypeError) as e:
        print(f"Warning: Cannot convert numerical features to float: {e}")
        print(f"  orig_num dtype: {orig_num.dtype}, shape: {orig_num.shape}")
        print(f"  cf_num dtype: {cf_num.dtype}, shape: {cf_num.shape}")
        print(f"  First few values of orig_num: {orig_num[0, :5] if orig_num.shape[1] >= 5 else orig_num[0]}")
        print(f"  First few values of cf_num: {cf_num[0, :5] if cf_num.shape[1] >= 5 else cf_num[0]}")
        return None
    
    # 计算L2距离
    # 确保diff是numpy数组
    diff = np.asarray(orig_num - cf_num, dtype=np.float64)
    
    # 检查是否有NaN或Inf
    if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
        print(f"Warning: NaN or Inf values in diff. Replacing with 0.")
        diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 计算L2距离
    try:
        l2_dist = np.linalg.norm(diff, axis=1)
        return float(np.mean(l2_dist))
    except Exception as e:
        print(f"Error in l2_distance calculation: {e}")
        print(f"  orig_num shape: {orig_num.shape}, dtype: {orig_num.dtype}")
        print(f"  cf_num shape: {cf_num.shape}, dtype: {cf_num.dtype}")
        print(f"  diff shape: {diff.shape}, dtype: {diff.dtype}")
        print(f"  diff type: {type(diff)}")
        raise


def diversity(counterfactual, numerical_indices=None):
    """
    计算多样性（连续特征）
    
    Args:
        counterfactual: 反事实样本 (N, D)
        numerical_indices: 连续特征的索引列表
    
    Returns:
        float: 多样性值
    """
    # 确保输入是numpy数组（可能包含字符串）
    counterfactual = np.asarray(counterfactual)
    
    # 确保numerical_indices是列表
    if numerical_indices is not None:
        if isinstance(numerical_indices, (int, np.integer)):
            numerical_indices = [numerical_indices]
        numerical_indices = list(numerical_indices)
    
    if numerical_indices is not None and len(numerical_indices) > 0:
        # 确保索引是有效的
        numerical_indices = [idx for idx in numerical_indices if idx < counterfactual.shape[1]]
        if len(numerical_indices) > 0:
            cf_num = counterfactual[:, numerical_indices]
        else:
            return 0.0
    else:
        cf_num = counterfactual
    
    # 确保是二维数组
    if cf_num.ndim == 1:
        cf_num = cf_num.reshape(-1, 1)
    
    # 确保数据类型是float（只对数值特征部分）
    try:
        cf_num = np.asarray(cf_num, dtype=np.float64)
    except (ValueError, TypeError):
        print(f"Warning: Cannot convert counterfactual numerical features to float. Skipping diversity.")
        return None
    
    # 计算两两之间的欧氏距离
    if cf_num.shape[0] < 2:
        return 0.0
    distances = pdist(cf_num, metric='euclidean')
    return float(np.mean(distances))


def instability(original, counterfactual, y_original, numerical_indices=None):
    """
    计算不稳定性（连续特征）
    
    Args:
        original: 原始样本 (N, D)
        counterfactual: 反事实样本 (N, D)
        y_original: 原始标签 (N,)
        numerical_indices: 连续特征的索引列表
    
    Returns:
        float: 不稳定性值
    """
    # 确保输入是numpy数组（可能包含字符串）
    original = np.asarray(original)
    counterfactual = np.asarray(counterfactual)
    y_original = np.asarray(y_original)
    
    # 确保numerical_indices是列表
    if numerical_indices is not None:
        if isinstance(numerical_indices, (int, np.integer)):
            numerical_indices = [numerical_indices]
        numerical_indices = list(numerical_indices)
    
    if numerical_indices is not None and len(numerical_indices) > 0:
        # 确保索引是有效的
        numerical_indices = [idx for idx in numerical_indices if idx < original.shape[1] and idx < counterfactual.shape[1]]
        if len(numerical_indices) > 0:
            orig_num = original[:, numerical_indices]
            cf_num = counterfactual[:, numerical_indices]
        else:
            return 0.0
    else:
        orig_num = original
        cf_num = counterfactual
    
    # 确保是二维数组
    if orig_num.ndim == 1:
        orig_num = orig_num.reshape(-1, 1)
    if cf_num.ndim == 1:
        cf_num = cf_num.reshape(-1, 1)
    
    # 确保数据类型是float（只对数值特征部分）
    try:
        orig_num = np.asarray(orig_num, dtype=np.float64)
        cf_num = np.asarray(cf_num, dtype=np.float64)
    except (ValueError, TypeError):
        print(f"Warning: Cannot convert numerical features to float. Skipping instability.")
        return None
    
    instability_values = []
    
    for i in range(len(original)):
        # 找到与原始样本标签相同且最接近的样本
        same_label_mask = (y_original == y_original[i])
        if not np.any(same_label_mask):
            continue
        
        same_label_indices = np.where(same_label_mask)[0]
        same_label_indices = same_label_indices[same_label_indices != i]
        
        if len(same_label_indices) == 0:
            continue
        
        # 计算距离
        distances = np.linalg.norm(orig_num[same_label_indices] - orig_num[i], axis=1)
        closest_idx = same_label_indices[np.argmin(distances)]
        
        # 计算反事实样本的距离
        cf_distance = np.linalg.norm(cf_num[closest_idx] - cf_num[i])
        orig_distance = np.linalg.norm(orig_num[closest_idx] - orig_num[i])
        
        instability_values.append(cf_distance / (1 + orig_distance))
    
    return np.mean(instability_values) if instability_values else 0.0


def js_divergence(counterfactual, target_class_data, categorical_indices=None):
    """
    计算JS散度（分类特征）
    
    Args:
        counterfactual: 反事实样本 (N, D)
        target_class_data: 目标类样本 (M, D)
        categorical_indices: 分类特征的索引列表
    
    Returns:
        float: JS散度均值
    """
    if categorical_indices is None or len(categorical_indices) == 0:
        return 0.0
    
    if target_class_data is None or len(target_class_data) == 0:
        return 0.0
    
    js_values = []
    
    for idx in categorical_indices:
        # 获取分类特征的值
        cf_cat = counterfactual[:, idx].astype(int)
        target_cat = target_class_data[:, idx].astype(int)
        
        # 计算分布
        cf_values, cf_counts = np.unique(cf_cat, return_counts=True)
        target_values, target_counts = np.unique(target_cat, return_counts=True)
        
        # 创建统一的概率分布
        all_values = np.unique(np.concatenate([cf_values, target_values]))
        cf_probs = np.zeros(len(all_values))
        target_probs = np.zeros(len(all_values))
        
        for i, val in enumerate(all_values):
            if val in cf_values:
                cf_probs[i] = cf_counts[np.where(cf_values == val)[0][0]] / len(cf_cat)
            if val in target_values:
                target_probs[i] = target_counts[np.where(target_values == val)[0][0]] / len(target_cat)
        
        # 归一化（确保概率和为1）
        cf_probs = cf_probs / (cf_probs.sum() + 1e-10)
        target_probs = target_probs / (target_probs.sum() + 1e-10)
        
        # 计算JS散度
        # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5 * (P + Q)
        m = 0.5 * (cf_probs + target_probs)
        m = m / (m.sum() + 1e-10)  # 归一化
        
        # 避免log(0)
        epsilon = 1e-10
        cf_probs = cf_probs + epsilon
        target_probs = target_probs + epsilon
        m = m + epsilon
        
        # 归一化
        cf_probs = cf_probs / cf_probs.sum()
        target_probs = target_probs / target_probs.sum()
        m = m / m.sum()
        
        # 计算KL散度
        kl_pm = np.sum(cf_probs * np.log(cf_probs / m))
        kl_qm = np.sum(target_probs * np.log(target_probs / m))
        js = 0.5 * kl_pm + 0.5 * kl_qm
        
        js_values.append(js)
    
    return np.mean(js_values) if js_values else 0.0


def validity(counterfactual, classifier, target_y, device='cpu', transform_fn=None):
    """
    计算有效性（通用）
    
    Args:
        counterfactual: 反事实样本 (N, D) - 原始数据空间
        classifier: 分类器模型
        target_y: 目标标签
        device: 设备
        transform_fn: 数据变换函数（将原始数据转换为分类器输入）
    
    Returns:
        float: 有效性值（0-1之间）
    """
    classifier.eval()
    
    # 如果提供了变换函数，先变换数据
    if transform_fn is not None:
        cf_transformed = transform_fn(counterfactual)
    else:
        cf_transformed = counterfactual
    
    with torch.no_grad():
        cf_tensor = torch.from_numpy(cf_transformed).float().to(device)
        logits = classifier(cf_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
    # 计算预测为目标标签的比例
    valid = (predictions == target_y).sum()
    return valid / len(counterfactual)


def im1_im2(counterfactual, ae_o, ae_t, ae, device='cpu'):
    """
    计算可解释性指标IM1和IM2（通用）
    
    注意：需要预训练的自编码器AE_o、AE_t、AE
    
    Args:
        counterfactual: 反事实样本 (N, D)
        ae_o: 仅用原始类样本训练的自编码器
        ae_t: 仅用目标类样本训练的自编码器
        ae: 用全量数据集训练的自编码器
        device: 设备
    
    Returns:
        tuple: (IM1, IM2)
    """
    if ae_o is None or ae_t is None or ae is None:
        return None, None
    
    cf_tensor = torch.from_numpy(counterfactual).float().to(device)
    
    # 计算重构误差
    with torch.no_grad():
        recon_o = ae_o(cf_tensor)
        recon_t = ae_t(cf_tensor)
        recon = ae(cf_tensor)
        
        error_o = torch.norm(cf_tensor - recon_o, dim=1) ** 2
        error_t = torch.norm(cf_tensor - recon_t, dim=1) ** 2
        error_diff = torch.norm(recon_t - recon, dim=1) ** 2
        norm_cf = torch.norm(cf_tensor, dim=1, p=1)
    
    # IM1
    epsilon = 1e-10
    im1 = torch.mean(error_t / (error_o + epsilon)).item()
    
    # IM2
    im2 = torch.mean(error_diff / (norm_cf + epsilon)).item()
    
    return im1, im2


def evaluate_counterfactuals(
    original_samples,
    counterfactual_samples,
    classifier,
    target_y,
    y_original=None,
    target_class_data=None,
    numerical_indices=None,
    categorical_indices=None,
    ae_o=None,
    ae_t=None,
    ae=None,
    device='cpu',
    transform_fn=None
):
    """
    评估反事实样本
    
    Args:
        original_samples: 原始样本 (N, D)
        counterfactual_samples: 反事实样本 (N, D)
        classifier: 分类器模型
        target_y: 目标标签
        y_original: 原始标签 (N,)
        target_class_data: 目标类样本 (M, D)
        numerical_indices: 连续特征的索引列表
        categorical_indices: 分类特征的索引列表
        ae_o: 仅用原始类样本训练的自编码器
        ae_t: 仅用目标类样本训练的自编码器
        ae: 用全量数据集训练的自编码器
        device: 设备
    
    Returns:
        dict: 评估指标结果
    """
    results = {}
    
    # 1. L2距离（连续特征）
    if numerical_indices is not None and len(numerical_indices) > 0:
        try:
            results['L2_distance'] = l2_distance(original_samples, counterfactual_samples, numerical_indices)
            results['Diversity'] = diversity(counterfactual_samples, numerical_indices)
            if y_original is not None and len(y_original) > 0:
                results['Instability'] = instability(original_samples, counterfactual_samples, y_original, numerical_indices)
        except Exception as e:
            print(f"Warning: Error calculating numerical metrics: {e}")
            print(f"  numerical_indices: {numerical_indices}")
            print(f"  original_samples shape: {original_samples.shape}")
            print(f"  counterfactual_samples shape: {counterfactual_samples.shape}")
            if 'L2_distance' not in results:
                results['L2_distance'] = None
            if 'Diversity' not in results:
                results['Diversity'] = None
            if 'Instability' not in results:
                results['Instability'] = None
    
    # 2. JS散度（分类特征）
    if categorical_indices is not None and target_class_data is not None:
        results['JS_divergence'] = js_divergence(counterfactual_samples, target_class_data, categorical_indices)
    
    # 3. 有效性（通用）
    # 注意：counterfactual_samples是原始数据空间，但分类器期望变换后的数据
    # 需要将反事实样本转换回分类器输入空间
    if classifier is not None:
        # 有效性计算需要transform_fn，如果未提供则跳过
        if transform_fn is not None:
            results['Validity'] = validity(counterfactual_samples, classifier, target_y, device, transform_fn)
        else:
            print("Note: Validity calculation requires transform_fn to convert counterfactual samples to classifier input space.")
            results['Validity'] = None
    
    # 4. 可解释性（通用）
    if ae_o is not None and ae_t is not None and ae is not None:
        im1, im2 = im1_im2(counterfactual_samples, ae_o, ae_t, ae, device)
        results['IM1'] = im1
        results['IM2'] = im2
    
    return results


def main():
    parser = argparse.ArgumentParser(description='TDCE反事实评估脚本')
    parser.add_argument('--original_samples', type=str, required=True,
                       help='原始样本文件路径（.npy格式）')
    parser.add_argument('--counterfactual_samples', type=str, required=True,
                       help='反事实样本文件路径（.npy格式）')
    parser.add_argument('--classifier_path', type=str, required=True,
                       help='分类器模型路径（.pt格式）')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径（.toml格式）')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据集路径')
    parser.add_argument('--target_y', type=int, required=True,
                       help='目标标签')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备（cpu/cuda:0）')
    parser.add_argument('--output', type=str, default='evaluation_results.txt',
                       help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 加载数据
    print("Loading data...")
    try:
        original_samples = np.load(args.original_samples, allow_pickle=True)
        counterfactual_samples = np.load(args.counterfactual_samples, allow_pickle=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying with allow_pickle=False...")
        original_samples = np.load(args.original_samples, allow_pickle=False)
        counterfactual_samples = np.load(args.counterfactual_samples, allow_pickle=False)
    
    print(f"Original samples shape: {original_samples.shape}")
    print(f"Counterfactual samples shape: {counterfactual_samples.shape}")
    
    # 加载配置
    config = lib.load_config(args.config)
    
    # 注意：original_samples是变换后的数据空间，counterfactual_samples是原始数据空间
    # 为了评估，我们需要将original_samples也反归一化到原始数据空间
    print("Note: Original samples are in transformed data space.")
    print("      Counterfactual samples are in original data space.")
    print("      We need to inverse transform original samples for evaluation.")
    
    # 加载数据集以获取特征信息
    T = lib.Transformations(**config['train']['T'])
    dataset = make_dataset(
        args.data_path,
        T,
        num_classes=config['model_params']['num_classes'],
        is_y_cond=config['model_params'].get('is_y_cond', False),
        change_val=False
    )
    
    # 获取目标类样本（原始数据空间）
    # 需要反归一化目标类样本以便与counterfactual_samples比较
    if dataset.y is not None and 'test' in dataset.y:
        target_mask = (dataset.y['test'] == args.target_y).flatten()
        target_count = target_mask.sum()
        print(f"Target class samples in test set: {target_count}")
        
        if target_count == 0:
            print(f"Warning: No samples with target_y={args.target_y} in test set. JS divergence will be skipped.")
            target_class_data = None
        elif dataset.X_num is not None:
            target_class_data_transformed = dataset.X_num['test'][target_mask]
            # 反归一化目标类样本到原始数据空间
            if dataset.num_transform is not None:
                # 分离target、数值特征、分类特征（one-hot）
                is_y_cond = config['model_params'].get('is_y_cond', False)
                num_numerical_features_cfg = config.get('num_numerical_features', 4)
                
                if is_y_cond:
                    # 第一列是target，需要移除
                    target_class_data_transformed = target_class_data_transformed[:, 1:]
                
                # 分离数值特征和分类特征（one-hot）
                num_features_in_data = target_class_data_transformed.shape[1]
                if target_class_data_transformed.shape[0] == 0:
                    # 目标类数据为空
                    target_class_data = None
                elif num_features_in_data > num_numerical_features_cfg:
                    # 有分类特征（one-hot）
                    num_only = target_class_data_transformed[:, :num_numerical_features_cfg]
                    cat_ohe = target_class_data_transformed[:, num_numerical_features_cfg:]
                    
                    # 反归一化数值特征
                    if num_only.shape[0] > 0 and num_only.shape[1] > 0:
                        num_inv = dataset.num_transform.inverse_transform(num_only)
                    else:
                        num_inv = num_only
                    
                    # 反归一化分类特征
                    if dataset.cat_transform is not None and cat_ohe.shape[1] > 0 and cat_ohe.shape[0] > 0:
                        from scripts.sample import to_good_ohe
                        cat_ohe_fixed = to_good_ohe(dataset.cat_transform.steps[0][1], cat_ohe)
                        
                        # 检查维度是否匹配
                        ohe = dataset.cat_transform.steps[0][1]
                        if hasattr(ohe, 'categories_'):
                            expected_cat_ohe_cols = sum(len(cats) for cats in ohe.categories_)
                        elif hasattr(ohe, '_n_features_outs'):
                            expected_cat_ohe_cols = sum(ohe._n_features_outs)
                        elif hasattr(ohe, 'n_features_out_'):
                            expected_cat_ohe_cols = ohe.n_features_out_
                        else:
                            expected_cat_ohe_cols = cat_ohe_fixed.shape[1]
                        
                        # 如果维度不匹配，调整
                        if cat_ohe_fixed.shape[1] != expected_cat_ohe_cols:
                            if cat_ohe_fixed.shape[1] < expected_cat_ohe_cols:
                                # 添加缺失的列（全部为0）
                                missing_cols = expected_cat_ohe_cols - cat_ohe_fixed.shape[1]
                                missing_data = np.zeros((cat_ohe_fixed.shape[0], missing_cols), dtype=cat_ohe_fixed.dtype)
                                cat_ohe_fixed = np.hstack([cat_ohe_fixed, missing_data])
                            else:
                                # 截取到期望的列数
                                cat_ohe_fixed = cat_ohe_fixed[:, :expected_cat_ohe_cols]
                        
                        cat_inv = dataset.cat_transform.inverse_transform(cat_ohe_fixed)
                        target_class_data = np.hstack([num_inv, cat_inv])
                    else:
                        target_class_data = num_inv
                else:
                    # 只有数值特征
                    if target_class_data_transformed.shape[0] > 0:
                        target_class_data = dataset.num_transform.inverse_transform(target_class_data_transformed)
                    else:
                        target_class_data = None
            else:
                target_class_data = target_class_data_transformed
        else:
            target_class_data = None
    else:
        target_class_data = None
    
    # 获取原始标签
    original_indices_path = args.original_samples.replace('.npy', '_indices.npy')
    if os.path.exists(original_indices_path):
        try:
            original_indices = np.load(original_indices_path, allow_pickle=True)
        except:
            original_indices = np.load(original_indices_path, allow_pickle=False)
        
        if dataset.y is not None and 'test' in dataset.y:
            y_original = dataset.y['test'][original_indices].flatten()
        else:
            y_original = None
    else:
        y_original = None
    
    # 反归一化original_samples到原始数据空间，以便与counterfactual_samples比较
    print("Inverse transforming original samples to original data space...")
    is_y_cond = config['model_params'].get('is_y_cond', False)
    num_numerical_features_cfg = config.get('num_numerical_features', 4)
    
    if is_y_cond:
        # 移除target列
        original_samples_transformed = original_samples[:, 1:]
    else:
        original_samples_transformed = original_samples
    
    # 分离数值特征和分类特征（one-hot）
    num_features_in_data = original_samples_transformed.shape[1]
    if original_samples_transformed.shape[0] == 0:
        # 原始样本为空
        original_samples_original = original_samples_transformed
    elif num_features_in_data > num_numerical_features_cfg:
        # 有分类特征（one-hot）
        num_only = original_samples_transformed[:, :num_numerical_features_cfg]
        cat_ohe = original_samples_transformed[:, num_numerical_features_cfg:]
        
        # 反归一化数值特征
        if dataset.num_transform is not None and num_only.shape[0] > 0 and num_only.shape[1] > 0:
            num_inv = dataset.num_transform.inverse_transform(num_only)
        else:
            num_inv = num_only
        
        # 反归一化分类特征
        if dataset.cat_transform is not None and cat_ohe.shape[1] > 0 and cat_ohe.shape[0] > 0:
            from scripts.sample import to_good_ohe
            cat_ohe_fixed = to_good_ohe(dataset.cat_transform.steps[0][1], cat_ohe)
            
            # 检查维度是否匹配
            ohe = dataset.cat_transform.steps[0][1]
            if hasattr(ohe, 'categories_'):
                expected_cat_ohe_cols = sum(len(cats) for cats in ohe.categories_)
            elif hasattr(ohe, '_n_features_outs'):
                expected_cat_ohe_cols = sum(ohe._n_features_outs)
            elif hasattr(ohe, 'n_features_out_'):
                expected_cat_ohe_cols = ohe.n_features_out_
            else:
                expected_cat_ohe_cols = cat_ohe_fixed.shape[1]
            
            # 如果维度不匹配，调整
            if cat_ohe_fixed.shape[1] != expected_cat_ohe_cols:
                if cat_ohe_fixed.shape[1] < expected_cat_ohe_cols:
                    # 添加缺失的列（全部为0）
                    missing_cols = expected_cat_ohe_cols - cat_ohe_fixed.shape[1]
                    missing_data = np.zeros((cat_ohe_fixed.shape[0], missing_cols), dtype=cat_ohe_fixed.dtype)
                    cat_ohe_fixed = np.hstack([cat_ohe_fixed, missing_data])
                else:
                    # 截取到期望的列数
                    cat_ohe_fixed = cat_ohe_fixed[:, :expected_cat_ohe_cols]
            
            cat_inv = dataset.cat_transform.inverse_transform(cat_ohe_fixed)
            original_samples_original = np.hstack([num_inv, cat_inv])
        else:
            original_samples_original = num_inv
    else:
        # 只有数值特征
        if dataset.num_transform is not None and original_samples_transformed.shape[0] > 0:
            original_samples_original = dataset.num_transform.inverse_transform(original_samples_transformed)
        else:
            original_samples_original = original_samples_transformed
    
    print(f"Original samples (inverse transformed) shape: {original_samples_original.shape}")
    
    # 确定特征索引（原始数据空间）
    num_numerical_features = num_numerical_features_cfg
    # 确保索引在有效范围内
    max_feature_idx = min(num_numerical_features, counterfactual_samples.shape[1], original_samples_original.shape[1])
    numerical_indices = list(range(max_feature_idx))
    if counterfactual_samples.shape[1] > num_numerical_features:
        categorical_indices = list(range(num_numerical_features, counterfactual_samples.shape[1]))
    else:
        categorical_indices = []
    
    print(f"Numerical indices: {numerical_indices} (max: {max_feature_idx})")
    print(f"Categorical indices: {categorical_indices}")
    
    # 加载分类器
    print("Loading classifier...")
    # 注意：original_samples是变换后的数据空间（66维），这是分类器的输入维度
    # 需要先构建分类器模型结构，然后加载权重
    from tdce.modules import MLP, UNetClassifier
    
    # 获取分类器输入维度（从original_samples推断）
    classifier_input_dim = original_samples.shape[1]
    print(f"Classifier input dimension: {classifier_input_dim}")
    
    # 确定分类器架构类型
    classifier_model_type = config['model_params'].get('classifier_model_type', 'mlp')  # 'mlp' or 'unet'
    is_y_cond = config['model_params'].get('is_y_cond', False)
    
    # 获取分类器参数
    classifier_params = config['model_params'].get('classifier_params', {
        'd_layers': [256, 256],
        'dropout': 0.1
    })
    
    # 根据配置选择架构
    if classifier_model_type == 'unet':
        # 使用U-Net架构的分类器
        print(f"  Loading UNetClassifier (same architecture as diffusion model)")
        classifier = UNetClassifier(
            d_in=classifier_input_dim,
            num_classes=config['model_params'].get('num_classes', 0),
            is_y_cond=is_y_cond,
            rtdl_params={
                'd_layers': classifier_params.get('d_layers', [256, 256]),
                'dropout': classifier_params.get('dropout', 0.1)
            },
            dim_t=config['model_params'].get('dim_t', 128),
            num_output_classes=config['model_params'].get('num_classes', 2)
        )
    else:
        # 使用MLP架构的分类器（默认）
        print(f"  Loading MLP classifier")
        classifier = MLP.make_baseline(
            d_in=classifier_input_dim,
            d_out=config['model_params'].get('num_classes', 2),
            d_layers=classifier_params.get('d_layers', [256, 256]),
            dropout=classifier_params.get('dropout', 0.1)
        )
    
    # 加载权重
    checkpoint = torch.load(args.classifier_path, map_location=args.device)
    if isinstance(checkpoint, dict):
        # 如果是state_dict
        classifier.load_state_dict(checkpoint)
    else:
        # 如果是完整模型
        classifier = checkpoint
    
    classifier.to(args.device)
    classifier.eval()
    print(f"Classifier loaded successfully")
    
    # 创建数据变换函数：将原始数据空间的反事实样本转换为分类器输入空间
    def transform_fn(data):
        """
        将原始数据空间的反事实样本转换为分类器输入空间
        
        Args:
            data: shape (N, D_original) - 原始数据空间（12维：4数值+8分类）
        
        Returns:
            transformed: shape (N, D_transformed) - 变换后的数据空间（66维：标准化+one-hot）
        """
        data = np.asarray(data)
        batch_size = data.shape[0]
        
        # 分离数值特征和分类特征
        num_features_cfg = num_numerical_features_cfg
        if data.shape[1] > num_features_cfg:
            # 有分类特征
            num_data = data[:, :num_features_cfg]  # shape: (N, 4)
            cat_data = data[:, num_features_cfg:]  # shape: (N, 8) - 分类特征（字符串或整数）
        else:
            # 只有数值特征
            num_data = data
            cat_data = None
        
        # 1. 归一化数值特征
        if dataset.num_transform is not None and num_data.shape[1] > 0:
            num_transformed = dataset.num_transform.transform(num_data)
        else:
            num_transformed = num_data
        
        # 2. 编码分类特征（one-hot）
        if cat_data is not None and dataset.cat_transform is not None:
            # 分类特征需要转换为one-hot编码
            # 注意：cat_data可能是字符串数组，需要先转换为整数索引
            cat_transformed = dataset.cat_transform.transform(cat_data)
            
            # 如果使用one-hot编码，cat_transformed已经是one-hot向量
            # 组合数值特征和分类特征
            if num_transformed.shape[1] > 0:
                transformed = np.hstack([num_transformed, cat_transformed])
            else:
                transformed = cat_transformed
        else:
            # 没有分类特征
            transformed = num_transformed
        
        # 3. 如果使用is_y_cond，需要添加target列（第一列）
        # 注意：有效性计算时，我们需要知道反事实的目标标签
        # 但这里我们假设反事实已经满足目标标签，所以target列可以设为0或目标标签值
        # 为了简化，我们假设target列已经在数据中，或者需要添加
        # 实际上，分类器在训练时如果is_y_cond=True，第一列是target
        # 但在评估反事实时，我们通常不需要target列，因为分类器已经知道目标标签
        # 检查原始样本的维度以确定是否需要添加target列
        if is_y_cond:
            # 如果原始样本有target列（第一列），我们需要检查
            # 但反事实样本通常不包含target列，所以不需要添加
            # 分类器期望的输入应该是：如果is_y_cond=True，输入应该包含target列
            # 但为了有效性计算，我们可以使用目标标签作为target列
            # 这里我们假设分类器输入不需要target列（因为分类器已经训练好了）
            # 如果原始样本维度是66，而反事实是12，说明原始样本有target列（1）+ 数值(4) + one-hot(61) = 66
            # 反事实样本是 数值(4) + 分类(8) = 12
            # 转换后应该是 数值标准化(4) + one-hot(61) = 65，但原始样本是66，说明有target列
            # 为了匹配分类器输入，我们需要添加target列（使用目标标签）
            if transformed.shape[1] == original_samples.shape[1] - 1:
                # 缺少target列，添加目标标签列
                target_col = np.full((batch_size, 1), args.target_y, dtype=transformed.dtype)
                transformed = np.hstack([target_col, transformed])
        
        return transformed
    
    # 评估
    print("Evaluating counterfactuals...")
    print("Note: Counterfactual samples are in original data space.")
    print("      For validity calculation, transforming them to classifier input space.")
    
    results = evaluate_counterfactuals(
        original_samples=original_samples_original,  # 使用反归一化后的原始样本
        counterfactual_samples=counterfactual_samples,
        classifier=classifier,
        target_y=args.target_y,
        y_original=y_original,
        target_class_data=target_class_data,
        numerical_indices=numerical_indices,
        categorical_indices=categorical_indices,
        ae_o=None,  # 需要预训练
        ae_t=None,  # 需要预训练
        ae=None,   # 需要预训练
        device=args.device,
        transform_fn=transform_fn  # 传入变换函数
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for metric, value in results.items():
        if value is not None:
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: N/A (not calculated)")
    print("="*50)
    
    # 保存结果
    with open(args.output, 'w') as f:
        f.write("TDCE Counterfactual Evaluation Results\n")
        f.write("="*50 + "\n")
        for metric, value in results.items():
            if value is not None:
                f.write(f"{metric}: {value:.6f}\n")
            else:
                f.write(f"{metric}: N/A (not calculated)\n")
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

