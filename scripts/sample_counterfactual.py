"""
TDCE: 反事实生成脚本
从原始样本生成满足目标标签的反事实样本
"""

import torch
import numpy as np
import argparse
import os
import sys

# Add parent directory to path to import tdce and lib modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib
from tdce.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tdce.classifier_guidance import ClassifierWrapper
from scripts.utils_train import get_model, make_dataset
from lib.data import create_immutable_mask_split


def load_classifier(
    classifier_path: str,
    model_params: dict,
    num_numerical_features: int,
    num_classes: list,
    device: torch.device
) -> torch.nn.Module:
    """
    加载训练好的分类器
    
    Args:
        classifier_path: 分类器模型路径
        model_params: 模型参数（包含分类器相关配置）
        num_numerical_features: 数值特征数量
        num_classes: 分类特征类别数列表
        device: 设备
    
    Returns:
        classifier: 分类器模型
    """
    from tdce.modules import MLP
    
    # 构建分类器（使用MLP）
    d_in = num_numerical_features + sum(num_classes)
    classifier_params = model_params.get('classifier_params', {
        'd_layers': [256, 256],
        'dropout': 0.1
    })
    
    classifier = MLP.make_baseline(
        d_in=d_in,
        d_out=model_params.get('num_classes', 2),  # 分类任务的类别数
        d_layers=classifier_params.get('d_layers', [256, 256]),
        dropout=classifier_params.get('dropout', 0.1)
    )
    
    # 加载模型权重
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    else:
        print(f"Warning: Classifier path {classifier_path} not found. Using random weights.")
    
    classifier.to(device)
    classifier.eval()
    
    # 包装为ClassifierWrapper以便使用
    classifier_wrapped = ClassifierWrapper(
        classifier,
        num_numerical_features,
        num_classes
    )
    classifier_wrapped.to(device)
    classifier_wrapped.eval()
    
    return classifier_wrapped


def sample_counterfactual(
    config_path: str,
    original_data_path: str,
    classifier_path: str,
    immutable_indices: list = None,
    target_y: int = 1,
    output_path: str = None,
    lambda_guidance: float = 1.0,
    device: str = 'cpu',
    batch_size: int = 32,
    start_from_noise: bool = True
):
    """
    生成反事实样本的主函数
    
    Args:
        config_path: 配置文件路径
        original_data_path: 原始样本文件路径（numpy数组）
        classifier_path: 分类器模型路径
        immutable_indices: 不可变特征索引列表（在总特征中的索引）
        target_y: 目标标签
        output_path: 输出文件路径
        lambda_guidance: 引导权重
        device: 设备
        batch_size: 批量大小
        start_from_noise: 是否从完全噪声开始
    
    Returns:
        counterfactuals: 反事实样本（numpy数组）
    """
    device = torch.device(device)
    
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
    if original_data_path.endswith('.npy'):
        original_data = np.load(original_data_path)  # shape: (n_samples, n_features)
    else:
        # 假设是CSV文件或其他格式
        import pandas as pd
        df = pd.read_csv(original_data_path)
        original_data = df.values
    
    original_samples = torch.from_numpy(original_data).float().to(device)
    print(f"Loaded {len(original_samples)} original samples")
    
    # 4. 加载扩散模型
    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0:
        K = np.array([0])
    
    # 注意：如果使用one-hot编码，分类特征已经合并到数值特征中
    # 所以num_numerical_features应该是转换后的维度（包含所有特征）
    num_numerical_features_original = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    
    # 检查是否使用is_y_cond
    is_y_cond = config['model_params'].get('is_y_cond', False)
    
    # 如果使用is_y_cond，原始数据的第一列是target，需要减去1
    # 但这里num_numerical_features_original是转换后的数据维度（可能包含target）
    # 所以我们需要计算原始数据（不含target）的维度
    # 注意：num_numerical_features_original是转换后的数据维度（包括target，如果is_y_cond）
    # 我们需要知道原始数据中有几个数值特征（不含target）
    # 这里我们使用配置中的num_numerical_features（如果可用）
    if is_y_cond:
        # 如果使用is_y_cond，第一列是target
        # 我们需要从配置中获取原始数值特征数，或者从数据维度推断
        # 如果配置中有num_numerical_features，使用它
        if 'num_numerical_features' in config:
            num_features_original = config['num_numerical_features']  # 原始数值特征数（不含target）
        elif 'num_numerical_features' in config.get('model_params', {}):
            num_features_original = config['model_params']['num_numerical_features']
        else:
            # 如果无法从配置获取，假设只有数值特征，没有分类特征（one-hot已合并）
            # 这种情况下，num_numerical_features_original = 1 (target) + num_features_original
            num_features_original = num_numerical_features_original - 1  # 减去target列
    else:
        num_features_original = num_numerical_features_original
    num_cat_features = len(dataset.get_category_sizes('train'))
    
    # 计算实际输入维度
    # 如果使用one-hot编码（K全为0），则所有特征都在X_num中
    if len(K) == 0 or (len(K) == 1 and K[0] == 0):
        # 使用one-hot编码，所有特征都在数值特征中
        num_numerical_features = num_numerical_features_original  # 包含所有特征（68维）
        d_in = num_numerical_features
    else:
        # 有分类特征，按原始方式计算
        num_numerical_features = num_numerical_features_original
        d_in = np.sum(K) + num_numerical_features
    
    config['model_params']['d_in'] = int(d_in)
    print(f"Diffusion model input dimension: {d_in}")
    print(f"  num_numerical_features: {num_numerical_features}")
    print(f"  category_sizes (K): {list(K)}")
    
    model = get_model(
        config['model_type'],
        config['model_params'],
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    
    # 加载模型权重
    model_path = config.get('model_path', os.path.join(os.path.dirname(config_path), 'model.pt'))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model path {model_path} not found. Using random weights.")
    
    # 创建扩散模型（使用Gumbel-Softmax）
    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        num_timesteps=config['diffusion_params'].get('num_timesteps', 1000),
        gaussian_loss_type=config['diffusion_params'].get('gaussian_loss_type', 'mse'),
        scheduler=config['diffusion_params'].get('scheduler', 'cosine'),
        device=device,
        # TDCE参数
        use_gumbel_softmax=True,
        tau_init=config.get('tau_init', 1.0),
        tau_final=config.get('tau_final', 0.3),
        tau_schedule=config.get('tau_schedule', 'anneal')
    )
    diffusion.to(device)
    diffusion.eval()
    
    # 5. 加载分类器
    # 注意：分类器是在转换后的数据上训练的（one-hot编码后）
    # 所以输入维度应该是转换后的数据维度，而不是原始特征维度
    # 从原始样本数据推断分类器的实际输入维度
    actual_input_dim = original_samples.shape[1]
    print(f"Classifier input dimension (from data): {actual_input_dim}")
    print(f"  num_numerical_features (original): {num_numerical_features}")
    print(f"  category_sizes: {list(K)}")
    
    # 如果使用one-hot编码，分类特征已经合并到数值特征中
    # 所以实际输入维度就是转换后的数据维度
    if len(K) == 0 or (len(K) == 1 and K[0] == 0):
        # 使用one-hot编码，分类特征已合并
        print(f"  Using one-hot encoding: input_dim = {actual_input_dim}")
        classifier_num_features = actual_input_dim
        classifier_category_sizes = []
    else:
        # 使用其他编码方式
        calculated_dim = num_numerical_features + sum(K)
        print(f"  Using category sizes: calculated_dim = {calculated_dim}, actual_dim = {actual_input_dim}")
        if calculated_dim != actual_input_dim:
            print(f"  Warning: Dimension mismatch. Using actual dimension from data: {actual_input_dim}")
        classifier_num_features = actual_input_dim
        classifier_category_sizes = list(K)
    
    classifier = load_classifier(
        classifier_path,
        config['model_params'],
        classifier_num_features,
        classifier_category_sizes,
        device
    )
    print(f"Loaded classifier from {classifier_path}")
    
    # 6. 创建不可变特征掩码
    immutable_mask = None
    immutable_mask_num = None
    immutable_mask_cat = None
    
    if immutable_indices:
        # 计算分类特征的数量（one-hot展开后）
        num_cat_features_expanded = int(np.sum(K))
        
        # 分离数值和分类特征的不可变索引
        immutable_indices_num = [i for i in immutable_indices if i < num_numerical_features]
        immutable_indices_cat = [i - num_numerical_features for i in immutable_indices 
                                 if i >= num_numerical_features and i < num_numerical_features + num_cat_features]
        
        # 创建分离的掩码
        immutable_mask_num, immutable_mask_cat = create_immutable_mask_split(
            immutable_indices_num,
            immutable_indices_cat,
            num_numerical_features,
            num_cat_features,
            num_classes=list(K),
            device=device
        )
        
        # 创建总掩码（用于传递给sample_counterfactual）
        total_features = num_numerical_features + num_cat_features_expanded
        immutable_mask = torch.ones(total_features, device=device)
        
        # 设置不可变特征的掩码
        for idx in immutable_indices:
            if idx < total_features:
                immutable_mask[idx] = 0.0
        
        print(f"Created immutable mask for {len(immutable_indices)} features")
    
    # 7. 获取原始标签（如果可用）
    y_original = None
    if dataset.y is not None and 'test' in dataset.y:
        # 尝试从数据集获取标签（需要匹配样本）
        # 这里简化处理，假设原始数据包含标签
        pass
    
    # 8. 生成反事实样本
    counterfactuals = []
    print(f"Generating counterfactuals for target_y={target_y}...")
    
    for i in range(0, len(original_samples), batch_size):
        batch_original = original_samples[i:i+batch_size]
        batch_size_actual = batch_original.shape[0]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(original_samples) + batch_size - 1)//batch_size}")
        
        # 创建batch级别的掩码
        batch_immutable_mask = None
        if immutable_mask is not None:
            batch_immutable_mask = immutable_mask.unsqueeze(0).expand(batch_size_actual, -1)
        
        # 调用sample_counterfactual生成反事实样本
        batch_cf = diffusion.sample_counterfactual(
            x_original=batch_original,
            y_original=y_original,  # 如果可用
            target_y=target_y,
            classifier=classifier,
            immutable_mask=batch_immutable_mask,
            lambda_guidance=lambda_guidance,
            num_steps=None,  # 使用全部时间步
            start_from_noise=start_from_noise
        )
        
        counterfactuals.append(batch_cf.cpu().numpy())
    
    counterfactuals = np.concatenate(counterfactuals, axis=0)
    
    # 9. 反归一化：将反事实样本转换回原始数据空间
    print("Inverse transforming counterfactuals to original data space...")
    
    # 检查是否使用is_y_cond（target是否包含在特征中）
    is_y_cond = config['model_params'].get('is_y_cond', False)
    
    # 分离数值和分类特征（如果使用one-hot编码，分类特征已合并到数值特征中）
    if len(K) == 0 or (len(K) == 1 and K[0] == 0):
        # 使用one-hot编码，所有特征都在数值特征中
        counterfactuals_num = counterfactuals
        counterfactuals_cat = None
    else:
        # 有分离的分类特征
        counterfactuals_num = counterfactuals[:, :num_numerical_features]
        counterfactuals_cat = counterfactuals[:, num_numerical_features:]
    
    # 反归一化数值特征
    if dataset.num_transform is not None and counterfactuals_num.shape[1] > 0:
        # 如果使用one-hot编码，分类特征已经合并到数值特征中
        # 需要处理one-hot编码的分类特征
        if len(K) == 0 or (len(K) == 1 and K[0] == 0):
            # 使用one-hot编码，需要对one-hot部分进行转换
            # 注意：counterfactuals_num包含target（第一列，如果is_y_cond）和数值特征+分类特征（one-hot）
            # 需要分离：target（如果is_y_cond）、数值特征、分类特征（one-hot）
            start_idx = 1 if is_y_cond else 0  # target列的位置
            num_features_in_data = num_numerical_features_original  # 转换后的数据维度（包括target）
            num_features_actual = num_features_in_data - (1 if is_y_cond else 0)  # 实际数值+分类特征数
            
            if num_features_original < num_features_actual:
                # 有分类特征（one-hot编码后）
                # 分离：target（如果is_y_cond）、数值特征、分类特征（one-hot）
                # 注意：实际数据中的分类特征列数可能与OneHotEncoder期望的不同
                # 使用实际数据中的列数，而不是期望的列数
                if is_y_cond:
                    cf_target = counterfactuals_num[:, 0:1]  # target列
                    cf_num_only = counterfactuals_num[:, 1:1+num_features_original]
                    # 使用实际数据中的列数（剩余的列）
                    cf_cat_ohe = counterfactuals_num[:, 1+num_features_original:]
                else:
                    cf_target = None
                    cf_num_only = counterfactuals_num[:, :num_features_original]
                    cf_cat_ohe = counterfactuals_num[:, num_features_original:]
                
                # 检查分类特征列数是否与OneHotEncoder期望的匹配
                if dataset.cat_transform is not None:
                    ohe = dataset.cat_transform.steps[0][1]
                    if hasattr(ohe, 'categories_'):
                        expected_cat_ohe_cols = sum(len(cats) for cats in ohe.categories_)
                    elif hasattr(ohe, '_n_features_outs'):
                        expected_cat_ohe_cols = sum(ohe._n_features_outs)
                    elif hasattr(ohe, 'n_features_out_'):
                        expected_cat_ohe_cols = ohe.n_features_out_
                    else:
                        expected_cat_ohe_cols = cf_cat_ohe.shape[1]
                    
                    # 如果列数不匹配，需要调整
                    if cf_cat_ohe.shape[1] != expected_cat_ohe_cols:
                        print(f"  Warning: Categorical one-hot columns mismatch: actual={cf_cat_ohe.shape[1]}, expected={expected_cat_ohe_cols}")
                        # 如果实际列数少于期望，需要填充缺失的列（通常是最少出现的类别列）
                        if cf_cat_ohe.shape[1] < expected_cat_ohe_cols:
                            # 添加缺失的列（全部为0，表示最少出现的类别）
                            missing_cols = expected_cat_ohe_cols - cf_cat_ohe.shape[1]
                            print(f"  Adding {missing_cols} missing column(s) (all zeros)")
                            missing_data = np.zeros((cf_cat_ohe.shape[0], missing_cols), dtype=cf_cat_ohe.dtype)
                            cf_cat_ohe = np.hstack([cf_cat_ohe, missing_data])
                        else:
                            # 实际列数多于期望，截取
                            print(f"  Truncating to {expected_cat_ohe_cols} columns")
                            cf_cat_ohe = cf_cat_ohe[:, :expected_cat_ohe_cols]
                
                # 反归一化数值部分
                if num_features_original > 0:
                    cf_num_inv = dataset.num_transform.inverse_transform(cf_num_only)
                else:
                    cf_num_inv = cf_num_only
                
                # 处理one-hot编码的分类特征
                if dataset.cat_transform is not None and cf_cat_ohe.shape[1] > 0:
                    # 将one-hot转换为正确的one-hot格式
                    from scripts.sample import to_good_ohe
                    cf_cat_ohe_fixed = to_good_ohe(dataset.cat_transform.steps[0][1], cf_cat_ohe)
                    cf_cat_inv = dataset.cat_transform.inverse_transform(cf_cat_ohe_fixed)
                    
                    # 合并：target（如果is_y_cond）、数值特征、分类特征
                    if cf_target is not None:
                        counterfactuals_inv = np.hstack([cf_target, cf_num_inv, cf_cat_inv])
                    else:
                        counterfactuals_inv = np.hstack([cf_num_inv, cf_cat_inv])
                else:
                    # 没有分类特征
                    if cf_target is not None:
                        counterfactuals_inv = np.hstack([cf_target, cf_num_inv])
                    else:
                        counterfactuals_inv = cf_num_inv
            else:
                # 只有数值特征（没有分类特征）
                if is_y_cond:
                    # 分离target和数值特征
                    cf_target = counterfactuals_num[:, 0:1]
                    cf_num_only = counterfactuals_num[:, 1:]
                    cf_num_inv = dataset.num_transform.inverse_transform(cf_num_only)
                    counterfactuals_inv = np.hstack([cf_target, cf_num_inv])
                else:
                    counterfactuals_inv = dataset.num_transform.inverse_transform(counterfactuals_num)
        else:
            # 没有使用one-hot编码，直接反归一化
            counterfactuals_inv = dataset.num_transform.inverse_transform(counterfactuals_num)
            
            # 反归一化分类特征
            if dataset.cat_transform is not None and counterfactuals_cat is not None:
                cf_cat_inv = dataset.cat_transform.inverse_transform(counterfactuals_cat)
                counterfactuals_inv = np.hstack([counterfactuals_inv, cf_cat_inv])
    else:
        # 没有数值特征或没有变换
        counterfactuals_inv = counterfactuals
    
    # 如果使用is_y_cond，第一个特征可能是target，需要移除
    if is_y_cond:
        # 移除target列（第一列）
        counterfactuals_inv = counterfactuals_inv[:, 1:]
        print(f"  Removed target column (is_y_cond=True), final shape: {counterfactuals_inv.shape}")
    else:
        print(f"  Final shape: {counterfactuals_inv.shape}")
    
    # 10. 保存结果
    if output_path:
        np.save(output_path, counterfactuals_inv)
        print(f"Saved {len(counterfactuals_inv)} counterfactual samples (inverse transformed) to {output_path}")
    
    return counterfactuals_inv


def main():
    parser = argparse.ArgumentParser(description='TDCE: 反事实生成脚本')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径（.toml格式）')
    parser.add_argument('--original_data', type=str, required=True,
                       help='原始样本文件路径（.npy或.csv格式）')
    parser.add_argument('--classifier_path', type=str, required=True,
                       help='分类器模型路径（.pt格式）')
    parser.add_argument('--output', type=str, default='counterfactuals.npy',
                       help='输出文件路径（默认：counterfactuals.npy）')
    parser.add_argument('--immutable_indices', type=int, nargs='+', default=None,
                       help='不可变特征的索引列表（在总特征中的索引）')
    parser.add_argument('--target_y', type=int, default=1,
                       help='目标标签（默认：1）')
    parser.add_argument('--lambda_guidance', type=float, default=1.0,
                       help='引导权重（默认：1.0）')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备（默认：cuda:0）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小（默认：32）')
    parser.add_argument('--start_from_noise', action='store_true',
                       help='从完全噪声开始生成（默认：False，从部分加噪的原始样本开始）')
    
    args = parser.parse_args()
    
    counterfactuals = sample_counterfactual(
        config_path=args.config,
        original_data_path=args.original_data,
        classifier_path=args.classifier_path,
        immutable_indices=args.immutable_indices,
        target_y=args.target_y,
        output_path=args.output,
        lambda_guidance=args.lambda_guidance,
        device=args.device,
        batch_size=args.batch_size,
        start_from_noise=args.start_from_noise
    )
    
    print(f"\n✅ Generated {len(counterfactuals)} counterfactual samples")
    print(f"   Shape: {counterfactuals.shape}")
    print(f"   Saved to: {args.output}")


if __name__ == '__main__':
    main()

