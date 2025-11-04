"""
准备反事实生成所需的原始样本数据
从测试集中选择一些样本，转换为正确的格式
"""

import numpy as np
import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib
from scripts.utils_train import make_dataset


def prepare_original_samples(
    data_path: str,
    config_path: str,
    num_samples: int = 100,
    output_path: str = 'original_samples.npy',
    split: str = 'test'
):
    """
    准备反事实生成的原始样本
    
    Args:
        data_path: 数据路径
        config_path: 配置文件路径
        num_samples: 要选择的样本数量
        output_path: 输出文件路径
        split: 数据集分割（'train', 'val', 'test'）
    """
    # 1. 加载配置
    config = lib.load_config(config_path)
    
    # 2. 加载数据集（获取转换后的数据）
    # 注意：必须使用与扩散模型训练时相同的配置
    # 使用配置文件中的is_y_cond设置（而不是固定为False）
    T = lib.Transformations(**config['train']['T'])
    dataset = make_dataset(
        data_path,
        T,
        num_classes=config['model_params']['num_classes'],
        is_y_cond=config['model_params'].get('is_y_cond', False),  # 使用配置文件中的设置
        change_val=False
    )
    
    # 3. 获取转换后的数据
    X_num = dataset.X_num[split] if dataset.X_num is not None else None
    X_cat = dataset.X_cat[split] if dataset.X_cat is not None else None
    
    # 4. 合并特征（与训练时一致）
    if X_cat is not None:
        if X_num is not None:
            X = np.concatenate([X_num, X_cat], axis=1)
        else:
            X = X_cat
    else:
        X = X_num
    
    print(f"Loaded {split} set: {X.shape}")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Total features: {X.shape[1]}")
    
    # 5. 选择样本（选择前num_samples个，或者随机选择）
    if num_samples > X.shape[0]:
        num_samples = X.shape[0]
        print(f"Warning: Requested {num_samples} samples but only {X.shape[0]} available. Using all samples.")
    
    # 随机选择样本
    indices = np.random.choice(X.shape[0], size=num_samples, replace=False)
    selected_samples = X[indices]
    
    print(f"Selected {num_samples} samples")
    
    # 6. 保存
    np.save(output_path, selected_samples)
    print(f"Saved original samples to {output_path}")
    print(f"  Shape: {selected_samples.shape}")
    
    # 7. 保存选择的索引（可选，用于后续分析）
    indices_path = output_path.replace('.npy', '_indices.npy')
    np.save(indices_path, indices)
    print(f"Saved sample indices to {indices_path}")
    
    return selected_samples, indices


def main():
    parser = argparse.ArgumentParser(description='准备反事实生成的原始样本数据')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据路径')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径（.toml格式）')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='要选择的样本数量（默认：100）')
    parser.add_argument('--output', type=str, default='original_samples.npy',
                       help='输出文件路径（默认：original_samples.npy）')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='数据集分割（默认：test）')
    parser.add_argument('--seed', type=int, default=0,
                       help='随机种子（默认：0）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    prepare_original_samples(
        data_path=args.data_path,
        config_path=args.config,
        num_samples=args.num_samples,
        output_path=args.output,
        split=args.split
    )
    
    print(f"\n✅ Original samples prepared!")


if __name__ == '__main__':
    main()

