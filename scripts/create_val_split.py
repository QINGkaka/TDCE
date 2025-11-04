"""
从训练集中划分出验证集
如果val文件不存在，从train中划分出20%作为val
"""

import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

def create_val_split(data_path, val_size=0.2):
    """从训练集划分出验证集"""
    data_path = os.path.normpath(data_path)
    
    # 检查val文件是否已存在
    if os.path.exists(os.path.join(data_path, 'y_val.npy')):
        print(f"验证集文件已存在：{data_path}")
        return
    
    # 加载训练集
    print(f"从训练集划分验证集：{data_path}")
    y_train = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True)
    
    X_num_train = None
    X_cat_train = None
    if os.path.exists(os.path.join(data_path, 'X_num_train.npy')):
        X_num_train = np.load(os.path.join(data_path, 'X_num_train.npy'), allow_pickle=True)
    if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')):
        X_cat_train = np.load(os.path.join(data_path, 'X_cat_train.npy'), allow_pickle=True)
    
    # 划分数据集
    n_samples = len(y_train)
    indices = np.arange(n_samples)
    
    # 检查是否为分类任务（用于分层抽样）
    unique_labels = np.unique(y_train)
    is_classification = len(unique_labels) <= 20  # 启发式：如果标签种类少，认为是分类
    
    if is_classification and len(unique_labels) > 1:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_size, random_state=777, stratify=y_train
        )
    else:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_size, random_state=777
        )
    
    # 划分标签
    y_val = y_train[val_idx]
    y_train_new = y_train[train_idx]
    
    # 保存新的训练集
    np.save(os.path.join(data_path, 'y_train_new.npy'), y_train_new)
    print(f"原始训练集大小: {n_samples}, 新训练集大小: {len(y_train_new)}, 验证集大小: {len(y_val)}")
    
    # 划分数值特征
    if X_num_train is not None:
        X_num_val = X_num_train[val_idx]
        X_num_train_new = X_num_train[train_idx]
        # 备份原始文件
        np.save(os.path.join(data_path, 'X_num_train_backup.npy'), X_num_train)
        np.save(os.path.join(data_path, 'X_num_train_new.npy'), X_num_train_new)
        np.save(os.path.join(data_path, 'X_num_val.npy'), X_num_val)
        print(f"数值特征 - 新训练集: {X_num_train_new.shape}, 验证集: {X_num_val.shape}")
    
    # 划分分类特征
    if X_cat_train is not None:
        X_cat_val = X_cat_train[val_idx]
        X_cat_train_new = X_cat_train[train_idx]
        # 备份原始文件
        np.save(os.path.join(data_path, 'X_cat_train_backup.npy'), X_cat_train)
        np.save(os.path.join(data_path, 'X_cat_train_new.npy'), X_cat_train_new)
        np.save(os.path.join(data_path, 'X_cat_val.npy'), X_cat_val)
        print(f"分类特征 - 新训练集: {X_cat_train_new.shape}, 验证集: {X_cat_val.shape}")
    
    # 保存验证集标签
    np.save(os.path.join(data_path, 'y_val.npy'), y_val)
    
    # 替换原始训练集（可选，先只创建val）
    print(f"✅ 验证集创建完成！")
    print(f"   原始训练集大小: {n_samples}")
    print(f"   新训练集大小: {len(y_train_new)}")
    print(f"   验证集大小: {len(y_val)}")
    print(f"\n注意：原始训练集已备份为 *_backup.npy")
    print(f"如果确认无误，可以手动替换原始文件")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从训练集划分出验证集')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--val_size', type=float, default=0.2, help='验证集比例')
    args = parser.parse_args()
    
    create_val_split(args.data_path, args.val_size)

