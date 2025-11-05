#!/usr/bin/env python3
"""
测试逆变换过程是否正确
验证：正向变换 → 逆变换 = 原始数据
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib
from scripts.utils_train import make_dataset


def test_inverse_transform():
    """测试逆变换过程"""
    print("=" * 60)
    print("逆变换测试")
    print("=" * 60)
    
    # 1. 加载配置和数据
    print("\n【步骤1】加载配置和数据")
    print("-" * 60)
    
    config = lib.load_config('exp/adult/tdce_test/config.toml')
    T_dict = config['train']['T']
    T = lib.Transformations(**T_dict)
    num_classes = config['model_params']['num_classes']
    is_y_cond = config['model_params'].get('is_y_cond', False)
    
    dataset = make_dataset('data/adult', T, num_classes, is_y_cond, change_val=False)
    
    print(f"数据集信息:")
    print(f"  数值特征数量: {len(dataset.X_num['test']) if dataset.X_num else 0}")
    print(f"  分类特征数量: {len(dataset.X_cat['test']) if dataset.X_cat else 0}")
    print(f"  测试集大小: {len(dataset.y['test'])}")
    
    # 2. 获取原始数据（未转换）
    print("\n【步骤2】获取原始数据（未转换）")
    print("-" * 60)
    
    from lib.data import read_pure_data
    X_num_orig, X_cat_orig, y_orig = read_pure_data('data/adult', 'test')
    
    # 取前100个样本进行测试
    n_test = 100
    if X_num_orig is not None:
        X_num_test = X_num_orig[:n_test]
        print(f"原始数值特征 shape: {X_num_test.shape}")
        print(f"原始数值特征范围: [{X_num_test.min():.4f}, {X_num_test.max():.4f}]")
        print(f"原始数值特征均值: {X_num_test.mean(axis=0)[:4]}")
    else:
        X_num_test = None
    
    if X_cat_orig is not None:
        X_cat_test = X_cat_orig[:n_test]
        print(f"原始分类特征 shape: {X_cat_test.shape}")
    else:
        X_cat_test = None
    
    # 3. 测试数值特征的逆变换
    print("\n【步骤3】测试数值特征的逆变换")
    print("-" * 60)
    
    if dataset.num_transform is not None and X_num_test is not None:
        # 正向变换：原始数据 → 标准化数据
        X_num_transformed = dataset.num_transform.transform(X_num_test)
        print(f"标准化后数值特征 shape: {X_num_transformed.shape}")
        print(f"标准化后数值特征范围: [{X_num_transformed.min():.4f}, {X_num_transformed.max():.4f}]")
        print(f"标准化后数值特征均值: {X_num_transformed.mean(axis=0)[:4]}")
        print(f"标准化后数值特征标准差: {X_num_transformed.std(axis=0)[:4]}")
        
        # 逆变换：标准化数据 → 原始数据
        X_num_inverse = dataset.num_transform.inverse_transform(X_num_transformed)
        print(f"\n逆变换后数值特征 shape: {X_num_inverse.shape}")
        print(f"逆变换后数值特征范围: [{X_num_inverse.min():.4f}, {X_num_inverse.max():.4f}]")
        print(f"逆变换后数值特征均值: {X_num_inverse.mean(axis=0)[:4]}")
        
        # 计算误差
        error = np.abs(X_num_test - X_num_inverse)
        print(f"\n误差分析:")
        print(f"  平均绝对误差: {error.mean():.4f}")
        print(f"  最大绝对误差: {error.max():.4f}")
        print(f"  每列的平均误差: {error.mean(axis=0)[:4]}")
        print(f"  每列的最大误差: {error.max(axis=0)[:4]}")
        
        # 检查是否有异常大的误差
        large_error_threshold = 1.0
        large_errors = error > large_error_threshold
        if large_errors.any():
            print(f"\n⚠️  警告: 发现 {large_errors.sum()} 个误差 > {large_error_threshold}")
            print(f"  误差 > {large_error_threshold} 的列: {np.where(large_errors.any(axis=0))[0].tolist()}")
            print(f"  最大误差位置: {np.unravel_index(error.argmax(), error.shape)}")
        else:
            print(f"\n✅ 数值特征逆变换正确（所有误差 < {large_error_threshold}）")
        
        # 检查数值范围是否合理
        print(f"\n数值范围检查:")
        print(f"  原始数据范围: [{X_num_test.min():.4f}, {X_num_test.max():.4f}]")
        print(f"  逆变换后范围: [{X_num_inverse.min():.4f}, {X_num_inverse.max():.4f}]")
        
        # 检查是否有异常值
        if X_num_inverse.max() > X_num_test.max() * 10 or X_num_inverse.min() < X_num_test.min() * 10:
            print(f"  ⚠️  警告: 逆变换后的数值范围异常")
            print(f"     可能原因: 标准化数据包含异常值（超出训练时的范围）")
        else:
            print(f"  ✅ 数值范围合理")
    else:
        print("⚠️  没有数值特征或没有数值变换器")
    
    # 4. 测试分类特征的逆变换（如果使用one-hot编码）
    print("\n【步骤4】测试分类特征的逆变换（one-hot）")
    print("-" * 60)
    
    if dataset.cat_transform is not None and X_cat_test is not None:
        # 正向变换：原始分类数据 → one-hot编码
        X_cat_transformed = dataset.cat_transform.transform(X_cat_test)
        print(f"One-hot编码后分类特征 shape: {X_cat_transformed.shape}")
        
        # 逆变换：one-hot编码 → 原始分类数据
        X_cat_inverse = dataset.cat_transform.inverse_transform(X_cat_transformed)
        print(f"逆变换后分类特征 shape: {X_cat_inverse.shape}")
        
        # 检查是否一致（one-hot编码可能有精度问题）
        if X_cat_inverse.shape == X_cat_test.shape:
            matches = (X_cat_inverse == X_cat_test).all(axis=1)
            match_rate = matches.sum() / len(matches)
            print(f"\n分类特征匹配率: {match_rate:.4f} ({matches.sum()}/{len(matches)})")
            
            if match_rate > 0.95:
                print(f"✅ 分类特征逆变换正确（匹配率 > 95%）")
            else:
                print(f"⚠️  警告: 分类特征逆变换匹配率较低")
                print(f"  不匹配的样本索引: {np.where(~matches)[0][:10]}")
        else:
            print(f"⚠️  警告: 形状不匹配")
    else:
        print("⚠️  没有分类特征或没有分类变换器")
    
    # 5. 测试反事实样本的逆变换
    print("\n【步骤5】测试反事实样本的逆变换")
    print("-" * 60)
    
    try:
        cf = np.load('exp/adult/counterfactuals_fixed.npy', allow_pickle=True)
        
        # 转换数据类型
        if cf.dtype == object:
            cf_numeric = []
            for row in cf:
                cf_numeric.append([float(x) if isinstance(x, (int, float, np.number)) else 0.0 for x in row])
            cf = np.array(cf_numeric, dtype=np.float64)
        
        # 检查反事实样本在逆变换前的范围
        print(f"反事实样本（逆变换前）shape: {cf.shape}")
        
        # 假设反事实样本已经是逆变换后的（原始空间）
        # 需要检查这些值的合理性
        if dataset.num_transform is not None and cf.shape[1] >= 4:
            # 假设前4列是数值特征
            cf_num = cf[:n_test, :4]
            
            # 将这些值转换到标准化空间（正向变换）
            try:
                cf_num_transformed = dataset.num_transform.transform(cf_num)
                print(f"反事实数值特征（标准化后）范围: [{cf_num_transformed.min():.4f}, {cf_num_transformed.max():.4f}]")
                print(f"反事实数值特征（标准化后）均值: {cf_num_transformed.mean(axis=0)}")
                print(f"反事实数值特征（标准化后）标准差: {cf_num_transformed.std(axis=0)}")
                
                # 检查是否在合理范围内（标准化数据通常在[-3, 3]之间）
                if cf_num_transformed.max() > 10 or cf_num_transformed.min() < -10:
                    print(f"\n⚠️  警告: 反事实样本在标准化空间的数值范围异常")
                    print(f"  可能原因: 反事实样本在逆变换前就包含了异常值")
                    print(f"  建议: 检查采样过程中的数值范围")
                else:
                    print(f"\n✅ 反事实样本在标准化空间的数值范围合理")
                
            except Exception as e:
                print(f"⚠️  无法转换反事实样本: {e}")
                print(f"  可能原因: 反事实样本不在原始数据空间，或者格式不正确")
        
    except FileNotFoundError:
        print("⚠️  反事实样本文件不存在，跳过此测试")
    except Exception as e:
        print(f"⚠️  测试反事实样本时出错: {e}")
    
    # 6. 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("逆变换测试完成！")
    print("\n如果发现以下问题：")
    print("1. 数值特征逆变换误差过大 → 检查StandardScaler的拟合数据")
    print("2. 反事实样本在标准化空间数值异常 → 检查采样过程中的数值范围")
    print("3. 数值范围异常 → 检查tanh裁剪是否过度限制了数值")


if __name__ == '__main__':
    test_inverse_transform()

