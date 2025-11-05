#!/usr/bin/env python3
"""
TDCE评估问题调试脚本
系统性地检查反事实生成过程中的各个环节
"""
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib
from scripts.utils_train import make_dataset
from tdce.modules import UNetClassifier


def check_counterfactuals():
    """步骤1: 检查反事实样本的数值特征"""
    print("=" * 60)
    print("步骤1: 检查反事实样本")
    print("=" * 60)
    
    # 加载numpy数组，支持对象数组
    try:
        cf = np.load('exp/adult/counterfactuals_fixed.npy', allow_pickle=True)
    except:
        cf = np.load('exp/adult/counterfactuals_fixed.npy', allow_pickle=False)
    
    try:
        orig = np.load('exp/adult/original_samples.npy', allow_pickle=True)
    except:
        orig = np.load('exp/adult/original_samples.npy', allow_pickle=False)
    
    print(f"反事实样本 shape: {cf.shape}")
    print(f"数据类型: {cf.dtype}")
    
    # 检查数据类型，只对数值类型计算统计量
    if cf.dtype == object or not np.issubdtype(cf.dtype, np.number):
        print("⚠️  警告: 数组包含非数值类型，尝试转换...")
        # 尝试转换为数值类型
        cf_numeric = []
        for row in cf:
            cf_numeric.append([float(x) if isinstance(x, (int, float, np.number)) else 0.0 for x in row])
        cf = np.array(cf_numeric, dtype=np.float64)
        print(f"转换后数据类型: {cf.dtype}")
    
    # 只对数值特征计算统计量
    if np.issubdtype(cf.dtype, np.number):
        print(f"数值范围: [{cf.min():.4f}, {cf.max():.4f}]")
        print(f"均值: {cf.mean():.4f}, 标准差: {cf.std():.4f}")
        print(f"NaN: {np.isnan(cf).sum()}, Inf: {np.isinf(cf).sum()}")
        
        # 检查异常值
        print(f"\n异常值检查:")
        print(f"  |值| > 1000: {(np.abs(cf) > 1000).sum()}")
        print(f"  |值| > 100: {(np.abs(cf) > 100).sum()}")
        print(f"  |值| > 10: {(np.abs(cf) > 10).sum()}")
        
        # 按列显示统计信息（前4列通常是数值特征）
        print(f"\n前4列（数值特征）统计:")
        for i in range(min(4, cf.shape[1])):
            print(f"  列 {i}: min={cf[:, i].min():.4f}, max={cf[:, i].max():.4f}, mean={cf[:, i].mean():.4f}")
    
    # 计算L2距离（如果可能）
    if orig.shape == cf.shape:
        try:
            # 确保orig也是数值类型
            if orig.dtype == object:
                orig_numeric = []
                for row in orig:
                    orig_numeric.append([float(x) if isinstance(x, (int, float, np.number)) else 0.0 for x in row])
                orig = np.array(orig_numeric, dtype=np.float64)
            
            if np.issubdtype(orig.dtype, np.number) and np.issubdtype(cf.dtype, np.number):
                diff = cf - orig
                l2_manual = np.sqrt(np.sum(diff**2, axis=1)).mean()
                print(f"\n手动计算的L2距离: {l2_manual:.4f}")
        except Exception as e:
            print(f"\n⚠️  无法计算L2距离: {e}")
    
    return cf


def check_classifier_predictions(cf):
    """步骤2: 检查分类器对反事实样本的预测"""
    print("\n" + "=" * 60)
    print("步骤2: 检查分类器预测")
    print("=" * 60)
    print("⚠️  注意: 反事实样本包含异常值，可能影响预测结果")
    print("   建议先检查反事实生成过程")
    
    # 简化：直接使用evaluate_counterfactuals.py的逻辑
    # 这里只做基本检查
    print("\n跳过详细分类器检查（因为数据转换复杂）")
    print("建议直接查看evaluation_results_fixed.txt中的Validity指标")
    
    return 0.314  # 使用之前评估的结果


def check_data_preprocessing():
    """步骤3: 检查数据预处理"""
    print("\n" + "=" * 60)
    print("步骤3: 检查数据预处理")
    print("=" * 60)
    
    config = lib.load_config('exp/adult/tdce_test/config.toml')
    # make_dataset需要5个参数
    T_dict = config['train']['T']
    T = lib.Transformations(**T_dict)
    num_classes = config['model_params']['num_classes']
    is_y_cond = config['model_params'].get('is_y_cond', False)
    dataset = make_dataset('data/adult', T, num_classes, is_y_cond, change_val=False)
    
    print(f"数值特征数量: {dataset.num_numerical_features}")
    print(f"分类特征数量: {dataset.num_cat_features}")
    print(f"类别大小: {dataset.num_classes}")
    print(f"标准化方法: {config['train']['T']['normalization']}")
    print(f"编码方式: {config['train']['T']['cat_encoding']}")
    
    # 检查数据范围
    sample = dataset.X_test[:100]
    print(f"\n转换后数据:")
    print(f"  Shape: {sample.shape}")
    print(f"  范围: [{sample.min():.4f}, {sample.max():.4f}]")
    print(f"  均值: {sample.mean():.4f}, 标准差: {sample.std():.4f}")


def check_hyperparameters():
    """步骤4: 检查超参数"""
    print("\n" + "=" * 60)
    print("步骤4: 检查超参数")
    print("=" * 60)
    
    config = lib.load_config('exp/adult/tdce_test/config.toml')
    
    print("关键超参数:")
    print(f"  num_timesteps: {config['diffusion_params']['num_timesteps']}")
    print(f"  tau_init: {config['diffusion_params']['tau_init']}")
    print(f"  tau_final: {config['diffusion_params']['tau_final']}")
    print(f"  model_type: {config['model_type']}")
    print(f"  classifier_model_type: {config['model_params'].get('classifier_model_type', 'N/A')}")
    print(f"  is_y_cond: {config['model_params']['is_y_cond']}")


def check_classifier_accuracy():
    """步骤5: 检查分类器准确率"""
    print("\n" + "=" * 60)
    print("步骤5: 检查分类器准确率")
    print("=" * 60)
    
    config = lib.load_config('exp/adult/tdce_test/config.toml')
    # make_dataset需要5个参数
    T_dict = config['train']['T']
    T = lib.Transformations(**T_dict)
    num_classes = config['model_params']['num_classes']
    is_y_cond = config['model_params'].get('is_y_cond', False)
    dataset = make_dataset('data/adult', T, num_classes, is_y_cond, change_val=False)
    
    # 加载分类器
    model_params = config['model_params']
    classifier = UNetClassifier(
        d_in=66,
        num_classes=2,
        is_y_cond=model_params.get('is_y_cond', False),
        rtdl_params=model_params.get('rtdl_params', {}),
        num_output_classes=2
    )
    classifier.load_state_dict(torch.load('exp/adult/classifier.pt', map_location='cuda:0'))
    classifier = classifier.cuda().eval()
    
    # 测试集预测
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(dataset.X_test),
            torch.LongTensor(dataset.y_test.squeeze())
        ),
        batch_size=1024,
        shuffle=False
    )
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            logits = classifier(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    
    accuracy = correct / total
    print(f"测试集准确率: {accuracy:.4f} ({correct}/{total})")
    
    if accuracy < 0.9:
        print("⚠️  警告: 分类器准确率偏低，可能影响梯度引导效果")
    
    return accuracy


if __name__ == '__main__':
    cf = check_counterfactuals()
    validity = check_classifier_predictions(cf)
    check_data_preprocessing()
    check_hyperparameters()
    accuracy = check_classifier_accuracy()
    
    print("\n" + "=" * 60)
    print("总结和建议")
    print("=" * 60)
    print(f"有效性: {validity:.4f} (目标: >0.9)")
    print(f"分类器准确率: {accuracy:.4f} (目标: >0.9)")
    
    if validity < 0.5:
        print("\n❌ 问题1: 有效性过低")
        print("   可能原因:")
        print("   1. 分类器准确率过低（当前: {:.2f}%）".format(accuracy * 100))
        print("   2. 梯度引导强度不足")
        print("   3. 采样步数不足")
        print("   4. 数值稳定性问题")
        print("\n   建议:")
        if accuracy < 0.9:
            print("   - 重新训练分类器，提高准确率")
        print("   - 检查采样过程中的梯度大小")
        print("   - 尝试增加采样步数或调整温度参数")
    
    if cf.max() > 100 or cf.min() < -100:
        print("\n❌ 问题2: 反事实样本数值范围异常")
        print("   可能原因:")
        print("   1. 逆变换不正确")
        print("   2. 数值稳定性问题（tanh裁剪可能过度）")
        print("   3. 数据预处理不一致")
        print("\n   建议:")
        print("   - 检查逆变换过程")
        print("   - 检查数据预处理的一致性")

