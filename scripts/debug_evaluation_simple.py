#!/usr/bin/env python3
"""
TDCE评估问题快速调试脚本（简化版）
专注于关键问题：异常值分析
"""
import numpy as np

print("=" * 60)
print("TDCE评估问题快速诊断")
print("=" * 60)

# 1. 检查反事实样本
print("\n【步骤1】反事实样本检查")
print("-" * 60)

try:
    cf = np.load('exp/adult/counterfactuals_fixed.npy', allow_pickle=True)
    
    # 转换数据类型
    if cf.dtype == object:
        cf_numeric = []
        for row in cf:
            cf_numeric.append([float(x) if isinstance(x, (int, float, np.number)) else 0.0 for x in row])
        cf = np.array(cf_numeric, dtype=np.float64)
    
    print(f"Shape: {cf.shape}")
    print(f"数值范围: [{cf.min():.4f}, {cf.max():.4f}]")
    print(f"均值: {cf.mean():.4f}, 标准差: {cf.std():.4f}")
    
    # 异常值统计
    print(f"\n异常值统计:")
    print(f"  |值| > 1000: {(np.abs(cf) > 1000).sum()} ({(np.abs(cf) > 1000).sum() / cf.size * 100:.1f}%)")
    print(f"  |值| > 100: {(np.abs(cf) > 100).sum()} ({(np.abs(cf) > 100).sum() / cf.size * 100:.1f}%)")
    print(f"  |值| > 10: {(np.abs(cf) > 10).sum()} ({(np.abs(cf) > 10).sum() / cf.size * 100:.1f}%)")
    
    # 按列分析（前4列是数值特征）
    print(f"\n按列分析（前4列是数值特征）:")
    for i in range(min(4, cf.shape[1])):
        col = cf[:, i]
        print(f"  列 {i}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, std={col.std():.4f}")
        print(f"    |值| > 100: {(np.abs(col) > 100).sum()} ({(np.abs(col) > 100).sum() / len(col) * 100:.1f}%)")
    
    # 检查是否有负值（对于某些特征不应该有负值）
    print(f"\n负值检查:")
    for i in range(min(4, cf.shape[1])):
        neg_count = (cf[:, i] < 0).sum()
        if neg_count > 0:
            print(f"  列 {i}: {neg_count}个负值 (最小值: {cf[:, i].min():.4f})")
    
except Exception as e:
    print(f"❌ 错误: {e}")

# 2. 检查原始样本（对比）
print("\n【步骤2】原始样本检查（对比）")
print("-" * 60)

try:
    orig = np.load('exp/adult/original_samples.npy', allow_pickle=True)
    
    if orig.dtype == object:
        orig_numeric = []
        for row in orig:
            orig_numeric.append([float(x) if isinstance(x, (int, float, np.number)) else 0.0 for x in row])
        orig = np.array(orig_numeric, dtype=np.float64)
    
    print(f"Shape: {orig.shape}")
    print(f"数值范围: [{orig.min():.4f}, {orig.max():.4f}]")
    print(f"均值: {orig.mean():.4f}, 标准差: {orig.std():.4f}")
    
    # 对比分析
    if orig.shape == cf.shape:
        diff = np.abs(cf - orig)
        print(f"\n差异分析:")
        print(f"  平均绝对差异: {diff.mean():.4f}")
        print(f"  最大绝对差异: {diff.max():.4f}")
        print(f"  差异 > 1000: {(diff > 1000).sum()}")
        print(f"  差异 > 100: {(diff > 100).sum()}")
        
        # 计算L2距离（只对数值特征）
        if cf.shape[1] >= 4:
            l2_manual = np.sqrt(np.sum(diff[:, :4]**2, axis=1)).mean()
            print(f"  手动计算的L2距离（前4列）: {l2_manual:.4f}")
        
except Exception as e:
    print(f"❌ 错误: {e}")

# 3. 问题总结
print("\n【步骤3】问题总结")
print("-" * 60)

print("🔍 发现的关键问题：")
print("1. ❌ 反事实样本包含大量异常值（>1000的值）")
print("2. ❌ 列1的数值范围异常（-989到13814）")
print("3. ❌ 列2也有较大异常值（-504到733）")
print()
print("可能原因：")
print("1. 逆变换不正确（StandardScaler的逆变换可能有问题）")
print("2. 采样过程中的数值不稳定（tanh裁剪可能过度）")
print("3. 数据预处理不一致（训练和采样时的预处理不同）")
print()
print("建议：")
print("1. 检查逆变换过程（sample_counterfactual.py中的inverse_transform）")
print("2. 检查采样过程中的数值范围（gaussian_multinomial_diffsuion.py）")
print("3. 检查数据预处理的一致性（训练和采样时是否使用相同的预处理）")
print("4. 考虑减少或移除tanh裁剪，查看是否改善")

print("\n" + "=" * 60)

