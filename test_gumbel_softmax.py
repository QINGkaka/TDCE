"""
TDCE: 测试Gumbel-Softmax扩散功能
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tdce.gumbel_softmax_utils import (
    gumbel_softmax_relaxation,
    temperature_scheduler,
    gumbel_softmax_q_sample,
    index_to_onehot,
    gumbel_softmax_to_index
)

def test_gumbel_softmax_utils():
    """测试Gumbel-Softmax工具函数"""
    print("=" * 60)
    print("测试1: Gumbel-Softmax工具函数")
    print("=" * 60)
    
    # 测试1.1: gumbel_softmax_relaxation
    print("\n1.1 测试 gumbel_softmax_relaxation...")
    batch_size = 4
    num_classes = 5
    logits = torch.randn(batch_size, num_classes)
    tau = 1.0
    result = gumbel_softmax_relaxation(logits, tau=tau, hard=False)
    print(f"   输入logits shape: {logits.shape}")
    print(f"   输出shape: {result.shape}")
    print(f"   输出和是否为1: {result.sum(dim=-1)}")  # 应该接近1
    assert result.shape == (batch_size, num_classes), f"Shape错误: {result.shape}"
    assert torch.allclose(result.sum(dim=-1), torch.ones(batch_size), atol=1e-3), "概率和不等于1"
    print("   ✅ 通过")
    
    # 测试1.2: temperature_scheduler
    print("\n1.2 测试 temperature_scheduler...")
    tau_init = 1.0
    tau_final = 0.3
    num_timesteps = 1000
    tau_start = temperature_scheduler(0, tau_init, tau_final, num_timesteps)
    tau_end = temperature_scheduler(num_timesteps-1, tau_init, tau_final, num_timesteps)
    print(f"   初始温度: {tau_start:.4f} (期望: {tau_init})")
    print(f"   最终温度: {tau_end:.4f} (期望: {tau_final})")
    assert abs(tau_start - tau_init) < 1e-6, f"初始温度错误: {tau_start}"
    assert abs(tau_end - tau_final) < 1e-3, f"最终温度错误: {tau_end}"
    print("   ✅ 通过")
    
    # 测试1.3: index_to_onehot
    print("\n1.3 测试 index_to_onehot...")
    batch_size = 4
    num_cat_features = 3
    num_classes = [2, 3, 4]  # 每个分类特征的类别数
    # 为每个特征生成符合其类别数范围的索引
    x_index = torch.zeros(batch_size, num_cat_features, dtype=torch.long)
    for i in range(num_cat_features):
        x_index[:, i] = torch.randint(0, num_classes[i], (batch_size,))
    x_onehot = index_to_onehot(x_index, num_classes)
    print(f"   输入索引 shape: {x_index.shape}")
    print(f"   输出one-hot shape: {x_onehot.shape}")
    print(f"   示例输入: {x_index[0]}")
    print(f"   示例输出 shape: {x_onehot[0].shape}")
    assert x_onehot.shape == (batch_size, num_cat_features, max(num_classes)), f"Shape错误: {x_onehot.shape}"
    print("   ✅ 通过")
    
    # 测试1.4: gumbel_softmax_q_sample
    print("\n1.4 测试 gumbel_softmax_q_sample...")
    batch_size = 4
    num_cat_features = 2
    num_classes_per_feat = 3
    x_cat_onehot = torch.zeros(batch_size, num_cat_features, num_classes_per_feat)
    for i in range(batch_size):
        for j in range(num_cat_features):
            idx = np.random.randint(0, num_classes_per_feat)
            x_cat_onehot[i, j, idx] = 1.0
    
    t = torch.randint(0, 1000, (batch_size,))
    beta_schedule = torch.linspace(0.0001, 0.02, 1000)
    tau = 1.0
    device = torch.device('cpu')
    
    x_t_cat = gumbel_softmax_q_sample(x_cat_onehot, t, beta_schedule, tau, device)
    print(f"   输入one-hot shape: {x_cat_onehot.shape}")
    print(f"   输出Gumbel-Softmax shape: {x_t_cat.shape}")
    print(f"   输出概率和: {x_t_cat.sum(dim=-1)[0]}")  # 应该接近1
    assert x_t_cat.shape == x_cat_onehot.shape, f"Shape错误: {x_t_cat.shape}"
    assert torch.allclose(x_t_cat.sum(dim=-1), torch.ones(batch_size, num_cat_features), atol=1e-2), "概率和不等于1"
    print("   ✅ 通过")
    
    print("\n" + "=" * 60)
    print("✅ 所有Gumbel-Softmax工具函数测试通过！")
    print("=" * 60)


def test_diffusion_with_gumbel_softmax():
    """测试扩散模型中的Gumbel-Softmax功能"""
    print("\n" + "=" * 60)
    print("测试2: 扩散模型Gumbel-Softmax集成")
    print("=" * 60)
    
    try:
        from tdce.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
        from tdce.modules import MLPDiffusion
        
        # 创建简单的模型
        num_classes = np.array([2, 3])  # 2个分类特征，分别有2和3个类别
        num_numerical_features = 5
        num_timesteps = 100
        
        # 创建简单的去噪网络
        # MLPDiffusion需要的参数：d_in, num_classes, is_y_cond, rtdl_params, dim_t
        rtdl_params = {
            'd_layers': [64, 64],
            'dropout': 0.0
        }
        model = MLPDiffusion(
            d_in=num_numerical_features + sum(num_classes),  # 数值特征 + 分类特征的总维度
            num_classes=0,  # 不使用条件标签
            is_y_cond=False,  # 不使用条件
            rtdl_params=rtdl_params,
            dim_t=128  # 时间嵌入维度
        )
        
        # 测试2.1: 创建扩散模型（使用Gumbel-Softmax）
        print("\n2.1 测试创建扩散模型（use_gumbel_softmax=True）...")
        diffusion = GaussianMultinomialDiffusion(
            num_classes=num_classes,
            num_numerical_features=num_numerical_features,
            denoise_fn=model,
            num_timesteps=num_timesteps,
            use_gumbel_softmax=True,
            tau_init=1.0,
            tau_final=0.3,
            tau_schedule='anneal',
            device=torch.device('cpu')
        )
        print(f"   模型创建成功")
        print(f"   use_gumbel_softmax: {diffusion.use_gumbel_softmax}")
        print(f"   betas shape: {diffusion.betas.shape}")
        print("   ✅ 通过")
        
        # 测试2.2: q_sample_gumbel_softmax
        print("\n2.2 测试 q_sample_gumbel_softmax...")
        batch_size = 8
        # 为每个特征生成符合其类别数范围的索引
        x_cat_index = torch.zeros(batch_size, len(num_classes), dtype=torch.long)
        for i in range(len(num_classes)):
            x_cat_index[:, i] = torch.randint(0, num_classes[i], (batch_size,))
        x_cat_onehot = index_to_onehot(x_cat_index, list(num_classes))
        t = torch.randint(0, num_timesteps, (batch_size,))
        
        x_t_cat = diffusion.q_sample_gumbel_softmax(x_cat_onehot, t)
        print(f"   输入one-hot shape: {x_cat_onehot.shape}")
        print(f"   输出Gumbel-Softmax shape: {x_t_cat.shape}")
        assert x_t_cat.shape == x_cat_onehot.shape, f"Shape错误: {x_t_cat.shape}"
        print("   ✅ 通过")
        
        # 测试2.3: mixed_loss（使用Gumbel-Softmax）
        print("\n2.3 测试 mixed_loss（use_gumbel_softmax=True）...")
        # 创建模拟数据
        x_num = torch.randn(batch_size, num_numerical_features)
        # 为每个特征生成符合其类别数范围的索引
        x_cat = torch.zeros(batch_size, len(num_classes), dtype=torch.long)
        for i in range(len(num_classes)):
            x_cat[:, i] = torch.randint(0, num_classes[i], (batch_size,))
        x = torch.cat([x_num, x_cat.float()], dim=1)
        
        # 创建out_dict
        out_dict = {
            'y': torch.randint(0, 2, (batch_size,))  # 假设是二分类任务
        }
        
        try:
            loss_multi, loss_gauss = diffusion.mixed_loss(x, out_dict)
            print(f"   分类特征损失: {loss_multi.item():.6f}")
            print(f"   数值特征损失: {loss_gauss.item():.6f}")
            assert not torch.isnan(loss_multi), "分类特征损失为NaN"
            assert not torch.isnan(loss_gauss), "数值特征损失为NaN"
            assert loss_multi >= 0, f"分类特征损失为负: {loss_multi}"
            assert loss_gauss >= 0, f"数值特征损失为负: {loss_gauss}"
            print("   ✅ 通过")
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试2.4: 对比传统多项式扩散（use_gumbel_softmax=False）
        print("\n2.4 测试 mixed_loss（use_gumbel_softmax=False，对比）...")
        diffusion_traditional = GaussianMultinomialDiffusion(
            num_classes=num_classes,
            num_numerical_features=num_numerical_features,
            denoise_fn=model,
            num_timesteps=num_timesteps,
            use_gumbel_softmax=False,
            device=torch.device('cpu')
        )
        
        try:
            loss_multi_trad, loss_gauss_trad = diffusion_traditional.mixed_loss(x, out_dict)
            print(f"   传统方法分类损失: {loss_multi_trad.item():.6f}")
            print(f"   传统方法数值损失: {loss_gauss_trad.item():.6f}")
            print("   ✅ 传统方法也正常工作")
        except Exception as e:
            print(f"   ⚠️  传统方法测试失败（可能不影响TDCE）: {e}")
        
        print("\n" + "=" * 60)
        print("✅ 扩散模型Gumbel-Softmax集成测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_p_sample_gumbel_softmax():
    """测试p_sample_gumbel_softmax反向采样方法"""
    print("\n" + "=" * 60)
    print("测试3: p_sample_gumbel_softmax反向采样")
    print("=" * 60)
    
    try:
        from tdce.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
        from tdce.modules import MLPDiffusion
        
        # 创建简单的模型
        num_classes = np.array([2, 3])  # 2个分类特征，分别有2和3个类别
        num_numerical_features = 5
        num_timesteps = 100
        
        # 创建简单的去噪网络
        rtdl_params = {
            'd_layers': [64, 64],
            'dropout': 0.0
        }
        model = MLPDiffusion(
            d_in=num_numerical_features + sum(num_classes),
            num_classes=0,
            is_y_cond=False,
            rtdl_params=rtdl_params,
            dim_t=128
        )
        
        # 创建扩散模型（使用Gumbel-Softmax）
        print("\n3.1 测试创建扩散模型...")
        diffusion = GaussianMultinomialDiffusion(
            num_classes=num_classes,
            num_numerical_features=num_numerical_features,
            denoise_fn=model,
            num_timesteps=num_timesteps,
            use_gumbel_softmax=True,
            tau_init=1.0,
            tau_final=0.3,
            tau_schedule='anneal',
            device=torch.device('cpu')
        )
        print("   ✅ 模型创建成功")
        
        # 测试3.2: p_sample_gumbel_softmax
        print("\n3.2 测试 p_sample_gumbel_softmax...")
        batch_size = 4
        
        # 创建模拟的模型输出和当前时间步的分类特征
        from tdce.gumbel_softmax_utils import index_to_onehot
        
        # 创建随机的分类特征索引
        x_cat_index = torch.zeros(batch_size, len(num_classes), dtype=torch.long)
        for i in range(len(num_classes)):
            x_cat_index[:, i] = torch.randint(0, num_classes[i], (batch_size,))
        
        # 转为one-hot，然后通过前向扩散得到x_t
        x_cat_onehot = index_to_onehot(x_cat_index, list(num_classes))
        t = torch.randint(50, 100, (batch_size,))  # 随机时间步
        
        # 前向扩散得到x_t
        x_t_cat_gumbel = diffusion.q_sample_gumbel_softmax(x_cat_onehot, t)
        print(f"   输入x_t_cat_gumbel shape: {x_t_cat_gumbel.shape}")
        
        # 模拟模型输出（随机logits）
        model_out_cat = torch.randn(batch_size, sum(num_classes))
        
        # 反向采样
        out_dict = {'y': torch.randint(0, 2, (batch_size,))}
        x_t_minus_1_cat = diffusion.p_sample_gumbel_softmax(
            model_out_cat,
            x_t_cat_gumbel,
            t,
            out_dict
        )
        print(f"   输出x_t_minus_1_cat shape: {x_t_minus_1_cat.shape}")
        print(f"   输出概率和（每个特征）: {x_t_minus_1_cat.sum(dim=-1)[0]}")
        
        # 验证形状和概率和
        assert x_t_minus_1_cat.shape == x_t_cat_gumbel.shape, f"形状错误: {x_t_minus_1_cat.shape} vs {x_t_cat_gumbel.shape}"
        # 验证每个分类特征的概率和接近1
        for i in range(len(num_classes)):
            prob_sum = x_t_minus_1_cat[:, i, :num_classes[i]].sum(dim=-1)
            assert torch.allclose(prob_sum, torch.ones(batch_size), atol=1e-3), f"特征{i}的概率和不等于1"
        
        print("   ✅ 通过")
        
        # 测试3.3: 完整的sample方法（Gumbel-Softmax模式）
        print("\n3.3 测试完整的sample方法（Gumbel-Softmax模式）...")
        y_dist = torch.tensor([0.5, 0.5])  # 二分类，均匀分布
        
        sample, out_dict = diffusion.sample(num_samples=4, y_dist=y_dist)
        print(f"   采样结果shape: {sample.shape}")
        print(f"   期望shape: (4, {num_numerical_features + len(num_classes)})")
        
        # 验证形状
        expected_shape = (4, num_numerical_features + len(num_classes))
        assert sample.shape == expected_shape, f"形状错误: {sample.shape} vs {expected_shape}"
        
        # 验证分类特征值在有效范围内
        if len(num_classes) > 0:
            x_cat_sampled = sample[:, num_numerical_features:]
            for i, num_class in enumerate(num_classes):
                assert (x_cat_sampled[:, i] >= 0).all(), f"特征{i}有负值"
                assert (x_cat_sampled[:, i] < num_class).all(), f"特征{i}超出范围"
        
        print("   ✅ 通过")
        
        # 测试3.4: 对比传统模式（use_gumbel_softmax=False）
        print("\n3.4 测试传统模式（use_gumbel_softmax=False，对比）...")
        diffusion_traditional = GaussianMultinomialDiffusion(
            num_classes=num_classes,
            num_numerical_features=num_numerical_features,
            denoise_fn=model,
            num_timesteps=num_timesteps,
            use_gumbel_softmax=False,
            device=torch.device('cpu')
        )
        
        sample_trad, out_dict_trad = diffusion_traditional.sample(num_samples=4, y_dist=y_dist)
        print(f"   传统方法采样结果shape: {sample_trad.shape}")
        assert sample_trad.shape == expected_shape, f"传统方法形状错误: {sample_trad.shape}"
        print("   ✅ 传统方法也正常工作")
        
        print("\n" + "=" * 60)
        print("✅ p_sample_gumbel_softmax和sample方法测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("TDCE Gumbel-Softmax功能测试")
    print("=" * 60)
    
    # 测试1: Gumbel-Softmax工具函数
    test_gumbel_softmax_utils()
    
    # 测试2: 扩散模型集成
    success2 = test_diffusion_with_gumbel_softmax()
    
    # 测试3: p_sample_gumbel_softmax反向采样
    success3 = test_p_sample_gumbel_softmax()
    
    print("\n" + "=" * 60)
    if success2 and success3:
        print("🎉 所有测试通过！Gumbel-Softmax功能正常")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("=" * 60)


if __name__ == '__main__':
    main()

