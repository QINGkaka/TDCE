"""
TDCE: 分类器训练脚本
训练分类器用于TDCE的梯度引导
"""

import sys
import os

# Add parent directory to path to import tdce and lib modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("开始导入模块...", flush=True)
import torch
print("✓ torch 导入完成", flush=True)
import torch.nn as nn
import torch.optim as optim
print("✓ torch.nn, torch.optim 导入完成", flush=True)
import numpy as np
print("✓ numpy 导入完成", flush=True)
import argparse
print("✓ argparse 导入完成", flush=True)

print("开始导入项目模块...", flush=True)
import lib
print("✓ lib 导入完成", flush=True)
from tdce.modules import MLP
print("✓ MLP 导入完成", flush=True)
from scripts.utils_train import make_dataset
print("✓ make_dataset 导入完成", flush=True)
from lib.data import prepare_fast_dataloader
print("✓ prepare_fast_dataloader 导入完成", flush=True)
print("所有模块导入完成！", flush=True)


def train_classifier(
    data_path: str,
    model_params: dict,
    num_numerical_features: int,
    num_classes: list,
    num_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 0.001,
    device: str = 'cuda:0',
    output_path: str = 'classifier.pt',
    seed: int = 0
):
    """
    训练分类器用于TDCE的梯度引导
    
    Args:
        data_path: 数据路径
        model_params: 模型参数字典
        num_numerical_features: 数值特征数量
        num_classes: 分类特征类别数列表
        num_epochs: 训练轮数
        batch_size: 批量大小
        lr: 学习率
        device: 设备
        output_path: 输出模型路径
        seed: 随机种子
    
    Returns:
        classifier: 训练好的分类器
    """
    import zero
    zero.improve_reproducibility(seed)
    
    device = torch.device(device)
    
    # 1. 加载数据
    print("Step 1: 准备数据转换配置...")
    T_dict = {
        'normalization': 'quantile',
        'cat_encoding': 'one-hot',  # 分类特征需要one-hot编码
        'y_policy': 'default',
        'num_nan_policy': model_params.get('num_nan_policy', '__none__'),
        'cat_nan_policy': model_params.get('cat_nan_policy', '__none__'),
        'cat_min_frequency': model_params.get('cat_min_frequency', '__none__')
    }
    T = lib.Transformations(**T_dict)
    
    print("Step 2: 加载和转换数据集（这可能需要一些时间）...")
    # 注意：如果使用配置文件，应该使用配置文件中的is_y_cond设置
    # 以匹配扩散模型的配置
    is_y_cond = model_params.get('is_y_cond', False)
    print(f"  Using is_y_cond={is_y_cond} (from model_params)")
    dataset = make_dataset(
        data_path,
        T,
        num_classes=model_params.get('num_classes', 0),
        is_y_cond=is_y_cond,  # 使用配置文件中的设置
        change_val=False
    )
    print("Step 3: 数据集加载完成，准备数据加载器...")
    
    # 2. 准备数据加载器
    # prepare_fast_dataloader内部已经处理shuffle（train默认shuffle=True）
    print("Step 4: 创建训练数据加载器...")
    train_loader = prepare_fast_dataloader(
        dataset, 
        split='train', 
        batch_size=batch_size
    )
    print("Step 5: 创建验证数据加载器...")
    val_loader = prepare_fast_dataloader(
        dataset, 
        split='val', 
        batch_size=batch_size
    )
    print("Step 6: 数据加载器创建完成")
    
    # 3. 构建分类器
    # 计算输入维度（数值特征 + 分类特征的one-hot展开）
    category_sizes = dataset.get_category_sizes('train')
    d_in = num_numerical_features + sum(category_sizes)
    d_out = model_params.get('num_classes', 2)  # 分类任务的类别数
    
    classifier_params = model_params.get('classifier_params', {
        'd_layers': [256, 256],
        'dropout': 0.1
    })
    
    classifier = MLP.make_baseline(
        d_in=d_in,
        d_out=d_out,
        d_layers=classifier_params.get('d_layers', [256, 256]),
        dropout=classifier_params.get('dropout', 0.1)
    )
    classifier.to(device)
    
    print(f"Classifier architecture:")
    print(f"  Input dimension: {d_in}")
    print(f"  Output dimension: {d_out}")
    print(f"  Hidden layers: {classifier_params.get('d_layers', [256, 256])}")
    print(f"  Dropout: {classifier_params.get('dropout', 0.1)}")
    
    # 4. 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    # Calculate number of batches from dataset size
    train_size = len(dataset.y['train'])
    val_size = len(dataset.y['val'])
    train_batches = (train_size + batch_size - 1) // batch_size
    val_batches = (val_size + batch_size - 1) // batch_size
    print(f"  Train batches: {train_batches} (dataset size: {train_size})")
    print(f"  Val batches: {val_batches} (dataset size: {val_size})")
    
    for epoch in range(num_epochs):
        # 训练阶段
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batch_count = 0
        
        # 限制训练批次数量，避免无限循环（prepare_fast_dataloader返回无限生成器）
        for x, y in train_loader:
            if train_batch_count >= train_batches:
                break
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            train_batch_count += 1
        
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0.0
        avg_train_loss = train_loss / train_batch_count if train_batch_count > 0 else 0.0
        
        # 验证阶段
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 限制验证批次数量，避免无限循环
        val_batch_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                if val_batch_count >= val_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                
                logits = classifier(x)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                val_batch_count += 1
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc or (val_acc == best_val_acc and avg_val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            torch.save(classifier.state_dict(), output_path)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  ✓ Saved best model with val_acc={val_acc:.2f}%, val_loss={avg_val_loss:.4f}")
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {output_path}")
    
    return classifier


def main():
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    print("\n=== 开始执行主函数 ===", flush=True)
    sys.stdout.flush()
    print("正在创建参数解析器...", flush=True)
    sys.stdout.flush()
    parser = argparse.ArgumentParser(description='TDCE: 分类器训练脚本')
    print("✓ 参数解析器创建完成", flush=True)
    parser.add_argument('--config', type=str, required=False,
                       help='配置文件路径（.toml格式）')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据路径')
    parser.add_argument('--output_path', type=str, default='classifier.pt',
                       help='输出模型路径（默认：classifier.pt）')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数（默认：100）')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='批量大小（默认：1024）')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率（默认：0.001）')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备（默认：cuda:0）')
    parser.add_argument('--seed', type=int, default=0,
                       help='随机种子（默认：0）')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='分类任务的类别数（默认：2）')
    parser.add_argument('--classifier_d_layers', type=int, nargs='+', default=[256, 256],
                       help='分类器隐藏层维度（默认：[256, 256]）')
    parser.add_argument('--classifier_dropout', type=float, default=0.1,
                       help='分类器dropout率（默认：0.1）')
    
    print("正在解析命令行参数...", flush=True)
    args = parser.parse_args()
    print(f"✓ 参数解析完成: data_path={args.data_path}, output_path={args.output_path}", flush=True)
    
    # 加载配置或使用默认值
    print("正在加载配置...", flush=True)
    if args.config:
        config = lib.load_config(args.config)
        model_params = config.get('model_params', {})
        print("✓ 从配置文件加载参数", flush=True)
        print(f"  is_y_cond={model_params.get('is_y_cond', False)}", flush=True)
    else:
        model_params = {}
        print("✓ 使用默认参数", flush=True)
        print("  ⚠️  警告：未提供配置文件，将使用默认is_y_cond=False", flush=True)
        print("  ⚠️  如果扩散模型使用is_y_cond=true，请使用--config参数", flush=True)
    
    # 设置模型参数
    print("正在设置模型参数...", flush=True)
    model_params['num_classes'] = args.num_classes
    model_params['classifier_params'] = {
        'd_layers': args.classifier_d_layers,
        'dropout': args.classifier_dropout
    }
    print("✓ 模型参数设置完成", flush=True)
    
    # 从数据集获取特征信息
    # 先加载数据集以获取特征信息
    print("正在准备数据转换配置...", flush=True)
    T_dict = {
        'normalization': 'quantile',
        'cat_encoding': 'one-hot',
        'y_policy': 'default',
        'num_nan_policy': '__none__',
        'cat_nan_policy': '__none__',
        'cat_min_frequency': '__none__'
    }
    T = lib.Transformations(**T_dict)
    print("✓ 数据转换配置完成", flush=True)
    
    print(f"正在加载数据集: {args.data_path}（这可能需要一些时间）...", flush=True)
    # 使用配置文件中的is_y_cond设置（如果提供）
    is_y_cond = model_params.get('is_y_cond', False)
    print(f"  Using is_y_cond={is_y_cond} (from model_params)", flush=True)
    dataset_temp = make_dataset(
        args.data_path,
        T,
        num_classes=model_params.get('num_classes', 0),
        is_y_cond=is_y_cond,  # 使用配置文件中的设置
        change_val=False
    )
    print("✓ 数据集加载完成", flush=True)
    
    num_numerical_features = dataset_temp.X_num['train'].shape[1] if dataset_temp.X_num is not None else 0
    num_classes_list = dataset_temp.get_category_sizes('train')
    
    print(f"Dataset information:")
    print(f"  Numerical features: {num_numerical_features}")
    print(f"  Categorical features: {len(num_classes_list)}")
    print(f"  Category sizes: {num_classes_list}")
    
    # 训练分类器
    classifier = train_classifier(
        data_path=args.data_path,
        model_params=model_params,
        num_numerical_features=num_numerical_features,
        num_classes=num_classes_list,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_path=args.output_path,
        seed=args.seed
    )
    
    print(f"\n✅ Classifier training completed!")


if __name__ == '__main__':
    print("脚本开始执行...", flush=True)
    main()
    print("脚本执行完成！", flush=True)

