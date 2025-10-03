#!/usr/bin/env python3
"""
使用Dummy环境快速验证训练流程

这个脚本用于:
1. 验证整个训练流程是否正常
2. 快速测试模型和环境的集成
3. 检查GPU/CPU设备是否正常工作
4. 预计运行时间: 10-20分钟

使用方法:
    python scripts/train_dummy.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from agsac.models import AGSACModel
from agsac.envs import DummyAGSACEnvironment
from agsac.training import AGSACTrainer

def print_separator(title="", char="=", width=70):
    """打印分隔线"""
    if title:
        title = f" {title} "
        padding = (width - len(title)) // 2
        print(char * padding + title + char * padding)
    else:
        print(char * width)

def main():
    print_separator("AGSAC Dummy环境训练验证")
    print(f"\n项目根目录: {project_root}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 设备检测
    print_separator("设备检测")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[OK] 使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建环境
    print_separator("创建环境")
    print("创建训练环境...")
    train_env = DummyAGSACEnvironment(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        pred_horizon=12,
        max_episode_steps=100,
        use_geometric_reward=True,
        device=device
    )
    print("[OK] 训练环境创建成功")
    
    print("创建评估环境...")
    eval_env = DummyAGSACEnvironment(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        pred_horizon=12,
        max_episode_steps=100,
        use_geometric_reward=True,
        device=device
    )
    print("[OK] 评估环境创建成功")
    
    # 测试环境
    print("\n测试环境reset和step...")
    obs = train_env.reset()
    print(f"[OK] Reset成功, observation keys: {list(obs.keys())}")
    
    action = np.random.randn(2)  # (linear_vel, angular_vel)
    obs, reward, done, info = train_env.step(action)
    print(f"[OK] Step成功, reward: {reward:.4f}, done: {done}")
    
    # 创建模型
    print_separator("创建模型")
    
    # 检查预训练权重
    pretrained_path = project_root / "pretrained/social_circle/evsczara1/torch_epoch150.pt"
    if not pretrained_path.exists():
        print(f"[WARNING] 预训练权重不存在: {pretrained_path}")
        print("[INFO] 将使用简化的轨迹预测器（fallback）")
        use_pretrained = False
        pretrained_weights_path = None
    else:
        print(f"[OK] 找到预训练权重: {pretrained_path}")
        use_pretrained = True
        pretrained_weights_path = pretrained_path
    
    print("\n创建AGSACModel...")
    model = AGSACModel(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        pred_horizon=12,
        action_dim=2,           # (linear_vel, angular_vel)
        hidden_dim=256,
        use_pretrained_predictor=use_pretrained,
        pretrained_weights_path=pretrained_weights_path,
        device=device
    ).to(device)
    
    print("[OK] 模型创建成功")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}")
    print(f"  冻结:   {frozen_params:,}")
    print(f"  参数预算满足: {'[OK]' if total_params < 2_000_000 else '[WARNING]'} ({total_params/1e6:.2f}M < 2M)")
    
    # 测试前向传播
    print("\n测试模型前向传播...")
    model.eval()
    with torch.no_grad():
        action_output = model(obs)
    print(f"[OK] 前向传播成功")
    print(f"  action shape: {action_output['action'].shape}")
    print(f"  log_prob shape: {action_output['log_prob'].shape}")
    
    # 创建训练器
    print_separator("创建训练器")
    print("配置训练器参数...")
    
    output_dir = project_root / 'outputs' / 'dummy_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = AGSACTrainer(
        model=model,
        env=train_env,
        eval_env=eval_env,
        buffer_capacity=1000,      # 小buffer快速验证
        seq_len=8,
        batch_size=16,
        warmup_episodes=5,         # 少量warmup
        updates_per_episode=20,    # 少量更新
        eval_interval=5,           # 频繁评估
        eval_episodes=3,
        save_interval=10,
        max_episodes=50,           # 快速验证50 episodes
        device=device,
        save_dir=str(output_dir),
        experiment_name='dummy_quick_test'
    )
    print("[OK] 训练器创建成功")
    print(f"\n训练配置:")
    print(f"  最大episodes: 50")
    print(f"  Warmup episodes: 5")
    print(f"  批次大小: 16")
    print(f"  序列长度: 8")
    print(f"  输出目录: {output_dir}")
    
    # 开始训练
    print_separator("开始训练")
    print("[INFO] 训练开始，预计10-20分钟...")
    print("[INFO] 按Ctrl+C可随时停止训练\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n[INFO] 训练被用户中断")
    except Exception as e:
        print(f"\n\n[ERROR] 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 训练完成
    print_separator("训练完成")
    print(f"\n训练结果已保存到: {output_dir}")
    print("\n生成的文件:")
    if output_dir.exists():
        for item in sorted(output_dir.rglob('*')):
            if item.is_file():
                size = item.stat().st_size
                print(f"  {item.relative_to(output_dir)} ({size:,} bytes)")
    
    print("\n后续步骤:")
    print("  1. 查看TensorBoard: tensorboard --logdir logs/tensorboard")
    print("  2. 查看checkpoint: ls outputs/dummy_test/checkpoints/")
    print("  3. 如果一切正常，可以开始完整训练:")
    print("     python scripts/train.py --config configs/training_full.yaml")
    
    print_separator()
    print("[SUCCESS] 验证完成！系统运行正常。")
    print_separator()
    
    return 0

if __name__ == "__main__":
    exit(main())


