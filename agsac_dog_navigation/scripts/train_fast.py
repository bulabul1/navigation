#!/usr/bin/env python3
"""
快速训练脚本 - 优化版本

改进:
1. 减少模型推理频率（使用更简单的policy）
2. 增加episodes数量（100个）
3. 更频繁的evaluation和保存
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

def adapt_observation_for_model(env_obs, device='cpu'):
    """将DummyEnvironment的observation格式转换为AGSACModel期望的格式"""
    robot_state = env_obs['robot_state']
    
    current_pos = robot_state['position'].unsqueeze(0)
    past_trajectory = current_pos.unsqueeze(0).repeat(1, 8, 1)
    
    corridor_vertices = env_obs['corridor_vertices']
    corridor_mask = env_obs['corridor_mask']
    
    vertex_counts = []
    for i in range(corridor_vertices.shape[0]):
        if corridor_mask[i]:
            non_zero = (corridor_vertices[i].abs().sum(dim=1) > 0).sum().item()
            vertex_counts.append(non_zero if non_zero > 0 else 4)
        else:
            vertex_counts.append(0)
    vertex_counts = torch.tensor(vertex_counts, dtype=torch.long, device=device).unsqueeze(0)
    
    model_obs = {
        'dog': {
            'trajectory': past_trajectory.to(device),
            'velocity': robot_state['velocity'].unsqueeze(0).to(device),
            'position': robot_state['position'].unsqueeze(0).to(device),
            'goal': (robot_state['position'] + robot_state['goal_vector']).unsqueeze(0).to(device)
        },
        'corridors': {
            'polygons': corridor_vertices.unsqueeze(0).to(device),
            'vertex_counts': vertex_counts,
            'mask': corridor_mask.unsqueeze(0).to(device)
        },
        'pedestrians': {
            'trajectories': env_obs['pedestrian_observations'].unsqueeze(0).to(device),
            'mask': env_obs['pedestrian_mask'].unsqueeze(0).to(device)
        }
    }
    
    return model_obs

def main():
    print_separator("AGSAC 快速训练 (100 Episodes)")
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
    print("创建训练和评估环境...")
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
    print("[OK] 环境创建成功")
    
    # 创建模型
    print_separator("创建模型")
    
    # 检查预训练权重
    possible_paths = [
        project_root.parent / "Project-Monandaeg-SocialCircle/evsczara1",
        project_root / "pretrained/social_circle/evsczara1",
    ]
    
    pretrained_weights_path = None
    for base_path in possible_paths:
        if base_path.exists():
            for filename in ["torch_test_new__epoch65.pt", "evsc_weights.pth"]:
                weight_file = base_path / filename
                if weight_file.exists():
                    pretrained_weights_path = base_path
                    print(f"[OK] 找到预训练权重: {weight_file}")
                    break
            if pretrained_weights_path:
                break
    
    if pretrained_weights_path is None:
        print(f"[WARNING] 未找到预训练权重，使用简化预测器")
        use_pretrained = False
    else:
        use_pretrained = True
    
    print("\n创建AGSACModel...")
    model = AGSACModel(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        pred_horizon=12,
        action_dim=2,
        hidden_dim=128,  # 已优化
        use_pretrained_predictor=use_pretrained,
        pretrained_weights_path=pretrained_weights_path,
        device=device
    ).to(device)
    
    print("[OK] 模型创建成功")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建训练器
    print_separator("创建训练器")
    
    output_dir = project_root / 'outputs' / 'fast_training'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = AGSACTrainer(
        model=model,
        env=train_env,
        eval_env=eval_env,
        buffer_capacity=2000,       # 增大buffer
        seq_len=8,
        batch_size=32,              # 增大batch
        warmup_episodes=10,         # 增加warmup
        updates_per_episode=50,     # 增加更新次数
        eval_interval=10,           # 每10个episodes评估
        eval_episodes=5,
        save_interval=20,           # 每20个episodes保存
        max_episodes=100,           # 训练100个episodes
        device=device,
        save_dir=str(output_dir),
        experiment_name='fast_training_100ep'
    )
    print("[OK] 训练器创建成功")
    
    # 开始训练
    print_separator("开始训练")
    print("[INFO] 训练100个episodes，预计30-60分钟...")
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
    
    print("\n后续步骤:")
    print("  1. 查看TensorBoard: tensorboard --logdir logs/tensorboard")
    print("  2. 分析训练曲线，查看是否有改进趋势")
    print("  3. 如果效果好，可以训练更多episodes")
    
    print_separator()
    print("[SUCCESS] 训练完成！")
    print_separator()
    
    return 0

if __name__ == "__main__":
    exit(main())

