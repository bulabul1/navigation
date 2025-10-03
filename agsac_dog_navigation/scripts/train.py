#!/usr/bin/env python3
"""
AGSAC训练脚本（基于配置文件）

使用方法：
    python scripts/train.py --config configs/fixed_scene.yaml
    python scripts/train.py --config configs/curriculum_learning.yaml
    python scripts/train.py --config configs/dynamic_scenes.yaml
    python scripts/train.py --config configs/debug.yaml
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import argparse
from datetime import datetime

from configs.train_config import AGSACConfig
from agsac.models import AGSACModel
from agsac.envs import DummyAGSACEnvironment
from agsac.training import AGSACTrainer


def main():
    parser = argparse.ArgumentParser(description="AGSAC训练（基于配置文件）")
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径 (例如: configs/fixed_scene.yaml)')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU（覆盖配置文件）')
    args = parser.parse_args()
    
    # ===== 1. 加载配置 =====
    print("=" * 70)
    print("AGSAC训练开始")
    print("=" * 70)
    
    config_path = project_root / args.config
    print(f"\n[配置] 加载配置文件: {config_path}")
    config = AGSACConfig.from_yaml(config_path)
    
    # 设备覆盖
    if args.cpu:
        config.device = 'cpu'
        config.env.device = 'cpu'
        config.model.device = 'cpu'
    else:
        # 自动检测CUDA
        if torch.cuda.is_available() and config.device == 'cuda':
            config.device = 'cuda'
            config.env.device = 'cuda'
            config.model.device = 'cuda'
        else:
            config.device = 'cpu'
            config.env.device = 'cpu'
            config.model.device = 'cpu'
    
    print(f"[配置] 训练模式: {config.mode}")
    print(f"[配置] 设备: {config.device}")
    print(f"[配置] Episodes: {config.training.episodes}")
    print(f"[配置] 使用Corridor生成器: {config.env.use_corridor_generator}")
    print(f"[配置] 课程学习: {config.env.curriculum_learning}")
    
    # 设置随机种子
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        print(f"[配置] 随机种子: {config.seed}")
    
    # ===== 2. 创建日志目录 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "logs" / f"{config.training.experiment_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config.to_yaml(log_dir / "config.yaml")
    config.to_json(log_dir / "config.json")
    print(f"[配置] 日志目录: {log_dir}")
    print("")
    
    # ===== 3. 创建环境 =====
    print("[环境] 创建训练环境...")
    env = DummyAGSACEnvironment(
        max_pedestrians=config.env.max_pedestrians,
        max_corridors=config.env.max_corridors,
        max_vertices=config.env.max_vertices,
        obs_horizon=config.env.obs_horizon,
        pred_horizon=config.env.pred_horizon,
        max_episode_steps=config.env.max_episode_steps,
        use_geometric_reward=config.env.use_geometric_reward,
        use_corridor_generator=config.env.use_corridor_generator,
        curriculum_learning=config.env.curriculum_learning,
        scenario_seed=config.env.scenario_seed,
        device=config.env.device
    )
    print(f"[环境] 创建成功")
    print("")
    
    # ===== 4. 创建模型 =====
    print("[模型] 创建AGSAC模型...")
    
    # 查找预训练权重
    if config.model.use_pretrained_predictor and config.model.pretrained_weights_path:
        pretrained_path = Path(config.model.pretrained_weights_path)
        if not pretrained_path.exists():
            print(f"[警告] 预训练权重不存在: {pretrained_path}")
            pretrained_path = None
    else:
        pretrained_path = None
    
    model = AGSACModel(
        action_dim=config.model.action_dim,
        hidden_dim=config.model.hidden_dim,
        num_modes=config.model.num_modes,
        max_pedestrians=config.model.max_pedestrians,
        max_corridors=config.model.max_corridors,
        max_vertices=config.model.max_vertices,
        obs_horizon=config.model.obs_horizon,
        pred_horizon=config.model.pred_horizon,
        actor_lr=config.training.actor_lr,
        critic_lr=config.training.critic_lr,
        gamma=config.training.gamma,
        tau=config.training.tau,
        use_pretrained_predictor=pretrained_path is not None,
        pretrained_weights_path=str(pretrained_path) if pretrained_path else None,
        device=config.model.device
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 可训练参数: {trainable_params:,}")
    print("")
    
    # ===== 5. 创建训练器 =====
    print("[训练器] 创建AGSACTrainer...")
    trainer = AGSACTrainer(
        model=model,
        env=env,
        eval_env=None,
        buffer_capacity=config.training.buffer_capacity,
        seq_len=config.training.seq_len,
        batch_size=config.training.batch_size,
        warmup_episodes=config.training.warmup_episodes,
        updates_per_episode=config.training.updates_per_episode,
        eval_interval=config.training.eval_interval,
        save_interval=config.training.save_interval,
        log_interval=config.training.log_interval,
        max_episodes=config.training.episodes,
        device=config.device,
        save_dir=str(log_dir),
        experiment_name=config.training.experiment_name,
        use_tensorboard=config.training.use_tensorboard
    )
    print(f"[训练器] 创建成功")
    print(f"  - Buffer容量: {config.training.buffer_capacity}")
    print(f"  - 序列长度: {config.training.seq_len}")
    print(f"  - Batch大小: {config.training.batch_size}")
    print("")
    
    # ===== 6. 开始训练 =====
    print("=" * 70)
    print("开始训练")
    print("=" * 70)
    print("")
    
    # AGSACTrainer的train方法不需要参数（已在init中设置）
    train_history = trainer.train()
    
    print("")
    print("=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"日志保存在: {log_dir}")
    print(f"模型保存在: {trainer.save_dir}")


if __name__ == '__main__':
    main()
