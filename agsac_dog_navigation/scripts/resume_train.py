#!/usr/bin/env python3
"""
AGSAC恢复训练脚本

使用方法：
    python scripts/resume_train.py --checkpoint logs/curriculum_training_20251004_124233/curriculum_training/checkpoint_ep201.pt
    python scripts/resume_train.py --checkpoint logs/curriculum_training_20251004_124233/curriculum_training/best_model.pt
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
    parser = argparse.ArgumentParser(description="AGSAC恢复训练")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint文件路径')
    parser.add_argument('--config', type=str, default='configs/curriculum_learning.yaml',
                        help='配置文件路径 (默认: configs/curriculum_learning.yaml)')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU（覆盖配置文件）')
    args = parser.parse_args()
    
    # ===== 1. 加载配置 =====
    print("=" * 70)
    print("AGSAC恢复训练开始")
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
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.env.device = config.device
        config.model.device = config.device
    
    print(f"[设备] 使用设备: {config.device}")
    
    # ===== 2. 创建环境 =====
    print(f"\n[环境] 创建环境...")
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
        corridor_constraint_mode=config.env.corridor_constraint_mode,
        corridor_penalty_weight=config.env.corridor_penalty_weight,
        corridor_penalty_cap=config.env.corridor_penalty_cap,
        progress_reward_weight=config.env.progress_reward_weight,
        step_penalty_weight=config.env.step_penalty_weight,
        enable_step_limit=config.env.enable_step_limit,
        device=config.env.device
    )
    print(f"[环境] 环境创建完成")
    
    # ===== 3. 创建模型 =====
    print(f"\n[模型] 创建模型...")
    
    # 预训练模型路径
    pretrained_path = None
    if config.model.use_pretrained_predictor and config.model.pretrained_weights_path:
        pretrained_path = project_root / config.model.pretrained_weights_path
    
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
        alpha=config.training.alpha,
        encoder_lr=getattr(config.training, 'encoder_lr', None),  # 可选参数
        gamma=config.training.gamma,
        tau=config.training.tau,
        auto_entropy=getattr(config.model, 'auto_entropy', True),
        use_pretrained_predictor=pretrained_path is not None,
        pretrained_weights_path=str(pretrained_path) if pretrained_path else None,
        device=config.model.device
    )
    print(f"[模型] 模型创建完成")
    
    # ===== 4. 创建训练器 =====
    print(f"\n[训练器] 创建训练器...")
    
    # 创建新的保存目录（添加时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / 'logs'
    experiment_name_with_timestamp = f"{config.training.experiment_name}_{timestamp}"
    
    # 固定评估环境：禁用课程学习并固定seed（若提供）
    eval_env = DummyAGSACEnvironment(
        max_pedestrians=config.env.max_pedestrians,
        max_corridors=config.env.max_corridors,
        max_vertices=config.env.max_vertices,
        obs_horizon=config.env.obs_horizon,
        pred_horizon=config.env.pred_horizon,
        max_episode_steps=config.env.max_episode_steps,
        use_geometric_reward=config.env.use_geometric_reward,
        use_corridor_generator=True,
        curriculum_learning=False,
        scenario_seed=config.seed if config.seed is not None else 42,
        corridor_constraint_mode=config.env.corridor_constraint_mode,
        corridor_penalty_weight=config.env.corridor_penalty_weight,
        corridor_penalty_cap=config.env.corridor_penalty_cap,
        progress_reward_weight=config.env.progress_reward_weight,
        step_penalty_weight=config.env.step_penalty_weight,
        enable_step_limit=config.env.enable_step_limit,
        device=config.env.device
    )

    trainer = AGSACTrainer(
        model=model,
        env=env,
        eval_env=eval_env,
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
        save_dir=str(save_dir),
        experiment_name=experiment_name_with_timestamp,
        use_tensorboard=config.training.use_tensorboard,
        eval_seed=(config.seed if config.seed is not None else 1234)
    )
    
    # ===== 5. 加载checkpoint =====
    print(f"\n[恢复] 加载checkpoint: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint)
    
    current_episode = trainer.episode_count
    print(f"[恢复] 当前状态:")
    print(f"  - Episode: {current_episode}")
    print(f"  - Total steps: {trainer.total_steps}")
    print(f"  - Best eval return: {trainer.best_eval_return:.2f}")
    
    # ===== 5.5. 保存配置文件（便于追溯） =====
    print(f"\n[配置] 保存配置文件...")
    config_save_path = trainer.save_dir
    
    # 保存为YAML格式（可读性好）
    config_yaml_path = config_save_path / 'config.yaml'
    config.to_yaml(str(config_yaml_path))
    print(f"  - YAML: {config_yaml_path}")
    
    # 保存为JSON格式（程序可读）
    config_json_path = config_save_path / 'config.json'
    config.to_json(str(config_json_path))
    print(f"  - JSON: {config_json_path}")
    
    # ===== 6. 调整训练目标（补到总集数） =====
    target_episodes = config.training.episodes
    remaining_episodes = max(0, target_episodes - current_episode)
    
    print(f"\n[训练] 继续训练...")
    print(f"  - 已完成: {current_episode} episodes")
    print(f"  - 目标总数: {target_episodes} episodes")
    print(f"  - 将再训练: {remaining_episodes} episodes")
    
    # 调整max_episodes：补到总集数而非再训练episodes次
    trainer.max_episodes = remaining_episodes
    
    if remaining_episodes == 0:
        print(f"\n[完成] 已达到目标episodes数，无需继续训练")
    else:
        # train()方法不接受参数，使用trainer初始化时的配置
        trainer.train()
    
    print(f"\n[完成] 训练完成！")
    print(f"  - 最终 Episode: {trainer.episode_count}")
    print(f"  - 最终 Total steps: {trainer.total_steps}")
    print(f"  - 最佳 eval return: {trainer.best_eval_return:.2f}")


if __name__ == "__main__":
    main()
