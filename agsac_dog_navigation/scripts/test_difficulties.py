#!/usr/bin/env python3
"""
测试模型在不同难度下的表现

测试easy/medium/hard三种难度，了解模型的学习程度
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import argparse
from typing import Dict

from configs.train_config import AGSACConfig
from agsac.models import AGSACModel
from agsac.envs import DummyAGSACEnvironment


def _add_batch_dim(obs: Dict, env, device: str) -> Dict:
    """转换观察格式"""
    robot_state = obs['robot_state']
    position = robot_state['position'].unsqueeze(0).to(device)
    velocity = robot_state['velocity'].unsqueeze(0).to(device)
    goal = obs['goal'].unsqueeze(0).to(device)
    
    obs_horizon = env.obs_horizon
    trajectory = position.unsqueeze(1).repeat(1, obs_horizon, 1)
    
    model_obs = {
        'dog': {
            'trajectory': trajectory,
            'velocity': velocity,
            'position': position,
            'goal': goal
        },
        'pedestrians': {
            'trajectories': obs['pedestrian_observations'].unsqueeze(0).to(device),
            'mask': obs['pedestrian_mask'].unsqueeze(0).to(device)
        },
        'corridors': {
            'polygons': obs['corridor_vertices'].unsqueeze(0).to(device),
            'vertex_counts': torch.full((1, 10), obs['corridor_vertices'].shape[1], dtype=torch.long, device=device),
            'mask': obs['corridor_mask'].unsqueeze(0).to(device)
        }
    }
    
    return model_obs


def test_difficulty(model, env, difficulty: str, num_episodes: int = 10, device: str = 'cuda'):
    """
    测试特定难度下的模型表现
    
    Args:
        model: 训练好的模型
        env: 环境
        difficulty: 难度级别 ('easy', 'medium', 'hard')
        num_episodes: 测试episodes数量
        device: 设备
    """
    
    print(f"\n🎯 测试 {difficulty.upper()} 难度 ({num_episodes} episodes)")
    print("-" * 50)
    
    # 设置难度
    if difficulty == 'easy':
        env.episode_count = 0  # 0-50 episodes
    elif difficulty == 'medium':
        env.episode_count = 100  # 50-150 episodes
    else:  # hard
        env.episode_count = 200  # 150+ episodes
    
    returns = []
    lengths = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < 200:
            # 转换观察格式
            model_obs = _add_batch_dim(obs, env, device)
            
            # 获取动作
            with torch.no_grad():
                model_output = model(
                    model_obs,
                    hidden_states=None,
                    deterministic=True,
                    return_attention=False
                )
            
            action = model_output['action'].squeeze(0).cpu().numpy()
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            episode_return += reward
            episode_length += 1
        
        # 记录结果
        returns.append(episode_return)
        lengths.append(episode_length)
        
        # 统计终止原因
        if info.get('goal_reached', False):
            success_count += 1
            status = "✅ 成功"
        elif info.get('collision', False):
            collision_count += 1
            status = "❌ 碰撞"
        else:
            timeout_count += 1
            status = "⏰ 超时"
        
        print(f"  Episode {episode + 1:2d}: Return={episode_return:8.2f}, Length={episode_length:3d}, {status}")
    
    # 计算统计指标
    returns = np.array(returns)
    lengths = np.array(lengths)
    
    success_rate = success_count / num_episodes
    collision_rate = collision_count / num_episodes
    timeout_rate = timeout_count / num_episodes
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(lengths)
    
    print(f"\n📊 {difficulty.upper()} 难度结果:")
    print(f"  成功率: {success_rate:.1%} ({success_count}/{num_episodes})")
    print(f"  碰撞率: {collision_rate:.1%} ({collision_count}/{num_episodes})")
    print(f"  超时率: {timeout_rate:.1%} ({timeout_count}/{num_episodes})")
    print(f"  平均Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"  平均Length: {mean_length:.1f}")
    print(f"  最佳Return: {np.max(returns):.2f}")
    print(f"  最差Return: {np.min(returns):.2f}")
    
    return {
        'difficulty': difficulty,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_length': mean_length,
        'best_return': np.max(returns),
        'worst_return': np.min(returns)
    }


def main():
    parser = argparse.ArgumentParser(description="测试模型在不同难度下的表现")
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=10,
                        help='每个难度测试的episodes数量')
    parser.add_argument('--cpu', action='store_true',
                        help='使用CPU')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AGSAC模型难度测试")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"每难度测试episodes: {args.episodes}")
    print()
    
    # 设置设备
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    config = AGSACConfig.from_yaml('configs/default.yaml')
    
    # 创建环境
    env = DummyAGSACEnvironment(
        use_corridor_generator=True,
        curriculum_learning=True,  # 启用课程学习
        scenario_seed=42
    )
    
    # 创建模型
    model = AGSACModel(
        device=device,
        action_dim=config.model.action_dim,
        hidden_dim=config.model.hidden_dim,
        num_modes=config.model.num_modes,
        max_pedestrians=config.model.max_pedestrians,
        max_corridors=config.model.max_corridors,
        max_vertices=config.model.max_vertices,
        obs_horizon=config.model.obs_horizon,
        pred_horizon=config.model.pred_horizon,
        use_pretrained_predictor=config.model.use_pretrained_predictor,
        pretrained_weights_path=config.model.pretrained_weights_path
    )
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ 模型加载成功")
    
    # 测试不同难度
    results = []
    
    for difficulty in ['easy', 'medium', 'hard']:
        result = test_difficulty(model, env, difficulty, args.episodes, device)
        results.append(result)
    
    # 总结对比
    print("\n" + "=" * 60)
    print("难度对比总结")
    print("=" * 60)
    print(f"{'难度':<8} {'成功率':<8} {'碰撞率':<8} {'平均Return':<12} {'平均Length':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['difficulty']:<8} "
              f"{result['success_rate']:<8.1%} "
              f"{result['collision_rate']:<8.1%} "
              f"{result['mean_return']:<12.2f} "
              f"{result['mean_length']:<10.1f}")
    
    print("\n🎉 测试完成！")


if __name__ == "__main__":
    main()
