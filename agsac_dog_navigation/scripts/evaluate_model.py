#!/usr/bin/env python3
"""
AGSAC模型评估脚本

评估训练好的模型在不同场景下的性能：
1. 基本导航能力
2. 不同方向导航
3. 复杂环境适应
4. 统计性能指标
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
from typing import Dict

from configs.train_config import AGSACConfig
from agsac.models import AGSACModel
from agsac.envs import DummyAGSACEnvironment
from agsac.training import AGSACTrainer


def _add_batch_dim(obs: Dict, env, device: str) -> Dict:
    """
    为观测添加batch维度并转换为AGSACModel期望的格式
    """
    # 提取机器狗信息
    robot_state = obs['robot_state']
    position = robot_state['position'].unsqueeze(0).to(device)  # (1, 2)
    velocity = robot_state['velocity'].unsqueeze(0).to(device)  # (1, 2)
    goal = obs['goal'].unsqueeze(0).to(device)  # (1, 2)
    
    # 构造trajectory（使用position复制obs_horizon次）
    obs_horizon = env.obs_horizon
    trajectory = position.unsqueeze(1).repeat(1, obs_horizon, 1)  # (1, obs_horizon, 2)
    
    # 构造模型期望的格式
    model_obs = {
        'dog': {
            'trajectory': trajectory,  # (1, obs_horizon, 2)
            'velocity': velocity,      # (1, 2)
            'position': position,      # (1, 2)
            'goal': goal               # (1, 2)
        },
        'pedestrians': {
            'trajectories': obs['pedestrian_observations'].unsqueeze(0).to(device),  # (1, 10, 8, 2)
            'mask': obs['pedestrian_mask'].unsqueeze(0).to(device)  # (1, 10)
        },
        'corridors': {
            'polygons': obs['corridor_vertices'].unsqueeze(0).to(device),  # (1, 10, 20, 2)
            'vertex_counts': torch.full((1, 10), obs['corridor_vertices'].shape[1], dtype=torch.long, device=device),  # (1, 10)
            'mask': obs['corridor_mask'].unsqueeze(0).to(device)  # (1, 10)
        }
    }
    
    return model_obs


def evaluate_model(model_path: str, num_episodes: int = 50, save_results: bool = True):
    """
    评估模型性能
    
    Args:
        model_path: 模型文件路径
        num_episodes: 评估episodes数量
        save_results: 是否保存结果
    """
    
    print("=" * 60)
    print("AGSAC模型评估")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"评估episodes: {num_episodes}")
    print()
    
    # 1. 加载配置（使用与训练时相同的配置）
    config = AGSACConfig.from_yaml('configs/default.yaml')
    
    # 2. 创建环境
    env = DummyAGSACEnvironment(
        use_corridor_generator=True,
        curriculum_learning=False,  # 关闭课程学习，使用固定难度
        scenario_seed=42
    )
    
    # 3. 创建模型（使用与训练时完全相同的配置）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    # 4. 加载模型
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载模型权重成功")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ 加载模型权重成功")
    
    model.eval()
    
    # 5. 评估统计
    results = {
        'episodes': [],
        'returns': [],
        'lengths': [],
        'success_rate': 0.0,
        'collision_rate': 0.0,
        'timeout_rate': 0.0,
        'mean_return': 0.0,
        'std_return': 0.0,
        'mean_length': 0.0,
        'std_length': 0.0
    }
    
    print("\n开始评估...")
    print("-" * 60)
    
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    with torch.no_grad():
        for episode in range(num_episodes):
            # 重置环境
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
            results['episodes'].append(episode + 1)
            results['returns'].append(episode_return)
            results['lengths'].append(episode_length)
            
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
            
            # 打印进度
            if (episode + 1) % 10 == 0 or episode < 5:
                print(f"Episode {episode + 1:3d}: Return={episode_return:8.2f}, Length={episode_length:3d}, {status}")
    
    # 6. 计算统计指标
    returns = np.array(results['returns'])
    lengths = np.array(results['lengths'])
    
    results['success_rate'] = success_count / num_episodes
    results['collision_rate'] = collision_count / num_episodes
    results['timeout_rate'] = timeout_count / num_episodes
    results['mean_return'] = float(np.mean(returns))
    results['std_return'] = float(np.std(returns))
    results['mean_length'] = float(np.mean(lengths))
    results['std_length'] = float(np.std(lengths))
    
    # 7. 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总episodes: {num_episodes}")
    print(f"成功率: {results['success_rate']:.1%} ({success_count}/{num_episodes})")
    print(f"碰撞率: {results['collision_rate']:.1%} ({collision_count}/{num_episodes})")
    print(f"超时率: {results['timeout_rate']:.1%} ({timeout_count}/{num_episodes})")
    print()
    print(f"平均Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"平均Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"最佳Return: {np.max(returns):.2f}")
    print(f"最差Return: {np.min(returns):.2f}")
    
    # 8. 保存结果
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        
        # 保存详细结果
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n结果已保存到: {results_file}")
        
        # 生成简单报告
        report_file = f"evaluation_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("AGSAC模型评估报告\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估episodes: {num_episodes}\n\n")
            f.write("性能指标:\n")
            f.write(f"  成功率: {results['success_rate']:.1%}\n")
            f.write(f"  碰撞率: {results['collision_rate']:.1%}\n")
            f.write(f"  超时率: {results['timeout_rate']:.1%}\n")
            f.write(f"  平均Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}\n")
            f.write(f"  平均Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}\n")
            f.write(f"  最佳Return: {np.max(returns):.2f}\n")
            f.write(f"  最差Return: {np.min(returns):.2f}\n")
        
        print(f"报告已保存到: {report_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="AGSAC模型评估")
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=50,
                        help='评估episodes数量 (默认: 50)')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存结果文件')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    # 运行评估
    try:
        results = evaluate_model(
            model_path=args.model,
            num_episodes=args.episodes,
            save_results=not args.no_save
        )
        
        print("\n🎉 评估完成！")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
