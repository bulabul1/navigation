#!/usr/bin/env python3
"""
AGSAC评估脚本

使用方法:
    python scripts/evaluate.py --model outputs/models/best_model.pth
    python scripts/evaluate.py --model outputs/models/best_model.pth --num_episodes 100
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from omegaconf import OmegaConf

from agsac.models.agent import AGSACAgent
from agsac.envs.gazebo_env import GazeboNavigationEnv
from agsac.utils.logger import setup_logger
from agsac.utils.data_processing import set_seed
from agsac.utils.visualization import save_evaluation_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AGSAC评估脚本")
    
    # 模型路径
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    
    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="配置文件路径"
    )
    
    # 评估参数
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="评估回合数"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="每回合最大步数"
    )
    
    # 输出配置
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="输出目录"
    )
    
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="保存评估视频"
    )
    
    parser.add_argument(
        "--save_trajectories",
        action="store_true",
        help="保存轨迹数据"
    )
    
    # 设备配置
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="计算设备"
    )
    
    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    return parser.parse_args()


def setup_device(device_arg):
    """设置计算设备"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_arg
    
    print(f"使用设备: {device}")
    return device


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def evaluate_episode(agent, env, max_steps, episode_idx, logger):
    """评估单个回合"""
    obs = env.reset()
    done = False
    step = 0
    
    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'positions': [],
        'collisions': [],
        'success': False
    }
    
    total_reward = 0
    
    while not done and step < max_steps:
        # 获取动作
        with torch.no_grad():
            action = agent.select_action(obs, deterministic=True)
        
        # 执行动作
        next_obs, reward, done, info = env.step(action)
        
        # 记录数据
        episode_data['observations'].append(obs)
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        episode_data['positions'].append(info.get('position', [0, 0]))
        episode_data['collisions'].append(info.get('collision', False))
        
        total_reward += reward
        obs = next_obs
        step += 1
    
    # 检查是否成功到达目标
    episode_data['success'] = info.get('success', False)
    episode_data['total_reward'] = total_reward
    episode_data['steps'] = step
    
    logger.info(f"回合 {episode_idx + 1}: 步数={step}, 奖励={total_reward:.2f}, 成功={episode_data['success']}")
    
    return episode_data


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = setup_device(args.device)
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        name="AGSAC_Evaluator",
        log_dir=output_dir / "logs",
        level="INFO"
    )
    
    logger.info("=" * 50)
    logger.info("AGSAC评估开始")
    logger.info("=" * 50)
    logger.info(f"模型路径: {args.model}")
    logger.info(f"评估回合数: {args.num_episodes}")
    logger.info(f"每回合最大步数: {args.max_steps}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"设备: {device}")
    
    try:
        # 创建环境
        env = GazeboNavigationEnv(config.environment)
        logger.info("环境创建成功")
        
        # 创建智能体
        agent = AGSACAgent(config.model, device=device)
        logger.info("智能体创建成功")
        
        # 加载模型
        checkpoint = torch.load(args.model, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        logger.info("模型加载成功")
        
        # 开始评估
        all_episodes = []
        success_count = 0
        collision_count = 0
        total_rewards = []
        episode_lengths = []
        
        for episode_idx in range(args.num_episodes):
            episode_data = evaluate_episode(
                agent, env, args.max_steps, episode_idx, logger
            )
            
            all_episodes.append(episode_data)
            
            # 统计信息
            if episode_data['success']:
                success_count += 1
            if any(episode_data['collisions']):
                collision_count += 1
            
            total_rewards.append(episode_data['total_reward'])
            episode_lengths.append(episode_data['steps'])
        
        # 计算统计指标
        success_rate = success_count / args.num_episodes
        collision_rate = collision_count / args.num_episodes
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        # 输出结果
        logger.info("=" * 50)
        logger.info("评估结果:")
        logger.info(f"成功率: {success_rate:.2%}")
        logger.info(f"碰撞率: {collision_rate:.2%}")
        logger.info(f"平均奖励: {avg_reward:.2f}")
        logger.info(f"平均步数: {avg_length:.1f}")
        logger.info("=" * 50)
        
        # 保存结果
        results = {
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'episodes': all_episodes
        }
        
        save_evaluation_results(results, output_dir, args)
        
        logger.info("评估完成!")
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        raise
    finally:
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
