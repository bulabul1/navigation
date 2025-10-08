#!/usr/bin/env python3
"""
训练历史分析脚本

分析训练过程中的关键指标，了解模型学习趋势
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def analyze_training_history(history_file: str):
    """
    分析训练历史数据
    
    Args:
        history_file: 训练历史JSON文件路径
    """
    
    print("=" * 60)
    print("AGSAC训练历史分析")
    print("=" * 60)
    
    # 加载数据
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    # 提取关键指标
    episode_returns = np.array(data['episode_returns'])
    episode_lengths = np.array(data['episode_lengths'])
    eval_returns = np.array(data['eval_returns'])
    
    # 计算成功率（基于Return > 0）
    eval_success_rates = (eval_returns > 0).astype(float)
    
    print(f"训练episodes: {len(episode_returns)}")
    print(f"评估次数: {len(eval_returns)}")
    print()
    
    # 1. 训练Return分析
    print("📈 训练Return分析:")
    print(f"  初始Return (前10个episodes): {np.mean(episode_returns[:10]):.2f} ± {np.std(episode_returns[:10]):.2f}")
    print(f"  最终Return (后10个episodes): {np.mean(episode_returns[-10:]):.2f} ± {np.std(episode_returns[-10:]):.2f}")
    print(f"  最佳Return: {np.max(episode_returns):.2f}")
    print(f"  最差Return: {np.min(episode_returns):.2f}")
    print()
    
    # 2. 评估Return分析
    print("📊 评估Return分析:")
    print(f"  初始评估Return: {eval_returns[0]:.2f}")
    print(f"  最终评估Return: {eval_returns[-1]:.2f}")
    print(f"  最佳评估Return: {np.max(eval_returns):.2f}")
    print(f"  平均评估Return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")
    print()
    
    # 3. 成功率分析
    print("🎯 成功率分析:")
    print(f"  初始成功率: {eval_success_rates[0]:.1%}")
    print(f"  最终成功率: {eval_success_rates[-1]:.1%}")
    print(f"  最佳成功率: {np.max(eval_success_rates):.1%}")
    print(f"  平均成功率: {np.mean(eval_success_rates):.1%} ± {np.std(eval_success_rates):.1%}")
    print()
    
    # 4. Episode长度分析
    print("⏱️ Episode长度分析:")
    print(f"  初始平均长度: {episode_lengths[0]:.1f}")
    print(f"  最终平均长度: {episode_lengths[-1]:.1f}")
    print(f"  平均长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print()
    
    # 5. 学习趋势分析
    print("📈 学习趋势分析:")
    
    # 计算滑动平均
    window_size = 20
    if len(episode_returns) >= window_size:
        moving_avg = np.convolve(episode_returns, np.ones(window_size)/window_size, mode='valid')
        print(f"  训练Return趋势 (20-episode滑动平均):")
        print(f"    开始: {moving_avg[0]:.2f}")
        print(f"    结束: {moving_avg[-1]:.2f}")
        print(f"    改善: {moving_avg[-1] - moving_avg[0]:.2f}")
    
    # 评估趋势
    if len(eval_returns) > 1:
        eval_improvement = eval_returns[-1] - eval_returns[0]
        print(f"  评估Return改善: {eval_improvement:.2f}")
        
        success_improvement = eval_success_rates[-1] - eval_success_rates[0]
        print(f"  成功率改善: {success_improvement:.1%}")
    
    print()
    
    # 6. 课程学习阶段分析
    print("🎓 课程学习阶段分析:")
    total_episodes = len(episode_returns)
    
    # 根据配置文件，课程学习阶段：
    # - Easy: 0-50 episodes
    # - Medium: 50-150 episodes  
    # - Hard: 150+ episodes
    
    if total_episodes > 50:
        easy_returns = episode_returns[:50]
        print(f"  Easy阶段 (0-50): {np.mean(easy_returns):.2f} ± {np.std(easy_returns):.2f}")
    
    if total_episodes > 150:
        medium_returns = episode_returns[50:150]
        print(f"  Medium阶段 (50-150): {np.mean(medium_returns):.2f} ± {np.std(medium_returns):.2f}")
        
        hard_returns = episode_returns[150:]
        print(f"  Hard阶段 (150+): {np.mean(hard_returns):.2f} ± {np.std(hard_returns):.2f}")
    
    print()
    
    # 7. 问题诊断
    print("🔍 问题诊断:")
    
    # 检查是否有学习迹象
    if len(eval_returns) > 1:
        if eval_returns[-1] > eval_returns[0]:
            print("  ✅ 评估Return有改善")
        else:
            print("  ❌ 评估Return没有改善")
    
    if eval_success_rates[-1] > 0.1:  # 10%以上成功率
        print("  ✅ 模型有一定成功率")
    else:
        print("  ❌ 模型成功率极低")
    
    # 检查训练稳定性
    recent_returns = episode_returns[-20:] if len(episode_returns) >= 20 else episode_returns
    return_std = np.std(recent_returns)
    if return_std < 200:  # 标准差小于200认为较稳定
        print("  ✅ 训练相对稳定")
    else:
        print("  ❌ 训练不够稳定")
    
    print()
    
    # 8. 建议
    print("💡 改进建议:")
    
    if eval_success_rates[-1] < 0.1:
        print("  1. 成功率极低，建议:")
        print("     - 增加训练episodes")
        print("     - 调整奖励函数")
        print("     - 检查环境设置")
    
    if np.mean(eval_returns) < 0:
        print("  2. 平均Return为负，建议:")
        print("     - 降低环境难度")
        print("     - 增加正向奖励")
        print("     - 检查碰撞惩罚")
    
    if return_std > 300:
        print("  3. 训练不稳定，建议:")
        print("     - 降低学习率")
        print("     - 增加经验回放")
        print("     - 调整网络结构")
    
    print("  4. 方向多样性问题:")
    print("     - 实现全方向目标生成")
    print("     - 继续训练提高泛化能力")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="分析训练历史")
    parser.add_argument('--history', type=str, 
                        default='logs/curriculum_training_20251004_124233/curriculum_training/training_history.json',
                        help='训练历史文件路径')
    
    args = parser.parse_args()
    
    if not Path(args.history).exists():
        print(f"❌ 文件不存在: {args.history}")
        return
    
    analyze_training_history(args.history)


if __name__ == "__main__":
    main()
