"""
全面分析最近一次训练的表现
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 读取训练历史
log_dir = Path("logs/resume_training_optimized_20251006_184735")
with open(log_dir / "training_history.json", 'r') as f:
    history = json.load(f)

episode_returns = np.array(history['episode_returns'])
episode_lengths = np.array(history['episode_lengths'])

print("=" * 80)
print("训练总结报告".center(80))
print("=" * 80)

# 基本统计
total_episodes = len(episode_returns)
print(f"\n📊 **训练基本信息**")
print(f"  总Episode数: {total_episodes}")
print(f"  Episode Return 范围: [{episode_returns.min():.2f}, {episode_returns.max():.2f}]")
print(f"  Episode Length 范围: [{episode_lengths.min():.0f}, {episode_lengths.max():.0f}]")

# 按阶段分析 (每100 episodes)
print(f"\n📈 **分阶段表现** (每100集统计)")
print(f"{'阶段':<15} {'平均Return':<15} {'平均Length':<15} {'成功率':<15}")
print("-" * 60)

for i in range(0, total_episodes, 100):
    end_idx = min(i + 100, total_episodes)
    stage_returns = episode_returns[i:end_idx]
    stage_lengths = episode_lengths[i:end_idx]
    
    # 成功率：Return > 0
    success_rate = (stage_returns > 0).sum() / len(stage_returns)
    
    print(f"Ep {i:3d}-{end_idx:3d}    "
          f"{stage_returns.mean():>10.2f}     "
          f"{stage_lengths.mean():>10.1f}     "
          f"{success_rate:>10.1%}")

# 最后50集详细分析
print(f"\n🎯 **最后50集详细分析**")
last_50_returns = episode_returns[-50:]
last_50_lengths = episode_lengths[-50:]

success_count = (last_50_returns > 0).sum()
collision_count = (last_50_returns < -50).sum()  # 假设碰撞导致大负奖励
timeout_count = 50 - success_count - collision_count

print(f"  平均Return: {last_50_returns.mean():.2f} (±{last_50_returns.std():.2f})")
print(f"  平均Length: {last_50_lengths.mean():.1f} (±{last_50_lengths.std():.1f})")
print(f"  成功到达: {success_count}/50 ({success_count/50:.1%})")
print(f"  碰撞终止: {collision_count}/50 ({collision_count/50:.1%})")
print(f"  超时终止: {timeout_count}/50 ({timeout_count/50:.1%})")

# 最佳/最差表现
print(f"\n🏆 **极值分析**")
best_idx = episode_returns.argmax()
worst_idx = episode_returns.argmin()

print(f"  最佳Episode: Ep {best_idx} (Return={episode_returns[best_idx]:.2f}, Length={episode_lengths[best_idx]:.0f})")
print(f"  最差Episode: Ep {worst_idx} (Return={episode_returns[worst_idx]:.2f}, Length={episode_lengths[worst_idx]:.0f})")

# Episode Length分布分析
print(f"\n⏱️  **Episode Length分布**")
very_short = (episode_lengths <= 10).sum()
short = ((episode_lengths > 10) & (episode_lengths <= 30)).sum()
medium = ((episode_lengths > 30) & (episode_lengths <= 100)).sum()
long = (episode_lengths > 100).sum()

print(f"  极短 (≤10步): {very_short}/{total_episodes} ({very_short/total_episodes:.1%})")
print(f"  短 (11-30步): {short}/{total_episodes} ({short/total_episodes:.1%})")
print(f"  中 (31-100步): {medium}/{total_episodes} ({medium/total_episodes:.1%})")
print(f"  长 (>100步): {long}/{total_episodes} ({long/total_episodes:.1%})")

# 趋势分析（使用移动平均）
window = 50
if len(episode_returns) >= window:
    moving_avg = np.convolve(episode_returns, np.ones(window)/window, mode='valid')
    
    print(f"\n📉 **学习趋势** (50-episode移动平均)")
    first_avg = moving_avg[:50].mean()
    last_avg = moving_avg[-50:].mean()
    improvement = last_avg - first_avg
    
    print(f"  初期平均Return (Ep 0-50): {first_avg:.2f}")
    print(f"  后期平均Return (最后50): {last_avg:.2f}")
    print(f"  改进幅度: {improvement:+.2f} ({improvement/abs(first_avg)*100:+.1f}%)")
    
    if improvement > 0:
        print(f"  ✅ 训练有改进!")
    else:
        print(f"  ⚠️  训练未见明显改进")

# 碰撞分析（基于Episode Length）
print(f"\n💥 **碰撞推断分析**")
print(f"  注意：新版本日志会显示具体碰撞类型（行人/corridor/边界）")

# 极短episode (≤10步) 很可能是立即碰撞
immediate_collision = (episode_lengths <= 10).sum()
print(f"  疑似立即碰撞 (≤10步): {immediate_collision}/{total_episodes} ({immediate_collision/total_episodes:.1%})")

# 中等长度但失败的episode
medium_fail = ((episode_lengths > 10) & (episode_lengths < 100) & (episode_returns < 0)).sum()
print(f"  中途碰撞 (10-100步且失败): {medium_fail}/{total_episodes} ({medium_fail/total_episodes:.1%})")

# 评估结论
print(f"\n" + "=" * 80)
print("📝 **总体评估**".center(80))
print("=" * 80)

final_success_rate = (last_50_returns > 0).sum() / 50
final_avg_return = last_50_returns.mean()

if final_success_rate >= 0.5:
    print("  ✅ **训练效果良好**")
    print(f"     - 最后50集成功率: {final_success_rate:.1%}")
elif final_success_rate >= 0.2:
    print("  ⚠️  **训练效果一般**")
    print(f"     - 最后50集成功率: {final_success_rate:.1%}")
    print(f"     - 建议：继续训练或调整奖励函数")
else:
    print("  ❌ **训练效果不佳**")
    print(f"     - 最后50集成功率: {final_success_rate:.1%}")
    print(f"     - 建议：检查奖励函数、网络结构或环境设置")

if immediate_collision / total_episodes > 0.3:
    print(f"\n  ⚠️  **高碰撞率警告**")
    print(f"     - {immediate_collision/total_episodes:.1%} 的episode在10步内结束")
    print(f"     - 建议查看新日志确认是'行人碰撞'还是'corridor碰撞'")
    print(f"     - 如果主要是行人碰撞：增大行人生成距离")
    print(f"     - 如果主要是corridor碰撞：检查起点是否在corridor内")

print("\n" + "=" * 80)
print(f"💡 下一步建议：运行以下命令查看详细日志，确认碰撞类型")
print(f"   python scripts/resume_train.py --checkpoint logs/resume_training_optimized_20251006_184735/checkpoint_final.pt --config configs/resume_training_tuned.yaml")
print("=" * 80)

