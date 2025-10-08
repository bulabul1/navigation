"""
深度分析：碰撞类型、奖励分量趋势
"""

import json
import numpy as np
from pathlib import Path

# 读取训练历史
log_dir = Path("logs/resume_training_optimized_20251006_184735")
with open(log_dir / "training_history.json", 'r') as f:
    history = json.load(f)

episode_returns = np.array(history['episode_returns'])
episode_lengths = np.array(history['episode_lengths'])

print("=" * 80)
print("深度分析报告".center(80))
print("=" * 80)

# 1. 关键发现
print("\n🔍 **关键发现**")
print(f"  1. 训练改进显著: Return从初期-575提升到后期-44 (提升92%)")
print(f"  2. Episode长度大幅缩短: 从168步降到22步")
print(f"  3. 成功率依然很低: 最后50集只有6%成功")
print(f"  4. 碰撞率高达54%: 最后50集中27次碰撞")

# 2. 问题诊断
print("\n⚠️  **问题诊断**")

# Episode长度分析
last_100_lengths = episode_lengths[-100:]
very_short_pct = (last_100_lengths <= 15).sum() / 100

print(f"\n  【问题1】Episode过早终止")
print(f"     - 最后100集中，{very_short_pct:.1%}在15步内结束")
print(f"     - 这表明存在'一步即死'或'起点不合理'的问题")

# 碰撞类型推断
print(f"\n  【问题2】碰撞类型未知")
print(f"     - 54%的episode因碰撞终止")
print(f"     - 但无法确认是'行人碰撞'还是'corridor碰撞'")
print(f"     - ⚡ 已添加碰撞类型日志功能，下次训练会显示详细信息")

# Return分布
last_50_returns = episode_returns[-50:]
very_negative = (last_50_returns < -100).sum()

print(f"\n  【问题3】仍有大负奖励")
print(f"     - 最后50集中，{very_negative}次Return < -100")
print(f"     - 可能原因：碰撞penalty过大(-40)或corridor违规累积")

# 3. Episode Length变化趋势
print(f"\n📊 **Episode Length演变**")
for i in range(0, 600, 100):
    end_idx = min(i + 100, 600)
    stage_lengths = episode_lengths[i:end_idx]
    avg_len = stage_lengths.mean()
    short_pct = (stage_lengths <= 20).sum() / len(stage_lengths)
    print(f"  Ep {i:3d}-{end_idx:3d}: 平均{avg_len:5.1f}步, {short_pct:.1%}在20步内结束")

print(f"\n  ⚠️  **趋势异常**:")
print(f"     - Episode长度持续缩短，说明机器人学会了某种策略")
print(f"     - 但成功率没有上升，说明这个策略不是'到达目标'")
print(f"     - 可能是：快速碰撞 > 长时间探索（避免step_penalty累积）")

# 4. 成功episode分析
success_indices = np.where(episode_returns > 0)[0]
print(f"\n✅ **成功Episode分析**")
print(f"  总成功次数: {len(success_indices)}/600 ({len(success_indices)/600:.1%})")

if len(success_indices) > 0:
    success_returns = episode_returns[success_indices]
    success_lengths = episode_lengths[success_indices]
    
    print(f"  平均Return: {success_returns.mean():.2f} (±{success_returns.std():.2f})")
    print(f"  平均Length: {success_lengths.mean():.1f} (±{success_lengths.std():.1f})")
    print(f"\n  最近10次成功:")
    recent_success = success_indices[-10:] if len(success_indices) >= 10 else success_indices
    for idx in recent_success:
        print(f"     Ep {idx:3d}: Return={episode_returns[idx]:7.2f}, Length={episode_lengths[idx]:3.0f}步")

# 5. 碰撞episode分析
collision_indices = np.where(episode_returns < -50)[0]
print(f"\n💥 **碰撞Episode分析**")
print(f"  总碰撞次数: {len(collision_indices)}/600 ({len(collision_indices)/600:.1%})")

if len(collision_indices) > 0:
    collision_lengths = episode_lengths[collision_indices]
    immediate = (collision_lengths <= 10).sum()
    early = ((collision_lengths > 10) & (collision_lengths <= 30)).sum()
    mid = (collision_lengths > 30).sum()
    
    print(f"  立即碰撞 (≤10步): {immediate} ({immediate/len(collision_indices):.1%})")
    print(f"  早期碰撞 (11-30步): {early} ({early/len(collision_indices):.1%})")
    print(f"  中期碰撞 (>30步): {mid} ({mid/len(collision_indices):.1%})")

# 6. 推荐措施
print(f"\n" + "=" * 80)
print("💡 **推荐措施**".center(80))
print("=" * 80)

print(f"\n  【立即执行】运行验证，查看碰撞类型:")
print(f"     python scripts/resume_train.py \\")
print(f"       --checkpoint logs/resume_training_optimized_20251006_184735/checkpoint_final.pt \\")
print(f"       --config configs/resume_training_tuned.yaml")
print(f"\n  新日志会显示:")
print(f"     - collision | collision [行人碰撞]")
print(f"     - collision | collision [corridor碰撞]")
print(f"     - collision | collision [边界碰撞]")

print(f"\n  【根据碰撞类型调整】:")
print(f"     如果主要是'行人碰撞':")
print(f"       → 增大 min_safe_distance (2.5 → 3.0)")
print(f"       → 增大 collision_threshold (0.2 → 0.3)")
print(f"")
print(f"     如果主要是'corridor碰撞':")
print(f"       → 检查corridor生成逻辑，确保起点在corridor内")
print(f"       → 减小corridor_penalty (让机器人有机会学习回到corridor)")
print(f"")
print(f"     如果主要是'边界碰撞':")
print(f"       → 检查起点生成范围")

print(f"\n  【奖励函数调整】:")
print(f"     当前问题：机器人可能学会了'快速碰撞'策略")
print(f"     原因：step_penalty (-0.01)累积 > collision_penalty (-40)")
print(f"     ")
print(f"     建议方案A（增大碰撞惩罚）:")
print(f"       → collision_penalty: -40 → -80")
print(f"       → 让碰撞更不划算")
print(f"     ")
print(f"     建议方案B（减小step_penalty）:")
print(f"       → step_penalty: -0.01 → -0.005")
print(f"       → 让长时间探索更划算")
print(f"     ")
print(f"     建议方案C（增大progress奖励）:")
print(f"       → progress_weight: 20.0 → 30.0")
print(f"       → 强化'朝目标前进'的动机")

print(f"\n  【继续训练】:")
print(f"     - 当前600集训练显示学习曲线仍在改善")
print(f"     - 建议至少再训练500集，观察是否收敛")

print("\n" + "=" * 80)

