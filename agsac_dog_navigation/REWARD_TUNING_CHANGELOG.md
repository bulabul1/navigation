# 奖励函数温和校准 - 修改记录

**修改时间**: 2025-10-06  
**目标**: 50集快速验证  
**策略**: 温和校准，先观察效果再决定是否添加动态调整

---

## 📊 修改对比

### 1. **Corridor约束惩罚（配置文件）**

| 参数 | 原值 | 新值 | 变化 |
|------|------|------|------|
| `corridor_penalty_weight` | 8.0 | **4.0** | ↓ 50% |
| `corridor_penalty_cap` | 12.0 | **6.0** | ↓ 50% |

**文件**: `configs/resume_training_tuned.yaml` (Line 23-24)

---

### 2. **步数惩罚（配置文件）**

| 参数 | 原值 | 新值 | 变化 |
|------|------|------|------|
| `step_penalty_weight` | 0.02 | **0.01** | ↓ 50% |

**文件**: `configs/resume_training_tuned.yaml` (Line 28)

**作用**: 避免进一步压低episode回报

---

### 3. **稀疏奖励/惩罚（代码）**

| 参数 | 原值 | 新值 | 变化 |
|------|------|------|------|
| `collision_penalty` | -100.0 | **-40.0** | ↓ 60% |
| `goal_reached_reward` | +100.0 | **+40.0** | ↓ 60% |

**文件**: `agsac/envs/agsac_environment.py` (Line 1188, 1191)

**作用**: 降低稀疏信号主导地位，让密集信号更可见

---

### 4. **GDE权重（代码）**

| 参数 | 原值 | 新值 | 变化 |
|------|------|------|------|
| `direction_reward` 权重 | 0.3 | **0.5** | ↑ 67% |
| `curvature_reward` 权重 | 0.5 | **0.8** | ↑ 60% |

**文件**: `agsac/envs/agsac_environment.py` (Line 1212, 1223)

**作用**: 提升路径质量评估的影响力

---

## 📈 预期奖励分布变化

### 原奖励分布（Episode 454）

```
collision_penalty:   -100.00 (占98.0%)  ← 完全主导
progress_reward:       -0.44 (占1.5%)
curvature_reward:      -0.50 (占0.5%)
direction_reward:      +0.003 (占0.0%)
goal_reached:           0.00 (占0.0%)
─────────────────────────────────────────
Total:                -98.07
```

### 预期新分布（碰撞Episode）

```
collision_penalty:    -40.00 (占50-60%)  ← 降压到主导但不淹没
progress_reward:       +1~+4 (占30-40%)  ← 开始可见
curvature_reward:      ±0.8  (占3-5%)    ← 提升到可观察
direction_reward:      ±0.5  (占2-3%)    ← 提升到可观察
corridor_penalty:      0~-6  (占5-10%)   ← 降压
step_penalty:          -0.2  (占1-2%)
─────────────────────────────────────────
Total:                -35~-40（碰撞）
                      +10~+20（成功探索）
```

### 预期新分布（成功Episode）

```
progress_reward:      +80~120 (占60-70%)  ← 主导正向信号
goal_reached:         +40     (占25-30%)  ← 成功奖励
direction_reward:     +5~10   (占5-10%)   ← GDE正反馈
curvature_reward:     +5~10   (占5-10%)   ← GDE正反馈
step_penalty:         -2~-3   (占1-2%)
─────────────────────────────────────────
Total:                +120~170
```

---

## 🎯 关键改进点

### 1. **奖励尺度平衡**

**变化**:
- Collision主导度: 98% → 50-60%
- Progress可见度: 1.5% → 30-40%
- GDE可见度: 0.5% → 5-10%

**效果**:
- 密集信号不再被完全淹没
- GDE开始提供有效反馈
- 学习目标更加明确

---

### 2. **探索激励**

**原问题**: 
- 碰撞惩罚太大（-100），完全压制探索
- 机器人学会"快速碰撞结束"

**修复**:
- 碰撞惩罚降到-40，仍然显著但不是毁灭性
- 探索过程中的progress奖励（+1~+4/步）现在可以部分抵消碰撞风险
- 增加成功探索的可能性

---

### 3. **GDE有效性**

**原问题**:
- Direction权重0.3，被100:0.3 = 333倍差距淹没
- Curvature权重0.5，同样无影响

**修复**:
- Direction: 0.3 → 0.5，提升67%
- Curvature: 0.5 → 0.8，提升60%
- 新比例：40:0.8 = 50倍（仍是辅助，但开始可观察）

---

## 🔬 验证计划

### 阶段1：50集快速验证（~30分钟）

**观察指标**:
```python
# TensorBoard关注项
1. Episode Return:        目标从-98提升到-40~-50
2. Collision Rate:        目标从98%降到70-80%
3. Goal Reached Rate:     目标从0%升到5-10%
4. Episode Length:        目标从25步升到40-60步
5. Progress Reward:       观察是否从-0.44升到+1~+2
6. Direction/Curvature:   观察是否从0开始有波动
```

**成功标准**（50集后）:
- ✅ Episode Return > -50（改善50%）
- ✅ Collision Rate < 80%（降低18%）
- ✅ 出现至少1-2个成功案例
- ✅ GDE奖励开始有非零变化

**失败标准**:
- ❌ Episode Return仍然 < -90
- ❌ Collision Rate仍然 > 95%
- ❌ 完全无成功案例
- ❌ GDE奖励仍然恒定

---

### 阶段2：动态调整（如果需要）

如果50集验证效果好，可以实现**渐进式奖励调整**:

```python
# 伪代码示例
def get_adaptive_rewards(episode_num, success_rate):
    """根据训练进度动态调整奖励"""
    
    if episode_num < 100:
        # 初期：使用降压值
        collision = -40.0
        goal = 40.0
    elif success_rate > 0.4:
        # 成功率>40%后，线性恢复
        progress = (episode_num - 100) / 400  # 0~1 over 400 eps
        collision = -40 - 60 * progress       # -40 → -100
        goal = 40 + 60 * progress             # 40 → 100
    else:
        # 成功率不足，保持降压
        collision = -40.0
        goal = 40.0
    
    return collision, goal
```

**实现位置**: 可在`agsac_environment.py`的`__init__`中添加episode计数和success率追踪

---

## 📝 代码修改位置

### 配置文件
```yaml
# configs/resume_training_tuned.yaml

env:
  corridor_penalty_weight: 4.0      # Line 23
  corridor_penalty_cap: 6.0         # Line 24
  step_penalty_weight: 0.01         # Line 28
```

### 环境代码
```python
# agsac/envs/agsac_environment.py

# Line 1188: 目标奖励
goal_reached_reward = 40.0

# Line 1191: 碰撞惩罚
collision_penalty = -40.0 if collision else 0.0

# Line 1212: 方向权重
direction_reward = direction_normalized * 0.5

# Line 1223: 曲率权重
curvature_reward = normalized_curvature * 0.8
```

---

## ⚠️ 注意事项

### 1. **Curvature Bug待修复**

**问题**: `curvature_reward`恒为-0.8（原-0.5）

**原因**: `curvature_score_raw`总是0

**待查**:
- `_evaluate_path_curvature()`实现
- `current_planned_path`质量

**影响**: 
- 目前会给-0.8的恒定惩罚
- 如果能修复，可提供更丰富的反馈

---

### 2. **温和 vs 激进**

**本方案（温和）**:
- Collision: -100 → -40（降60%）
- GDE: 0.3→0.5, 0.5→0.8（升60-67%）

**之前提议（激进）**:
- Collision: -100 → -20（降80%）
- GDE: 0.3→3.0, 0.5→5.0（升10倍）

**选择原因**: 
- 温和方案风险更低
- 如果效果不够，再加大调整幅度
- 如果效果过度，更容易回调

---

### 3. **日志监控**

**关键监控点**:
```bash
# TensorBoard启动
python -m tensorboard.main --logdir logs/resume_training_optimized_XXXXXX/tensorboard --port 6006

# 关注曲线
- train/episode_return（期望上升）
- train/episode_length（期望上升）
- eval/success_rate（期望从0开始增长）
- train/collision_rate（期望下降）
- train/progress_reward（期望从负变正）
- train/direction_reward（期望开始波动）
- train/curvature_reward（期望不再恒定-0.8）
```

---

## 📊 预期训练曲线

### Episode Return（目标）

```
Episode 0-50:    -98 → -60~-70（快速改善）
Episode 50-100:  -60 → -40~-50（持续改善）
Episode 100-200: -40 → -20~0（稳定学习）
Episode 200+:    0 → +50~+100（开始成功）
```

### Collision Rate（目标）

```
Episode 0-50:    98% → 80-85%（初期改善）
Episode 50-100:  80% → 60-70%（明显下降）
Episode 100-200: 60% → 40-50%（稳定下降）
Episode 200+:    40% → 20-30%（接近实用）
```

---

## ✅ 检查清单

在启动训练前确认：

- [x] 配置文件已更新（`resume_training_tuned.yaml`）
- [x] 环境代码已更新（`agsac_environment.py`）
- [x] 注释已更新（反映新的奖励范围）
- [ ] Git提交更改（建议）
- [ ] TensorBoard已启动
- [ ] GPU可用（`nvidia-smi`检查）
- [ ] 准备监控前50集指标

---

## 🚀 启动命令

```powershell
# 1. 启动TensorBoard（新窗口）
python -m tensorboard.main --logdir logs --port 6006

# 2. 启动训练（设定50集验证）
python scripts/resume_train.py `
  --checkpoint logs/curriculum_training_20251004_124233/curriculum_training/best_model.pt `
  --config configs/resume_training_tuned.yaml `
  --episodes 50

# 3. 浏览器打开
# http://localhost:6006
```

---

## 📈 后续决策树

### 如果50集后效果好（Return>-50, Collision<80%）

**选项A**: 继续训练500集，观察长期效果

**选项B**: 实现动态调整，渐进恢复到-100/+100

**选项C**: 微调参数（如GDE权重再提升10-20%）

---

### 如果50集后效果一般（-80 < Return < -50）

**选项A**: 继续观察50集（共100集）

**选项B**: 适度加大调整（collision→-30, goal→+30）

**选项C**: 检查并修复curvature bug

---

### 如果50集后效果差（Return<-80）

**选项A**: 采用激进方案（collision→-20, GDE权重×10）

**选项B**: 检查环境设置（corridor太难？行人太多？）

**选项C**: 回到原始参数，重新诊断问题

---

**修改完成！准备启动50集验证！** 🚀


