# 奖励函数微调指南

**更新时间**: 2025-10-05  
**状态**: ✅ 已实现

---

## 🎯 微调目标

解决当前奖励函数的三个潜在问题：
1. **进展与惩罚平衡** - 防止走出corridor后仍有净正回报
2. **步长限幅** - 减少抖动与超冲，提高稳定性
3. **收敛节奏** - 训练初期降低激进性，提高稳定性

---

## 📊 优化1: 进展与惩罚平衡

### **问题分析**

```python
每步移动: ~0.15m
Progress奖励: +0.15 * 20 = +3.0
GDE合计: ±0.8
步惩罚: -0.01

如果偏离corridor 2米:
  Corridor惩罚: -2 * 10 = -20.0
  
但如果同时前进2米:
  净收益: +40 - 20 - 0.01 = +19.99 ✅ 还是正的！
  
→ 机器狗可能学会"无视corridor，直线前进"
```

### **解决方案**

**方案A: 增强corridor惩罚权重**
```yaml
env:
  corridor_penalty_weight: 15.0  # 原10.0 → 15.0
```

**效果：**
```
偏离2米前进2米:
  净收益: +40 - 30 - 0.01 = +9.99 (仍正，但收益降低)
  
偏离3米前进2米:
  净收益: +40 - 45 - 0.01 = -5.01 (开始亏损) ✅
```

**方案B: 添加惩罚上限（推荐）**
```yaml
env:
  corridor_penalty_cap: 30.0  # 每步最多扣30分
```

**效果：**
- 防止单步惩罚过大导致训练不稳定
- 保持主导但不失控
- 上限30 vs progress最大3，仍有约束力

**实现代码：**
```python
# 裁剪惩罚：防止单步惩罚过大
corridor_penalty = max(raw_penalty, -self.corridor_penalty_cap)
```

---

## 🎮 优化2: 步长限幅

### **问题分析**

```python
当前实现:
  displacement = direction * speed * dt
  = direction * 1.5 * 0.1
  = direction * 0.15米

问题1: 接近目标时可能超冲
  距离0.05米 → 移动0.15米 → 超过0.10米

问题2: 规划路径与实际路径不一致
  → 导致GDE评分不准确
  → 影响direction/curvature reward
```

### **解决方案**

```python
# 步长限幅：防止超冲和抖动
if self.enable_step_limit:
    remaining_distance = np.linalg.norm(self.goal_pos - self.robot_position)
    actual_displacement = min(max_displacement, remaining_distance)
else:
    actual_displacement = max_displacement

displacement = direction * actual_displacement
```

**效果：**
- ✅ 接近目标时自动减速，避免超冲
- ✅ 实际移动与规划更一致，GDE评分更稳定
- ✅ 减少"来回抖动"现象

**配置：**
```yaml
env:
  enable_step_limit: true  # 默认启用
```

---

## 📉 优化3: 收敛节奏

### **问题分析**

```python
训练初期（Episode 0-100）:
  - 模型随机策略
  - Progress波动大 (+20/-20)
  - 可能过于激进，忽略安全

训练中后期（Episode 100+）:
  - 策略逐渐稳定
  - 需要更强的progress信号
  - 可以恢复高权重
```

### **解决方案**

**训练初期配置（Episode 201-350）：**
```yaml
env:
  progress_reward_weight: 15.0   # 降低（原20.0 → 15.0）
  step_penalty_weight: 0.02      # 增加（原0.01 → 0.02）
```

**预期效果：**
```
降低progress权重:
  - 每米+15分（原+20）
  - 降低激进性，更关注安全和路径质量
  
增加step惩罚:
  - 每步-0.02（原-0.01）
  - 200步 = -4.0（原-2.0）
  - 更强的时间压力
```

**训练中后期配置（Episode 350+）：**
```yaml
env:
  progress_reward_weight: 20.0   # 恢复原值
  step_penalty_weight: 0.01      # 恢复原值
```

---

## 📋 完整配置对比

| 参数 | 原配置 | 优化配置（初期） | 优化配置（后期） |
|------|--------|-----------------|-----------------|
| **corridor_penalty_weight** | 10.0 | 15.0 ✅ | 15.0 |
| **corridor_penalty_cap** | ∞ | 30.0 ✅ | 30.0 |
| **progress_reward_weight** | 20.0 | 15.0 ✅ | 20.0 |
| **step_penalty_weight** | 0.01 | 0.02 ✅ | 0.01 |
| **enable_step_limit** | false | true ✅ | true |

---

## 🎯 使用方法

### **方案1: 使用优化配置文件**

```bash
cd agsac_dog_navigation
python scripts/resume_train.py \
  --checkpoint logs/curriculum_training_20251004_124233/curriculum_training/best_model.pt \
  --config configs/resume_training_tuned.yaml
```

### **方案2: 手动调整训练阶段**

**Episode 201-350（适应期）：**
```yaml
# configs/resume_training_early.yaml
env:
  progress_reward_weight: 15.0
  step_penalty_weight: 0.02
  corridor_penalty_weight: 15.0
  corridor_penalty_cap: 30.0
  enable_step_limit: true
```

**Episode 350+（优化期）：**
```yaml
# configs/resume_training_late.yaml  
env:
  progress_reward_weight: 20.0    # 恢复
  step_penalty_weight: 0.01       # 恢复
  corridor_penalty_weight: 15.0
  corridor_penalty_cap: 30.0
  enable_step_limit: true
```

---

## 📊 预期效果对比

### **优化前（原配置）**

```
典型episode (Episode 201-250):
  - 成功率: 10-20%
  - 经常走出corridor: 40% episodes
  - 接近目标时抖动: 常见
  - 平均回报: 可能不稳定

风险:
  - 可能学会"无视corridor"策略
  - 超冲导致GDE评分不准
```

### **优化后（微调配置）**

```
预期改善 (Episode 201-350):
  - 成功率: 20-35% ↑
  - Corridor违规: 20% episodes ↓
  - 抖动减少: 步长限幅生效
  - 平均回报: 更稳定

Episode 350+（恢复原权重）:
  - 成功率: 40-60% ↑↑
  - 学会平衡速度与安全
```

---

## 🔍 监控指标

### **关键指标**

训练时监控以下指标判断是否需要调整：

```python
# 1. Corridor违规率
corridor_violation_rate = corridor_violations / total_steps
目标: < 20%

# 2. 平均corridor惩罚
avg_corridor_penalty = mean(corridor_penalty_per_step)
期望: -2 ~ -5（有约束但不过分）

# 3. 进展稳定性
progress_std = std(progress_per_episode)
目标: 逐渐下降（策略稳定化）

# 4. 到达目标距离
final_distance = distance_at_episode_end
目标: 接近目标时 < 0.5m（无抖动）
```

### **调整信号**

**如果corridor违规率 > 40%：**
→ 增加 `corridor_penalty_weight` (15 → 20)

**如果训练不稳定（回报波动大）：**
→ 进一步降低 `progress_reward_weight` (15 → 12)

**如果接近目标时频繁抖动：**
→ 确认 `enable_step_limit: true`

**如果成功率低但很稳定：**
→ 恢复 `progress_reward_weight` 到20.0

---

## 💡 专家建议总结

1. **进展与惩罚平衡** ✅
   - 增强corridor约束（10 → 15）
   - 添加惩罚上限（30.0）

2. **步长限幅** ✅
   - 启用 `enable_step_limit`
   - 防止超冲和抖动

3. **收敛节奏** ✅
   - 初期降低progress（20 → 15）
   - 初期增加step惩罚（0.01 → 0.02）
   - 中后期恢复原值

---

## 📌 快速开始

**推荐：直接使用优化配置**

```bash
cd agsac_dog_navigation

# 运行验证
python verify_resume_config.py

# 开始训练（使用优化配置）
python scripts/resume_train.py \
  --checkpoint logs/curriculum_training_20251004_124233/curriculum_training/best_model.pt \
  --config configs/resume_training_tuned.yaml
```

**预期训练时间：** 2-3小时（299 episodes）

**监控TensorBoard：**
```bash
tensorboard --logdir=logs/resume_training_tuned/tensorboard
```

---

✅ **所有优化已实现并准备就绪！**
