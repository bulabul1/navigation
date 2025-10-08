# Easy难度配置说明

**更新时间**: 2025-10-05  
**配置文件**: `configs/resume_training_tuned.yaml`

---

## 🎯 训练配置

### **难度设置**
- **难度**: 固定 `easy`
- **课程学习**: `禁用` (curriculum_learning: false)
- **Corridor约束**: 固定 `soft`
- **Corridor惩罚权重**: 固定 `8.0`

---

## 🗺️ 场景生成配置

### **Easy难度参数**

| 参数 | 配置 | 说明 |
|------|------|------|
| **Corridors** | 2-4个（随机） | 每个episode随机生成2-4条通路 |
| **行人** | 2-3个（随机） | 每个episode随机生成2-3个行人 |
| **障碍物** | 0个 | 完全移除障碍物 |
| **起点** | 随机生成 | 地图10%-90%范围内随机 |
| **终点** | 随机生成 | 地图10%-90%范围内随机，任意方向 |
| **最小距离** | 地图对角线50% | 确保起点终点距离足够远 |

---

## 📝 代码修改记录

### **1. corridor_generator.py (Line 59)**
```python
# 修改前
num_corridors = 2  # 固定2个

# 修改后
num_corridors = self.rng.randint(2, 5)  # 2-4个走廊（随机）
```

### **2. agsac_environment.py (Line 592)**
```python
# 修改前
self.current_difficulty = 'easy' if curriculum_learning else 'medium'

# 修改后
self.current_difficulty = 'easy' if curriculum_learning else 'easy'
# 注：禁用课程学习时固定使用easy
```

### **3. resume_training_tuned.yaml (Line 17)**
```yaml
# 修改前
curriculum_learning: true  # 启用课程学习

# 修改后
curriculum_learning: false  # 禁用课程学习，固定easy难度
```

---

## ✅ 验证清单

- [x] Corridor数量: 2-4个（随机）
- [x] 行人数量: 2-3个（随机）
- [x] 障碍物数量: 0个
- [x] 起点终点: 随机生成（任意方向）
- [x] 难度固定: easy（不会增加）
- [x] 约束模式: soft（不会收紧）
- [x] 惩罚权重: 8.0（不会递增）

---

## 🚀 使用方法

### **启动训练**
```bash
cd agsac_dog_navigation

python scripts/resume_train.py \
  --checkpoint logs/resume_training_tuned/checkpoint_ep302.pt \
  --config configs/resume_training_tuned.yaml
```

### **预期行为**

每个episode：
1. ✅ 随机生成起点和终点（任意方向）
2. ✅ 随机生成2-4条corridor
3. ✅ 随机生成2-3个行人
4. ✅ 0个障碍物
5. ✅ Soft约束（corridor_penalty_weight=8.0）
6. ✅ 固定easy难度（不会增加）

---

## 📊 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **max_episode_steps** | 120 | 每个episode最多120步 |
| **progress_reward_weight** | 20.0 | 进展奖励权重 |
| **corridor_penalty_weight** | 8.0 | Corridor惩罚权重（固定） |
| **corridor_penalty_cap** | 12.0 | Corridor惩罚上限 |
| **step_penalty_weight** | 0.02 | 步数惩罚 |
| **enable_step_limit** | true | 启用步长限幅 |

---

## 💡 设计理念

### **为什么选择Easy难度？**

1. **简化学习任务**
   - 2-4条corridor：提供多种路径选择，但不会过于复杂
   - 2-3个行人：足够练习避障，但不会拥挤
   - 0个障碍物：专注于行人避障和路径规划

2. **稳定训练**
   - 固定难度：避免课程学习带来的学习信号波动
   - Soft约束：鼓励探索，惩罚适中

3. **随机性保证泛化**
   - 随机起点终点（任意方向）：避免方向偏见
   - 随机corridor数量：学习适应不同路径复杂度
   - 随机行人数量：学习适应不同人群密度

---

## 🎯 训练目标

- **成功率**: ≥80%
- **平均回报**: ≥50
- **碰撞率**: ≤10%
- **Episode长度**: ≤100步

---

## 📌 注意事项

1. ⚠️ **不要启用课程学习**  
   如果启用，难度会自动从easy → medium → hard

2. ⚠️ **episode_count同步**  
   Resume训练时，确保环境的episode_count与trainer同步

3. ⚠️ **日志路径**  
   每次训练会创建带时间戳的新目录，避免覆盖

4. ✅ **所有代码已修复**  
   - episode_count同步 ✅
   - path_history管理 ✅
   - 步长限幅修正 ✅
   - 奖励函数优化 ✅

