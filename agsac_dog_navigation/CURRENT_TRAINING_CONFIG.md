# 当前训练环境与奖励函数详解

**更新时间**: 2025-10-05  
**配置文件**: `configs/resume_training_tuned.yaml`  
**难度**: **Easy (固定)**

---

## 🗺️ 训练环境配置

### **场景生成 (每个Episode)**

| 参数 | 配置 | 说明 |
|------|------|------|
| **地图大小** | 12m × 12m | 固定大小 |
| **起点位置** | 随机 (10%-90%范围) | 避免靠近边界 |
| **终点位置** | 随机 (10%-90%范围) | 任意方向，距起点≥50%对角线 |
| **Corridor数量** | 2-4条 (随机) | Easy难度的通路数 |
| **行人数量** | 2-3个 (随机) | 动态避障目标 |
| **障碍物数量** | 0个 | 完全移除 |
| **难度** | **固定Easy** | 不会增加 |
| **课程学习** | **禁用** | 始终保持Easy |

### **Episode限制**

| 参数 | 值 | 说明 |
|------|-----|------|
| **最大步数** | 120步 | 超过则timeout |
| **到达阈值** | 1.0米 | 距离目标<1.0m视为成功 |
| **连续违规阈值** | 20步 | 连续20步离开corridor提前终止 |

---

## 🤖 机器狗运动控制

### **动作空间**

- **维度**: 22 (11个路径点 × 2坐标)
- **输出范围**: [-1, 1] (tanh激活)
- **坐标转换**: 
  ```python
  归一化值 → 相对坐标
  scale = 2.0米
  path_relative = action * 2.0  # [-2m, +2m]
  path_global = robot_position + path_relative
  ```

### **路径执行**

```python
# 1. 提取第一个路径点作为短期目标
target_point = path_global[0]

# 2. 计算朝向目标的方向
direction = (target_point - robot_position) / ||target_point - robot_position||

# 3. 固定速度移动
speed = 1.5 m/s
dt = 0.1 s
max_displacement = 0.15米/步

# 4. 步长限幅（防止超冲）
if enable_step_limit:
    remaining_distance = ||target_point - robot_position||
    actual_displacement = min(0.15米, remaining_distance)
else:
    actual_displacement = 0.15米

# 5. 更新位置
robot_position += direction * actual_displacement
```

### **运动特性**

- **速度**: 1.5 m/s (每步最多0.15米)
- **时间步长**: 0.1秒
- **步长限幅**: ✅ 启用 (防止超过第一个路径点)
- **朝向更新**: 自动根据移动方向更新

---

## 💰 奖励函数详解

### **奖励组成 (7个部分)**

```python
total_reward = (
    progress_reward +       # 进展奖励
    direction_reward +      # 方向一致性
    curvature_reward +      # 路径平滑度
    corridor_penalty +      # Corridor约束
    goal_reached_reward +   # 到达目标
    collision_penalty +     # 碰撞惩罚
    step_penalty            # 步数惩罚
)
```

---

### **1️⃣ 进展奖励 (主导信号)**

```python
progress = last_distance - current_distance  # 到目标的距离变化
progress_reward = progress * 20.0
```

**特点**:
- ✅ **主导奖励**：每米进展 ≈ +20分
- ✅ **双向反馈**：靠近得正奖励，远离得负奖励
- 📊 **典型值**：每步0.15米 → +3.0分

**权重**: `progress_reward_weight = 20.0`

---

### **2️⃣ 方向一致性奖励 (GDE)**

```python
# 使用几何微分评估器(GDE)评估路径与目标方向的一致性
direction_score = GDE(planned_path, goal_direction)  # [0, 1]
direction_normalized = 2.0 * direction_score - 1.0   # [-1, 1]
direction_reward = direction_normalized * 0.3        # [-0.3, +0.3]
```

**特点**:
- ✅ **对称奖励**：好的方向得正分，差的方向扣分
- ✅ **稳定反馈**：不会主导总奖励
- 📊 **典型值**：±0.3分之间

---

### **3️⃣ 路径平滑度奖励 (曲率)**

```python
curvature_score = evaluate_path_curvature(planned_path)  # (0, 1]
normalized = 2.0 * curvature_score - 1.0                 # [-1, 1]
curvature_reward = normalized * 0.5                      # [-0.5, +0.5]
```

**特点**:
- ✅ **鼓励平滑路径**：减少抖动和急转弯
- ✅ **适度权重**：不会过度限制灵活性
- 📊 **典型值**：±0.5分之间

---

### **4️⃣ Corridor约束惩罚**

```python
if not in_corridor:
    distance = distance_to_nearest_corridor(robot_position)
    
    # 根据约束模式计算惩罚
    if mode == 'soft':
        raw_penalty = -distance * 8.0
    elif mode == 'medium':
        raw_penalty = -distance * 16.0
    elif mode == 'hard':
        return collision  # 直接视为碰撞
    
    # 裁剪上限
    corridor_penalty = max(raw_penalty, -12.0)  # 最多-12分/步
```

**当前配置 (Easy难度)**:
- **约束模式**: `soft` (软约束)
- **惩罚权重**: `8.0` (固定)
- **惩罚上限**: `-12.0` (每步最多扣12分)
- **连续违规**: >20步提前终止

**特点**:
- ✅ **软约束**：鼓励在corridor内，但允许探索
- ✅ **惩罚裁剪**：防止单步惩罚过大
- ✅ **早期终止**：连续严重违规会提前结束episode
- 📊 **典型值**：
  - 在corridor内: 0分
  - 距离1米: -8分
  - 距离2米: -12分 (上限)

---

### **5️⃣ 到达目标奖励 (稀疏)**

```python
if distance_to_goal < 1.0:
    goal_reached_reward = +100.0
```

**特点**:
- ✅ **大幅奖励**：成功到达给予+100分
- ✅ **明确反馈**：清晰的成功信号
- 📊 **触发条件**：距离目标<1.0米

---

### **6️⃣ 碰撞惩罚 (稀疏)**

```python
if collision:
    collision_penalty = -100.0
```

**碰撞检测**:
- 与行人碰撞 (半径0.5米)
- Hard模式下离开corridor

**特点**:
- ❌ **严重惩罚**：碰撞扣100分并结束episode
- ✅ **强调安全**：与到达奖励对称
- 📊 **触发条件**：任何碰撞

---

### **7️⃣ 步数惩罚**

```python
step_penalty = -0.02  # 每步
```

**特点**:
- ⏱️ **鼓励效率**：减少无效探索
- ✅ **适度权重**：不会过度限制
- 📊 **累积效果**：100步累计-2分

**权重**: `step_penalty_weight = 0.02`

---

## 📊 奖励函数数值示例

### **典型场景分析**

#### **场景1: 正常前进（在corridor内）**
```python
progress: 0.15米前进        → +3.0
direction: 0.8一致性        → +0.24
curvature: 0.7平滑度        → +0.20
corridor: 在内              → 0.0
goal: 未到达                → 0.0
collision: 无               → 0.0
step: 1步                   → -0.02
─────────────────────────────────
总奖励: +3.42
```

#### **场景2: 轻微偏离corridor**
```python
progress: 0.15米前进        → +3.0
direction: 0.6一致性        → +0.06
curvature: 0.5平滑度        → 0.0
corridor: 距离0.5米         → -4.0  (0.5*8)
goal: 未到达                → 0.0
collision: 无               → 0.0
step: 1步                   → -0.02
─────────────────────────────────
总奖励: -0.96
```

#### **场景3: 严重偏离corridor**
```python
progress: 0.10米前进        → +2.0
direction: 0.4一致性        → -0.06
curvature: 0.3平滑度        → -0.20
corridor: 距离2.0米         → -12.0  (上限)
goal: 未到达                → 0.0
collision: 无               → 0.0
step: 1步                   → -0.02
─────────────────────────────────
总奖励: -10.28
```

#### **场景4: 成功到达目标**
```python
progress: 0.20米前进        → +4.0
direction: 0.9一致性        → +0.27
curvature: 0.8平滑度        → +0.30
corridor: 在内              → 0.0
goal: 到达！                → +100.0
collision: 无               → 0.0
step: 1步                   → -0.02
─────────────────────────────────
总奖励: +104.55
Episode结束: goal_reached
```

#### **场景5: 碰撞失败**
```python
progress: -0.05米后退       → -1.0
direction: 0.3一致性        → -0.12
curvature: 0.4平滑度        → -0.10
corridor: 在内              → 0.0
goal: 未到达                → 0.0
collision: 碰撞！           → -100.0
step: 1步                   → -0.02
─────────────────────────────────
总奖励: -101.24
Episode结束: collision
```

---

## 🎯 奖励函数设计理念

### **平衡原则**

1. **进展主导** (Progress-Driven)
   - 进展奖励(~3分/步)远大于其他奖励
   - 确保agent专注于到达目标

2. **约束辅助** (Constraint-Assisted)
   - Corridor惩罚有上限(-12分/步)
   - 不会完全压制进展奖励
   - 允许短暂偏离以探索

3. **质量引导** (Quality-Guided)
   - 方向和曲率奖励提供细微反馈
   - 不主导决策，但引导优化

4. **安全优先** (Safety-First)
   - 碰撞惩罚(-100)与到达奖励(+100)对称
   - 连续违规提前终止

### **渐进策略**

| Episode阶段 | 约束模式 | 惩罚权重 | 学习目标 |
|------------|---------|---------|---------|
| **当前(Easy)** | soft | 8.0 | 探索+基础导航 |
| (如启用课程) | medium | 16.0 | 精细控制 |
| (如启用课程) | hard | 视为碰撞 | 严格约束 |

**注**: 当前配置禁用课程学习，固定使用Easy/soft配置

---

## 🔧 可调参数汇总

| 参数 | 当前值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `progress_reward_weight` | 20.0 | 15-25 | 进展奖励权重 |
| `corridor_penalty_weight` | 8.0 | 5-15 | Corridor惩罚权重 |
| `corridor_penalty_cap` | 12.0 | 10-30 | 单步惩罚上限 |
| `step_penalty_weight` | 0.02 | 0.01-0.05 | 步数惩罚 |
| `enable_step_limit` | true | - | 步长限幅开关 |
| `corridor_constraint_mode` | soft | soft/medium/hard | 约束模式 |

---

## 📈 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **总Episodes** | 500 | 训练目标 |
| **Updates/Episode** | 20 | 每episode更新20次 |
| **Buffer容量** | 10000 | Replay Buffer大小 |
| **Batch大小** | 16 | 训练批次 |
| **Sequence长度** | 16 | RNN序列长度 |
| **Actor学习率** | 0.0001 | 较低以稳定训练 |
| **Critic学习率** | 0.0001 | 较低以稳定训练 |
| **Gamma (折扣因子)** | 0.99 | 标准值 |
| **Tau (软更新)** | 0.005 | 目标网络更新率 |
| **Alpha (熵权重)** | 0.2 | SAC温度参数 |
| **Warmup Episodes** | 0 | Resume训练无需预热 |
| **Eval间隔** | 25 episodes | 评估频率 |
| **Save间隔** | 50 episodes | 保存频率 |

---

## 🎲 随机性来源

1. **起点位置**: 每episode随机
2. **终点位置**: 每episode随机（任意方向）
3. **Corridor数量**: 2-4条随机
4. **Corridor形状**: 路径规划随机
5. **行人数量**: 2-3个随机
6. **行人位置**: 在corridor内随机生成
7. **模型探索**: SAC算法的熵正则化

---

## ✅ 总结

### **当前配置特点**

✅ **固定Easy难度** - 不会自动增加  
✅ **Soft约束** - 鼓励探索，适度惩罚  
✅ **进展主导** - 明确的到达目标信号  
✅ **质量引导** - 路径方向和平滑度优化  
✅ **安全机制** - 碰撞严重惩罚，连续违规提前终止  
✅ **步长限幅** - 防止超冲，提高精度  
✅ **随机场景** - 起点终点任意方向，避免偏见

### **适用场景**

- ✅ 初期训练，建立基础导航能力
- ✅ 探索性学习，发现有效策略
- ✅ 稳定训练，避免课程学习波动
- ✅ 泛化能力，适应不同起点终点方向

### **潜在调整方向**

如果训练稳定后想提高难度:
- 提高`corridor_penalty_weight` (8.0 → 12.0)
- 切换到`medium`约束模式
- 减少`corridor_penalty_cap` (但不低于10.0)
- 启用课程学习 (`curriculum_learning: true`)

