# 🚪 Corridor约束使用指南

**更新时间**: 2025-10-04  
**状态**: ✅ **已实现，可立即使用**

---

## 🎯 快速开始

### **方式1：默认配置（软约束）**

```python
from agsac.envs import DummyAGSACEnvironment

# 创建环境（默认软约束）
env = DummyAGSACEnvironment(
    max_pedestrians=10,
    max_corridors=10,
    max_vertices=20,
    corridor_constraint_mode='soft',      # 软约束（默认）
    corridor_penalty_weight=10.0,         # 每米扣10分（默认）
    device='cpu'
)
```

---

### **方式2：中等约束**

```python
env = DummyAGSACEnvironment(
    corridor_constraint_mode='medium',    # 中等约束
    corridor_penalty_weight=10.0,         # 实际会×2 = 20分/米
    device='cpu'
)
```

---

### **方式3：硬约束（训练后期）**

```python
env = DummyAGSACEnvironment(
    corridor_constraint_mode='hard',      # 硬约束
    # 离开corridor直接碰撞，无需penalty_weight
    device='cpu'
)
```

---

## 📊 约束模式对比

| 模式 | 惩罚方式 | 惩罚强度 | 适用阶段 | Episode终止 |
|------|----------|----------|----------|------------|
| **soft** | 距离惩罚 | -10分/米 | 训练初期 | ❌ 继续 |
| **medium** | 距离惩罚 | -20分/米 | 训练中期 | ❌ 继续 |
| **hard** | 立即碰撞 | Episode终止 | 训练后期 | ✅ 立即终止 |

---

## 🔧 配置参数详解

### **corridor_constraint_mode**

**类型**: `str`  
**可选值**: `'soft'`, `'medium'`, `'hard'`  
**默认值**: `'soft'`

**作用**：
```python
if mode == 'soft':
    # 轻微惩罚
    penalty = -distance × weight
elif mode == 'medium':
    # 更大惩罚
    penalty = -distance × (weight × 2)
elif mode == 'hard':
    # 立即碰撞（在_check_collision中处理）
    if not in_corridor:
        return True  # 碰撞
```

---

### **corridor_penalty_weight**

**类型**: `float`  
**默认值**: `10.0`  
**范围**: `0.0 ~ 50.0`（推荐）

**效果示例**：
```python
# weight=10.0
离开corridor 1米 → penalty = -10.0
离开corridor 3米 → penalty = -30.0

# weight=20.0
离开corridor 1米 → penalty = -20.0
离开corridor 3米 → penalty = -60.0
```

**推荐值**：
- 训练初期：`5.0 ~ 10.0`（轻微引导）
- 训练中期：`10.0 ~ 20.0`（强烈引导）
- 训练后期：切换到`hard`模式

---

## 📈 训练策略

### **策略1：固定软约束（最简单）**

```python
# 全程使用软约束
env = DummyAGSACEnvironment(
    corridor_constraint_mode='soft',
    corridor_penalty_weight=10.0,
    ...
)

# 训练
trainer.train()
```

**优点**：
- ✅ 简单易用
- ✅ 训练稳定
- ✅ 不需要手动切换

**缺点**：
- ⚠️ 可能仍有5-15%的违规率
- ⚠️ 不能100%保证在corridor内

**适用场景**：
- 原型验证
- 快速迭代
- 不要求绝对遵守规则

---

### **策略2：渐进式加强（推荐）⭐**

```python
# 方案A：手动切换
# Episode 0-100: soft, weight=10
env = DummyAGSACEnvironment(
    corridor_constraint_mode='soft',
    corridor_penalty_weight=10.0
)
trainer.train(max_episodes=100)

# Episode 100-200: medium, weight=10 (实际20)
env.corridor_constraint_mode = 'medium'
trainer.train(max_episodes=100)

# Episode 200+: hard
env.corridor_constraint_mode = 'hard'
trainer.train(max_episodes=100)
```

```python
# 方案B：自动课程学习（需修改环境）
class CurriculumCorridorEnv(DummyAGSACEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_episodes = 0
    
    def reset(self):
        # 根据episode数量自动调整
        if self.total_episodes < 100:
            self.corridor_constraint_mode = 'soft'
        elif self.total_episodes < 200:
            self.corridor_constraint_mode = 'medium'
        else:
            self.corridor_constraint_mode = 'hard'
        
        self.total_episodes += 1
        return super().reset()
```

**优点**：
- ✅ 最佳学习效率
- ✅ 最终性能最好
- ✅ 违规率 < 5%

**缺点**：
- ⚠️ 需要更多episode
- ⚠️ 需要手动或自动切换

**适用场景**：
- 生产级训练
- 需要严格遵守规则
- 有足够训练时间

---

### **策略3：立即硬约束（不推荐）**

```python
# 从一开始就用硬约束
env = DummyAGSACEnvironment(
    corridor_constraint_mode='hard',
    ...
)

trainer.train()
```

**问题**：
- ❌ 初期探索困难
- ❌ 训练极其缓慢
- ❌ 可能长时间无法学习

**适用场景**：
- ✅ 微调已训练好的模型
- ✅ 强化安全性
- ❌ **不适合从头训练**

---

## 📊 监控指标

### **控制台日志**

```bash
[Episode   42] Return= -15.23 Length= 87 Buffer= 100 | Actor=0.3421 Critic=1.2345 Alpha=0.2000 | Time=1.23s
  ├─ Start: ( 0.00, 0.00) → Goal: (10.00,10.00) | Dist:  12.34m (直线:14.14m) | 剩余: 3.45m | timeout
  ├─ Rewards: Prog=-0.120 Dir=0.045 Curv=0.120 Corr=-5.23 Goal=0.0 Coll=0.0 Step=-0.010
  ├─ Corridor: Violations=38/87 (43.7%)  ← 新增
  └─ Path: ( 0.00, 0.00) → ( 1.23, 1.45) → ... → ( 8.90, 7.12)
```

**解读**：
- `Corr=-5.23`: 平均每步的corridor惩罚（负值表示有违规）
- `Violations=38/87 (43.7%)`: 87步中有38步离开corridor，违规率43.7%

---

### **TensorBoard指标**

打开TensorBoard后，会看到新增的指标：

```
reward/
├─ corridor              ← 平均corridor惩罚（应逐渐接近0）
└─ ...

corridor/
├─ violations            ← 每episode的违规步数（应下降）
└─ violation_rate        ← 违规率（应下降）
```

**健康的训练曲线**：
```
Episode 0-50:
  corridor/violation_rate: 60% → 40%

Episode 50-150:
  corridor/violation_rate: 40% → 15%

Episode 150+:
  corridor/violation_rate: < 10%
```

---

## 🔍 故障排查

### **问题1：Violation rate一直很高（>50%）**

**可能原因**：
1. Corridor penalty太小
2. Progress reward太大，覆盖了corridor penalty
3. Corridor设计有问题（太窄或不合理）

**解决方案**：
```python
# 方案A：增加惩罚权重
env.corridor_penalty_weight = 20.0  # 从10增到20

# 方案B：降低progress权重
# 修改_compute_base_reward中
progress_reward = progress × 15.0  # 从20降到15

# 方案C：检查corridor是否合理
env.reset()
print("Corridor data:", env.corridor_data)
# 确保corridor足够宽，且覆盖起点和终点
```

---

### **问题2：训练完全不收敛**

**可能原因**：
- 使用了`hard`模式，探索受限

**解决方案**：
```python
# 切换到soft模式重新训练
env.corridor_constraint_mode = 'soft'
env.corridor_penalty_weight = 5.0  # 降低惩罚
```

---

### **问题3：Corridor penalty总是0**

**可能原因**：
1. 没有corridor数据（使用了不含corridor的场景）
2. Corridor设计太宽，机器狗永远在里面

**检查方法**：
```python
env.reset()

# 检查是否有corridor数据
if not env.corridor_data:
    print("❌ 没有corridor数据！")
else:
    print(f"✅ 有{len(env.corridor_data)}条corridor")
    
    # 检查是否在corridor内
    pos = env.robot_position
    in_corridor = env._is_in_any_corridor(pos)
    print(f"机器狗在corridor内: {in_corridor}")
```

---

### **问题4：Corridor collision太频繁（hard模式）**

**可能原因**：
- 过早使用hard模式
- Corridor太窄

**解决方案**：
```python
# 方案A：延后使用hard模式
# Episode 0-200: soft/medium
# Episode 200+: hard

# 方案B：扩大corridor
# 修改corridor_generator.py中的width参数
corridor = self._upper_detour_corridor(..., width=2.5)  # 从1.5增到2.5
```

---

## 🧪 测试corridor约束

### **测试脚本**

```python
import numpy as np
from agsac.envs import DummyAGSACEnvironment

# 创建环境
env = DummyAGSACEnvironment(
    corridor_constraint_mode='soft',
    corridor_penalty_weight=10.0,
    device='cpu'
)

# 重置环境
obs = env.reset()

print(f"Corridors: {len(env.corridor_data)}")
print(f"Start: {env.start_pos}")
print(f"Goal: {env.goal_pos}")

# 测试点在多边形内判断
test_points = [
    env.start_pos,              # 起点（应在corridor内）
    env.goal_pos,               # 终点（应在corridor内）
    np.array([5.0, 5.0]),       # 中心点（可能在障碍物内）
    np.array([5.0, 10.0]),      # 上方（可能在corridor内）
]

print("\n点在corridor内测试：")
for i, point in enumerate(test_points):
    in_corridor = env._is_in_any_corridor(point)
    distance = env._distance_to_nearest_corridor(point)
    print(f"Point {i}: {point} → In corridor: {in_corridor}, Distance: {distance:.2f}m")

# 测试step
print("\n测试step：")
for step in range(10):
    action = np.random.randn(22) * 0.1
    obs, reward, done, info = env.step(action)
    
    print(f"Step {step}: "
          f"In corridor: {info.get('in_corridor', 'N/A')}, "
          f"Corridor penalty: {info.get('corridor_penalty', 0):.2f}, "
          f"Total reward: {reward:.2f}")
    
    if done:
        print(f"Episode terminated: {info['done_reason']}")
        break
```

---

## 📋 配置文件示例

### **YAML配置**

```yaml
# configs/corridor_training.yaml

environment:
  type: 'DummyAGSACEnvironment'
  max_pedestrians: 10
  max_corridors: 10
  max_vertices: 20
  obs_horizon: 8
  pred_horizon: 12
  max_episode_steps: 500
  device: 'cpu'
  
  # Corridor约束配置
  use_corridor_generator: false  # 使用固定场景
  corridor_constraint_mode: 'soft'  # soft/medium/hard
  corridor_penalty_weight: 10.0
  
  # 奖励权重
  reward_weights:
    progress: 20.0
    direction: 0.3
    curvature: 0.5
    goal: 100.0
    collision: -100.0
    step: -0.01

training:
  max_episodes: 300
  warmup_episodes: 10
  updates_per_episode: 100
  eval_interval: 50
  
  # 课程学习（可选）
  curriculum:
    enabled: true
    stages:
      - episodes: 100
        corridor_mode: 'soft'
        corridor_weight: 10.0
      - episodes: 100
        corridor_mode: 'medium'
        corridor_weight: 10.0
      - episodes: 100
        corridor_mode: 'hard'
```

---

## ✅ 最佳实践

### **1. 从软约束开始**

```python
# 初次训练：使用软约束
env = DummyAGSACEnvironment(
    corridor_constraint_mode='soft',
    corridor_penalty_weight=10.0
)
```

### **2. 监控violation rate**

```bash
# 训练时观察日志
Corridor: Violations=38/87 (43.7%)  ← 应逐渐下降
```

### **3. 根据情况调整**

```python
# 如果violation rate > 60% → 增加惩罚
env.corridor_penalty_weight = 20.0

# 如果violation rate < 10% → 可以切换到medium/hard
env.corridor_constraint_mode = 'medium'
```

### **4. TensorBoard可视化**

```bash
tensorboard --logdir outputs/your_experiment/tensorboard

# 重点关注：
# - corridor/violation_rate 曲线（应下降）
# - reward/corridor 曲线（应接近0）
```

### **5. 渐进式训练**

```python
# 分阶段训练
Episode 0-100: soft (10分/米)
Episode 100-200: medium (20分/米)
Episode 200+: hard (立即碰撞)
```

---

## 🎓 总结

### **推荐配置**

| 训练阶段 | Mode | Weight | 预期Violation Rate |
|---------|------|--------|-------------------|
| **初期** | soft | 10.0 | 40-60% → 20-30% |
| **中期** | medium | 10.0 | 20-30% → 5-15% |
| **后期** | hard | - | < 5% |

### **关键指标**

- ✅ **Violation Rate < 10%** - 基本可用
- ✅ **Violation Rate < 5%** - 生产级
- ⚠️ **Violation Rate > 30%** - 需调整

### **故障诊断流程**

```
1. 检查是否有corridor数据
   ↓
2. 检查corridor是否合理（宽度、覆盖范围）
   ↓
3. 调整penalty weight（10 → 20 → 30）
   ↓
4. 如果仍无效，降低progress weight
   ↓
5. 考虑切换到medium/hard模式
```

---

**开始训练吧！corridor约束已经完全集成到你的环境中了！** 🚀

