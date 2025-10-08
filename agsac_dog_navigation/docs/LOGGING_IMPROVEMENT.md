# 📊 日志和奖励可视化改进

**更新时间**: 2025-10-04  
**状态**: ✅ **已完成**

---

## 🎯 改进目标

**问题**：虽然奖励函数设计合理，但训练时无法看到各个奖励分量的具体值，难以诊断训练问题。

**解决方案**：增强日志输出和TensorBoard记录，让所有奖励分量可见。

---

## 📝 改进内容

### 1. 奖励函数返回详细信息

**修改文件**: `agsac/envs/agsac_environment.py`

**变更**:
```python
# 之前：只返回总奖励
def _compute_base_reward(self, action, collision) -> float:
    # ... 计算各个分量 ...
    return total_reward

# 之后：返回总奖励 + 详细分量
def _compute_base_reward(self, action, collision) -> Tuple[float, Dict]:
    # ... 计算各个分量 ...
    reward_components = {
        'progress_reward': progress_reward,
        'progress_meters': progress,
        'direction_reward': direction_reward,
        'direction_score': direction_score_raw,
        'curvature_reward': curvature_reward,
        'curvature_score': curvature_score_raw,
        'goal_reached_reward': goal_reached_reward,
        'collision_penalty': collision_penalty,
        'step_penalty': step_penalty,
        'current_distance': current_distance
    }
    return total_reward, reward_components
```

**好处**:
- ✅ 保留所有奖励分量的详细信息
- ✅ 包含原始分数（direction_score, curvature_score）和加权后的奖励
- ✅ 便于调试和分析

---

### 2. 更新奖励信息传递

**修改文件**: `agsac/envs/agsac_environment.py`

**变更**:
```python
def _compute_reward(self, action, collision):
    total_reward, reward_components = self._compute_base_reward(action, collision)
    
    reward_info = {
        'total_reward': total_reward,
        **reward_components  # 展开所有分量
    }
    
    return total_reward, reward_info
```

**好处**:
- ✅ 通过`info`字典将奖励分量传递给trainer
- ✅ 保持接口不变（仍然返回`reward, info`）

---

### 3. Trainer收集奖励详情

**修改文件**: `agsac/training/trainer.py`

**变更**:
```python
# 在collect_episode中收集奖励详情
reward_infos = []  # 新增

while not done:
    # ... step ...
    
    # 收集每步的奖励分量
    reward_infos.append({
        'progress_reward': info.get('progress_reward', 0.0),
        'direction_reward': info.get('direction_reward', 0.0),
        'curvature_reward': info.get('curvature_reward', 0.0),
        'goal_reached_reward': info.get('goal_reached_reward', 0.0),
        'collision_penalty': info.get('collision_penalty', 0.0),
        'step_penalty': info.get('step_penalty', 0.0),
    })

episode_data['reward_infos'] = reward_infos  # 保存到episode数据
```

**好处**:
- ✅ 记录每一步的奖励分量
- ✅ 可以计算平均值或总和
- ✅ 便于分析奖励的时间变化

---

### 4. 增强日志输出

**修改文件**: `agsac/training/trainer.py`

**变更**:
```python
# 新增辅助方法
def _compute_average_reward_components(self, episode_data):
    """计算episode中奖励分量的平均值"""
    # ... 收集所有步骤的奖励 ...
    avg_rewards = {
        'progress': np.mean(progress_rewards),
        'direction': np.mean(direction_rewards),
        'curvature': np.mean(curvature_rewards),
        'goal': np.sum(goal_rewards),  # 稀疏奖励用sum
        'collision': np.sum(collision_penalties),
        'step': np.mean(step_penalties)
    }
    return avg_rewards

# 在_log_episode中增加奖励分量输出
def _log_episode(self, ...):
    # ... 第一行：基础信息 ...
    # ... 第二行：路径信息 ...
    
    # 第三行：奖励分量详情（新增）
    if 'reward_infos' in episode_data:
        avg_rewards = self._compute_average_reward_components(episode_data)
        
        reward_str = f"  ├─ Rewards: "
        reward_str += f"Prog={avg_rewards['progress']:.3f} "
        reward_str += f"Dir={avg_rewards['direction']:.3f} "
        reward_str += f"Curv={avg_rewards['curvature']:.3f} "
        reward_str += f"Goal={avg_rewards['goal']:.1f} "
        reward_str += f"Coll={avg_rewards['collision']:.1f} "
        reward_str += f"Step={avg_rewards['step']:.3f}"
        print(reward_str)
    
    # ... 第四行：路径点 ...
```

**效果示例**:
```
[Episode   42] Return= -15.23 Length= 87 Buffer= 100 | Actor=0.3421 Critic=1.2345 Alpha=0.2000 | Time=1.23s
  ├─ Start: ( 0.00, 0.00) → Goal: (10.00,10.00) | Dist:  12.34m (直线:14.14m) | 剩余: 3.45m | timeout
  ├─ Rewards: Prog=-0.120 Dir=0.045 Curv=0.120 Goal=0.0 Coll=0.0 Step=-0.010
  └─ Path: ( 0.00, 0.00) → ( 1.23, 1.45) → ( 3.45, 3.67) → ... → ( 8.90, 7.12)
```

**好处**:
- ✅ 一目了然地看到各个奖励分量
- ✅ 可以快速诊断问题（例如：direction奖励总是负数？）
- ✅ 清晰的层级结构（├─ 和 └─）

---

### 5. TensorBoard可视化

**修改文件**: `agsac/training/trainer.py`

**变更**:
```python
# 在_log_episode中增加TensorBoard记录
if self.use_tensorboard and self.writer is not None:
    # ... 原有的记录 ...
    
    # 奖励分量记录（新增）
    if episode_data is not None and 'reward_infos' in episode_data:
        avg_rewards = self._compute_average_reward_components(episode_data)
        if avg_rewards:
            self.writer.add_scalar('reward/progress', avg_rewards['progress'], episode)
            self.writer.add_scalar('reward/direction', avg_rewards['direction'], episode)
            self.writer.add_scalar('reward/curvature', avg_rewards['curvature'], episode)
            self.writer.add_scalar('reward/goal', avg_rewards['goal'], episode)
            self.writer.add_scalar('reward/collision', avg_rewards['collision'], episode)
            self.writer.add_scalar('reward/step', avg_rewards['step'], episode)
```

**TensorBoard面板**:
```
Scalars:
├─ train/
│  ├─ episode_return
│  ├─ episode_length
│  ├─ actor_loss
│  ├─ critic_loss
│  └─ alpha
└─ reward/  ← 新增
   ├─ progress
   ├─ direction
   ├─ curvature
   ├─ goal
   ├─ collision
   └─ step
```

**好处**:
- ✅ 可以绘制奖励分量的训练曲线
- ✅ 发现奖励失衡问题（某个分量过大/过小）
- ✅ 对比不同实验的奖励策略

---

## 📊 奖励分量说明

| 分量 | 含义 | 范围 | 统计方式 |
|------|------|------|----------|
| **progress** | 每步朝目标靠近的距离×20 | -∞ ~ +∞ | 平均值 |
| **direction** | 路径方向与目标方向一致性 | -0.3 ~ +0.3 | 平均值 |
| **curvature** | 路径平滑度（夹角积分） | -0.5 ~ +0.5 | 平均值 |
| **goal** | 到达目标的奖励 | 0 或 100.0 | 总和（稀疏） |
| **collision** | 碰撞惩罚 | 0 或 -100.0 | 总和（稀疏） |
| **step** | 每步固定惩罚 | -0.01 | 平均值 |

**注意**:
- `progress`、`direction`、`curvature`、`step` 是密集奖励，显示**平均值**
- `goal`、`collision` 是稀疏奖励，显示**总和**（一个episode最多出现一次）

---

## 🔍 使用示例

### 1. 查看控制台日志

运行训练后，你会看到：

```bash
[Episode    5] Return= -12.45 Length= 95 Buffer=  10 | Actor=0.2345 Critic=0.8765 Alpha=0.2000 | Time=1.20s
  ├─ Start: ( 0.00, 0.00) → Goal: (10.00,10.00) | Dist:  13.45m (直线:14.14m) | 剩余: 2.34m | timeout
  ├─ Rewards: Prog=-0.098 Dir=0.034 Curv=0.156 Goal=0.0 Coll=0.0 Step=-0.010
  └─ Path: ( 0.00, 0.00) → ( 1.34, 1.23) → ( 2.67, 2.56) → ... → ( 8.90, 8.12)
```

**分析**:
- `Prog=-0.098`: 平均每步略微后退（可能在绕障碍物）
- `Dir=0.034`: 方向稍微朝向目标（正值）
- `Curv=0.156`: 路径比较平滑（正值）
- `Goal=0.0`: 没有到达目标
- `Coll=0.0`: 没有碰撞
- `Step=-0.010`: 每步的固定惩罚

---

### 2. TensorBoard可视化

```bash
tensorboard --logdir outputs/your_experiment/tensorboard
```

打开浏览器访问 `http://localhost:6006`，你会看到：

**Scalars标签页**:
- `reward/progress`: 查看进展奖励的趋势（应该逐渐增大）
- `reward/direction`: 方向一致性的变化
- `reward/curvature`: 路径平滑度的变化
- `reward/goal`: 到达目标的频率
- `reward/collision`: 碰撞的频率（希望为0）

---

### 3. 诊断训练问题

**场景1: 总奖励很低，但不知道原因**

查看日志：
```
Rewards: Prog=-0.500 Dir=-0.120 Curv=-0.230 Goal=0.0 Coll=0.0 Step=-0.010
```

**诊断**:
- ❌ `Prog=-0.500`: **主要问题**！机器人每步都在后退
- ❌ `Dir=-0.120`: 方向错误（负值）
- ❌ `Curv=-0.230`: 路径弯曲（负值）
- ✅ 没有碰撞

**结论**: 策略还没有学会朝目标前进，需要继续训练。

---

**场景2: 经常碰撞**

查看日志：
```
[Episode   15] Return=-110.45 ...
  ├─ Rewards: Prog=0.234 Dir=0.045 Curv=0.120 Goal=0.0 Coll=-100.0 Step=-0.010
```

**诊断**:
- ✅ 进展、方向、曲率都不错
- ❌ `Coll=-100.0`: **主要问题**！发生了碰撞

**结论**: 需要增加碰撞避免训练，或提高碰撞惩罚。

---

**场景3: 训练停滞不前**

在TensorBoard中发现：
- `reward/progress` 一直在0附近波动，不增长
- `reward/direction` 持续为负

**诊断**: 策略陷入局部最优，可能需要：
1. 增加探索（调高alpha）
2. 调整奖励权重
3. 改变环境难度（课程学习）

---

## ✅ 改进效果

### 之前（无详细日志）

```bash
[Episode    5] Return= -12.45 Length= 95 ...
```

**问题**:
- ❌ 只能看到总奖励，不知道哪个分量有问题
- ❌ 无法诊断训练停滞的原因
- ❌ 难以调整奖励权重

---

### 之后（详细日志）

```bash
[Episode    5] Return= -12.45 Length= 95 ...
  ├─ Start: ( 0.00, 0.00) → Goal: (10.00,10.00) | Dist:  13.45m | 剩余: 2.34m | timeout
  ├─ Rewards: Prog=-0.098 Dir=0.034 Curv=0.156 Goal=0.0 Coll=0.0 Step=-0.010
  └─ Path: ( 0.00, 0.00) → ( 1.34, 1.23) → ... → ( 8.90, 8.12)
```

**好处**:
- ✅ 一眼看出各个奖励分量的贡献
- ✅ 快速定位问题（例如方向错误、碰撞频繁）
- ✅ 便于调整奖励权重和超参数
- ✅ TensorBoard可视化训练曲线

---

## 📈 未来可扩展性

如果需要添加新的奖励分量（例如能量消耗、舒适度等），只需：

1. 在`_compute_base_reward`中计算新分量
2. 加入`reward_components`字典
3. 在`_compute_average_reward_components`中收集
4. 在日志和TensorBoard中显示

**示例**:
```python
# 1. 计算能量消耗奖励
energy_penalty = -0.05 * np.linalg.norm(action)

# 2. 加入字典
reward_components['energy_penalty'] = energy_penalty

# 3. 在日志中显示
reward_str += f"Energy={avg_rewards['energy']:.3f} "
```

---

## 🎓 总结

### 主要改进

1. ✅ **奖励函数返回详细信息** - 保留所有分量
2. ✅ **Trainer收集奖励历史** - 每步的详细记录
3. ✅ **增强控制台日志** - 一目了然的分量显示
4. ✅ **TensorBoard可视化** - 训练曲线分析

### 奖励函数本身

- ✅ **设计合理** - 已经根据REWARD_FIX_SUMMARY.md修复
- ✅ **对称奖励** - direction和curvature都是[-x, +x]
- ✅ **权重平衡** - progress主导，GDE作为正则化

### 用户体验

- ✅ **诊断方便** - 快速定位训练问题
- ✅ **调试高效** - 不用修改代码就能看到详情
- ✅ **可扩展** - 容易添加新的奖励分量

---

**结论**: 奖励函数设计本身很好，现在日志输出也完善了！🎉

