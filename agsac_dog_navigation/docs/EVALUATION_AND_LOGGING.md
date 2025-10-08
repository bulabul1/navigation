# 📊 评估与日志系统说明

**更新时间**: 2025-10-04 00:30  
**状态**: ✅ 完整说明

---

## 🎯 评估系统

### **评估触发条件**

```python
# configs/default.yaml
training:
  eval_interval: 50      # 每50个episodes评估一次
  eval_episodes: 5       # 每次评估运行5个episodes
```

**触发时机**：
- Episode 50, 100, 150, 200, 250

---

### **评估流程**

```python
def evaluate(self) -> Dict[str, float]:
    """评估当前策略"""
    
    # 1. 切换到评估模式
    self.model.eval()  # 关闭dropout等
    
    # 2. 收集5个评估episodes
    eval_returns = []
    eval_lengths = []
    
    for _ in range(self.eval_episodes):  # 默认5次
        episode_data = self.collect_episode(deterministic=True)
        #                                   ↑ 关键：使用确定性策略
        eval_returns.append(episode_data['episode_return'])
        eval_lengths.append(episode_data['episode_length'])
    
    # 3. 恢复训练模式
    self.model.train()
    
    # 4. 计算统计量
    eval_stats = {
        'eval_return_mean': np.mean(eval_returns),   # 平均回报
        'eval_return_std': np.std(eval_returns),     # 标准差
        'eval_length_mean': np.mean(eval_lengths)    # 平均长度
    }
    
    return eval_stats
```

---

### **❓ 评估使用固定数据吗？**

**答案：❌ 不是固定数据，是动态随机生成的场景**

#### **评估场景生成方式**

```python
# 当 use_corridor_generator=True (default.yaml的设置)
def _reset_env(self):
    if self.use_corridor_generator:
        self._generate_dynamic_scenario()  # ← 每次reset都随机生成
        
        if self.curriculum_learning:
            # 根据episode_count选择难度
            if self.episode_count < 50:
                difficulty = 'easy'
            elif self.episode_count < 150:
                difficulty = 'medium'
            else:
                difficulty = 'hard'
        
        # 生成随机场景
        scenario = self.corridor_generator.generate_scenario(
            difficulty=self.current_difficulty
        )
        # ↓ 每次都不一样
        self.start_pos = scenario['start']      # 随机起点
        self.goal_pos = scenario['goal']        # 随机终点
        self.corridor_data = scenario['corridors']  # 随机通路
        self.obstacles = scenario['obstacles']   # 随机障碍物
        self.pedestrians = ...                   # 随机行人
```

#### **评估时的场景特点**

| 特性 | 说明 |
|------|------|
| **场景数据** | ❌ 不固定 - 每次reset随机生成 |
| **难度级别** | ✅ 固定 - 根据当前episode数确定 |
| **策略模式** | ✅ 确定性 - `deterministic=True` |
| **环境** | ⚠️ 与训练共享 - `eval_env=env` |

**示例**：
```
Episode 50评估（medium难度）：
  - 评估1: 起点(1.2, 3.4) → 终点(9.8, 7.2), 2条通路, 4个行人
  - 评估2: 起点(0.8, 2.1) → 终点(10.5, 8.9), 3条通路, 3个行人
  - 评估3: 起点(2.3, 4.5) → 终点(8.7, 6.3), 2条通路, 5个行人
  - 评估4: 起点(1.5, 3.8) → 终点(9.2, 7.8), 3条通路, 4个行人
  - 评估5: 起点(0.9, 2.7) → 终点(10.1, 8.2), 2条通路, 3个行人
  
每次场景都不同，但都是medium难度
```

---

### **确定性策略 vs 随机策略**

```python
# 训练时（探索）
deterministic=False
  ↓
action = mean + std * noise  # 带噪声的随机采样
log_prob = ...               # 计算概率用于SAC

# 评估时（利用）
deterministic=True
  ↓
action = mean                # 只用均值，无噪声
更稳定、可重复
```

---

## 📋 日志输出系统

### **1. 训练日志**（每个episode）

**触发频率**：
```python
log_interval: 10  # 每10个episodes打印一次
```

**输出内容**：
```
[Episode   10] Return= -15.23 Length= 87 Buffer= 10 | Actor=-112.34 Critic=2543.21 Alpha=0.9985 | Time=19.45s
           ↑        ↑          ↑          ↑              ↑           ↑              ↑              ↑
        episode  总回报    步数长度   buffer大小    Actor损失   Critic损失   温度系数     耗时
```

**参数说明**：

| 参数 | 含义 | 期望趋势 |
|------|------|----------|
| **Episode** | 当前episode编号 | 递增 |
| **Return** | Episode总回报 | 📈 上升 |
| **Length** | Episode步数 | 稳定或略增 |
| **Buffer** | 经验池大小 | 递增到上限 |
| **Actor** | Actor损失 | 逐渐稳定 |
| **Critic** | Critic损失 | 📉 下降 |
| **Alpha** | SAC温度系数 | 自动调整 |
| **Time** | Episode耗时 | 稳定 |

---

### **2. 评估日志**（每50个episodes）

**输出内容**：
```
============================================================
[Evaluation @ Episode 50]
  Mean Return: 25.34 ± 3.21
  Mean Length: 142.5
============================================================
```

**参数说明**：

| 参数 | 含义 | 说明 |
|------|------|------|
| **Mean Return** | 5次评估的平均回报 | 性能指标 |
| **± Std** | 标准差 | 稳定性指标 |
| **Mean Length** | 平均episode长度 | 效率指标 |

---

### **3. TensorBoard日志**（实时）

#### **训练指标** (`train/`)

```python
writer.add_scalar('train/episode_return', episode_return, episode)
writer.add_scalar('train/episode_length', episode_length, episode)
writer.add_scalar('train/buffer_size', len(buffer), episode)
writer.add_scalar('train/actor_loss', actor_loss, episode)
writer.add_scalar('train/critic_loss', critic_loss, episode)
writer.add_scalar('train/alpha', alpha, episode)
writer.add_scalar('train/episode_time', episode_time, episode)
```

**可视化**：
- 📈 `episode_return` - 回报曲线（最重要）
- 📊 `actor_loss` / `critic_loss` - 损失曲线
- 📉 `alpha` - 温度系数变化

#### **评估指标** (`eval/`)

```python
writer.add_scalar('eval/mean_return', mean_return, episode)
writer.add_scalar('eval/std_return', std_return, episode)
writer.add_scalar('eval/mean_length', mean_length, episode)
```

**可视化**：
- 📈 `mean_return` - 评估性能（关键指标）
- 📊 `std_return` - 评估稳定性
- 📉 `mean_length` - 完成效率

---

## 🔍 评估的优缺点

### **当前设计的优势** ✅

1. **真实反映泛化能力**
   - 每次评估的场景都不同
   - 测试模型在新场景中的表现
   - 不会过拟合特定测试场景

2. **与训练一致**
   - 使用相同的难度级别
   - 相同的场景生成器
   - 评估更有代表性

3. **确定性策略**
   - 使用mean，无噪声
   - 结果更稳定
   - 易于比较

### **当前设计的缺点** ⚠️

1. **评估结果有随机性**
   - 每次场景不同
   - 标准差可能较大
   - 难以精确比较不同checkpoint

2. **无法追踪特定场景的进展**
   - 不能看到"同一个场景"的性能改善
   - 难以debug特定失败案例

3. **评估场景可能偏简单/困难**
   - 随机生成可能运气好/坏
   - 5个episodes可能不够统计显著

---

## 💡 改进建议

### **建议1: 添加固定测试集**（可选）

```python
# 在训练开始时生成固定的测试场景
def create_fixed_test_scenarios(num_scenarios=10):
    """生成固定的测试场景用于一致性评估"""
    test_scenarios = []
    for i in range(num_scenarios):
        scenario = corridor_generator.generate_scenario(
            difficulty='medium',
            seed=42+i  # 固定种子
        )
        test_scenarios.append(scenario)
    return test_scenarios

# 评估时使用固定场景
def evaluate_fixed(self, test_scenarios):
    """在固定场景上评估"""
    for scenario in test_scenarios:
        # 加载固定场景
        # 运行episode
        # 记录结果
```

**优势**：
- ✅ 可精确比较不同checkpoint
- ✅ 可追踪特定场景的改善
- ✅ 结果更可重复

**劣势**：
- ❌ 可能过拟合测试集
- ❌ 需要额外存储场景

---

### **建议2: 增加评估episodes数量**

```yaml
training:
  eval_episodes: 10  # 从5增加到10
```

**优势**：
- ✅ 更稳定的统计量
- ✅ 降低随机性影响

**劣势**：
- ❌ 评估时间翻倍

---

### **建议3: 分层评估**（推荐）

```python
def evaluate_comprehensive(self):
    """综合评估：固定场景 + 随机场景"""
    # 1. 固定场景评估
    fixed_stats = self.evaluate_fixed(self.test_scenarios)
    
    # 2. 随机场景评估
    random_stats = self.evaluate_random(num_episodes=5)
    
    return {
        'fixed': fixed_stats,   # 可重复性
        'random': random_stats  # 泛化能力
    }
```

---

## 📊 总结

### **当前评估系统**

| 方面 | 配置 |
|------|------|
| **数据来源** | ❌ 非固定 - 每次随机生成 |
| **难度级别** | ✅ 根据训练进度确定 |
| **策略模式** | ✅ 确定性（无噪声） |
| **评估频率** | 每50 episodes |
| **评估次数** | 5个episodes |
| **环境** | 与训练共享 |

### **日志输出**

**控制台**：
- 每10 episodes：训练日志
- 每50 episodes：评估日志

**TensorBoard**：
- 每个episode：训练指标
- 每50 episodes：评估指标

### **适用性**

✅ **当前设计适合**：
- 初期训练和开发
- 快速迭代
- 泛化能力评估

⚠️ **可能需要改进**：
- 精确性能比较
- 论文发表
- 生产部署

---

**对于当前的训练任务，现有评估系统已经足够！** 如需要更严格的评估，可以在训练完成后再添加固定测试集。🎯

