# 🔄 AGSAC训练流程详解

**更新时间**: 2025-10-04 00:10  
**状态**: ✅ 完整训练流程说明

---

## 📊 模型参数分布

### **可训练参数** (476,590个，24%预算)

| 模块 | 参数量 | 占比 | 是否训练 |
|------|--------|------|----------|
| **DogEncoder** | 65,216 | 13.7% | ✅ **训练** |
| **PointNet** | 116,608 | 24.5% | ✅ **训练** |
| **CorridorEncoder** | 59,200 | 12.4% | ✅ **训练** |
| **TrajectoryPredictor** | 0 | 0% | ❌ **冻结（预训练）** |
| **PedestrianEncoder** | 224,704 | 47.2% | ✅ **训练** |
| **Fusion** | 49,920 | 10.5% | ✅ **训练** |
| **SAC_Actor** | 146,092 | 30.7% | ✅ **训练** |
| **SAC_Critic** | 319,746 | 67.1% | ✅ **训练** |
| **Critic_Target** | ~319,746 | - | ❌ **冻结（target网络）** |
| **GDE** | 0 | 0% | ❌ **无参数（规则）** |

### **预训练参数（冻结）** (~930,000个)
- **EVSC Model** (SocialCircle + E-V2-Net)
  - 在ETH-UCY Zara1数据集上预训练
  - 用于预测行人未来轨迹
  - `freeze=True` → `requires_grad=False`

---

## 🔄 完整训练流程

### **阶段1: 初始化** (Episode 0)

```
1. 加载预训练模型
   └─ PretrainedTrajectoryPredictor
      ├─ 加载EVSC权重 (evsczara1)
      ├─ 冻结所有参数 (requires_grad=False)
      └─ 验证加载成功

2. 初始化可训练模块
   ├─ DogEncoder (随机初始化)
   ├─ PointNet (随机初始化)
   ├─ CorridorEncoder (随机初始化)
   ├─ PedestrianEncoder (随机初始化)
   ├─ Fusion (随机初始化)
   ├─ SAC_Actor (LSTM，随机初始化)
   └─ SAC_Critic (Twin Q-Networks，随机初始化)

3. 初始化优化器
   ├─ actor_optimizer (Adam, lr=3e-4)
   ├─ critic_optimizer (Adam, lr=3e-4)
   └─ alpha_optimizer (Adam, lr=3e-4)  # 自动调整温度系数

4. 初始化ReplayBuffer
   └─ SequenceReplayBuffer (capacity=10,000)
```

---

### **阶段2: 数据收集** (每个Episode)

```
1. 环境Reset
   ├─ 根据episode数量选择难度
   │  ├─ Episode 0-49: Easy
   │  ├─ Episode 50-149: Medium
   │  └─ Episode 150-299: Hard
   │
   ├─ 生成场景
   │  ├─ CorridorGenerator生成通路
   │  ├─ 随机放置行人
   │  └─ 设置起点/终点
   │
   └─ 初始化LSTM隐藏状态
      ├─ actor_hidden: (h, c) = (0, 0)
      ├─ critic1_hidden: (h, c) = (0, 0)
      └─ critic2_hidden: (h, c) = (0, 0)

2. Episode循环 (最多200步)
   
   For step in range(max_episode_steps):
   
   ┌────────────────────────────────────────────────────┐
   │ 2.1 观测处理                                        │
   └────────────────────────────────────────────────────┘
   环境观测 → adapt_observation_for_model()
   ├─ dog_obs
   ├─ pedestrians (trajectories, mask)
   ├─ corridors (polygons, vertex_counts, mask)
   └─ reference_line
   
   ┌────────────────────────────────────────────────────┐
   │ 2.2 前向传播（推理，不计算梯度）                      │
   └────────────────────────────────────────────────────┘
   
   with torch.no_grad():  # 推理阶段不需要梯度
   
   a) DogEncoder
      dog_obs → dog_features (64维)
   
   b) PointNet + CorridorEncoder
      corridors → corridor_features (128维)
   
   c) 预训练TrajectoryPredictor（冻结）
      pedestrians → pedestrian_predictions
      (max_peds, pred_horizon=12, 2, num_modes=3)
      ⚠️ 此模块不参与训练，权重固定
   
   d) PedestrianEncoder
      pedestrian_predictions → pedestrian_features (128维)
   
   e) MultiModalFusion
      [dog_features, corridor_features, pedestrian_features]
      → fused_state (64维)
   
   f) SAC_Actor (LSTM)
      fused_state + hidden_actor
      → action (22维), log_prob, new_hidden_actor
      ⚠️ action = tanh(mean + std * noise)
   
   ┌────────────────────────────────────────────────────┐
   │ 2.3 环境交互                                        │
   └────────────────────────────────────────────────────┘
   
   action (22维) → 环境执行
   ├─ 坐标转换 (相对→全局)
   ├─ 只执行第一个点 (MPC策略)
   ├─ 评估完整路径 (GDE)
   └─ 计算奖励
      ├─ progress_reward (主导)
      ├─ direction_reward (GDE方向)
      ├─ curvature_reward (GDE平滑)
      ├─ goal_reached_reward
      ├─ collision_penalty
      └─ step_penalty
   
   → next_obs, reward, done, info
   
   ┌────────────────────────────────────────────────────┐
   │ 2.4 存储经验                                        │
   └────────────────────────────────────────────────────┘
   
   transition = {
       'observation': obs,
       'action': action,
       'reward': reward,
       'next_observation': next_obs,
       'done': done,
       'hidden_states': {
           'actor': hidden_actor,
           'critic1': hidden_critic1,
           'critic2': hidden_critic2
       }
   }
   
   episode_buffer.append(transition)
   
   If done:
       break

3. Episode结束
   └─ buffer.add_episode(episode_data)
      ├─ 计算return
      ├─ 标准化rewards
      └─ 存入ReplayBuffer
```

---

### **阶段3: 模型更新** (每个Episode后)

```
If len(buffer) >= warmup_episodes (30):

For update in range(updates_per_episode=10):

┌────────────────────────────────────────────────────┐
│ 3.1 采样Sequence Batch                            │
└────────────────────────────────────────────────────┘

segment_batch = buffer.sample(batch_size=16)
├─ observations: (16, seq_len=16, ...)
├─ actions: (16, 16, 22)
├─ rewards: (16, 16)
├─ dones: (16, 16)
└─ init_hidden_states: {
       'actor': (16, hidden_dim),
       'critic1': (16, hidden_dim),
       'critic2': (16, hidden_dim)
    }

┌────────────────────────────────────────────────────┐
│ 3.2 Critic更新（计算梯度）                          │
└────────────────────────────────────────────────────┘

For t in range(seq_len):

  # 当前Q值
  with gradients:  # 需要梯度
    obs_t → 编码器 → fused_state
    Q1(s_t, a_t), Q2(s_t, a_t) ← Critic(fused_state, action)
  
  # 目标Q值
  with torch.no_grad():  # 不需要梯度
    next_obs_t → 编码器 → next_fused_state
    next_action, next_log_prob ← Actor(next_fused_state)
    target_Q1, target_Q2 ← Critic_Target(next_fused_state, next_action)
    target_Q = min(target_Q1, target_Q2) - alpha * next_log_prob
    target = reward + gamma * (1 - done) * target_Q
  
  # Critic损失
  critic_loss = MSE(Q1, target) + MSE(Q2, target)

# 反向传播
critic_optimizer.zero_grad()
critic_loss.backward()  # ✅ 更新以下模块的梯度:
                        #    - DogEncoder
                        #    - PointNet
                        #    - CorridorEncoder
                        #    - PedestrianEncoder (⚠️ 不是Predictor!)
                        #    - Fusion
                        #    - Critic
torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
critic_optimizer.step()  # ✅ 更新参数

┌────────────────────────────────────────────────────┐
│ 3.3 Actor更新（计算梯度）                           │
└────────────────────────────────────────────────────┘

For t in range(seq_len):

  with gradients:  # 需要梯度
    obs_t → 编码器 → fused_state
    action, log_prob ← Actor(fused_state)  # 重参数化采样
    Q1, Q2 ← Critic(fused_state, action)
    Q = min(Q1, Q2)
  
  # Actor损失（最大化Q - alpha*entropy）
  actor_loss = -(Q - alpha * log_prob).mean()

# 反向传播
actor_optimizer.zero_grad()
actor_loss.backward()  # ✅ 更新以下模块的梯度:
                       #    - DogEncoder (共享)
                       #    - PointNet (共享)
                       #    - CorridorEncoder (共享)
                       #    - PedestrianEncoder (共享，⚠️ 不是Predictor!)
                       #    - Fusion (共享)
                       #    - Actor
torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
actor_optimizer.step()  # ✅ 更新参数

┌────────────────────────────────────────────────────┐
│ 3.4 Alpha更新（自动调整温度）                       │
└────────────────────────────────────────────────────┘

If auto_entropy:
  alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
  
  alpha_optimizer.zero_grad()
  alpha_loss.backward()
  alpha_optimizer.step()
  
  alpha = log_alpha.exp()

┌────────────────────────────────────────────────────┐
│ 3.5 Critic Target网络软更新                        │
└────────────────────────────────────────────────────┘

For param, target_param in zip(critic, critic_target):
  target_param.data = tau * param.data + (1-tau) * target_param.data

⚠️ target_param.requires_grad = False (不参与训练)
```

---

### **阶段4: 评估与保存** (每50个episodes)

```
If episode % eval_interval == 0:

1. 评估模式
   model.eval()
   
   For eval_episode in range(eval_episodes=5):
     ├─ 确定性策略 (deterministic=True, no noise)
     ├─ 收集episode
     └─ 记录return

2. 计算统计
   ├─ mean_return
   ├─ std_return
   └─ mean_length

3. TensorBoard记录
   ├─ eval/mean_return
   ├─ eval/std_return
   └─ eval/mean_length

4. 保存最佳模型
   If mean_return > best_eval_return:
     └─ save_checkpoint(is_best=True)

5. 恢复训练模式
   model.train()

If episode % save_interval == 0:
  └─ save_checkpoint()
```

---

## 🎯 关键训练机制

### **1. 共享编码器**
```
Actor和Critic共享以下编码器:
├─ DogEncoder
├─ PointNet
├─ CorridorEncoder
├─ PedestrianEncoder
└─ Fusion

✅ 优势:
  - 参数共享，减少总参数量
  - 编码器从两个角度学习（策略+价值）
  - 更好的特征表示

⚠️ 注意:
  - 编码器梯度来自Actor和Critic两个源
  - 需要适当的学习率和梯度裁剪
```

### **2. LSTM时序建模**
```
Actor和Critic都使用LSTM:

每个时间步:
  input: fused_state (64维)
  hidden: (h, c)  # 来自上一步
  ↓
  LSTM Cell
  ↓
  output: new_hidden (h', c')

✅ 优势:
  - 捕捉时序依赖
  - 记忆历史信息
  - 更平滑的决策

⚠️ 训练:
  - 使用序列段 (seq_len=16)
  - 保存初始隐藏状态
  - 梯度通过时间反向传播(BPTT)
```

### **3. 预训练模型冻结**
```
TrajectoryPredictor (EVSC):
  ├─ SocialCircle: 社交上下文编码
  └─ E-V2-Net: 轨迹预测

冻结策略:
  for param in trajectory_predictor.parameters():
      param.requires_grad = False

✅ 优势:
  - 保留真实人类行为先验知识
  - 减少训练参数 (66%减少)
  - 更稳定的训练

❌ 限制:
  - 预测器不会适应特定环境
  - 依赖预训练数据的质量
```

### **4. 双Q网络**
```
Critic = Twin Q-Networks:
  ├─ Q1(s, a)
  └─ Q2(s, a)

Target Q = min(Q1, Q2) - alpha * log_prob

✅ 优势:
  - 减少Q值过估计
  - 更稳定的训练
  - SAC标准做法
```

### **5. 梯度裁剪**
```
每次更新前:
  torch.nn.utils.clip_grad_norm_(
      parameters,
      max_norm=1.0
  )

✅ 防止梯度爆炸
✅ 提高训练稳定性
```

---

## 📈 训练进度监控

### **TensorBoard指标**

**训练阶段** (每个episode):
- `train/episode_return` - 应逐渐上升
- `train/actor_loss` - 应逐渐稳定
- `train/critic_loss` - 应逐渐下降
- `train/alpha` - 自动调整（通常下降）
- `train/buffer_size` - 逐渐增长到容量上限

**评估阶段** (每50 episodes):
- `eval/mean_return` - 应持续上升
- `eval/std_return` - 可能先增后减
- `eval/mean_length` - Episode长度变化

---

## 🔍 总结

### **会被训练的部分** ✅
1. **DogEncoder** - 学习编码机器狗状态
2. **PointNet + CorridorEncoder** - 学习编码环境几何
3. **PedestrianEncoder** - 学习编码行人预测轨迹
4. **Fusion** - 学习融合多模态特征
5. **SAC_Actor** - 学习最优策略
6. **SAC_Critic** - 学习价值函数
7. **Alpha** - 自动调整探索vs利用

### **不会被训练的部分** ❌
1. **TrajectoryPredictor (EVSC)** - 预训练冻结
2. **Critic_Target** - Target网络冻结
3. **GDE** - 无参数规则评估器

### **训练目标**
```
最大化累积回报:
  J(θ) = E[Σ γ^t * r_t]

其中 r_t 包括:
  - 向目标的进展
  - 路径方向一致性 (GDE)
  - 路径平滑度 (GDE)
  - 到达目标奖励
  - 避免碰撞惩罚
```

**整个系统通过SAC算法学习如何在动态环境中导航，同时利用预训练模型提供的准确行人预测！** 🎯

