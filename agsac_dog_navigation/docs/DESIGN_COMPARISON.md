# 原方案 vs 当前实现对比

## 📊 关键差异总结

### 1. **训练方式** ⭐ 核心变化

| 维度 | 原方案 | 当前实现 |
|-----|--------|---------|
| 采样单位 | 单个transition | 序列段segment (seq_len=16) |
| 存储格式 | `{'state', 'action', 'reward', 'next_state', 'done', 'hidden_states'}` | Episode轨迹，采样时提取segment |
| Hidden管理 | 每个transition存储h | Segment起始存储h，内部按序展开 |
| SAC更新 | 单步Q-learning | 序列段展开LSTM进行Q-learning |
| Burn-in | 无 | 支持burn-in预热hidden state |

**理由**: LSTM需要时序上下文才能充分利用记忆能力。单步采样会丢失序列信息。

---

### 2. **ReplayBuffer设计**

#### 原方案（单步）
```python
buffer.add({
    'state': fused_state,      # (64,)
    'action': action,           # (22,)
    'reward': reward,
    'next_state': next_state,
    'done': done,
    'hidden_states': {
        'h_actor': (h, c),
        'h_critic1': (h, c),
        'h_critic2': (h, c)
    }
})

batch = buffer.sample(batch_size=256)  # 256个独立transition
```

#### 当前实现（序列段）
```python
# 存储完整episode
episode_data = {
    'observations': [...],      # T个观测
    'actions': [...],           # T个动作
    'rewards': [...],           # T个奖励
    'dones': [...],            # T个done标志
    'hidden_states': [...]     # T个隐藏状态
}
buffer.add_episode(episode_data)

# 采样segment
segment_batch = buffer.sample(batch_size=32)  # 32个segment
# 每个segment:
# {
#     'states': (seq_len, 64),
#     'actions': (seq_len, 22),
#     'rewards': (seq_len,),
#     'next_states': (seq_len, 64),
#     'dones': (seq_len,),
#     'init_hidden': segment起始的hidden state
# }
```

---

### 3. **SAC Agent更新流程**

#### 原方案（单步更新）
```python
def update(self, batch):
    states = batch['states']          # (256, 64)
    actions = batch['actions']        # (256, 22)
    rewards = batch['rewards']        # (256,)
    next_states = batch['next_states'] # (256, 64)
    dones = batch['dones']            # (256,)
    
    # 每个样本独立处理，不考虑时序
    with torch.no_grad():
        next_actions, next_log_probs = actor(next_states)
        target_q1 = critic1_target(next_states, next_actions)
        target_q2 = critic2_target(next_states, next_actions)
        target_q = min(target_q1, target_q2) - alpha * next_log_probs
        target = rewards + gamma * (1 - dones) * target_q
    
    current_q1 = critic1(states, actions)
    current_q2 = critic2(states, actions)
    
    critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
    # ...
```

#### 当前实现（序列段更新）
```python
def update(self, segment_batch):
    """
    segment_batch: List[Dict]，每个segment是一个序列
    """
    for segment in segment_batch:
        states = segment['states']          # (seq_len, 64)
        actions = segment['actions']        # (seq_len, 22)
        rewards = segment['rewards']        # (seq_len,)
        next_states = segment['next_states'] # (seq_len, 64)
        dones = segment['dones']            # (seq_len,)
        init_hidden = segment['init_hidden'] # 起始隐藏状态
        
        # 按时序展开LSTM
        seq_len = states.shape[0]
        h_actor = init_hidden['actor']
        h_critic1 = init_hidden['critic1']
        h_critic2 = init_hidden['critic2']
        
        # 逐步展开（或使用LSTM的batch_first模式）
        for t in range(seq_len):
            state_t = states[t]
            action_t = actions[t]
            reward_t = rewards[t]
            next_state_t = next_states[t]
            done_t = dones[t]
            
            # Actor前向（更新h_actor）
            next_action, next_log_prob, h_actor = actor(next_state_t, h_actor)
            
            # Target Q计算
            target_q1, _ = critic1_target(next_state_t, next_action, ...)
            target_q2, _ = critic2_target(next_state_t, next_action, ...)
            target_q = min(target_q1, target_q2) - alpha * next_log_prob
            target = reward_t + gamma * (1 - done_t) * target_q
            
            # Current Q计算（更新h_critic）
            current_q1, h_critic1 = critic1(state_t, action_t, h_critic1)
            current_q2, h_critic2 = critic2(state_t, action_t, h_critic2)
            
            # 累积损失
            critic_loss += F.mse_loss(current_q1, target)
            critic_loss += F.mse_loss(current_q2, target)
        
        # 平均损失
        critic_loss /= seq_len
```

**关键区别**:
- 原方案: 256个独立样本并行计算
- 当前实现: 32个segment，每个segment内部按时序展开

---

### 4. **参数量问题** ⚠️

| 模块 | 原方案预期 | 当前实现 | 差异 |
|-----|-----------|---------|-----|
| SocialCircle | 20K (冻结) | 90K (内置于Predictor) | +70K |
| E-V2-Net | 300K (冻结) | 1.96M (SimplifiedE_V2_Net) | +1.66M |
| **总计** | **1.73M** (含冻结) | **3.03M** | **+1.3M** |
| **可训练** | **1.41M** | **3.03M** | **超出预算** |

**原因分析**:
- 原方案假设使用预训练的轻量级E-V2-Net (300K)
- 当前实现的SimpleE_V2_Net为每个模态创建独立的GRU解码器
  - 20个模态 × 每个~100K = 2M参数

**解决方案**:
1. 使用真实的预训练E-V2-Net（需下载开源权重）
2. 重构为共享解码器 + 模态嵌入
3. 减少模态数 (20 → 10)

---

### 5. **DogStateEncoder隐藏状态**

| 维度 | 原方案 | 当前实现 |
|-----|--------|---------|
| Hidden返回 | 明确返回 | 不返回（内部管理） |
| 接口 | `forward(...) -> (features, hidden)` | `forward(...) -> features` |
| SAC Hidden | Actor/Critic独立管理 | 只有Actor/Critic有hidden |

**当前实现合理性**: DogEncoder的GRU是特征提取用，不需要跨时间步记忆。SAC的LSTM才是真正的决策记忆。

---

### 6. **Alpha Loss计算** (已修正)

#### 原方案（可能有bug）
```python
alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
```

#### 当前实现（用户反馈后修正）
```python
# 累积所有segment的log_prob
total_log_prob_for_alpha = 0
for segment in segment_batch:
    for t in range(seq_len):
        action, log_prob, _ = actor(state_t, h)
        total_log_prob_for_alpha += (log_prob + target_entropy).detach()

# 计算总样本数
total_samples = sum(segment['states'].shape[0] for segment in segment_batch)

# 正确的平均
avg_log_prob = total_log_prob_for_alpha / total_samples
alpha_loss = -(log_alpha * avg_log_prob)
```

**关键**: 除以总样本数，而非batch_size。

---

### 7. **模块接口差异**

#### 原方案预期
```python
# SocialCircle独立模块
social_feat = social_circle(target_traj, neighbor_trajs, angles)

# E-V2-Net独立模块
future_pred = e_v2_net(social_feat)
```

#### 当前实现
```python
# 整合为TrajectoryPredictor
future_pred = trajectory_predictor(
    target_trajectory=target_traj,
    neighbor_trajectories=neighbor_trajs,
    neighbor_angles=angles,
    neighbor_mask=mask
)
# 内部调用: SocialCircle → E-V2-Net
```

**优势**: 接口更清晰，便于切换预训练/简化实现。

---

### 8. **观测格式统一**

#### 原方案
```python
inputs_raw = {
    'pedestrian_past_trajs': [Tensor(8,2), ...],
    'corridors': [Tensor(N_i,2), ...],
    'dog_past_traj': Tensor(8,2),
    ...
}
```

#### 当前实现
```python
observation = {
    'dog': {
        'trajectory': (batch, 8, 2),
        'velocity': (batch, 2),
        'position': (batch, 2),
        'goal': (batch, 2)
    },
    'pedestrians': {
        'trajectories': (batch, max_peds, 8, 2),
        'mask': (batch, max_peds)
    },
    'corridors': {
        'polygons': (batch, max_corridors, max_vertices, 2),
        'vertex_counts': (batch, max_corridors),
        'mask': (batch, max_corridors)
    },
    'reference_line': (batch, 2, 2)
}
```

**改进**: 
- 已经包含batch维度
- 已经完成padding和mask
- 结构化更清晰

---

## ✅ 保持一致的部分

1. ✅ **网络架构**: DogEncoder, CorridorEncoder, PedestrianEncoder, Fusion结构基本一致
2. ✅ **特征维度**: 
   - Dog: 64
   - Pedestrian: 64
   - Corridor: 128
   - Fusion: 64
3. ✅ **SAC结构**: Actor/Critic都是PreFC + LSTM + Head
4. ✅ **GDE**: 几何微分评估器完全一致
5. ✅ **动作空间**: (11, 2) = 22维
6. ✅ **Padding策略**: max_pedestrians=10, max_corridors=5
7. ✅ **Mask机制**: 所有编码器正确处理mask

---

## 🎯 需要补充的功能

### 当前缺失（按优先级）

1. **SequenceReplayBuffer** - 序列段采样
2. **AGSACEnvironment** - 环境接口
3. **AGSACTrainer** - 训练循环
4. **参数优化** - 将3M降至2M以内
5. **配置系统** - YAML超参数管理
6. **训练脚本** - train.py, evaluate.py

---

## 📝 建议的下一步

### 选项A: 完善当前架构（推荐）
1. 优化TrajectoryPredictor参数量
2. 实现SequenceReplayBuffer
3. 实现Environment和Trainer
4. 端到端测试

### 选项B: 回归原方案
1. 改为单步采样
2. 移除segment逻辑
3. 简化hidden state管理

**推荐选项A**: 序列段训练更适合LSTM，已投入的工作可以保留。

