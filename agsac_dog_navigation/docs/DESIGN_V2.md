# AGSAC混合方案完整设计文档 V2.0

> **版本**: 2.0  
> **更新**: 2025-10-02  
> **主要变化**: 序列段训练 + Burn-in机制 + 模块化接口

---

## 一、核心设定

```yaml
任务: 机器狗在动态环境中的路径规划
输入:
  - 行人历史轨迹: 2-10个行人，每个(8,2)
  - 通路多边形: 2-10个，顶点数可变
  - 机器狗状态: 历史轨迹(8,2)，速度(2)，位置(2)，目标(2)
输出:
  - 规划路径: (11,2) 全局坐标
约束:
  - 参数量 < 2M
  - 推理时间 < 50ms
  - 超过max时截断（选择最近的N个）
  
训练方式:
  - 序列段采样 (seq_len=16)
  - Burn-in支持LSTM预热
  - Episode级别存储
```

---

## 二、整体架构

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        输入层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

观测格式（已padding + mask）:
observation = {
    'dog': {
        'trajectory': (batch, 8, 2),
        'velocity': (batch, 2),
        'position': (batch, 2),
        'goal': (batch, 2)
    },
    'pedestrians': {
        'trajectories': (batch, max_peds, 8, 2),
        'mask': (batch, max_peds)  # 1=有效, 0=padding
    },
    'corridors': {
        'polygons': (batch, max_corridors, max_vertices, 2),
        'vertex_counts': (batch, max_corridors),
        'mask': (batch, max_corridors)
    },
    'reference_line': (batch, 2, 2)  # GDE用
}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  特征提取层（三路并行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[路径1] 行人未来预测
  ├─ TrajectoryPredictor (整合SocialCircle + E-V2-Net):
  │   ├─ 对每个有效行人i:
  │   │   ├─ 提取target_traj: (batch, 8, 2)
  │   │   ├─ 提取neighbor_trajs: (batch, N-1, 8, 2)
  │   │   ├─ 计算neighbor_angles: (batch, N-1)
  │   │   ├─ SocialCircle编码: → (batch, 128)
  │   │   └─ E-V2-Net预测: → (batch, 12, 2, 20)
  │   └─ Stack所有行人: (batch, max_peds, 12, 2, 20)
  ├─ PedestrianEncoder:
  │   ├─ 逐行人逐模态GRU编码
  │   ├─ 多模态注意力聚合 (20→1)
  │   └─ 跨行人注意力聚合 (N→1, with mask)
  └─ 输出: (batch, 64)

[路径2] 通路几何
  ├─ 逐多边形PointNet编码: (N_i,2) → (64)
  ├─ Stack + Padding: (batch, max_corridors, 64)
  ├─ CorridorEncoder:
  │   ├─ 位置编码
  │   ├─ 自注意力 (with mask)
  │   └─ 聚合有效走廊
  └─ 输出: (batch, 128)

[路径3] 机器狗状态
  ├─ DogStateEncoder:
  │   ├─ GRU编码历史轨迹: (8,2) → (64)
  │   ├─ MLP编码速度: (2) → (32)
  │   ├─ MLP编码相对目标: (goal-pos) → (32)
  │   └─ 融合: (64+32+32) → (64)
  └─ 输出: (batch, 64)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      融合层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MultiModalFusion:
  ├─ 投影统一维度:
  │   - dog: (64) → (64)
  │   - pedestrian: (64) → (64)
  │   - corridor: (128) → (64)
  ├─ 注意力融合:
  │   Query: dog_features
  │   Key/Value: [pedestrian_features, corridor_features]
  │   → attended_env: (batch, 64)
  ├─ 拼接: cat([dog, attended_env]) → (128)
  └─ MLP: (128) → (64)
  
输出: fused_state (batch, 64) + attention_weights


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  SAC决策层（混合架构）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HybridActor:
  ├─ PreFC: (64) → (128) → ReLU
  ├─ LSTM: (128) + hidden → (128) + new_hidden
  ├─ MeanHead: (128) → (22)
  ├─ LogStdHead: (128) → (22), clamp(-20, 2)
  └─ 重参数化采样 + tanh
  
  输出: action (batch, 22), log_prob, new_hidden

HybridCritic (×2):
  ├─ Concat: [state, action] → (86)
  ├─ PreFC: (86) → (128) → ReLU
  ├─ LSTM: (128) + hidden → (128) + new_hidden
  └─ QHead: (128) → (128) → (1)
  
  输出: Q值 (batch, 1), new_hidden

Temperature: log_alpha (可学习)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    评估层（训练时）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GeometricDifferentialEvaluator:
  ├─ 离散微分: path[i+1] - path[i]
  ├─ 夹角计算: arccos(d_norm · ref_norm)
  ├─ 指数加权: exp(-k/M)
  └─ 归一化评分: geo_score ∈ [0, 1]
  
奖励塑造:
  r_final = r_base + η * geo_score
  (η = 0.5)
```

---

## 三、详细数据流（单步推理 + Batch处理）

### 3.1 输入预处理（环境→标准格式）

```python
# 环境原始输出（可变长度）
env_output = {
    'pedestrians': [Tensor(8,2), ...],     # 2-10个
    'corridors': [Tensor(N_i,2), ...],     # 2-10个，N_i可变
    'dog_state': {
        'trajectory': Tensor(8,2),
        'velocity': Tensor(2),
        'position': Tensor(2),
        'goal': Tensor(2)
    },
    'reference_line': Tensor(2)
}

# 预处理为标准格式
observation = preprocess_observation(env_output)
# → 转换为 observation = {...} 如第二节所示
```

**预处理步骤**:
1. **截断超限**: 如果行人>10或走廊>10，选择最近的N个
2. **Padding行人**: 补齐到(max_peds, 8, 2)，生成mask
3. **PointNet编码走廊**: 逐个编码为(64)，补齐到(max_corridors, 64)
4. **添加batch维度**: 推理时batch=1，训练时batch>1

---

### 3.2 模型前向传播

```python
# 初始化隐藏状态（episode开始时）
hidden_states = model.init_hidden_states(batch_size=1)
# {
#     'actor': (h, c),      # (1, 1, 128)
#     'critic1': (h, c),
#     'critic2': (h, c)
# }

# 前向传播
result = model.forward(
    observation=observation,
    hidden_states=hidden_states,
    deterministic=False,
    return_attention=False
)

# 输出
action = result['action']                # (batch, 22)
log_prob = result['log_prob']            # (batch,)
q1, q2 = result['q1'], result['q2']      # (batch,), (batch,)
fused_state = result['fused_state']      # (batch, 64)
new_hidden = result['hidden_states']     # 更新后的隐藏状态
```

**内部流程**:
1. **编码机器狗状态**: DogEncoder → (batch, 64)
2. **编码走廊几何**: PointNet + CorridorEncoder → (batch, 128)
3. **预测行人轨迹**: TrajectoryPredictor → (batch, max_peds, 12, 2, 20)
4. **编码行人特征**: PedestrianEncoder → (batch, 64)
5. **多模态融合**: MultiModalFusion → (batch, 64)
6. **Actor生成动作**: HybridActor + LSTM → (batch, 22)
7. **Critic评估Q值**: HybridCritic + LSTM → (batch, 1)

---

### 3.3 路径转换与执行

```python
# 动作是归一化的路径 [-1, 1]^22
action_normalized = result['action']  # (batch, 22)

# 转换为全局坐标路径
path_global = convert_action_to_path(
    action_normalized,        # (batch, 22)
    robot_position,           # (batch, 2)
    robot_yaw                 # (batch,)
)  # → (batch, 11, 2) 全局坐标

# 执行第一个路径点
next_obs, reward, done, info = env.step(path_global[:, 0, :])
```

---

## 四、序列段训练流程 ⭐ 核心改进

### 4.1 SequenceReplayBuffer设计

```python
class SequenceReplayBuffer:
    """
    存储完整episode，采样固定长度segment
    """
    def __init__(
        self,
        capacity: int = 100000,     # episode容量
        seq_len: int = 16,          # segment长度
        burn_in: int = 4            # burn-in长度（可选）
    ):
        self.episodes = []          # List of episodes
        self.capacity = capacity
        self.seq_len = seq_len
        self.burn_in = burn_in
    
    def add_episode(self, episode_data):
        """
        存储完整episode
        
        episode_data = {
            'observations': List[Dict],      # T个观测
            'fused_states': List[Tensor],    # T个融合状态 (64)
            'actions': List[Tensor],         # T个动作 (22)
            'rewards': List[float],          # T个奖励
            'dones': List[bool],             # T个终止标志
            'hidden_states': List[Dict],     # T个隐藏状态
            'episode_return': float,
            'episode_length': int
        }
        """
        self.episodes.append(episode_data)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """
        采样segment batch
        
        Returns:
            segment_batch: List of segments
            [
                {
                    'states': (seq_len, 64),
                    'actions': (seq_len, 22),
                    'rewards': (seq_len,),
                    'next_states': (seq_len, 64),
                    'dones': (seq_len,),
                    'init_hidden': {
                        'actor': (h, c),
                        'critic1': (h, c),
                        'critic2': (h, c)
                    }
                },
                ...  # batch_size个
            ]
        """
        segments = []
        for _ in range(batch_size):
            # 1. 随机选择一个episode
            episode = random.choice(self.episodes)
            
            # 2. 随机选择起始位置
            max_start = len(episode['fused_states']) - self.seq_len
            if max_start <= 0:
                continue  # episode太短，跳过
            
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + self.seq_len
            
            # 3. 提取segment
            segment = {
                'states': torch.stack(
                    episode['fused_states'][start_idx:end_idx]
                ),  # (seq_len, 64)
                'actions': torch.stack(
                    episode['actions'][start_idx:end_idx]
                ),  # (seq_len, 22)
                'rewards': torch.tensor(
                    episode['rewards'][start_idx:end_idx]
                ),  # (seq_len,)
                'next_states': torch.stack(
                    episode['fused_states'][start_idx+1:end_idx+1]
                ),  # (seq_len, 64)
                'dones': torch.tensor(
                    episode['dones'][start_idx:end_idx]
                ),  # (seq_len,)
                'init_hidden': episode['hidden_states'][start_idx]
            }
            
            segments.append(segment)
        
        return segments
```

**关键设计**:
- 存储粒度: **Episode级别**
- 采样粒度: **Segment级别** (seq_len=16)
- Hidden管理: 存储segment起始的hidden state
- Burn-in支持: 可选地从start_idx前若干步开始warm-up

---

### 4.2 SAC序列段更新

```python
class SACAgent:
    def update(self, segment_batch: List[Dict]) -> Dict[str, float]:
        """
        序列段更新（核心训练逻辑）
        
        Args:
            segment_batch: List of segments from buffer
        
        Returns:
            losses: {'critic_loss', 'actor_loss', 'alpha_loss', 'alpha'}
        """
        total_critic_loss = 0
        total_actor_loss = 0
        total_log_prob_for_alpha = 0
        total_samples = 0
        
        for segment in segment_batch:
            states = segment['states']          # (seq_len, 64)
            actions = segment['actions']        # (seq_len, 22)
            rewards = segment['rewards']        # (seq_len,)
            next_states = segment['next_states'] # (seq_len, 64)
            dones = segment['dones']            # (seq_len,)
            init_hidden = segment['init_hidden']
            
            seq_len = states.shape[0]
            total_samples += seq_len
            
            # ============ Critic Update ============
            # 初始化hidden states
            h_critic1 = init_hidden['critic1']
            h_critic2 = init_hidden['critic2']
            h_critic1_target = init_hidden['critic1']  # 目标网络用相同初始hidden
            h_critic2_target = init_hidden['critic2']
            h_actor_for_target = init_hidden['actor']
            
            # 按时序展开
            for t in range(seq_len):
                state_t = states[t:t+1]          # (1, 64)
                action_t = actions[t:t+1]        # (1, 22)
                reward_t = rewards[t]
                next_state_t = next_states[t:t+1]
                done_t = dones[t]
                
                # Target Q计算
                with torch.no_grad():
                    next_action, next_log_prob, h_actor_for_target = self.actor(
                        next_state_t, h_actor_for_target
                    )
                    
                    target_q1, h_critic1_target = self.critic_target.critic1(
                        next_state_t, next_action, h_critic1_target
                    )
                    target_q2, h_critic2_target = self.critic_target.critic2(
                        next_state_t, next_action, h_critic2_target
                    )
                    
                    target_q = torch.min(target_q1, target_q2)
                    target_q = target_q - self.alpha * next_log_prob
                    target = reward_t + self.gamma * (1 - done_t) * target_q
                
                # Current Q计算
                curr_q1, h_critic1 = self.critic.critic1(
                    state_t, action_t, h_critic1
                )
                curr_q2, h_critic2 = self.critic.critic2(
                    state_t, action_t, h_critic2
                )
                
                # Critic loss
                critic_loss_t = F.mse_loss(curr_q1.squeeze(), target.squeeze())
                critic_loss_t += F.mse_loss(curr_q2.squeeze(), target.squeeze())
                
                total_critic_loss += critic_loss_t
            
            # ============ Actor Update ============
            h_actor = init_hidden['actor']
            h_critic1_for_actor = init_hidden['critic1']
            h_critic2_for_actor = init_hidden['critic2']
            
            for t in range(seq_len):
                state_t = states[t:t+1]
                
                # Actor前向
                action_new, log_prob, h_actor = self.actor(
                    state_t, h_actor
                )
                
                # Q值评估
                q1, h_critic1_for_actor = self.critic.critic1(
                    state_t, action_new, h_critic1_for_actor
                )
                q2, h_critic2_for_actor = self.critic.critic2(
                    state_t, action_new, h_critic2_for_actor
                )
                q = torch.min(q1, q2)
                
                # Actor loss
                actor_loss_t = (self.alpha * log_prob - q).mean()
                total_actor_loss += actor_loss_t
                
                # Alpha用
                total_log_prob_for_alpha += (log_prob + self.target_entropy).detach()
        
        # 平均损失
        avg_critic_loss = total_critic_loss / len(segment_batch)
        avg_actor_loss = total_actor_loss / len(segment_batch)
        
        # Backward
        self.critic_optimizer.zero_grad()
        avg_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        avg_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # ============ Alpha Update ============
        if self.auto_entropy:
            avg_log_prob = total_log_prob_for_alpha / total_samples  # 关键：除以总样本数
            alpha_loss = -(self.log_alpha * avg_log_prob)
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        return {
            'critic_loss': avg_critic_loss.item(),
            'actor_loss': avg_actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy else 0,
            'alpha': self.alpha
        }
```

**关键点**:
1. ✅ 按segment循环，segment内按时序展开
2. ✅ Hidden state在segment起始初始化，内部递进
3. ✅ Target网络使用相同的初始hidden（保持一致性）
4. ✅ Alpha loss除以总样本数（修正后）
5. ✅ 梯度裁剪防止exploding gradients

---

### 4.3 完整训练循环

```python
class AGSACTrainer:
    def train(self, num_episodes: int):
        for episode in range(num_episodes):
            # ========== Episode开始 ==========
            observation = self.env.reset()
            hidden_states = self.model.init_hidden_states(batch_size=1)
            
            episode_data = {
                'observations': [],
                'fused_states': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'hidden_states': [],
                'geo_scores': []
            }
            
            done = False
            episode_return = 0
            
            # ========== Episode执行 ==========
            while not done:
                # 1. 选择动作
                action, log_prob, new_hidden = self.model.select_action(
                    observation, hidden_states, deterministic=False
                )
                
                # 2. 转换并执行
                path_global = convert_action_to_path(action, ...)
                next_obs, reward_base, done, info = self.env.step(path_global[0])
                
                # 3. 计算geo_score
                geo_score = self.model.geo_evaluator(
                    path_global, observation['reference_line']
                )
                
                # 4. 奖励塑造
                reward_final = reward_base + 0.5 * geo_score
                
                # 5. 获取fused_state（用于存储）
                with torch.no_grad():
                    result = self.model.forward(
                        observation, hidden_states, deterministic=False
                    )
                    fused_state = result['fused_state']
                
                # 6. 存储transition
                episode_data['observations'].append(observation)
                episode_data['fused_states'].append(fused_state)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward_final)
                episode_data['dones'].append(done)
                episode_data['hidden_states'].append(hidden_states)
                episode_data['geo_scores'].append(geo_score)
                
                # 7. 更新状态
                observation = next_obs
                hidden_states = new_hidden
                episode_return += reward_final
            
            # ========== Episode结束 ==========
            episode_data['episode_return'] = episode_return
            episode_data['episode_length'] = len(episode_data['rewards'])
            
            # 存储到buffer
            self.buffer.add_episode(episode_data)
            
            # ========== 训练更新 ==========
            if self.buffer.size() >= self.warmup_steps:
                for _ in range(self.updates_per_episode):
                    # 采样segment batch
                    segment_batch = self.buffer.sample(batch_size=32)
                    
                    # SAC更新
                    losses = self.model.update(segment_batch)
                    
                    # 软更新target
                    self.model.soft_update_target()
                    
                    # 记录
                    self.logger.log(losses)
            
            # ========== 定期评估 ==========
            if episode % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.logger.log_eval(eval_metrics)
                
            # ========== 保存检查点 ==========
            if episode % self.save_interval == 0:
                self.model.save_checkpoint(f'ckpt_ep{episode}.pth')
```

---

## 五、关键模块实现细节

### 5.1 TrajectoryPredictor（整合接口）

```python
class TrajectoryPredictorInterface(nn.Module):
    def forward(
        self,
        target_trajectory: Tensor,       # (batch, 8, 2)
        neighbor_trajectories: Tensor,   # (batch, N, 8, 2)
        neighbor_angles: Tensor,         # (batch, N)
        neighbor_mask: Tensor            # (batch, N)
    ) -> Tensor:
        # → (batch, pred_horizon, 2, num_modes)
        raise NotImplementedError

# 实现1: 简化版（当前）
class SimpleTrajectoryPredictor(TrajectoryPredictorInterface):
    def __init__(...):
        self.social_circle = SimpleSocialCircle(...)
        self.e_v2_net = SimpleE_V2_Net(...)
    
    def forward(...):
        social_feat = self.social_circle(...)
        predictions = self.e_v2_net(social_feat)
        return predictions

# 实现2: 预训练版（待整合）
class PretrainedTrajectoryPredictor(TrajectoryPredictorInterface):
    def __init__(self, weights_path):
        # 加载开源SocialCircle + E-V2-Net
        self.model = load_pretrained_model(weights_path)
    
    def forward(...):
        return self.model(...)
```

---

### 5.2 PedestrianEncoder（三层注意力）

```
输入: (batch, max_peds, pred_horizon, 2, num_modes)
输出: (batch, 64)

流程:
1. 对每个行人的每个模态:
   GRU编码时序: (pred_horizon, 2) → (hidden_dim)
   → (batch, max_peds, num_modes, hidden_dim)

2. 多模态注意力聚合:
   Query = learnable token
   Key/Value = 20个模态特征
   → (batch, max_peds, hidden_dim)

3. 跨行人注意力聚合:
   Query = learnable token
   Key/Value = N个行人特征（with mask）
   → (batch, hidden_dim)

4. MLP投影: (hidden_dim) → (64)
```

---

### 5.3 CorridorEncoder（位置编码+自注意力）

```
输入: (batch, max_corridors, 64) + mask
输出: (batch, 128)

流程:
1. 添加可学习位置编码
2. 自注意力（key_padding_mask处理无效走廊）
3. 残差连接 + LayerNorm
4. 前馈网络
5. 聚合有效走廊（masked mean pooling）
6. MLP投影到128维
```

---

### 5.4 MultiModalFusion（Query-based注意力）

```
输入: dog(batch,64), pedestrian(batch,64), corridor(batch,128)
输出: (batch, 64)

流程:
1. 投影统一维度: corridor(128) → (64)
2. Stack环境特征: [pedestrian, corridor] → (batch, 2, 64)
3. Query-based注意力:
   Query: dog_features (batch, 1, 64)
   Key/Value: env_features (batch, 2, 64)
   → attended_env (batch, 64)
4. 拼接: [dog, attended_env] → (batch, 128)
5. MLP: (128) → (64)
```

---

## 六、参数预算（目标 vs 当前）

| 模块 | 目标参数量 | 当前实现 | 状态 |
|-----|-----------|---------|-----|
| DogEncoder | 75K | 65K | ✅ |
| PointNet | 50K | 117K | ⚠️ |
| CorridorEncoder | 70K | 59K | ✅ |
| **TrajectoryPredictor** | **320K** | **2.05M** | ❌ |
| PedestrianEncoder | 120K | 225K | ⚠️ |
| Fusion | 50K | 50K | ✅ |
| Actor | 355K | 146K | ✅ |
| Critic×2 | 740K | 320K | ✅ |
| **总计** | **1.78M** | **3.03M** | ❌ 超出70% |

### 参数优化方案

**优先级1**: TrajectoryPredictor优化（必须）
- 当前: 2.05M (20个独立GRU解码器)
- 目标: <320K
- 方案:
  - A. 使用预训练E-V2-Net（推荐）
  - B. 共享解码器 + 模态嵌入
  - C. 减少模态数 (20→10)

**优先级2**: 其他模块微调
- PointNet: 117K → 80K（减少hidden dims）
- PedestrianEncoder: 225K → 150K（减少GRU层数）

**优化后预期**: **~1.5M** ✅ 符合<2M预算

---

## 七、超参数配置

```yaml
# 网络结构
network:
  dog_feature_dim: 64
  corridor_feature_dim: 128
  pedestrian_feature_dim: 64
  fusion_dim: 64
  hidden_dim: 128
  num_heads: 4
  dropout: 0.1

# 场景限制
scene:
  max_pedestrians: 10
  max_corridors: 5
  max_vertices: 20
  obs_horizon: 8
  pred_horizon: 12
  num_modes: 20

# SAC配置
sac:
  action_dim: 22
  gamma: 0.99
  tau: 0.005
  actor_lr: 1e-4
  critic_lr: 1e-4
  alpha_lr: 3e-4
  auto_entropy: true
  target_entropy: -22
  max_grad_norm: 1.0

# 训练配置
training:
  num_episodes: 10000
  seq_len: 16              # 序列段长度
  burn_in: 4               # Burn-in长度（可选）
  batch_size: 32           # segment batch size
  buffer_capacity: 100000  # episode容量
  warmup_steps: 1000
  updates_per_episode: 4
  eval_interval: 100
  save_interval: 500

# GDE配置
gde:
  eta: 0.5                 # 奖励塑造权重
  M: 10                    # 指数衰减参数

# 奖励函数
reward:
  arrival: 300
  collision: -200
  warning: -200
  progress_coeff: 500
```

---

## 八、实现检查清单 V2

### 核心模块 ✅
- [x] DogStateEncoder
- [x] PointNet
- [x] CorridorEncoder
- [x] SimpleSocialCircle + SimpleE_V2_Net
- [x] PedestrianEncoder
- [x] MultiModalFusion
- [x] HybridActor
- [x] HybridCritic
- [x] SACAgent (支持序列段更新)
- [x] GeometricDifferentialEvaluator
- [x] AGSACModel (主模型整合)

### 训练基础设施 ⏳
- [ ] SequenceReplayBuffer
- [ ] AGSACEnvironment
- [ ] AGSACTrainer
- [ ] 配置系统

### 参数优化 ⏳
- [ ] TrajectoryPredictor重构/预训练加载
- [ ] PointNet轻量化
- [ ] PedestrianEncoder优化
- [ ] 总参数量验证 <2M

### 测试验证 ⏳
- [ ] 序列段采样测试
- [ ] Hidden state传递测试
- [ ] 端到端训练测试
- [ ] 推理时间测试 <50ms

### 工具脚本 ⏳
- [ ] train.py
- [ ] evaluate.py
- [ ] demo.py
- [ ] 可视化工具

---

## 九、与原方案的主要改进

### ✅ 保留的设计
1. 三路并行特征提取
2. 多模态融合机制
3. 混合SAC架构（LSTM记忆）
4. 几何微分评估器
5. Padding + Mask处理可变输入
6. 奖励塑造策略

### ⭐ 核心改进
1. **序列段训练**: 单步 → segment
2. **Episode存储**: transition → full episode
3. **Hidden管理**: 每步存储 → segment起始存储
4. **模块接口**: 分散 → 整合（TrajectoryPredictor）
5. **观测格式**: 可变 → 标准化batch格式
6. **Alpha loss**: 简单平均 → 正确总样本数平均

### ⚠️ 待解决问题
1. 参数量超预算（3M vs 2M）
2. 缺少SequenceReplayBuffer实现
3. 缺少环境接口
4. 缺少训练器

---

## 十、下一步行动计划

### Phase 1: 参数优化（1-2天）
1. 重构TrajectoryPredictor
2. 验证参数量 <2M
3. 保证功能不变

### Phase 2: 训练基础（2-3天）
1. 实现SequenceReplayBuffer
2. 实现AGSACEnvironment
3. 实现AGSACTrainer
4. 简单训练测试

### Phase 3: 完善系统（1-2天）
1. 配置系统
2. 训练/评估脚本
3. 端到端测试
4. 性能优化

---

**预计完成时间**: 4-7天  
**当前完成度**: 85%  
**关键路径**: 参数优化 → 训练基础设施 → 端到端验证

