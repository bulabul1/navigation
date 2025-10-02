# AGSAC混合方案完整设计文档 V1.0 (原始方案)

> **版本**: 1.0 (原始设计)  
> **训练方式**: 单步transition采样  
> **Buffer**: Transition级别存储  

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
```

---

## 二、整体架构

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        输入层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

可变输入 → Padding + Mask → 固定格式

行人轨迹: List[Tensor(8,2)] 长度2-10
  → pad到(10,8,2) + mask(10)
  
通路多边形: List[Tensor(N_i,2)] 长度2-10，顶点数可变
  → 逐个PointNet编码到(10,64) + mask(10)
  
机器狗状态: 直接编码


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  特征提取层（三路并行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[路径1] 行人未来预测（预训练冻结）
  ├─ 对每个有效行人i:
  │   ├─ SocialCircle编码 (8,2) → (64)
  │   ├─ E-V2-Net预测 (64) → (12,2,20)
  │   └─ 汇总: (N_peds, 12,2,20)
  ├─ PedestrianEncoder:
  │   ├─ 逐行人时序GRU编码
  │   ├─ 多模态注意力聚合
  │   └─ 跨行人注意力聚合（with mask）
  └─ 输出: (64)

[路径2] 通路几何
  ├─ 逐多边形PointNet编码: (N_i,2) → (64)
  ├─ Stack: (N_corridors, 64)
  ├─ Padding: (10, 64) + mask(10)
  ├─ 多边形间注意力（with mask）
  └─ 输出: (128)

[路径3] 机器狗状态
  ├─ 历史轨迹GRU编码: (8,2) → (64)
  ├─ 速度MLP编码: (2) → (32)
  ├─ 相对目标MLP编码: (2) → (32)
  ├─ 融合: (64+32+32) → (64)
  └─ 输出: (64)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      融合层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

多模态注意力融合:
  Query: 机器狗特征(64)
  Key/Value: [行人特征(64), 通路特征(128)]
  输出: 融合状态(64)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  SAC决策层（带LSTM记忆）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Actor:
  融合状态(64) → FC(256) → LSTM(256,1层) → Mean/LogStd(22)
  输出: 路径(11,2) + log_prob + 隐藏状态

Critic Q1/Q2:
  [融合状态(64) + 动作(22)] → FC(256) → LSTM(256,1层) → Q(1)
  输出: Q值 + 隐藏状态

温度参数: log_alpha（可学习）


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    评估层（训练时）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

几何微分评估器(GDE):
  路径(11,2) + 参考线(2) → geo_score
  
奖励塑造:
  r_final = r_base + 0.5 * geo_score
```

---

## 三、详细数据流（单步推理）

### 输入格式
```python
# 原始输入（可变）
inputs_raw = {
    'pedestrian_past_trajs': [Tensor(8,2), ...],  # 2-10个
    'corridors': [Tensor(N_i,2), ...],            # 2-10个，N_i可变
    'dog_past_traj': Tensor(8,2),
    'dog_current_vel': Tensor(2),
    'dog_position': Tensor(2),
    'goal_position': Tensor(2),
    'reference_line': Tensor(2),
}

# 隐藏状态（前一时刻）
hidden_states = {
    'h_actor': (h, c),      # (1,1,256), (1,1,256)
    'h_critic1': (h, c),
    'h_critic2': (h, c),
}
```

### 步骤1: 输入预处理（可变→固定）
```python
# 截断超限输入
if len(pedestrians) > 10:
    pedestrians = select_closest_n(pedestrians, robot_pos, n=10)
if len(corridors) > 10:
    corridors = select_closest_n(corridors, robot_pos, n=10)

# Padding行人轨迹
N_peds = len(pedestrians)
ped_trajs_padded = torch.zeros(10, 8, 2)
ped_trajs_padded[:N_peds] = torch.stack(pedestrians)
ped_mask = torch.zeros(10)
ped_mask[:N_peds] = 1
# → (10,8,2), mask(10)

# 通路PointNet编码
corridor_features = []
for corridor in corridors:
    feat = pointnet_encode(corridor)  # (N_i,2) → (64)
    corridor_features.append(feat)
N_corridors = len(corridors)
corridor_feats_padded = torch.zeros(10, 64)
corridor_feats_padded[:N_corridors] = torch.stack(corridor_features)
corridor_mask = torch.zeros(10)
corridor_mask[:N_corridors] = 1
# → (10,64), mask(10)
```

### 步骤2: 行人未来预测（路径1）
```python
all_futures = []
for i in range(N_peds):
    ped_i = ped_trajs_padded[i]  # (8,2)
    other_peds = ped_trajs_padded  # (10,8,2)
    
    # 计算相对角度
    angles = compute_angles(ped_i[-1], other_peds[:,-1])  # (10,)
    
    # SocialCircle（预训练冻结）
    social_feat = social_circle(ped_i, other_peds, angles)  # (64)
    
    # E-V2-Net预测（预训练冻结）
    future_i = e_v2_net(social_feat)  # (12,2,20)
    all_futures.append(future_i)

all_futures = torch.stack(all_futures)  # (N_peds, 12,2,20)

# PedestrianEncoder聚合
ped_feature = pedestrian_encoder(all_futures, ped_mask)  # (64)
```

### 步骤3: 通路几何编码（路径2）
```python
# 输入已是 (10,64) + mask(10)
corridor_feature = corridor_encoder(
    corridor_feats_padded,  # (10,64)
    corridor_mask,          # (10)
    robot_position
)  # → (128)
```

### 步骤4: 机器狗状态编码（路径3）
```python
dog_feature = dog_encoder(
    past_traj=dog_past_traj,      # (8,2)
    current_vel=dog_current_vel,  # (2)
    current_pos=dog_position,     # (2)
    goal_pos=goal_position        # (2)
)  # → (64)
```

### 步骤5: 多模态融合
```python
fused_state, attn_weights = fusion(
    dog_feat=dog_feature,          # (64)
    ped_feat=ped_feature,          # (64)
    corridor_feat=corridor_feature # (128)
)  # → (64), (1,2)
```

### 步骤6: SAC Actor决策
```python
path_normalized, log_prob, h_actor_new = actor(
    state=fused_state,              # (64)
    hidden_state=hidden_states['h_actor']
)  # → (11,2) in [-1,1], scalar, (h,c)
```

### 步骤7: 路径转换
```python
# 增量式转换
path_global = convert_incremental_path(
    path_normalized,  # (11,2)
    robot_position,   # (2)
    robot_yaw         # scalar
)  # → (11,2) 全局坐标
```

### 步骤8: 几何评估（训练时）
```python
geo_score = gde(path_global, reference_line)  # scalar
r_final = r_base + 0.5 * geo_score
```

### 步骤9: Critic评估（训练时）
```python
q1, h_c1_new = critic1(
    fused_state,                  # (64)
    path_normalized.view(-1),     # (22)
    hidden_states['h_critic1']
)  # → scalar, (h,c)

q2, h_c2_new = critic2(
    fused_state,
    path_normalized.view(-1),
    hidden_states['h_critic2']
)  # → scalar, (h,c)
```

### 输出
```python
output = {
    'path': path_global,             # (11,2)
    'log_prob': log_prob,            # scalar
    'q_values': (q1, q2),           # scalar, scalar
    'geo_score': geo_score,          # scalar
    'new_hidden_states': {
        'h_actor': h_actor_new,
        'h_critic1': h_c1_new,
        'h_critic2': h_c2_new,
    }
}
```

---

## 四、关键模块实现细节

### 4.1 PointNet多边形编码器
```
输入: 可变顶点 (N_vertices, 2)
处理:
  1. 相对化: vertices - robot_pos → (N,2)
  2. 相对质心: vertices - centroid → (N,2)
  3. 拼接特征: [相对坐标, 局部坐标] → (N,4)
  4. MLP编码: (N,4) → (N,64)
  5. 对称聚合: max_pool + mean_pool → (128)
  6. MLP: (128) → (64)
输出: (64) 固定维度
```

### 4.2 行人未来编码器
```
输入: (N_peds, 12,2,20) + mask(N_peds)
处理:
  1. 对每个行人的每个模态:
     - GRU编码时序: (12,2) → (64)
  2. 多模态注意力聚合: (20,64) → (64)
  3. 跨行人注意力（with mask）: (N_peds,64) → (64)
输出: (64)
```

### 4.3 通路几何编码器
```
输入: (10,64) + mask(10)
处理:
  1. 位置编码: (10,64) + pos_embed
  2. 自注意力（key_padding_mask=~mask）: (10,64) → (10,64)
  3. 仅聚合有效项: features[mask].mean() → (64)
  4. MLP: (64) → (128)
输出: (128)
```

### 4.4 机器狗状态编码器
```
输入: 轨迹(8,2), 速度(2), 位置(2), 目标(2)
处理:
  1. GRU编码轨迹: (8,2) → (64)
  2. MLP编码速度: (2) → (32)
  3. MLP编码相对目标: (goal-pos) → (32)
  4. 拼接+融合: (64+32+32) → (64)
输出: (64)
```

### 4.5 多模态融合
```
输入: 机器狗(64), 行人(64), 通路(128)
处理:
  1. 投影统一维度: → (64), (64), (64)
  2. Query=机器狗, KV=[行人,通路]
  3. 注意力: (1,1,64) attend (1,2,64) → (1,1,64)
  4. 拼接原始机器狗特征: cat([dog,attended]) → (128)
  5. MLP: (128) → (64)
输出: (64) + attn_weights(1,2)
```

### 4.6 混合SAC Actor
```
输入: 状态(64), 隐藏状态(h,c)
处理:
  1. FC: (64) → (256) → ReLU
  2. 展开时序: (1,256)
  3. LSTM: (1,1,256) + (h,c) → (1,1,256) + (h',c')
  4. Mean Head: (256) → (22)
  5. LogStd Head: (256) → (22), clamp(-20,2)
  6. 重参数化采样 + tanh
输出: 动作(22), log_prob, (h',c')
```

### 4.7 混合SAC Critic
```
输入: 状态(64), 动作(22), 隐藏状态(h,c)
处理:
  1. 拼接: cat([state,action]) → (86)
  2. FC: (86) → (256) → ReLU
  3. LSTM: (1,1,256) + (h,c) → (1,1,256) + (h',c')
  4. FC: (256) → (128) → ReLU → (1)
输出: Q值(1), (h',c')
```

### 4.8 几何微分评估器
```
输入: 路径(11,2), 参考线(2)
处理:
  1. 离散微分: path[i+1]-path[i] → (10,2)
  2. 归一化: d_norm = d / ||d||
  3. 夹角: θ = arccos(d_norm · L_norm)
  4. 指数权重: w_k = exp(-k/10)
  5. 加权平均: weighted_mean(θ)
  6. 归一化: score = 1 - (θ_mean / π)
输出: geo_score ∈ [0,1]
```

---

## 五、训练流程

### 5.1 经验回放缓冲区（单步存储）
```
存储格式（每个transition）:
{
  'state': fused_state (64),
  'action': path_normalized (22),
  'reward': r_final,
  'next_state': next_fused_state (64),
  'done': bool,
  'hidden_states': {
    'h_actor': (h,c),
    'h_critic1': (h,c),
    'h_critic2': (h,c)
  }
}

容量: 100000
采样: batch_size=256（独立样本）
```

### 5.2 训练循环
```
Episode开始:
  1. 重置环境
  2. 初始化隐藏状态 h_actor/critic = None

每个时间步:
  1. 输入预处理（padding+mask）
  2. 特征提取（三路并行）
  3. 融合
  4. Actor生成路径（更新h_actor）
  5. 路径转换+执行
  6. 环境反馈+GDE评分
  7. 存储经验（含隐藏状态）
  8. SAC更新（if buffer>256）

Episode结束:
  - 隐藏状态重置

SAC更新（单步batch）:
  - 从buffer采样256个独立transition
  - Critic更新（MSE loss）
  - Actor更新（最大化Q-α*entropy）
  - Alpha更新（自适应熵）
  - Target软更新（τ=0.005）
```

### 5.3 SAC更新详细流程（单步版本）

```python
def update(self, batch):
    """
    单步transition batch更新
    
    Args:
        batch: 256个独立的transition
    """
    states = batch['states']          # (256, 64)
    actions = batch['actions']        # (256, 22)
    rewards = batch['rewards']        # (256,)
    next_states = batch['next_states'] # (256, 64)
    dones = batch['dones']            # (256,)
    # hidden_states 每个样本独立，不利用时序
    
    # ============ Critic Update ============
    with torch.no_grad():
        # 计算target Q（每个样本独立）
        next_actions, next_log_probs = actor(next_states)
        target_q1 = critic1_target(next_states, next_actions)
        target_q2 = critic2_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
        target = rewards + gamma * (1 - dones) * target_q
    
    # Current Q
    current_q1 = critic1(states, actions)
    current_q2 = critic2(states, actions)
    
    # Critic loss
    critic_loss = F.mse_loss(current_q1, target)
    critic_loss += F.mse_loss(current_q2, target)
    
    # ============ Actor Update ============
    new_actions, log_probs = actor(states)
    q1 = critic1(states, new_actions)
    q2 = critic2(states, new_actions)
    q = torch.min(q1, q2)
    
    actor_loss = (alpha * log_probs - q).mean()
    
    # ============ Alpha Update ============
    alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
    
    # 优化器步骤
    # ...
```

**关键特点**:
- 每个样本独立处理
- 不考虑时序关系
- Hidden state存储但训练时不传递
- 适合MLP但不适合LSTM

### 5.4 奖励函数
```python
r_base = {
    到达: +300,
    碰撞: -200,
    警告: -200,
    距离进步: 500*(prev_dist - curr_dist)
}
r_final = r_base + 0.5 * geo_score
```

---

## 六、参数统计

```
预训练模块（冻结）:
  - SocialCircle: ~20K
  - E-V2-Net: ~300K

可训练模块:
  - PedestrianEncoder: ~120K
  - CorridorEncoder: ~70K
  - DogEncoder: ~75K
  - Fusion: ~50K
  - Actor: ~355K (含LSTM)
  - Critic×2: ~740K (含LSTM)
  
总计: ~1.73M（含冻结）
可训练: ~1.41M
✅ 符合<2M约束
```

---

## 七、超参数配置

```yaml
网络:
  hidden_dim: 256
  lstm_layers: 1
  num_heads: 4
  
环境:
  max_pedestrians: 10
  max_corridors: 10
  max_steps: 500
  
SAC:
  gamma: 0.99
  tau: 0.005
  lr_actor: 1e-4
  lr_critic: 1e-4
  lr_alpha: 3e-4
  batch_size: 256         # 单步采样
  buffer_size: 100000     # transition容量
  target_entropy: -22
  
GDE:
  eta: 0.5
  M: 10
```

---

## 八、实现检查清单

**数据处理**：
- [ ] 截断超限输入（选择最近N个）
- [ ] Padding行人轨迹到(10,8,2)
- [ ] PointNet编码通路到(10,64)
- [ ] 生成mask并传递

**特征提取**：
- [ ] SocialCircle预训练模型加载
- [ ] E-V2-Net预训练模型加载
- [ ] PedestrianEncoder处理mask
- [ ] CorridorEncoder注意力mask
- [ ] DogEncoder GRU处理轨迹

**SAC模块**：
- [ ] Actor LSTM前向传播
- [ ] Critic LSTM前向传播
- [ ] 隐藏状态初始化和传递
- [ ] 经验回放存储隐藏状态（单步）
- [ ] Target网络软更新

**训练逻辑**：
- [ ] Episode开始时重置隐藏状态
- [ ] 奖励塑造（含GDE）
- [ ] 梯度裁剪（max_norm=1.0）
- [ ] Alpha自适应更新

**测试验证**：
- [ ] 单步前向传播无错
- [ ] Batch处理正确
- [ ] Mask生效验证
- [ ] 隐藏状态维度正确
- [ ] 推理时间<50ms

---

## 九、V1 vs V2 主要差异

### V1.0（本方案）的特点
✅ **简单直接**: 单步采样，实现容易  
✅ **内存效率高**: 只存储transition  
❌ **不利于LSTM**: 丢失时序信息  
❌ **Hidden state浪费**: 存储但不利用  

### 后续改进方向（V2.0）
⭐ **序列段训练**: segment采样保留时序  
⭐ **Episode存储**: 完整轨迹存储  
⭐ **Burn-in机制**: 预热LSTM隐藏状态  
⭐ **更好的LSTM利用**: 充分利用记忆能力  

---

**本方案作为初始设计，为V2.0的改进提供了坚实基础。**

