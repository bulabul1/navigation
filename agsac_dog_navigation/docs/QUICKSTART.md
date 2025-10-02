# AGSAC 快速开始指南

## 🚀 快速使用

### 1. 创建模型

```python
from agsac.models import AGSACModel
import torch

# 初始化模型
model = AGSACModel(
    # 特征维度
    dog_feature_dim=64,
    corridor_feature_dim=128,
    pedestrian_feature_dim=128,
    fusion_dim=64,
    action_dim=22,
    
    # 场景配置
    max_pedestrians=10,
    max_corridors=5,
    max_vertices=20,
    obs_horizon=8,
    pred_horizon=12,
    num_modes=20,
    
    # 网络配置
    hidden_dim=128,
    num_heads=4,
    dropout=0.1,
    
    # SAC配置
    actor_lr=1e-4,
    critic_lr=1e-4,
    alpha_lr=3e-4,
    gamma=0.99,
    tau=0.005,
    auto_entropy=True,
    max_grad_norm=1.0,
    
    # 设备
    device='cuda'  # 或 'cpu'
)

print(f"模型总参数: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. 准备观测数据

```python
batch_size = 4

observation = {
    # 机器狗状态
    'dog': {
        'trajectory': torch.randn(batch_size, 8, 2),  # 历史轨迹
        'velocity': torch.randn(batch_size, 2),        # 当前速度
        'position': torch.randn(batch_size, 2),        # 当前位置
        'goal': torch.randn(batch_size, 2)             # 目标位置
    },
    
    # 行人轨迹
    'pedestrians': {
        'trajectories': torch.randn(batch_size, 10, 8, 2),  # (batch, max_peds, obs_horizon, 2)
        'mask': torch.ones(batch_size, 10)                   # 1=有效, 0=padding
    },
    
    # 走廊几何
    'corridors': {
        'polygons': torch.randn(batch_size, 5, 20, 2),  # (batch, max_corridors, max_vertices, 2)
        'vertex_counts': torch.tensor([[10, 8, 6, 4, 3]] * batch_size),  # 每个走廊的实际顶点数
        'mask': torch.ones(batch_size, 5)               # 1=有效, 0=padding
    },
    
    # 参考线（用于GDE评估）
    'reference_line': torch.randn(batch_size, 2, 2)  # 起点和终点
}
```

### 3. 推理（数据收集）

```python
# 初始化隐藏状态
hidden_states = model.init_hidden_states(batch_size=1)

# 单步推理
action, log_prob, hidden_states = model.select_action(
    observation=observation,
    hidden_states=hidden_states,
    deterministic=False  # False=随机采样, True=确定性
)

print(f"动作: {action.shape}")        # (batch, 22)
print(f"对数概率: {log_prob.shape}")  # (batch,)
```

### 4. 完整前向传播（带调试信息）

```python
result = model.forward(
    observation=observation,
    hidden_states=None,  # None会自动初始化
    deterministic=False,
    return_attention=True  # 返回注意力权重
)

# 输出包含
print(f"动作: {result['action'].shape}")              # (batch, 22)
print(f"Q1值: {result['q1'].shape}")                  # (batch,)
print(f"Q2值: {result['q2'].shape}")                  # (batch,)
print(f"融合特征: {result['fused_state'].shape}")     # (batch, 64)
print(f"隐藏状态: {list(result['hidden_states'].keys())}")  # ['actor', 'critic1', 'critic2']

# 调试信息
debug = result['debug_info']
print(f"机器狗特征: {debug['dog_features'].shape}")           # (batch, 64)
print(f"走廊特征: {debug['corridor_features'].shape}")         # (batch, 128)
print(f"行人预测: {debug['pedestrian_predictions'].shape}")    # (batch, max_peds, 12, 2, 20)
print(f"行人特征: {debug['pedestrian_features'].shape}")       # (batch, 64)
```

### 5. 训练（使用序列段）

```python
# 准备序列段batch
segment_batch = [
    {
        'states': torch.randn(16, 64),     # 序列长度16
        'actions': torch.randn(16, 22),
        'rewards': torch.randn(16),
        'next_states': torch.randn(16, 64),
        'dones': torch.zeros(16),
        'init_hidden': {
            'actor': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128)),
            'critic1': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128)),
            'critic2': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
        }
    }
    # ... 更多segment
]

# 训练更新
model.train()
losses = model.update(segment_batch)

print(f"Critic损失: {losses['critic_loss']:.4f}")
print(f"Actor损失: {losses['actor_loss']:.4f}")
print(f"Alpha: {losses['alpha']:.4f}")

# 软更新目标网络
model.soft_update_target()
```

### 6. 保存和加载模型

```python
# 保存检查点
model.save_checkpoint('checkpoints/model_episode_1000.pth')

# 加载检查点
model.load_checkpoint(
    'checkpoints/model_episode_1000.pth',
    load_optimizers=True  # 是否加载优化器状态
)
```

---

## 📊 观测数据详细说明

### 机器狗状态 (`observation['dog']`)
- **trajectory**: (batch, 8, 2) - 过去8个时间步的位置
- **velocity**: (batch, 2) - 当前速度 [vx, vy]
- **position**: (batch, 2) - 当前位置 [x, y]
- **goal**: (batch, 2) - 目标位置 [x, y]

### 行人轨迹 (`observation['pedestrians']`)
- **trajectories**: (batch, max_peds, 8, 2) - 每个行人过去8步的位置
- **mask**: (batch, max_peds) - 有效性掩码
  - 1 = 该行人存在
  - 0 = padding（不存在的行人）

### 走廊几何 (`observation['corridors']`)
- **polygons**: (batch, max_corridors, max_vertices, 2)
  - 每个走廊是一个多边形
  - padding的顶点设为0
- **vertex_counts**: (batch, max_corridors) - 每个走廊的实际顶点数
- **mask**: (batch, max_corridors) - 走廊有效性掩码

### 参考线 (`observation['reference_line']`)
- (batch, 2, 2) - 起点 [x, y] 和 终点 [x, y]
- 用于几何微分评估器（GDE）计算路径质量

---

## 🎯 典型训练流程

```python
from agsac.models import AGSACModel

# 1. 创建模型
model = AGSACModel(...).to('cuda')

# 2. 初始化Replay Buffer
buffer = SequenceReplayBuffer(capacity=100000, seq_len=16)

# 3. 训练循环
for episode in range(num_episodes):
    # 收集数据
    observation = env.reset()
    hidden_states = model.init_hidden_states(batch_size=1)
    episode_data = []
    
    while not done:
        # 选择动作
        action, log_prob, hidden_states = model.select_action(
            observation, hidden_states, deterministic=False
        )
        
        # 执行动作
        next_obs, reward, done, info = env.step(action)
        
        # 存储transition
        episode_data.append({
            'observation': observation,
            'action': action,
            'reward': reward,
            'done': done,
            'hidden_states': hidden_states
        })
        
        observation = next_obs
    
    # 存储episode
    buffer.add_episode(episode_data)
    
    # 训练更新
    if buffer.size() >= warmup_steps:
        for _ in range(updates_per_episode):
            segment_batch = buffer.sample(batch_size)
            losses = model.update(segment_batch)
            model.soft_update_target()
    
    # 定期评估和保存
    if episode % eval_interval == 0:
        evaluate(model, eval_env)
    
    if episode % save_interval == 0:
        model.save_checkpoint(f'checkpoint_{episode}.pth')
```

---

## ⚠️ 注意事项

### 1. 参数量问题
当前模型约3M参数，超出2M预算。主要原因：
- TrajectoryPredictor占67.6% (2.05M参数)
- 建议使用预训练E-V2-Net或减少预测模态数

### 2. 坐标系统
- 所有坐标应在同一参考系下
- 建议使用机器狗当前位置为原点的局部坐标系

### 3. 数据归一化
- 速度、位置等建议归一化到合理范围
- 有助于训练稳定性

### 4. Hidden State管理
- 每个episode开始时需要重置hidden states
- 序列段采样时需要保存segment起始的hidden state

### 5. 设备一致性
- 确保observation和model在同一设备上（CPU或CUDA）
- 可以用`.to(device)`移动数据

---

## 🔧 下一步

1. **实现ReplayBuffer** - 序列段存储和采样
2. **实现Environment** - 环境接口封装
3. **实现Trainer** - 完整训练流程
4. **优化参数量** - 减少TrajectoryPredictor参数

详见 [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md)

