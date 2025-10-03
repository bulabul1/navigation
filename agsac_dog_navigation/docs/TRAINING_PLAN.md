# AGSAC训练方案

**日期**: 2025-10-03  
**状态**: 🟢 系统就绪，可以开始训练

---

## 🎯 训练目标

训练一个能够在复杂环境中导航的机器狗控制策略，该策略能够：
1. ✅ 避开动态行人
2. ✅ 在走廊中安全导航
3. ✅ 到达目标位置
4. ✅ 保持舒适的运动轨迹

---

## 📋 训练分阶段方案

### 阶段0: 预训练准备（已完成✓）

**状态**: ✅ 完成

- [x] 集成SocialCircle + E-V2-Net预训练模型
- [x] 验证PretrainedTrajectoryPredictor (4/4测试通过)
- [x] 所有模块测试通过 (207/207)
- [x] 参数量验证 (960K < 2M)

**预训练模型**:
- 位置: `pretrained/social_circle/evsczara1/`
- 权重文件: `P36_evscsdd/torch_epoch150.pt`
- 配置文件: `evsc_config.json`
- 状态: ✅ 已加载并验证

---

### 阶段1: 模拟环境训练（推荐从此开始）

**目标**: 使用`DummyAGSACEnvironment`快速验证整个训练流程

#### 1.1 准备工作

```bash
# 激活环境
conda activate agsac_dog_nav

# 检查GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 1.2 创建简单训练配置

创建 `configs/training_dummy.yaml`:
```yaml
# 继承默认配置
defaults:
  - default_config.yaml

# 简化训练参数（快速验证）
training:
  mode: "full"
  batch_size: 32          # 小batch，快速迭代
  learning_rate: 0.0003
  
rl_training:
  replay_buffer:
    size: 10000           # 小buffer
    batch_size: 32
    
evaluation:
  eval_frequency: 10      # 频繁评估
  num_eval_episodes: 3

experiment:
  name: "dummy_env_test"
  description: "使用DummyEnvironment验证训练流程"
  output_dir: "outputs/dummy_test"
```

#### 1.3 创建启动脚本

创建 `scripts/train_dummy.py`:
```python
#!/usr/bin/env python3
"""使用Dummy环境快速验证训练流程"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from agsac.models import AGSACModel
from agsac.envs import DummyAGSACEnvironment
from agsac.training import AGSACTrainer, SequenceReplayBuffer

def main():
    print("=" * 60)
    print("AGSAC Dummy环境训练验证")
    print("=" * 60)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建环境
    print("\n1. 创建环境...")
    train_env = DummyAGSACEnvironment(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        pred_horizon=12,
        max_episode_steps=100,
        use_geometric_reward=True,
        device=device
    )
    eval_env = DummyAGSACEnvironment(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        pred_horizon=12,
        max_episode_steps=100,
        use_geometric_reward=True,
        device=device
    )
    print("[OK] 环境创建成功")
    
    # 创建模型（使用预训练轨迹预测器）
    print("\n2. 创建模型...")
    model = AGSACModel(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        pred_horizon=12,
        action_dim=2,           # (linear_vel, angular_vel)
        hidden_dim=256,
        use_pretrained_predictor=True,
        pretrained_weights_path=project_root / "pretrained/social_circle/evsczara1/torch_epoch150.pt",
        device=device
    ).to(device)
    
    print("[OK] 模型创建成功")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    print("\n3. 创建训练器...")
    trainer = AGSACTrainer(
        model=model,
        env=train_env,
        eval_env=eval_env,
        buffer_capacity=1000,      # 小buffer快速验证
        seq_len=8,
        batch_size=16,
        warmup_episodes=5,         # 少量warmup
        updates_per_episode=20,    # 少量更新
        eval_interval=5,           # 频繁评估
        eval_episodes=3,
        save_interval=10,
        max_episodes=50,           # 快速验证50 episodes
        device=device,
        save_dir='./outputs/dummy_test',
        experiment_name='dummy_quick_test'
    )
    print("[OK] 训练器创建成功")
    
    # 开始训练
    print("\n4. 开始训练...")
    print("=" * 60)
    trainer.train()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

#### 1.4 运行验证

```bash
# 运行快速验证（预计10-20分钟）
python scripts/train_dummy.py
```

**预期输出**:
- Episode数据收集
- 模型更新
- 评估指标
- Checkpoint保存

**成功标志**:
- ✅ 无报错
- ✅ 能看到reward变化
- ✅ 生成checkpoint文件
- ✅ 评估指标正常

---

### 阶段2: 真实环境训练（推荐）

**前提**: 阶段1验证通过

#### 2.1 选择环境实现方案

**方案A: 使用ROS + Gazebo仿真**
- 优点: 真实物理模拟，接近实际部署
- 缺点: 需要安装ROS环境
- 适用: 有机器人仿真基础

**方案B: 自定义Python仿真环境**
- 优点: 简单灵活，快速迭代
- 缺点: 物理不够真实
- 适用: 快速原型验证

**方案C: 使用真实机器狗（最终目标）**
- 优点: 直接在真实环境部署
- 缺点: 需要硬件，训练慢
- 适用: 模型验证后的最终测试

#### 2.2 创建真实环境接口

创建 `agsac/envs/simulation_environment.py`:

```python
"""基于物理引擎的仿真环境"""

import numpy as np
from .agsac_environment import AGSACEnvironment

class SimulationEnvironment(AGSACEnvironment):
    """
    使用PyBullet或类似物理引擎的仿真环境
    
    特性:
    1. 真实物理碰撞检测
    2. 动态行人模拟
    3. 复杂场景支持
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: 初始化物理引擎
        # import pybullet as p
        # self.physics_client = p.connect(p.GUI)
        pass
    
    def _reset_env(self):
        # TODO: 重置仿真环境
        pass
    
    def _step_env(self, action):
        # TODO: 执行动作并更新物理状态
        pass
    
    def _get_raw_observation(self):
        # TODO: 从仿真环境获取观测
        pass
    
    def _check_collision(self):
        # TODO: 使用物理引擎检测碰撞
        pass
    
    def _compute_base_reward(self, action, collision):
        # TODO: 计算奖励
        pass
```

#### 2.3 数据集准备

如果有真实轨迹数据：

```bash
# 数据格式
data/trajectories/
├── eth/
│   ├── train/
│   └── test/
├── ucy/
│   ├── train/
│   └── test/
└── custom/  # 自定义数据集
    ├── scene_1.csv
    ├── scene_2.csv
    └── ...
```

**数据格式要求**:
```
frame_id, agent_id, x, y, vx, vy
0, 0, 0.0, 0.0, 1.0, 0.5
0, 1, 5.0, 2.0, -0.5, 1.0
...
```

#### 2.4 完整训练配置

创建 `configs/training_full.yaml`:
```yaml
# 完整训练配置
defaults:
  - default_config.yaml

training:
  mode: "full"
  batch_size: 128
  learning_rate: 0.0001
  max_epochs: 1000

rl_training:
  replay_buffer:
    size: 100000
    batch_size: 128
  
  exploration:
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 0.995

evaluation:
  eval_frequency: 50
  num_eval_episodes: 10
  visualize: true
  save_videos: true

experiment:
  name: "agsac_full_training"
  description: "完整的AGSAC训练实验"
  output_dir: "outputs/experiments"
```

#### 2.5 启动完整训练

```bash
# 使用官方训练脚本
python scripts/train.py \
    --config configs/training_full.yaml \
    --pretrained pretrained/social_circle/evsczara1/torch_epoch150.pt \
    --device auto \
    --seed 42
```

---

## 📊 训练监控

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# 浏览器打开
http://localhost:6006
```

**监控指标**:
- `episode_reward`: Episode总奖励
- `success_rate`: 成功到达目标的比例
- `collision_rate`: 碰撞率
- `actor_loss`: Actor网络损失
- `critic_loss`: Critic网络损失
- `alpha`: 熵系数

### 日志文件

```bash
# 查看训练日志
tail -f logs/training.log

# 查看评估结果
cat outputs/experiments/agsac_full_training/eval_results.json
```

---

## 🔧 训练调优

### 超参数调整建议

**如果训练不稳定**:
```yaml
training:
  learning_rate: 0.00005  # 降低学习率
  grad_clip_norm: 0.5      # 更严格的梯度裁剪

rl_training:
  target_update:
    tau: 0.001              # 更慢的目标网络更新
```

**如果收敛太慢**:
```yaml
training:
  batch_size: 256          # 增大batch size
  
rl_training:
  replay_buffer:
    size: 500000           # 更大的buffer
  exploration:
    epsilon_decay: 0.99    # 更快的探索衰减
```

**如果过拟合**:
```yaml
model:
  dropout: 0.2             # 增加dropout
  
training:
  weight_decay: 1e-3       # 增加L2正则化
```

---

## 📈 预期训练效果

### 阶段1 (Dummy环境)
- **Episodes**: 50-100
- **时间**: 10-30分钟
- **预期**:
  - ✅ 训练流程正常运行
  - ✅ Reward有上升趋势
  - ✅ 无内存泄漏或崩溃

### 阶段2 (完整训练)
- **Episodes**: 1000-5000
- **时间**: 数小时到数天（取决于环境复杂度）
- **预期**:
  - ✅ Success rate > 80%
  - ✅ Collision rate < 5%
  - ✅ Average reward稳定增长
  - ✅ 行为符合预期（避障、导航）

---

## 🐛 常见问题

### Q1: CUDA out of memory

**解决**:
```yaml
training:
  batch_size: 64  # 减小batch size

rl_training:
  replay_buffer:
    batch_size: 64
```

### Q2: 训练过程中模型发散

**解决**:
- 检查reward设计是否合理
- 降低学习率
- 增加reward归一化
- 检查梯度裁剪是否生效

### Q3: 评估性能不佳

**可能原因**:
- 探索不足 → 增加warmup episodes
- 过拟合 → 增加dropout和正则化
- 环境随机性过大 → 增加训练episodes

---

## 🚀 快速开始（推荐流程）

```bash
# 1. 验证环境
conda activate agsac_dog_nav
python -c "import torch; print(torch.cuda.is_available())"

# 2. 快速测试（5分钟）
python scripts/train_dummy.py

# 3. 查看结果
ls outputs/dummy_test/

# 4. 查看TensorBoard
tensorboard --logdir logs/tensorboard

# 5. 如果一切正常，开始完整训练
python scripts/train.py --config configs/training_full.yaml
```

---

## 📝 下一步工作

### 短期（1-2周）
1. [ ] 运行Dummy环境验证
2. [ ] 实现真实仿真环境
3. [ ] 完成至少100 episodes训练
4. [ ] 分析训练曲线和行为

### 中期（1-2月）
1. [ ] 完整训练到收敛（1000+ episodes）
2. [ ] 超参数调优
3. [ ] 多场景泛化测试
4. [ ] 可视化和视频生成

### 长期（3-6月）
1. [ ] 在真实机器狗上部署
2. [ ] 真实环境测试
3. [ ] 持续改进和优化
4. [ ] 论文撰写

---

**状态**: 🟢 系统完全就绪  
**建议**: 立即开始阶段1的Dummy环境验证  
**预计时间**: 15-30分钟即可看到初步结果

**祝训练顺利！🎉**


