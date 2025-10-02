# AGSAC开发任务清单 V1.0 (原始方案)

> **版本**: 1.0 (原始规划)  
> **训练方式**: 单步transition采样  
> **总任务数**: 26个任务，7个阶段  

---

## 开发流程概览

```
阶段1: 基础框架          (任务1-3)   ← 数据处理、工具
阶段2: 编码器            (任务4-7)   ← PointNet、DogEncoder、CorridorEncoder
阶段3: 预训练模块        (任务8-11)  ← SocialCircle、E-V2-Net、PedestrianEncoder
阶段4: 融合与SAC         (任务12-16) ← Fusion、Actor、Critic、Agent
阶段5: 评估与完整Agent   (任务17-19) ← GDE、完整Agent、集成测试
阶段6: 训练系统          (任务20-23) ← Buffer、Trainer、Environment
阶段7: 可视化与评估      (任务24-26) ← 可视化、评估、日志
```

---

## 第1阶段：基础框架

### 任务1: 数据处理模块
**文件**: `agsac/utils/data_processing.py`

**功能**:
- 输入预处理（可变→固定）
- Padding行人轨迹到(10, 8, 2)
- 生成mask
- 截断超限输入（选择最近N个）
- 坐标转换工具

**关键函数**:
```python
def preprocess_pedestrians(pedestrians, max_num=10) -> (Tensor, Tensor)
def preprocess_corridors(corridors, max_num=10) -> (List[Tensor], Tensor)
def select_closest_n(items, robot_pos, n) -> List[Tensor]
def convert_to_relative_coords(positions, robot_pos, robot_yaw) -> Tensor
```

**输入/输出**:
- 输入: 可变长度的list
- 输出: 固定shape的Tensor + mask

---

### 任务2: 几何工具
**文件**: `agsac/utils/geometry_utils.py`

**功能**:
- 计算相对角度
- 路径转换（归一化↔全局）
- 碰撞检测
- 距离计算

**关键函数**:
```python
def compute_relative_angles(target_pos, neighbor_pos) -> Tensor
def convert_incremental_path(path_normalized, robot_pos, robot_yaw) -> Tensor
def check_collision(path, obstacles) -> bool
def compute_distance_to_goal(pos, goal) -> float
```

---

### 任务3: 基础测试
**文件**: `tests/test_data_processing.py`, `tests/test_geometry_utils.py`

**测试内容**:
- Padding正确性
- Mask生成正确性
- 坐标转换可逆性
- 角度计算准确性

---

## 第2阶段：编码器

### 任务4: PointNet
**文件**: `agsac/models/encoders/pointnet.py`

**架构**:
```
输入: (N_vertices, 2) 可变顶点
↓
MLP: (N, 2) → (N, 64)
↓
对称聚合: max_pool + mean_pool → (128)
↓
MLP: (128) → (64)
↓
输出: (64) 固定维度
```

**参数量**: ~50K

**测试点**:
- 置换不变性
- 可变输入长度
- 输出维度正确

---

### 任务5: DogStateEncoder
**文件**: `agsac/models/encoders/dog_state_encoder.py`

**架构**:
```
输入: 轨迹(8,2), 速度(2), 位置(2), 目标(2)
↓
GRU编码轨迹: (8,2) → (64)
MLP编码速度: (2) → (32)
MLP编码相对目标: (2) → (32)
↓
拼接+融合: (128) → (64)
↓
输出: (64)
```

**参数量**: ~75K

**测试点**:
- Batch处理
- 梯度流
- 输出范围

---

### 任务6: CorridorEncoder
**文件**: `agsac/models/encoders/corridor_encoder.py`

**架构**:
```
输入: (10, 64) + mask(10)
↓
位置编码
↓
自注意力（with mask）
↓
聚合有效走廊
↓
MLP: (64) → (128)
↓
输出: (128)
```

**参数量**: ~70K

**测试点**:
- Mask有效性
- 注意力权重
- 边界情况（全mask/无mask）

---

### 任务7: 编码器测试
**文件**: `tests/test_encoders.py`

**测试内容**:
- PointNet编码一致性
- DogEncoder特征质量
- CorridorEncoder mask处理
- 参数量验证

---

## 第3阶段：预训练模块

### 任务8: SocialCircle
**文件**: `agsac/models/encoders/social_circle.py`

**架构**:
```
输入: 目标轨迹(8,2), 邻居轨迹(N,8,2), 角度(N)
↓
8扇区社交空间表示
↓
GRU编码目标: (8,2) → (64)
GRU编码邻居: (N,8,2) → (N,64)
↓
注意力聚合: (N,64) → (64)
↓
输出: (64) 社交特征
```

**参数量**: ~20K

**预训练**: 使用开源权重

---

### 任务9: E-V2-Net
**文件**: `agsac/models/predictors/e_v2_net.py`

**架构**:
```
输入: 社交特征(64)
↓
编码器: (64) → (256)
↓
多模态解码器: 20个GRU
↓
输出: (12, 2, 20) 多模态预测
```

**参数量**: ~300K

**预训练**: 使用开源权重

---

### 任务10: PedestrianEncoder
**文件**: `agsac/models/encoders/pedestrian_encoder.py`

**架构**:
```
输入: (N_peds, 12, 2, 20) + mask
↓
逐行人逐模态GRU编码
↓
多模态注意力聚合: 20 → 1
↓
跨行人注意力聚合: N → 1
↓
输出: (64)
```

**参数量**: ~120K

---

### 任务11: 预训练脚本
**文件**: `scripts/pretrain_trajectory.py`

**功能**:
- 加载预训练SocialCircle权重
- 加载预训练E-V2-Net权重
- 验证模型输出
- 冻结参数

---

## 第4阶段：融合与SAC

### 任务12: MultiModalFusion
**文件**: `agsac/models/fusion/multi_modal_fusion.py`

**架构**:
```
输入: dog(64), pedestrian(64), corridor(128)
↓
Query-based注意力
↓
MLP融合
↓
输出: (64)
```

**参数量**: ~50K

---

### 任务13: Hybrid Actor
**文件**: `agsac/models/sac/actor.py`

**架构**:
```
输入: 状态(64), hidden(h,c)
↓
FC: (64) → (256)
↓
LSTM: (256) → (256)
↓
Mean/LogStd: (256) → (22)×2
↓
输出: 动作(22), log_prob, new_hidden
```

**参数量**: ~355K

---

### 任务14: Hybrid Critic
**文件**: `agsac/models/sac/critic.py`

**架构**:
```
输入: 状态(64), 动作(22), hidden(h,c)
↓
拼接: (86)
↓
FC: (86) → (256)
↓
LSTM: (256) → (256)
↓
FC: (256) → (1)
↓
输出: Q值(1), new_hidden
```

**参数量**: ~370K × 2 = ~740K

---

### 任务15: SAC Agent
**文件**: `agsac/models/sac/sac_agent.py`

**功能**:
- 整合Actor和Critic
- SAC训练逻辑（单步版本）
- Target网络软更新
- Alpha自适应调节

**关键方法**:
```python
def update(self, batch) -> Dict[str, float]
def select_action(self, state, hidden, deterministic=False)
def soft_update_target(self)
```

---

### 任务16: SAC测试
**文件**: `tests/test_sac.py`

**测试内容**:
- Actor输出范围
- Critic Q值有限
- 隐藏状态传递
- Target更新正确

---

## 第5阶段：评估与完整Agent

### 任务17: GDE
**文件**: `agsac/models/evaluator/geometric_evaluator.py`

**功能**:
- 几何微分评估
- 路径-参考线对齐度
- 指数加权评分

**参数量**: 0（无可训练参数）

---

### 任务18: 完整Agent
**文件**: `agsac/models/agent.py`

**功能**:
- 整合所有子模块
- 端到端前向传播
- 推理接口
- 训练接口

**关键方法**:
```python
def forward(self, observation, hidden_states)
def select_action(self, observation, hidden_states)
def update(self, batch) -> Dict[str, float]
```

---

### 任务19: 集成测试
**文件**: `tests/test_integration.py`

**测试内容**:
- 完整前向传播
- 隐藏状态管理
- 参数量<2M
- 推理时间<50ms

---

## 第6阶段：训练系统

### 任务20: ReplayBuffer
**文件**: `agsac/training/replay_buffer.py`

**功能**:
- 存储transition
- 采样batch
- 隐藏状态管理

**容量**: 100K transitions

**采样**: batch_size=256

---

### 任务21: Trainer
**文件**: `agsac/training/trainer.py`

**功能**:
- 完整训练循环
- 数据收集
- SAC更新
- 定期评估
- 检查点保存

---

### 任务22: Environment
**文件**: `agsac/envs/gazebo_env.py`

**功能**:
- Gazebo仿真接口
- 观测预处理
- 动作执行
- 奖励计算（含GDE）

---

### 任务23: 训练脚本
**文件**: `scripts/train.py`

**功能**:
- 参数解析
- 模型初始化
- 训练启动
- 日志记录

---

## 第7阶段：可视化与评估

### 任务24: 可视化工具
**文件**: `agsac/utils/visualization.py`

**功能**:
- 轨迹可视化
- 注意力权重可视化
- 训练曲线绘制
- 实时监控

---

### 任务25: 评估脚本
**文件**: `scripts/evaluate.py`

**功能**:
- 加载训练好的模型
- 多场景评估
- 成功率统计
- 生成报告

---

### 任务26: Logger
**文件**: `agsac/utils/logger.py`

**功能**:
- TensorBoard集成
- 指标记录
- 性能监控
- 模型快照

---

## 依赖关系图

```
任务1,2 (基础工具)
    ↓
任务3 (基础测试)
    ↓
任务4,5,6 (编码器) → 任务7 (编码器测试)
    ↓
任务8,9 (预训练模型) → 任务11 (预训练脚本)
    ↓
任务10 (PedestrianEncoder)
    ↓
任务12 (Fusion)
    ↓
任务13,14,15 (SAC) → 任务16 (SAC测试)
    ↓
任务17 (GDE)
    ↓
任务18 (完整Agent) → 任务19 (集成测试)
    ↓
任务20,21,22,23 (训练系统)
    ↓
任务24,25,26 (可视化评估)
```

---

## 参数预算分配

| 模块 | 任务 | 预期参数 | 占比 |
|-----|------|---------|------|
| PointNet | 4 | 50K | 2.9% |
| DogEncoder | 5 | 75K | 4.3% |
| CorridorEncoder | 6 | 70K | 4.0% |
| SocialCircle | 8 | 20K | 1.2% (冻结) |
| E-V2-Net | 9 | 300K | 17.3% (冻结) |
| PedestrianEncoder | 10 | 120K | 6.9% |
| Fusion | 12 | 50K | 2.9% |
| Actor | 13 | 355K | 20.5% |
| Critic×2 | 14 | 740K | 42.8% |
| GDE | 17 | 0 | 0% |
| **总计** | - | **1.78M** | **100%** |
| **可训练** | - | **1.41M** | **79%** |

✅ 符合 <2M 预算

---

## 预计时间

- 第1阶段: 1-2天
- 第2阶段: 2-3天
- 第3阶段: 2-3天
- 第4阶段: 3-4天
- 第5阶段: 2-3天
- 第6阶段: 3-4天
- 第7阶段: 2-3天

**总计**: 15-22天

---

## 检查清单

### 阶段1
- [ ] 任务1: 数据处理模块
- [ ] 任务2: 几何工具
- [ ] 任务3: 基础测试

### 阶段2
- [ ] 任务4: PointNet
- [ ] 任务5: DogStateEncoder
- [ ] 任务6: CorridorEncoder
- [ ] 任务7: 编码器测试

### 阶段3
- [ ] 任务8: SocialCircle
- [ ] 任务9: E-V2-Net
- [ ] 任务10: PedestrianEncoder
- [ ] 任务11: 预训练脚本

### 阶段4
- [ ] 任务12: MultiModalFusion
- [ ] 任务13: Hybrid Actor
- [ ] 任务14: Hybrid Critic
- [ ] 任务15: SAC Agent
- [ ] 任务16: SAC测试

### 阶段5
- [ ] 任务17: GDE
- [ ] 任务18: 完整Agent
- [ ] 任务19: 集成测试

### 阶段6
- [ ] 任务20: ReplayBuffer
- [ ] 任务21: Trainer
- [ ] 任务22: Environment
- [ ] 任务23: 训练脚本

### 阶段7
- [ ] 任务24: 可视化工具
- [ ] 任务25: 评估脚本
- [ ] 任务26: Logger

---

**注**: 本清单基于V1.0原始设计（单步训练），实际实现中已调整为V2.0方案（序列段训练）。

