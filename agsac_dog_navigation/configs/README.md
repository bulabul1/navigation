# AGSAC训练配置文件

## 📁 配置文件说明

### `train_config.py`
Python配置类定义，包含：
- `EnvironmentConfig` - 环境参数
- `ModelConfig` - 模型参数
- `TrainingConfig` - 训练参数
- `AGSACConfig` - 完整配置（自动同步各部分）

### 预定义配置

#### `debug.yaml` - 调试配置 🐛
- **用途**：快速验证代码
- **Episodes**: 5
- **模式**: 固定场景
- **设备**: CPU
- **特点**: 小模型(hidden_dim=64)，短episode(50步)

#### `default.yaml` - 默认配置 ⭐
- **用途**：标准训练
- **Episodes**: 300
- **模式**: 课程学习
- **设备**: CUDA（自动检测）
- **特点**: 完整配置，从easy到hard渐进训练

---

## 🚀 使用方法

### 1. 基础训练
```bash
# 使用默认配置
python scripts/train.py --config configs/default.yaml

# 使用调试配置
python scripts/train.py --config configs/debug.yaml

# 强制使用CPU
python scripts/train.py --config configs/default.yaml --cpu
```

### 2. 自定义配置

复制一个现有配置并修改：
```bash
cp configs/default.yaml configs/my_config.yaml
# 编辑 my_config.yaml
python scripts/train.py --config configs/my_config.yaml
```

### 3. 配置参数说明

#### **训练模式** (`mode`)
- `fixed`: 固定场景（快速测试）
- `dynamic`: 随机场景（泛化训练）
- `curriculum`: 课程学习（完整训练）

#### **关键参数**
```yaml
env:
  max_episode_steps: 200        # Episode长度
  use_corridor_generator: true  # 是否使用随机场景
  curriculum_learning: true     # 是否启用课程学习

model:
  hidden_dim: 128               # 模型隐藏层维度
  action_dim: 22                # 动作维度（11个路径点）

training:
  episodes: 300                 # 训练episode数
  buffer_capacity: 10000        # 经验回放容量
  batch_size: 16                # 训练批量大小
```

---

## ⚙️ 配置一致性保证

配置系统**自动同步**以下参数，无需手动设置：
- ✅ `env.max_corridors` → `model.max_corridors`
- ✅ `env.max_pedestrians` → `model.max_pedestrians`  
- ✅ `device` → `env.device` & `model.device`
- ✅ `mode` → 自动配置生成器和课程学习

**示例**：修改`env.max_corridors`后，`model.max_corridors`会自动同步。

---

## 📊 训练输出

训练日志保存在：
```
logs/{experiment_name}_{timestamp}/
├── config.yaml          # 训练使用的完整配置
├── config.json          # JSON格式配置
├── training.log         # 文本日志
├── tensorboard/         # TensorBoard日志
└── checkpoints/         # 模型检查点
    ├── best_model.pth
    └── checkpoint_ep*.pth
```

---

## 💡 最佳实践

1. **快速验证**: 先用`debug.yaml`测试代码
2. **正式训练**: 使用`default.yaml`或自定义配置
3. **保存配置**: 每次训练会自动保存配置到日志目录
4. **复现实验**: 使用保存的`config.yaml`可完全复现

---

## 🔍 配置检查

```python
# 加载并检查配置
from configs.train_config import AGSACConfig

config = AGSACConfig.from_yaml('configs/default.yaml')
print(config.to_dict())
```

