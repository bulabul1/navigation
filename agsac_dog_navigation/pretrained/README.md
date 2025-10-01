# 预训练模型

本目录包含预训练的模型权重文件。

## 模型文件

### SocialCircle
- **文件**: `social_circle/social_circle_weights.pth`
- **用途**: 社会交互特征提取
- **预训练数据**: ETH/UCY行人轨迹数据集

### E-V2-Net
- **文件**: `e_v2_net/e_v2_net_weights.pth`
- **用途**: 轨迹预测骨干网络
- **预训练数据**: ETH/UCY行人轨迹数据集

## 使用方法

### 加载预训练模型

```python
import torch
from agsac.models.predictors.e_v2_net import EV2Net

# 加载E-V2-Net预训练权重
model = EV2Net(...)
checkpoint = torch.load('pretrained/e_v2_net/e_v2_net_weights.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 在训练中使用

```bash
# 使用预训练模型开始训练
python scripts/train.py \
    --config configs/training_config.yaml \
    --pretrained pretrained/e_v2_net/e_v2_net_weights.pth
```

## 模型信息

### SocialCircle
- **输入**: 行人轨迹 `(batch_size, max_pedestrians, trajectory_length, 2)`
- **输出**: 社会交互特征 `(batch_size, max_pedestrians, feature_dim)`
- **参数量**: ~50K

### E-V2-Net
- **输入**: 编码特征 `(batch_size, max_pedestrians, hidden_dim)`
- **输出**: 预测轨迹 `(batch_size, max_pedestrians, prediction_horizon, 2)`
- **参数量**: ~200K

## 预训练过程

1. **数据准备**: 使用ETH/UCY数据集
2. **模型训练**: 600个epoch，学习率0.001
3. **验证**: 使用20%数据作为验证集
4. **保存**: 保存最佳验证性能的模型

## 注意事项

1. 预训练模型需要与目标任务的输入格式匹配
2. 建议在目标任务上进行微调
3. 模型权重文件较大，请确保有足够的存储空间
4. 使用前请检查模型版本兼容性
