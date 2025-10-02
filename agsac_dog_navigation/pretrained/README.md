# 预训练模型目录

本目录用于存放预训练的模型权重和相关代码。

## 目录结构

```
pretrained/
├── social_circle/              # SocialCircle相关
│   ├── original_code/          # 从GitHub克隆的原始代码（可选）
│   ├── weights/                # 预训练权重
│   │   ├── social_circle.pth
│   │   └── e_v2_net.pth
│   └── README.md              # 使用说明
│
└── e_v2_net/                   # E-V2-Net相关
    ├── weights/
    │   └── e_v2_net_weights.pth
    └── README.md
```

## 如何获取预训练权重

### 1. SocialCircle + E-V2-Net

**查找开源仓库**:
1. 在GitHub搜索: "SocialCircle pedestrian trajectory"
2. 或搜索论文标题找到代码链接
3. 常见仓库名模式:
   - `social-circle`
   - `SocialCircle`
   - `angle-based-social-interaction`

**下载步骤**:
```bash
# 方法1: 如果找到GitHub仓库
cd pretrained/social_circle/
git clone [仓库链接] original_code

# 方法2: 如果只有权重文件
cd pretrained/social_circle/weights/
wget [权重下载链接]
```

### 2. 备选方案

如果找不到SocialCircle的开源代码，可以使用：

**其他行人轨迹预测模型**:
- **Social-LSTM**: https://github.com/quancore/social-lstm
- **Social-GAN**: https://github.com/agrimgupta92/sgan
- **Trajectron++**: https://github.com/StanfordASL/Trajectron-plus-plus

这些模型也有预训练权重，可以适配到我们的框架中。

## 数据集

预训练需要的数据集：

### ETH/UCY数据集
```bash
cd data/trajectories/

# 下载ETH
wget [ETH数据集链接]

# 下载UCY
wget [UCY数据集链接]
```

**数据集来源**:
- https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw
- https://github.com/agrimgupta92/sgan

## 使用说明

### 方式1: 使用开源预训练模型（推荐）

```python
from agsac.models.predictors import PretrainedTrajectoryPredictor

# 加载预训练模型
predictor = PretrainedTrajectoryPredictor(
    weights_path='pretrained/social_circle/weights/',
    freeze=True  # 冻结参数
)

# 预测
predictions = predictor(
    target_trajectory,
    neighbor_trajectories,
    neighbor_angles
)
```

### 方式2: 使用简化实现

如果暂时找不到开源代码，可以使用我们的简化实现：

```python
from agsac.models.predictors import SimpleTrajectoryPredictor

predictor = SimpleTrajectoryPredictor(
    feature_dim=64,
    prediction_horizon=12,
    num_modes=20
)

# 在ETH/UCY数据集上预训练
# python scripts/pretrain_trajectory.py
```

## 许可证

使用开源模型时请注意：
1. 检查原仓库的许可证
2. 在论文/项目中适当引用
3. 遵守许可证要求

## 引用

如果使用SocialCircle，请引用原论文：

```bibtex
@inproceedings{socialcircle,
  title={SocialCircle: Learning the Angle-based Social Interaction Representation for Pedestrian Trajectory Prediction},
  author={[作者名]},
  booktitle={[会议名]},
  year={[年份]}
}
```

## 当前状态

- [ ] 找到SocialCircle开源仓库
- [ ] 下载预训练权重
- [ ] 整合到项目中
- [ ] 测试前向传播
- [ ] 验证输出正确性

## 问题排查

### 找不到开源代码？
- 尝试联系论文作者
- 使用备选的开源模型
- 使用我们的简化实现并自己预训练

### 权重格式不兼容？
- 查看权重文件的keys: `torch.load(path).keys()`
- 创建权重转换脚本
- 参考原仓库的加载代码

### 版本不兼容？
- 检查PyTorch版本要求
- 使用虚拟环境隔离依赖

## 联系方式

如果您找到了SocialCircle的开源资源，请更新本文档！
