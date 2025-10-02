# 预训练模型整合指南

## SocialCircle + E-V2-Net

### 1. 开源资源

**论文**: SocialCircle: Learning the Angle-based Social Interaction Representation for Pedestrian Trajectory Prediction

**GitHub仓库**:
- 主仓库: https://github.com/[需要补充具体链接]
- 可能的仓库名: `social-circle`, `SocialCircle`, `E-V2-Net`

### 2. 整合策略

#### 方案A: 直接使用开源代码（推荐）

**优点**:
- 现成的实现，经过验证
- 可直接加载预训练权重
- 减少开发和调试时间

**步骤**:
1. 克隆SocialCircle仓库到 `pretrained/social_circle/`
2. 提取必要的模型代码
3. 创建包装类适配我们的接口
4. 下载预训练权重

**目录结构**:
```
pretrained/
├── social_circle/
│   ├── original_code/          # 原始代码
│   │   ├── models/
│   │   │   ├── social_circle.py
│   │   │   └── e_v2_net.py
│   │   └── utils/
│   ├── weights/                # 预训练权重
│   │   ├── social_circle.pth
│   │   └── e_v2_net.pth
│   └── README.md              # 使用说明
```

#### 方案B: 参考实现，自己编写

**优点**:
- 更好地理解模型
- 代码风格统一
- 易于定制

**缺点**:
- 开发时间长
- 需要自己训练或转换权重

### 3. 接口适配

我们需要创建统一的接口：

```python
# agsac/models/predictors/pretrained_predictor.py

class PretrainedTrajectoryPredictor(nn.Module):
    """
    预训练轨迹预测器包装类
    加载开源的SocialCircle+E-V2-Net模型
    """
    
    def __init__(self, weights_path: str, freeze: bool = True):
        super().__init__()
        
        # 加载原始模型
        self.social_circle = load_social_circle_model(weights_path)
        self.e_v2_net = load_e_v2_net_model(weights_path)
        
        # 冻结参数
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        target_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        neighbor_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        预测未来轨迹
        
        Returns:
            predicted_trajectories: (12, 2, 20)
                12个时间步，2维坐标，20个模态
        """
        # 1. SocialCircle编码
        social_features = self.social_circle(
            target_trajectory,
            neighbor_trajectories,
            neighbor_angles
        )
        
        # 2. E-V2-Net预测
        predictions = self.e_v2_net(social_features)
        
        return predictions
```

### 4. 实现清单

- [ ] **步骤1**: 找到并克隆SocialCircle GitHub仓库
- [ ] **步骤2**: 研究代码结构和模型定义
- [ ] **步骤3**: 下载预训练权重
- [ ] **步骤4**: 创建 `pretrained_predictor.py` 包装类
- [ ] **步骤5**: 实现权重加载函数
- [ ] **步骤6**: 测试前向传播
- [ ] **步骤7**: 验证输出维度和数值合理性

### 5. 许可证检查

⚠️ **重要**: 
- 检查SocialCircle的许可证类型（MIT, Apache, GPL等）
- 确保可以在项目中使用
- 适当添加引用和致谢

### 6. 备选方案

如果找不到开源代码或权重：

**选项1**: 使用现有的行人轨迹预测模型
- Social-LSTM
- Social-GAN
- Trajectron++

**选项2**: 简化实现
- 使用我们已经实现的 `SocialCircle`
- 实现简化版的E-V2-Net
- 在公开数据集上预训练（ETH/UCY）

### 7. 数据集

预训练通常使用：
- **ETH**: `data/trajectories/eth/`
- **UCY**: `data/trajectories/ucy/`

这些数据集可以从以下获取：
- https://github.com/StanfordASL/Trajectron-plus-plus
- https://github.com/agrimgupta92/sgan

### 8. 下一步行动

**立即行动**:
1. 搜索SocialCircle GitHub仓库
2. 查看是否有预训练权重
3. 评估代码复杂度和整合难度
4. 决定使用方案A还是方案B

**替代搜索关键词**:
- "SocialCircle pedestrian trajectory prediction"
- "E-V2-Net trajectory prediction github"
- "Angle-based social interaction representation"
- "SocialCircle pretrained weights"

---

## 参考资料

1. **论文**: SocialCircle: Learning the Angle-based Social Interaction Representation
2. **数据集**: ETH-UCY pedestrian trajectory dataset
3. **相关工作**: 
   - Social-LSTM
   - Social-GAN
   - Trajectron++

