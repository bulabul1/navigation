# SocialCircle开源代码整合指南

## 📋 目标

将开源的SocialCircle+E-V2-Net代码和预训练权重整合到我们的项目中。

## 🔍 第一步：查找开源资源

### 搜索关键词

在GitHub或Google Scholar搜索：
- `"SocialCircle" pedestrian trajectory prediction github`
- `"angle-based social interaction" trajectory`
- `"E-V2-Net" pedestrian prediction`
- 论文作者名 + `github`

### 可能的仓库位置

通常开源代码会在：
1. 论文页面提供的链接
2. 作者的GitHub主页
3. 会议/期刊的补充材料

### 如果找不到？

**备选方案**：
1. **联系作者**：发邮件请求代码和权重
2. **使用类似模型**：
   - Social-LSTM: https://github.com/quancore/social-lstm
   - Social-GAN: https://github.com/agrimgupta92/sgan
   - Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus
3. **使用我们的简化实现**：已经包含在项目中，可直接使用

## 📥 第二步：下载和放置

### 如果找到GitHub仓库

```bash
cd pretrained/social_circle/

# 克隆原始代码
git clone [仓库URL] original_code

# 查看结构
cd original_code
ls
```

### 如果只有权重文件

```bash
cd pretrained/social_circle/weights/

# 下载权重（根据实际链接）
wget [权重下载链接]
# 或
curl -O [权重下载链接]
```

### 期望的目录结构

```
pretrained/
└── social_circle/
    ├── original_code/          # 原始GitHub代码（如果有）
    │   ├── models/
    │   ├── utils/
    │   └── README.md
    └── weights/                # 预训练权重
        ├── social_circle.pth
        └── e_v2_net.pth
```

## 🔧 第三步：整合代码

### 方案A：权重文件可直接加载

如果权重文件格式兼容PyTorch的`torch.load()`：

1. **检查权重结构**：
```python
import torch

weights = torch.load('pretrained/social_circle/weights/model.pth', map_location='cpu')
print(weights.keys())
```

2. **修改加载函数**：
编辑 `agsac/models/predictors/trajectory_predictor.py` 中的 `_load_pretrained_model()` 方法：

```python
def _load_pretrained_model(self):
    checkpoint = torch.load(self.weights_path, map_location='cpu')
    
    # 根据实际的权重结构调整
    from agsac.models.encoders.social_circle import SocialCircle
    
    self.social_circle = SocialCircle()
    self.social_circle.load_state_dict(checkpoint['social_circle'])
    
    self.e_v2_net = SimpleE_V2_Net()
    self.e_v2_net.load_state_dict(checkpoint['e_v2_net'])
```

### 方案B：需要使用原始代码

如果需要使用原始仓库的模型定义：

1. **检查原始代码结构**：
```bash
cd pretrained/social_circle/original_code
ls models/
```

2. **创建适配器**：
创建 `agsac/models/predictors/pretrained_adapter.py`：

```python
"""适配原始SocialCircle代码"""
import sys
from pathlib import Path

# 添加原始代码路径
original_code_path = Path(__file__).parent.parent.parent.parent / 'pretrained/social_circle/original_code'
sys.path.insert(0, str(original_code_path))

# 导入原始模型
from models.social_circle import SocialCircleModel  # 根据实际调整
from models.e_v2_net import EV2NetModel  # 根据实际调整

# 创建包装器
class OriginalModelWrapper(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        self.social_circle = SocialCircleModel()
        self.e_v2_net = EV2NetModel()
        
        # 加载权重
        checkpoint = torch.load(weights_path)
        self.social_circle.load_state_dict(checkpoint['social_circle'])
        self.e_v2_net.load_state_dict(checkpoint['e_v2_net'])
    
    def forward(self, *args, **kwargs):
        # 适配接口
        ...
```

3. **在`PretrainedTrajectoryPredictor`中使用**：
```python
from agsac.models.predictors.pretrained_adapter import OriginalModelWrapper

def _load_pretrained_model(self):
    self.model = OriginalModelWrapper(self.weights_path)
```

## ✅ 第四步：测试

### 测试脚本

创建 `tests/test_pretrained_integration.py`：

```python
import torch
from agsac.models.predictors import PretrainedTrajectoryPredictor

def test_pretrained_predictor():
    # 加载预训练模型
    predictor = PretrainedTrajectoryPredictor(
        weights_path='pretrained/social_circle/weights/model.pth',
        freeze=True
    )
    
    # 测试数据
    target_traj = torch.randn(8, 2)
    neighbor_trajs = torch.randn(5, 8, 2)
    angles = torch.rand(5) * 2 * 3.14159
    
    # 预测
    predictions = predictor(target_traj, neighbor_trajs, angles)
    
    # 验证输出
    assert predictions.shape == (12, 2, 20), f"输出维度错误: {predictions.shape}"
    assert torch.isfinite(predictions).all(), "输出包含非有限值"
    
    print("✓ 预训练模型测试通过！")

if __name__ == '__main__':
    test_pretrained_predictor()
```

### 运行测试

```bash
python tests/test_pretrained_integration.py
```

## 🐛 常见问题

### 问题1：模块导入错误

**错误**：`ModuleNotFoundError: No module named 'xxx'`

**解决**：
1. 检查原始代码的依赖：`cat requirements.txt`
2. 安装缺失的包：`pip install xxx`
3. 或使用虚拟环境隔离

### 问题2：权重格式不兼容

**错误**：`KeyError` 或 `RuntimeError: Error(s) in loading state_dict`

**解决**：
1. 打印权重的keys：
```python
checkpoint = torch.load(path)
print("Keys:", checkpoint.keys())
if 'model' in checkpoint:
    print("Model keys:", checkpoint['model'].keys())
```

2. 手动映射keys：
```python
state_dict = checkpoint['model']
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('old_prefix', 'new_prefix')
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
```

### 问题3：输出维度不匹配

**解决**：
1. 检查原始模型的超参数
2. 调整我们的接口参数：
```python
predictor = PretrainedTrajectoryPredictor(
    prediction_horizon=12,  # 根据实际调整
    num_modes=20
)
```

### 问题4：找不到开源代码

**解决**：直接使用我们的简化实现：

```python
from agsac.models.predictors import SimpleTrajectoryPredictor

# 使用简化实现
predictor = SimpleTrajectoryPredictor(
    social_circle_dim=64,
    prediction_horizon=12,
    num_modes=20
)

# 可以在ETH/UCY数据集上预训练
# python scripts/pretrain_trajectory.py
```

## 📊 第五步：验证效果

### 定性检查

可视化预测结果：

```python
import matplotlib.pyplot as plt

# 预测
predictions = predictor(target_traj, neighbor_trajs, angles)  # (12, 2, 20)

# 可视化
plt.figure(figsize=(10, 10))
plt.plot(target_traj[:, 0], target_traj[:, 1], 'b-o', label='History')

# 绘制多个模态
for mode in range(20):
    pred = predictions[:, :, mode].detach().numpy()
    plt.plot(pred[:, 0], pred[:, 1], 'r-', alpha=0.3)

plt.legend()
plt.grid(True)
plt.axis('equal')
plt.title('Trajectory Prediction')
plt.show()
```

### 定量评估

如果有测试数据集：

```python
# 计算ADE (Average Displacement Error)
# 计算FDE (Final Displacement Error)
# 参考：scripts/evaluate.py
```

## 📝 更新清单

完成整合后，请更新以下文件：

- [ ] `pretrained/README.md` - 添加下载链接和说明
- [ ] `agsac/models/predictors/trajectory_predictor.py` - 完善加载逻辑
- [ ] `INTEGRATION_GUIDE.md` (本文件) - 记录实际步骤
- [ ] `README.md` - 更新安装和使用说明

## 🎯 当前状态

- [x] 创建预测器框架
- [x] 实现简化版本（作为后备）
- [x] 创建整合指南
- [ ] 找到开源仓库
- [ ] 下载预训练权重
- [ ] 整合并测试
- [ ] 验证效果

## 📞 需要帮助？

如果您找到了SocialCircle的开源代码或权重，请：
1. 更新本文档
2. 提交issue或PR
3. 分享给团队

---

**祝整合顺利！** 🚀

