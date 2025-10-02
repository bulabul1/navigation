# SocialCircle集成指南

> 将开源的SocialCircle PyTorch实现集成到AGSAC系统

**开源仓库**: [https://github.com/cocoon2wong/SocialCircle](https://github.com/cocoon2wong/SocialCircle)  
**论文**: SocialCircle: Learning the Angle-based Social Interaction Representation for Pedestrian Trajectory Prediction (CVPR2024)

---

## 📋 集成步骤

### **步骤1: 获取SocialCircle代码**

**方法A: 使用自动化脚本（推荐）**

```bash
# Windows
cd agsac_dog_navigation
scripts\setup_socialcircle.bat

# Linux/Mac
chmod +x scripts/setup_socialcircle.sh
./scripts/setup_socialcircle.sh
```

**方法B: 手动操作**

```bash
# 1. 在项目外clone PyTorch分支
cd C:\Users\13772\Desktop\myProjects\WWW_navigetion
git clone -b TorchVersion https://github.com/cocoon2wong/SocialCircle.git SocialCircle_temp

# 2. 创建external目录
cd agsac_dog_navigation
mkdir external\SocialCircle_original

# 3. 复制核心代码
xcopy /E /I SocialCircle_temp\socialCircle external\SocialCircle_original\socialCircle
xcopy /E /I SocialCircle_temp\qpid external\SocialCircle_original\qpid

# 4. 复制依赖文件
copy SocialCircle_temp\requirements.txt external\SocialCircle_original\
copy SocialCircle_temp\README.md external\SocialCircle_original\

# 5. 清理临时目录
cd ..
rmdir /s /q SocialCircle_temp
```

---

### **步骤2: 下载预训练权重**

```bash
# 创建权重目录
cd agsac_dog_navigation
mkdir pretrained\social_circle

# 下载权重（从GitHub Releases）
# https://github.com/cocoon2wong/SocialCircle/releases
# 下载后放到 pretrained/social_circle/weights.pth
```

**下载链接**: [SocialCircle Model Weights (Sep 25, 2023)](https://github.com/cocoon2wong/SocialCircle/releases)

---

### **步骤3: 安装依赖**

查看`external/SocialCircle_original/requirements.txt`并安装：

```bash
# 安装SocialCircle的依赖（如果有冲突，手动调整）
pip install -r external/SocialCircle_original/requirements.txt

# 可能需要的额外依赖
pip install opencv-python
pip install scipy
```

---

### **步骤4: 适配代码**

#### 4.1 检查目录结构

```
agsac_dog_navigation/
├── external/
│   └── SocialCircle_original/
│       ├── socialCircle/       # SocialCircle核心代码
│       ├── qpid/               # 依赖的qpid框架
│       ├── requirements.txt
│       └── README.md
├── pretrained/
│   └── social_circle/
│       └── weights.pth         # 预训练权重
└── agsac/
    └── models/
        └── encoders/
            └── social_circle_pretrained.py  # 我们的适配器
```

#### 4.2 完善适配器

打开`agsac/models/encoders/social_circle_pretrained.py`，根据实际的SocialCircle代码结构调整：

**需要修改的部分**:

1. **导入路径** (`_import_socialcircle`方法):
```python
# 根据实际代码结构调整import语句
from socialCircle.layers import SocialCircleLayer
from socialCircle.model import SocialCircleModel
# ...
```

2. **模型构建** (`_build_pretrained_model`方法):
```python
# 根据SocialCircle的实际API构建模型
self.model = SocialCircleModel(
    obs_len=self.obs_horizon,
    pred_len=12,  # 根据需要调整
    feature_dim=self.social_feature_dim,
    # 其他参数...
)
```

3. **输入格式适配** (`forward`方法):
```python
# 将我们的格式转换为SocialCircle期望的格式
# 我们的: (batch, obs_horizon, 2)
# SocialCircle可能需要: (batch, agents, obs_horizon, 2)
```

---

### **步骤5: 测试适配器**

```bash
# 测试适配器
python -m agsac.models.encoders.social_circle_pretrained
```

**预期输出**:
```
测试SocialCircle预训练模型适配器...
[Success] 成功导入开源SocialCircle from ...
模型创建成功: 使用预训练
输入:
  target_trajectory: torch.Size([2, 8, 2])
  neighbor_trajectories: torch.Size([2, 5, 8, 2])
  neighbor_mask: torch.Size([2, 5])
输出:
  social_features: torch.Size([2, 128])

[SUCCESS] SocialCircle预训练模型适配器测试通过！
```

---

### **步骤6: 集成到TrajectoryPredictor**

修改`agsac/models/predictors/trajectory_predictor.py`：

```python
from ..encoders.social_circle_pretrained import create_socialcircle_pretrained

class PretrainedTrajectoryPredictor(TrajectoryPredictorInterface):
    def __init__(
        self,
        social_circle_dim: int = 128,
        pred_horizon: int = 12,
        num_modes: int = 20,
        pretrained_path: Optional[str] = None,
        freeze_social_circle: bool = True
    ):
        super().__init__()
        
        # 使用预训练的SocialCircle
        self.social_circle = create_socialcircle_pretrained(
            use_pretrained=True,
            pretrained_path=pretrained_path,
            social_feature_dim=social_circle_dim
        )
        
        # 是否冻结SocialCircle参数
        if freeze_social_circle:
            for param in self.social_circle.parameters():
                param.requires_grad = False
        
        # E-V2-Net部分（可以是我们的简化实现或也使用预训练）
        # ...
```

---

### **步骤7: 更新.gitignore**

```bash
# 添加到.gitignore
echo "external/SocialCircle_original/" >> .gitignore
echo "pretrained/social_circle/*.pth" >> .gitignore
```

---

## 🔍 关键配置参数

根据[SocialCircle文档](https://github.com/cocoon2wong/SocialCircle)，关键参数包括：

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `partitions` | SocialCircle的扇区数 | -1 (自适应) |
| `Ts` | 变换类型 | `none`, `fft`, `haar`, `db2` |
| `use_direction` | 使用方向因子 | 1 |
| `use_distance` | 使用距离因子 | 1 |
| `use_velocity` | 使用速度因子 | 1 |
| `rel_speed` | 使用相对速度 | 0 (绝对速度) |

---

## 📊 参数量分析

使用预训练的SocialCircle后，模型参数量变化：

### 当前（SimpleTrajectoryPredictor）
```
TrajectoryPredictor: 2,048,770 (67.6%)
总计: 3,029,936
超出预算: 1,029,936
```

### 使用预训练SocialCircle（冻结）
```
SocialCircle (冻结): ~320,000 (不计入训练参数)
E-V2-Net: ~300,000
总TrajectoryPredictor: ~620,000
总计: ~1,600,000 ✅
在预算内！
```

---

## ⚠️ 常见问题

### Q1: 导入失败 "No module named 'socialCircle'"

**解决**:
```python
# 检查external目录是否存在
ls external/SocialCircle_original/socialCircle

# 手动添加到Python path
import sys
sys.path.insert(0, 'external/SocialCircle_original')
```

### Q2: 权重加载失败

**解决**:
```python
# 检查权重文件格式
checkpoint = torch.load('pretrained/social_circle/weights.pth')
print(checkpoint.keys())

# 可能需要适配key名称
# 参考social_circle_pretrained.py的_load_pretrained方法
```

### Q3: 输入格式不匹配

**解决**:
- 检查SocialCircle期望的输入shape
- 在适配器的`forward`方法中添加转换逻辑
- 参考SocialCircle的原始使用示例

---

## 📚 参考资料

- **论文**: [SocialCircle (CVPR2024)](https://cocoon2wong.github.io/SocialCircle/)
- **代码**: [GitHub - cocoon2wong/SocialCircle](https://github.com/cocoon2wong/SocialCircle)
- **PyTorch分支**: [TorchVersion branch](https://github.com/cocoon2wong/SocialCircle/tree/TorchVersion)
- **预训练权重**: [Releases](https://github.com/cocoon2wong/SocialCircle/releases)

---

## ✅ 验证清单

完成集成后，检查以下项目：

- [ ] `external/SocialCircle_original/`目录存在
- [ ] `socialCircle/`和`qpid/`子目录存在
- [ ] 预训练权重已下载到`pretrained/social_circle/weights.pth`
- [ ] 依赖已安装（无冲突）
- [ ] 适配器测试通过
- [ ] 模型参数量在2M以内
- [ ] 端到端测试通过

---

**下一步**: 运行完整的集成测试 `pytest tests/test_integration_e2e.py -v`

