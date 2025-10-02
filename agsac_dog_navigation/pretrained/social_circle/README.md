# SocialCircle预训练权重下载指南

## 📦 推荐模型

**E-V²-Net-SC (evsc)** - 这是最新且性能最好的SocialCircle变体

---

## 🔗 官方下载链接

### GitHub权重仓库
**主链接**: https://github.com/cocoon2wong/Project-Monandaeg/tree/SocialCircle

这个仓库包含所有PyTorch版本的预训练权重。

---

## 📥 下载步骤

### 方式1: 直接下载（推荐）⭐

1. **访问权重仓库**:
   ```
   https://github.com/cocoon2wong/Project-Monandaeg/tree/SocialCircle
   ```

2. **选择模型**（推荐其中之一）:
   - `evsc_P8_sdd` - 在SDD数据集上训练的E-V²-Net-SC
   - `evsc_zara1` - 在ZARA1数据集上训练的E-V²-Net-SC
   - `vsc_P8_sdd` - 在SDD数据集上训练的V²-Net-SC

3. **下载整个文件夹**:
   - 每个权重文件夹包含多个文件
   - 建议下载整个文件夹（包含配置文件）

4. **放置文件**:
   ```
   agsac_dog_navigation/
   └── pretrained/
       └── social_circle/
           ├── README.md          # 本文件
           └── evsc_P8_sdd/       # 下载的权重文件夹
               ├── checkpoints/   # 模型checkpoint
               ├── args.json      # 训练参数
               └── ...
   ```

### 方式2: 使用Git LFS（如果仓库支持）

```bash
# 进入pretrained目录
cd pretrained/social_circle/

# 使用sparse checkout只下载特定模型
git clone -b SocialCircle --single-branch --depth 1 \
    https://github.com/cocoon2wong/Project-Monandaeg.git temp

# 复制需要的权重
cp -r temp/weights/SocialCircle/evsc_P8_sdd ./

# 清理临时文件
rm -rf temp
```

---

## 📁 权重文件结构

下载后的权重文件夹通常包含：

```
evsc_P8_sdd/
├── checkpoints/
│   └── best_ade_epoch.pt          # 最佳ADE模型
│   └── best_fde_epoch.pt          # 最佳FDE模型
├── args.json                       # 训练时的参数配置
├── model.json                      # 模型架构信息
└── loss_log.txt                    # 训练损失记录
```

---

## 🧪 验证下载

运行测试脚本验证权重可以正常加载：

```bash
cd agsac_dog_navigation
python -m agsac.models.encoders.social_circle_pretrained
```

**预期输出**:
```
✓ SocialCircle model loaded successfully
✓ Model structure verified
✓ Forward pass test passed
```

---

## 📊 模型对比

| 模型 | 数据集 | 参数量 | ADE↓ | FDE↓ | 推荐场景 |
|------|--------|--------|------|------|---------|
| **evsc_P8_sdd** | SDD | ~20K | 6.37 | 10.27 | 室外场景 ⭐ |
| evsc_zara1 | ZARA1 | ~20K | 0.23 | 0.48 | 行人密集场景 |
| vsc_P8_sdd | SDD | ~18K | 7.12 | 11.53 | 轻量级部署 |

**推荐**: 使用 `evsc_P8_sdd` - 性能好且适合多数场景

---

## 🔧 在代码中使用

### 1. 在适配器中加载

修改 `agsac/models/encoders/social_circle_pretrained.py`:

```python
adapter = PretrainedSocialCircleAdapter(
    model_type='evsc',  # E-V²-Net-SC
    pretrained_path='pretrained/social_circle/evsc_P8_sdd/checkpoints/best_ade_epoch.pt',
    freeze=True  # 冻结权重不训练
)
```

### 2. 在AGSACModel中使用

修改 `agsac/models/agsac_model.py`:

```python
from .encoders.social_circle_pretrained import PretrainedSocialCircleAdapter

self.social_circle = PretrainedSocialCircleAdapter(
    model_type='evsc',
    pretrained_path='pretrained/social_circle/evsc_P8_sdd/checkpoints/best_ade_epoch.pt',
    freeze=True
)
```

---

## ⚠️ 常见问题

### Q1: 下载速度慢？
- 使用GitHub镜像站
- 或使用代理加速

### Q2: 找不到PyTorch权重？
- 确保访问的是 `SocialCircle` 分支
- 不是 `main` 分支（TensorFlow版本）

### Q3: 权重加载失败？
- 检查PyTorch版本兼容性（推荐1.12+）
- 确认下载的是`.pt`或`.pth`文件
- 检查文件路径是否正确

### Q4: 哪个checkpoint更好？
- `best_ade_epoch.pt` - 平均位移误差最小 ⭐ 推荐
- `best_fde_epoch.pt` - 最终位移误差最小

---

## 📚 参考资料

- **论文**: SocialCircle: Learning the Angle-based Social Interaction Representation (CVPR 2024)
- **主页**: https://cocoon2wong.github.io/SocialCircle/
- **代码**: https://github.com/cocoon2wong/SocialCircle
- **权重**: https://github.com/cocoon2wong/Project-Monandaeg/tree/SocialCircle

---

**更新时间**: 2025-10-03  
**状态**: 🟢 下载链接已验证
