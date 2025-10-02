# SocialCircle集成快速指南

> 5分钟完成SocialCircle集成！

---

## 🚀 快速开始

### 方法A: 自动化脚本（推荐） ⭐

**Windows用户**:
```bash
cd agsac_dog_navigation
scripts\setup_socialcircle.bat
```

**Linux/Mac用户**:
```bash
cd agsac_dog_navigation
chmod +x scripts/setup_socialcircle.sh
./scripts/setup_socialcircle.sh
```

---

### 方法B: 手动操作

```bash
# 1. Clone PyTorch分支（在项目外）
cd C:\Users\13772\Desktop\myProjects\WWW_navigetion
git clone -b TorchVersion https://github.com/cocoon2wong/SocialCircle.git SocialCircle_temp

# 2. 复制到项目里
cd agsac_dog_navigation
mkdir -p external\SocialCircle_original
xcopy /E /I ..\SocialCircle_temp\socialCircle external\SocialCircle_original\socialCircle
xcopy /E /I ..\SocialCircle_temp\qpid external\SocialCircle_original\qpid

# 3. 清理
cd ..
rmdir /s /q SocialCircle_temp
```

---

## 📥 下载预训练权重

1. 访问: https://github.com/cocoon2wong/SocialCircle/releases
2. 下载PyTorch版本权重
3. 放到: `pretrained/social_circle/weights.pth`

---

## ✅ 验证安装

```bash
# 测试适配器
python -m agsac.models.encoders.social_circle_pretrained

# 预期输出
# [Success] 成功导入开源SocialCircle from ...
# [SUCCESS] SocialCircle预训练模型适配器测试通过！
```

---

## 📂 最终目录结构

```
agsac_dog_navigation/
├── external/
│   └── SocialCircle_original/      # ✅ SocialCircle代码
│       ├── socialCircle/
│       └── qpid/
├── pretrained/
│   └── social_circle/
│       └── weights.pth             # ✅ 预训练权重
├── agsac/
│   └── models/
│       └── encoders/
│           └── social_circle_pretrained.py  # ✅ 适配器
└── docs/
    └── SOCIALCIRCLE_INTEGRATION.md  # 详细文档
```

---

## 🔧 下一步

### 完善适配器

打开`agsac/models/encoders/social_circle_pretrained.py`，根据实际代码调整：

1. **导入路径** (第49行)
2. **模型构建** (第68行)  
3. **输入格式转换** (第115行)

### 运行集成测试

```bash
pytest tests/test_integration_e2e.py -v
```

---

## 📖 详细文档

查看完整文档：`docs/SOCIALCIRCLE_INTEGRATION.md`

---

## ⚠️ 常见问题

**Q: 找不到SocialCircle模块？**

A: 检查`external/SocialCircle_original/`是否存在

**Q: 权重加载失败？**

A: 确保下载的是PyTorch版本（不是TensorFlow）

**Q: 没有权重可以用吗？**

A: 可以！会自动使用SimplifiedSocialCircle（效果略差）

---

**有问题？** 查看完整文档或提Issue！

