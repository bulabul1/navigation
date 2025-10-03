# AGSAC训练快速开始指南

**5分钟快速验证整个系统！**

---

## ✅ 前置检查

```bash
# 1. 确认环境已激活
conda activate agsac_dog_nav

# 2. 检查GPU（如果有）
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 3. 确认在项目根目录
cd agsac_dog_navigation
```

---

## 🚀 一键启动训练

```bash
python scripts/train_dummy.py
```

**就这么简单！** 脚本会自动：
- ✅ 检测设备（GPU/CPU）
- ✅ 创建Dummy环境
- ✅ 加载预训练模型（如果有）
- ✅ 开始训练（50 episodes，约15分钟）
- ✅ 保存checkpoint和日志

---

## 📊 查看训练进度

### 方法1: 实时日志
训练时会实时输出：
```
Episode 1/50 | Reward: 5.23 | Steps: 45 | Success: 0%
Episode 2/50 | Reward: 7.81 | Steps: 52 | Success: 0%
...
```

### 方法2: TensorBoard（推荐）

**新开一个终端窗口**:
```bash
# 启动TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# 浏览器打开
# http://localhost:6006
```

可以看到：
- 📈 Episode reward曲线
- 📉 Loss曲线
- 📊 成功率/碰撞率统计

---

## 📁 训练输出

```
outputs/dummy_test/
├── checkpoints/
│   ├── checkpoint_ep10.pt
│   ├── checkpoint_ep20.pt
│   └── ...
├── logs/
│   └── training.log
└── metrics/
    └── eval_results.json
```

---

## 🎯 预期结果

### 正常情况
- ✅ 训练顺利运行10-20分钟
- ✅ Reward逐渐上升
- ✅ 无内存泄漏或崩溃
- ✅ 生成checkpoint文件

### 如果看到错误

#### CUDA Out of Memory
```bash
# 解决：使用CPU
python scripts/train_dummy.py --device cpu
```

#### 预训练权重未找到
```
[WARNING] 预训练权重不存在
[INFO] 将使用简化的轨迹预测器（fallback）
```
**这是正常的！** 系统会自动使用简化版本继续训练。

---

## 🔥 下一步

### 验证成功后

1. **查看详细训练计划**：
   ```bash
   cat docs/TRAINING_PLAN.md
   ```

2. **开始完整训练**：
   ```bash
   python scripts/train.py --config configs/training_full.yaml
   ```

3. **自定义配置**：
   编辑 `configs/training_config.yaml`

---

## 📚 更多文档

- `docs/TRAINING_PLAN.md` - 详细训练方案
- `docs/ARCHITECTURE_VALIDATION.md` - 系统架构
- `docs/ALL_FIXES_COMPLETE.md` - 测试报告
- `README.md` - 项目总览

---

## 🆘 需要帮助？

**常见问题**:
1. **Q: 训练很慢？**  
   A: Dummy环境已经很快了。完整训练会更慢，建议使用GPU。

2. **Q: Reward不上升？**  
   A: 前几个episodes正常，需要50+episodes才能看到明显改进。

3. **Q: 想停止训练？**  
   A: 按 `Ctrl+C`，进度会自动保存。

**详细文档**: `docs/TRAINING_PLAN.md`

---

**🎉 祝训练顺利！**


