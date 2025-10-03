# TensorBoard使用指南

## 🎯 TensorBoard已集成！

训练时会自动记录以下指标：

### **训练指标** (`train/`)
- `episode_return` - 每个episode的总奖励
- `episode_length` - 每个episode的步数
- `buffer_size` - 经验回放buffer大小
- `actor_loss` - Actor网络损失
- `critic_loss` - Critic网络损失
- `alpha` - SAC温度参数
- `episode_time` - 每个episode耗时

### **评估指标** (`eval/`)
- `mean_return` - 评估平均奖励
- `std_return` - 评估奖励标准差
- `mean_length` - 评估平均长度

---

## 📊 如何查看TensorBoard

### **方法1：训练完成后查看**
```bash
# 训练完成后
tensorboard --logdir outputs/agsac_experiment/tensorboard

# 或者查看特定实验
tensorboard --logdir logs/curriculum_training_20251003_123456/
```

### **方法2：训练过程中实时查看**
```bash
# 终端1：启动训练
python scripts/train.py --config configs/default.yaml

# 终端2：启动TensorBoard
tensorboard --logdir outputs/agsac_experiment/tensorboard
```

然后在浏览器打开：`http://localhost:6006`

---

## 🔍 关键指标解读

### **1. Episode Return (train/episode_return)**
- **趋势**：应该逐渐上升
- **含义**：模型性能的直接指标
- **好的迹象**：从负数逐渐增加，最终接近正数

### **2. Actor Loss (train/actor_loss)**
- **趋势**：初期波动大，后期趋于稳定
- **含义**：策略网络的优化程度
- **正常范围**：-200 到 0之间

### **3. Critic Loss (train/critic_loss)**
- **趋势**：逐渐下降并稳定
- **含义**：值函数估计的准确性
- **正常范围**：初期几千，后期降到几百

### **4. Alpha (train/alpha)**
- **趋势**：自动调整，通常逐渐下降
- **含义**：探索vs利用的平衡
- **正常范围**：0.2 到 1.0之间

---

## 📈 典型训练曲线

### **良好的训练迹象**
```
Episode Return:    -5 → -3 → -1 → 0 → 2 → 5 → 10 ✓
Actor Loss:        -150 → -120 → -100 → -90 ✓
Critic Loss:       3000 → 2000 → 1000 → 500 ✓
```

### **需要注意的情况**
```
Episode Return:    -5 → -10 → -15 → -20 ✗ (性能下降)
Actor Loss:        -50 → -20 → -5 → 0 ✗ (可能过拟合)
Critic Loss:       保持3000不变 ✗ (没有学习)
```

---

## 🛠️ 配置TensorBoard

### **启用/禁用TensorBoard**

在配置文件中设置：
```yaml
training:
  use_tensorboard: true  # 启用
  # use_tensorboard: false  # 禁用
```

### **自定义日志目录**

TensorBoard日志保存在：
```
outputs/{experiment_name}/tensorboard/
```

或训练时指定的日志目录下的`tensorboard/`子目录。

---

## 💡 高级技巧

### **1. 比较多次实验**
```bash
tensorboard --logdir_spec=\
  exp1:logs/run1/,\
  exp2:logs/run2/,\
  exp3:logs/run3/
```

### **2. 平滑曲线**
在TensorBoard界面左侧调整`Smoothing`滑块（建议0.6-0.9）

### **3. 下载数据**
点击左下角的下载按钮可以导出CSV数据

---

## ❓ 常见问题

### **Q: TensorBoard显示"No dashboards are active"？**
A: 确保训练已经开始并至少运行了1个episode。

### **Q: 如何在服务器上使用TensorBoard？**
A: 
```bash
# 服务器上启动TensorBoard
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# 本地浏览器访问
http://服务器IP:6006
```

### **Q: TensorBoard占用内存太大？**
A: 
```bash
# 只加载最近的数据
tensorboard --logdir logs/ --reload_interval 30 --max_reload_threads 1
```

---

## 📚 更多资源

- [TensorBoard官方文档](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard教程](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)

