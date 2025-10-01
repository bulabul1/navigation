# AGSAC Dog Navigation

基于注意力机制和几何微分评估的机器狗导航系统

## 项目概述

AGSAC (Attention-based Geometric Soft Actor-Critic) 是一个专为机器狗设计的智能导航系统，结合了：

- **多模态注意力融合**：融合行人轨迹、通路几何和机器狗状态信息
- **几何微分评估器 (GDE)**：提供精确的几何约束评估
- **混合SAC算法**：结合连续和离散动作空间的强化学习
- **预训练轨迹预测**：基于E-V2-Net的行人轨迹预测

## 主要特性

- 🎯 **多模态感知**：同时处理行人、环境和机器狗状态
- 🧠 **注意力机制**：自适应关注重要信息
- 📐 **几何约束**：确保路径的几何可行性
- 🚀 **强化学习**：基于SAC的智能决策
- 🔄 **预训练模块**：可复用的轨迹预测组件

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 预训练轨迹预测模块

```bash
python scripts/pretrain_trajectory.py \
    --dataset data/trajectories/eth \
    --epochs 600 \
    --output pretrained/e_v2_net/
```

### 训练AGSAC模型

```bash
python scripts/train.py \
    --config configs/training_config.yaml \
    --pretrained pretrained/e_v2_net/e_v2_net_weights.pth
```

### 评估模型

```bash
python scripts/evaluate.py \
    --model outputs/models/agsac_epoch_1000.pth \
    --num_episodes 100
```

## 项目结构

```
agsac_dog_navigation/
├── agsac/                    # 核心代码包
│   ├── models/              # 模型组件
│   ├── envs/                # 环境接口
│   ├── training/            # 训练逻辑
│   └── utils/               # 工具函数
├── configs/                 # 配置文件
├── scripts/                 # 运行脚本
├── tests/                   # 单元测试
├── data/                    # 数据目录
├── pretrained/              # 预训练模型
└── outputs/                 # 输出结果
```

## 开发计划

### 第1阶段：基础框架 ✅
- [x] 数据处理模块
- [x] 几何工具
- [x] 基础测试

### 第2阶段：编码器
- [ ] PointNet实现
- [ ] DogStateEncoder
- [ ] CorridorEncoder
- [ ] 编码器测试

### 第3阶段：预训练模块
- [ ] SocialCircle
- [ ] E-V2-Net
- [ ] PedestrianEncoder
- [ ] 预训练脚本

### 第4阶段：融合与SAC
- [ ] MultiModalFusion
- [ ] Hybrid Actor
- [ ] Hybrid Critic
- [ ] SAC Agent

### 第5阶段：评估与完整Agent
- [ ] 几何微分评估器
- [ ] 完整Agent
- [ ] 集成测试

### 第6阶段：训练系统
- [ ] ReplayBuffer
- [ ] Trainer
- [ ] Environment
- [ ] 训练脚本

### 第7阶段：可视化与评估
- [ ] 可视化工具
- [ ] 评估脚本
- [ ] Logger

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。
