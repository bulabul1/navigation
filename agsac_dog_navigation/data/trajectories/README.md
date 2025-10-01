# 轨迹数据集

本目录包含用于预训练轨迹预测模块的数据集。

## 数据集结构

### ETH数据集
- **来源**: ETH行人轨迹数据集
- **格式**: 每个文件包含多行，每行格式为 `frame_id,person_id,x,y`
- **用途**: 预训练行人轨迹预测模块

### UCY数据集
- **来源**: UCY行人轨迹数据集
- **格式**: 每个文件包含多行，每行格式为 `frame_id,person_id,x,y`
- **用途**: 预训练行人轨迹预测模块

## 数据预处理

数据预处理脚本会自动：
1. 将轨迹按行人ID分组
2. 创建过去和未来轨迹对
3. 应用数据增强（噪声、旋转等）
4. 生成训练批次

## 使用方法

```bash
# 预训练轨迹预测模块
python scripts/pretrain_trajectory.py --dataset eth
python scripts/pretrain_trajectory.py --dataset ucy
python scripts/pretrain_trajectory.py --dataset both
```

## 数据格式

### 输入格式
- **过去轨迹**: `(batch_size, max_pedestrians, trajectory_length, 2)`
- **未来轨迹**: `(batch_size, max_pedestrians, prediction_horizon, 2)`
- **掩码**: `(batch_size, max_pedestrians)`

### 输出格式
- **预测轨迹**: `(batch_size, max_pedestrians, prediction_horizon, 2)`

## 注意事项

1. 确保数据文件格式正确
2. 轨迹长度需要一致
3. 坐标系统需要统一
4. 建议使用数据增强提高泛化能力
