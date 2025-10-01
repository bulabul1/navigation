#!/usr/bin/env python3
"""
轨迹预测模块预训练脚本

使用方法:
    python scripts/pretrain_trajectory.py --dataset data/trajectories/eth
    python scripts/pretrain_trajectory.py --dataset data/trajectories/eth --epochs 600
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from agsac.models.predictors.e_v2_net import EV2Net
from agsac.models.encoders.pedestrian_encoder import PedestrianEncoder
from agsac.models.encoders.social_circle import SocialCircle
from agsac.utils.logger import setup_logger
from agsac.utils.data_processing import set_seed, TrajectoryDataset, collate_fn


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="轨迹预测预训练脚本")
    
    # 数据集配置
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["eth", "ucy", "both"],
        help="使用的数据集"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/trajectories",
        help="数据路径"
    )
    
    # 训练配置
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="训练轮数"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批次大小"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="学习率"
    )
    
    # 模型配置
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="隐藏层维度"
    )
    
    parser.add_argument(
        "--prediction_horizon",
        type=int,
        default=12,
        help="预测时间步长"
    )
    
    # 输出配置
    parser.add_argument(
        "--output",
        type=str,
        default="pretrained/e_v2_net",
        help="输出目录"
    )
    
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="保存间隔"
    )
    
    # 设备配置
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="计算设备"
    )
    
    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    return parser.parse_args()


def setup_device(device_arg):
    """设置计算设备"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_arg
    
    print(f"使用设备: {device}")
    return device


def load_trajectory_data(data_path, dataset_name):
    """加载轨迹数据"""
    if dataset_name == "both":
        datasets = []
        for name in ["eth", "ucy"]:
            dataset_path = Path(data_path) / name
            if dataset_path.exists():
                datasets.append(TrajectoryDataset(dataset_path))
        return torch.utils.data.ConcatDataset(datasets)
    else:
        dataset_path = Path(data_path) / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")
        return TrajectoryDataset(dataset_path)


def create_model(config, device):
    """创建模型"""
    # 创建编码器
    pedestrian_encoder = PedestrianEncoder(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        num_layers=2
    )
    
    social_circle = SocialCircle(
        radius=2.0,
        num_circles=5,
        feature_dim=64,
        hidden_dim=config.hidden_dim
    )
    
    # 创建E-V2-Net
    e_v2_net = EV2Net(
        encoder_hidden=config.hidden_dim,
        decoder_hidden=config.hidden_dim,
        num_layers=2,
        prediction_horizon=config.prediction_horizon
    )
    
    # 组合模型
    model = nn.ModuleDict({
        'pedestrian_encoder': pedestrian_encoder,
        'social_circle': social_circle,
        'e_v2_net': e_v2_net
    })
    
    return model.to(device)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # 移动数据到设备
        past_trajectories = batch['past_trajectories'].to(device)
        future_trajectories = batch['future_trajectories'].to(device)
        masks = batch['masks'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        # 编码过去轨迹
        encoded_features = model['pedestrian_encoder'](past_trajectories, masks)
        
        # 社会交互特征
        social_features = model['social_circle'](past_trajectories, masks)
        
        # 融合特征
        combined_features = torch.cat([encoded_features, social_features], dim=-1)
        
        # 预测未来轨迹
        predicted_trajectories = model['e_v2_net'](combined_features)
        
        # 计算损失
        loss = criterion(predicted_trajectories, future_trajectories, masks)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            past_trajectories = batch['past_trajectories'].to(device)
            future_trajectories = batch['future_trajectories'].to(device)
            masks = batch['masks'].to(device)
            
            # 前向传播
            encoded_features = model['pedestrian_encoder'](past_trajectories, masks)
            social_features = model['social_circle'](past_trajectories, masks)
            combined_features = torch.cat([encoded_features, social_features], dim=-1)
            predicted_trajectories = model['e_v2_net'](combined_features)
            
            # 计算损失
            loss = criterion(predicted_trajectories, future_trajectories, masks)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = setup_device(args.device)
    
    # 设置输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        name="TrajectoryPretrain",
        log_dir=output_dir / "logs",
        level="INFO"
    )
    
    logger.info("=" * 50)
    logger.info("轨迹预测预训练开始")
    logger.info("=" * 50)
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"设备: {device}")
    
    try:
        # 加载数据
        logger.info("加载数据集...")
        dataset = load_trajectory_data(args.data_path, args.dataset)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        
        # 创建模型
        config = OmegaConf.create({
            'hidden_dim': args.hidden_dim,
            'prediction_horizon': args.prediction_horizon
        })
        
        model = create_model(config, device)
        logger.info("模型创建成功")
        
        # 创建优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            # 训练
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
            
            # 验证
            val_loss = validate_epoch(model, val_loader, criterion, device)
            
            logger.info(f"Epoch {epoch}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, output_dir / "best_model.pth")
                logger.info(f"保存最佳模型 (验证损失: {val_loss:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, output_dir / f"checkpoint_epoch_{epoch}.pth")
        
        # 保存最终模型
        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, output_dir / "final_model.pth")
        
        logger.info("预训练完成!")
        logger.info(f"最佳验证损失: {best_val_loss:.4f}")
        
    except Exception as e:
        logger.error(f"预训练过程中发生错误: {e}")
        raise
    finally:
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
