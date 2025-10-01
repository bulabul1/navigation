#!/usr/bin/env python3
"""
AGSAC训练主脚本

使用方法:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --pretrained pretrained/e_v2_net/e_v2_net_weights.pth
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from omegaconf import OmegaConf

from agsac.training.trainer import AGSACTrainer
from agsac.utils.logger import setup_logger
from agsac.utils.data_processing import set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AGSAC训练脚本")
    
    # 配置文件
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="配置文件路径"
    )
    
    # 预训练模型
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="预训练模型路径"
    )
    
    # 输出目录
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/experiments",
        help="输出目录"
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
    
    # 调试模式
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式"
    )
    
    # 恢复训练
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径"
    )
    
    return parser.parse_args()


def setup_device(device_arg, config):
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


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 使用OmegaConf加载配置
    config = OmegaConf.load(config_path)
    
    # 如果配置中有defaults，需要合并
    if hasattr(config, 'defaults') and config.defaults:
        base_config = OmegaConf.load(config.defaults[0])
        config = OmegaConf.merge(base_config, config)
    
    return config


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = setup_device(args.device, config)
    config.device.use_cuda = (device == "cuda")
    config.device.device_id = 0
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        name="AGSAC_Trainer",
        log_dir=output_dir / "logs",
        level=config.logging.level
    )
    
    logger.info("=" * 50)
    logger.info("AGSAC训练开始")
    logger.info("=" * 50)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"设备: {device}")
    logger.info(f"随机种子: {args.seed}")
    
    if args.pretrained:
        logger.info(f"预训练模型: {args.pretrained}")
    
    if args.resume:
        logger.info(f"恢复训练: {args.resume}")
    
    # 创建训练器
    try:
        trainer = AGSACTrainer(
            config=config,
            output_dir=output_dir,
            device=device,
            logger=logger
        )
        
        # 加载预训练模型
        if args.pretrained:
            trainer.load_pretrained(args.pretrained)
        
        # 恢复训练
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        trainer.train()
        
        logger.info("训练完成!")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise
    finally:
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
