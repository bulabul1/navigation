"""
编码器模块测试

测试各种编码器的前向传播和输出维度
"""

import pytest
import torch
import numpy as np
from agsac.models.encoders.pointnet import PointNet
from agsac.models.encoders.dog_state_encoder import DogStateEncoder
from agsac.models.encoders.corridor_encoder import CorridorEncoder
from agsac.models.encoders.pedestrian_encoder import PedestrianEncoder
from agsac.models.encoders.social_circle import SocialCircle


class TestEncoders:
    """编码器测试类"""
    
    def test_pointnet(self):
        """测试PointNet编码器"""
        # 创建PointNet
        pointnet = PointNet(
            input_dim=2,
            hidden_dims=[64, 128, 256],
            output_dim=256,
            dropout=0.1
        )
        
        # 创建测试数据
        batch_size = 4
        num_points = 100
        input_points = torch.randn(batch_size, num_points, 2)
        
        # 前向传播
        output = pointnet(input_points)
        
        # 检查输出形状
        assert output.shape == (batch_size, 256)
        
        # 检查输出值
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_dog_state_encoder(self):
        """测试机器狗状态编码器"""
        # 创建编码器
        encoder = DogStateEncoder(
            input_dim=6,  # (x, y, theta, vx, vy, omega)
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        )
        
        # 创建测试数据
        batch_size = 8
        sequence_length = 10
        input_states = torch.randn(batch_size, sequence_length, 6)
        
        # 前向传播
        output = encoder(input_states)
        
        # 检查输出形状
        assert output.shape == (batch_size, 128)
        
        # 检查输出值
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_corridor_encoder(self):
        """测试通路编码器"""
        # 创建编码器
        encoder = CorridorEncoder(
            pointnet_hidden=[64, 128, 256],
            attention_dim=128,
            dropout=0.1
        )
        
        # 创建测试数据
        batch_size = 4
        max_corridors = 5
        max_points = 50
        
        # 创建多边形点云数据
        corridor_points = torch.randn(batch_size, max_corridors, max_points, 2)
        masks = torch.ones(batch_size, max_corridors)
        
        # 前向传播
        output = encoder(corridor_points, masks)
        
        # 检查输出形状
        assert output.shape == (batch_size, max_corridors, 256)
        
        # 检查输出值
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_pedestrian_encoder(self):
        """测试行人编码器"""
        # 创建编码器
        encoder = PedestrianEncoder(
            input_dim=2,  # (x, y)
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        )
        
        # 创建测试数据
        batch_size = 4
        max_pedestrians = 10
        trajectory_length = 8
        
        trajectories = torch.randn(batch_size, max_pedestrians, trajectory_length, 2)
        masks = torch.ones(batch_size, max_pedestrians)
        
        # 前向传播
        output = encoder(trajectories, masks)
        
        # 检查输出形状
        assert output.shape == (batch_size, max_pedestrians, 128)
        
        # 检查输出值
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_social_circle(self):
        """测试SocialCircle编码器"""
        # 创建编码器
        encoder = SocialCircle(
            radius=2.0,
            num_circles=5,
            feature_dim=64,
            hidden_dim=128
        )
        
        # 创建测试数据
        batch_size = 4
        max_pedestrians = 10
        trajectory_length = 8
        
        trajectories = torch.randn(batch_size, max_pedestrians, trajectory_length, 2)
        masks = torch.ones(batch_size, max_pedestrians)
        
        # 前向传播
        output = encoder(trajectories, masks)
        
        # 检查输出形状
        assert output.shape == (batch_size, max_pedestrians, 128)
        
        # 检查输出值
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_encoder_with_masks(self):
        """测试编码器处理掩码的能力"""
        # 创建行人编码器
        encoder = PedestrianEncoder(
            input_dim=2,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        )
        
        # 创建测试数据
        batch_size = 2
        max_pedestrians = 5
        trajectory_length = 8
        
        trajectories = torch.randn(batch_size, max_pedestrians, trajectory_length, 2)
        
        # 创建掩码（第一个batch有3个行人，第二个batch有2个行人）
        masks = torch.tensor([
            [1, 1, 1, 0, 0],  # 前3个有效
            [1, 1, 0, 0, 0]   # 前2个有效
        ])
        
        # 前向传播
        output = encoder(trajectories, masks)
        
        # 检查输出形状
        assert output.shape == (batch_size, max_pedestrians, 128)
        
        # 检查掩码位置是否为零（或接近零）
        assert torch.allclose(output[0, 3:, :], torch.zeros_like(output[0, 3:, :]), atol=1e-6)
        assert torch.allclose(output[1, 2:, :], torch.zeros_like(output[1, 2:, :]), atol=1e-6)
    
    def test_encoder_gradient_flow(self):
        """测试编码器的梯度流"""
        # 创建PointNet
        pointnet = PointNet(
            input_dim=2,
            hidden_dims=[64, 128, 256],
            output_dim=256,
            dropout=0.1
        )
        
        # 创建测试数据
        input_points = torch.randn(2, 50, 2, requires_grad=True)
        
        # 前向传播
        output = pointnet(input_points)
        
        # 计算损失
        loss = output.sum()
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        assert input_points.grad is not None
        assert not torch.isnan(input_points.grad).any()
        assert not torch.isinf(input_points.grad).any()
    
    def test_encoder_different_batch_sizes(self):
        """测试编码器处理不同批次大小的能力"""
        # 创建编码器
        encoder = DogStateEncoder(
            input_dim=6,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        )
        
        # 测试不同批次大小
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            input_states = torch.randn(batch_size, 10, 6)
            output = encoder(input_states)
            
            assert output.shape == (batch_size, 128)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__])
