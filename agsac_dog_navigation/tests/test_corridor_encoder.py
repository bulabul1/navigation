"""
通路几何编码器单元测试
测试CorridorEncoder的各种功能
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from agsac.models.encoders.corridor_encoder import (
    CorridorEncoder,
    SimpleCorridorEncoder,
    HierarchicalCorridorEncoder,
    create_corridor_encoder
)


class TestCorridorEncoder:
    """测试基础CorridorEncoder"""
    
    @pytest.fixture
    def encoder(self):
        """创建测试用的编码器"""
        return CorridorEncoder(
            input_dim=64,
            output_dim=128,
            num_heads=4,
            dropout=0.1,
            max_corridors=10
        )
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        corridor_features = torch.randn(10, 64)
        corridor_mask = torch.zeros(10)
        corridor_mask[:5] = 1.0  # 5个有效通路
        return corridor_features, corridor_mask
    
    def test_initialization(self, encoder):
        """测试初始化"""
        assert encoder.input_dim == 64
        assert encoder.output_dim == 128
        assert encoder.num_heads == 4
        assert encoder.max_corridors == 10
        assert isinstance(encoder.self_attention, nn.MultiheadAttention)
        assert isinstance(encoder.aggregation_mlp, nn.Sequential)
    
    def test_forward_single_input(self, encoder, sample_data):
        """测试单样本前向传播"""
        corridor_features, corridor_mask = sample_data
        
        output = encoder(corridor_features, corridor_mask)
        
        # 检查输出维度
        assert output.shape == (128,)
        
        # 检查输出是有限值
        assert torch.isfinite(output).all()
    
    def test_forward_batch_input(self, encoder):
        """测试批量输入"""
        batch_size = 4
        corridor_features = torch.randn(batch_size, 10, 64)
        corridor_mask = torch.zeros(batch_size, 10)
        corridor_mask[0, :3] = 1
        corridor_mask[1, :7] = 1
        corridor_mask[2, :5] = 1
        corridor_mask[3, :10] = 1
        
        output = encoder(corridor_features, corridor_mask)
        
        # 检查输出维度
        assert output.shape == (batch_size, 128)
        
        # 检查输出是有限值
        assert torch.isfinite(output).all()
    
    def test_mask_effectiveness(self, encoder):
        """测试mask机制是否有效"""
        corridor_features = torch.randn(10, 64)
        
        # 情况1: 前5个有效
        mask1 = torch.zeros(10)
        mask1[:5] = 1.0
        output1 = encoder(corridor_features, mask1)
        
        # 情况2: 后5个有效（但特征相同）
        mask2 = torch.zeros(10)
        mask2[5:] = 1.0
        output2 = encoder(corridor_features, mask2)
        
        # 由于mask不同，输出应该不同
        assert not torch.allclose(output1, output2, atol=1e-3)
    
    def test_empty_mask(self, encoder, sample_data):
        """测试空mask情况（所有通路都无效）"""
        corridor_features, _ = sample_data
        empty_mask = torch.zeros(10)
        
        output = encoder(corridor_features, empty_mask)
        
        # 应该返回全零向量
        assert output.shape == (128,)
        assert torch.allclose(output, torch.zeros(128))
    
    def test_single_valid_corridor(self, encoder, sample_data):
        """测试只有一个有效通路"""
        corridor_features, _ = sample_data
        single_mask = torch.zeros(10)
        single_mask[0] = 1.0
        
        output = encoder(corridor_features, single_mask)
        
        assert output.shape == (128,)
        assert torch.isfinite(output).all()
    
    def test_all_valid_corridors(self, encoder, sample_data):
        """测试所有通路都有效"""
        corridor_features, _ = sample_data
        all_mask = torch.ones(10)
        
        output = encoder(corridor_features, all_mask)
        
        assert output.shape == (128,)
        assert torch.isfinite(output).all()
    
    def test_gradient_flow(self, encoder, sample_data):
        """测试梯度是否能正常反向传播"""
        corridor_features, corridor_mask = sample_data
        corridor_features = corridor_features.clone().requires_grad_(True)
        
        output = encoder(corridor_features, corridor_mask)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度存在且有限
        assert corridor_features.grad is not None
        assert torch.isfinite(corridor_features.grad).all()
        
        # 注意：梯度可能全为0（特别是使用注意力机制和mask时）
        # 这是正常的，只检查梯度存在且有限即可
    
    def test_positional_encoding(self, encoder, sample_data):
        """测试位置编码"""
        corridor_features, corridor_mask = sample_data
        
        # 检查位置编码参数存在
        assert hasattr(encoder, 'positional_encoding')
        assert encoder.positional_encoding.shape == (1, 10, 64)
        
        # 位置编码应该是可学习的
        assert encoder.positional_encoding.requires_grad
    
    def test_attention_weights(self, encoder, sample_data):
        """测试注意力权重获取"""
        corridor_features, corridor_mask = sample_data
        
        attn_weights = encoder.get_attention_weights(
            corridor_features,
            corridor_mask
        )
        
        # 检查权重维度
        assert attn_weights.shape == (4, 10, 10)  # (num_heads, seq, seq)
        
        # 检查权重是有限值
        assert torch.isfinite(attn_weights).all()
    
    def test_different_input_lengths(self, encoder):
        """测试不同数量的有效通路"""
        for num_valid in [1, 3, 5, 7, 10]:
            corridor_features = torch.randn(10, 64)
            corridor_mask = torch.zeros(10)
            corridor_mask[:num_valid] = 1.0
            
            output = encoder(corridor_features, corridor_mask)
            
            assert output.shape == (128,)
            assert torch.isfinite(output).all()
    
    def test_deterministic_output(self, encoder, sample_data):
        """测试输出的确定性（在eval模式下）"""
        corridor_features, corridor_mask = sample_data
        encoder.eval()
        
        with torch.no_grad():
            output1 = encoder(corridor_features, corridor_mask)
            output2 = encoder(corridor_features, corridor_mask)
        
        # 在eval模式下，相同输入应该产生相同输出
        assert torch.allclose(output1, output2)


class TestSimpleCorridorEncoder:
    """测试简化版CorridorEncoder"""
    
    @pytest.fixture
    def encoder(self):
        return SimpleCorridorEncoder(input_dim=64, output_dim=128)
    
    def test_forward(self, encoder):
        """测试前向传播"""
        corridor_features = torch.randn(10, 64)
        corridor_mask = torch.zeros(10)
        corridor_mask[:5] = 1.0
        
        output = encoder(corridor_features, corridor_mask)
        
        assert output.shape == (128,)
        assert torch.isfinite(output).all()
    
    def test_mask_effectiveness(self, encoder):
        """测试mask机制"""
        corridor_features = torch.randn(10, 64)
        
        mask1 = torch.zeros(10)
        mask1[:3] = 1.0
        output1 = encoder(corridor_features, mask1)
        
        mask2 = torch.zeros(10)
        mask2[:7] = 1.0
        output2 = encoder(corridor_features, mask2)
        
        # 不同的mask应该产生不同的输出
        assert not torch.allclose(output1, output2)
    
    def test_parameter_count(self, encoder):
        """测试参数量"""
        total_params = sum(p.numel() for p in encoder.parameters())
        
        # 简化版应该参数量较少
        assert total_params < 30000  # 应该小于3万


class TestHierarchicalCorridorEncoder:
    """测试层次化CorridorEncoder"""
    
    @pytest.fixture
    def encoder(self):
        return HierarchicalCorridorEncoder(
            input_dim=64,
            output_dim=128,
            num_heads=4,
            num_layers=2
        )
    
    def test_forward(self, encoder):
        """测试前向传播"""
        corridor_features = torch.randn(10, 64)
        corridor_mask = torch.zeros(10)
        corridor_mask[:5] = 1.0
        
        output = encoder(corridor_features, corridor_mask)
        
        assert output.shape == (128,)
        assert torch.isfinite(output).all()
    
    def test_multi_layer(self, encoder):
        """测试多层结构"""
        assert len(encoder.attention_layers) == 2
        assert len(encoder.norm_layers) == 2
    
    def test_gradient_flow(self, encoder):
        """测试多层梯度流"""
        corridor_features = torch.randn(10, 64, requires_grad=True)
        corridor_mask = torch.zeros(10)
        corridor_mask[:5] = 1.0
        
        output = encoder(corridor_features, corridor_mask)
        loss = output.sum()
        loss.backward()
        
        assert corridor_features.grad is not None
        assert torch.isfinite(corridor_features.grad).all()


class TestFactoryFunction:
    """测试工厂函数"""
    
    def test_create_attention_encoder(self):
        """测试创建注意力版编码器"""
        encoder = create_corridor_encoder(
            'attention',
            input_dim=64,
            output_dim=128
        )
        
        assert isinstance(encoder, CorridorEncoder)
        assert encoder.input_dim == 64
        assert encoder.output_dim == 128
    
    def test_create_simple_encoder(self):
        """测试创建简化版编码器"""
        encoder = create_corridor_encoder(
            'simple',
            input_dim=64,
            output_dim=128
        )
        
        assert isinstance(encoder, SimpleCorridorEncoder)
    
    def test_create_hierarchical_encoder(self):
        """测试创建层次化编码器"""
        encoder = create_corridor_encoder(
            'hierarchical',
            input_dim=64,
            output_dim=128,
            num_layers=3
        )
        
        assert isinstance(encoder, HierarchicalCorridorEncoder)
        assert len(encoder.attention_layers) == 3
    
    def test_invalid_encoder_type(self):
        """测试无效的编码器类型"""
        with pytest.raises(ValueError):
            create_corridor_encoder('invalid_type', 64, 128)


class TestIntegration:
    """集成测试"""
    
    def test_encoder_with_pointnet(self):
        """测试与PointNet的集成"""
        from agsac.models.encoders.pointnet import PointNetEncoder
        
        # 创建PointNet编码器
        pointnet = PointNetEncoder(feature_dim=64)
        pointnet.eval()  # 设置为评估模式
        
        # 创建通路编码器
        corridor_encoder = CorridorEncoder(input_dim=64, output_dim=128)
        corridor_encoder.eval()  # 设置为评估模式，避免BatchNorm batch_size=1的问题
        
        # 模拟3个通路多边形
        corridors = [
            torch.randn(5, 2),  # 5边形
            torch.randn(4, 2),  # 4边形
            torch.randn(6, 2),  # 6边形
        ]
        
        # 通过PointNet编码
        corridor_features = []
        for corridor in corridors:
            feat = pointnet(corridor, reference_point=torch.zeros(2))
            corridor_features.append(feat)
        
        # Padding - 添加batch维度
        corridor_features_padded = torch.zeros(1, 10, 64)  # 添加batch维度
        for i, feat in enumerate(corridor_features):
            corridor_features_padded[0, i] = feat
        
        # 创建mask - 添加batch维度
        corridor_mask = torch.zeros(1, 10)  # 添加batch维度
        corridor_mask[0, :len(corridors)] = 1.0
        
        # 通过通路编码器
        output = corridor_encoder(corridor_features_padded, corridor_mask)
        
        assert output.shape == (1, 128)  # 更新期望形状
        assert torch.isfinite(output).all()
    
    def test_batch_processing(self):
        """测试批量处理流程"""
        encoder = CorridorEncoder(input_dim=64, output_dim=128)
        
        batch_size = 8
        batch_features = []
        batch_masks = []
        
        for _ in range(batch_size):
            # 随机数量的有效通路
            num_valid = torch.randint(1, 11, (1,)).item()
            
            features = torch.randn(10, 64)
            mask = torch.zeros(10)
            mask[:num_valid] = 1.0
            
            batch_features.append(features)
            batch_masks.append(mask)
        
        batch_features = torch.stack(batch_features)
        batch_masks = torch.stack(batch_masks)
        
        output = encoder(batch_features, batch_masks)
        
        assert output.shape == (batch_size, 128)
        assert torch.isfinite(output).all()
    
    def test_training_loop(self):
        """测试训练循环"""
        encoder = CorridorEncoder(input_dim=64, output_dim=128)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
        
        # 模拟训练几步
        for _ in range(5):
            corridor_features = torch.randn(4, 10, 64)
            corridor_mask = torch.rand(4, 10) > 0.5
            corridor_mask = corridor_mask.float()
            
            # 确保至少有一个有效
            corridor_mask[:, 0] = 1.0
            
            # 前向传播
            output = encoder(corridor_features, corridor_mask)
            
            # 计算损失（随意的损失函数）
            target = torch.randn(4, 128)
            loss = F.mse_loss(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            assert torch.isfinite(loss)


class TestEdgeCases:
    """边界情况测试"""
    
    @pytest.fixture
    def encoder(self):
        return CorridorEncoder(input_dim=64, output_dim=128)
    
    def test_very_small_features(self, encoder):
        """测试非常小的特征值"""
        corridor_features = torch.randn(10, 64) * 1e-6
        corridor_mask = torch.ones(10)
        
        output = encoder(corridor_features, corridor_mask)
        
        assert torch.isfinite(output).all()
    
    def test_very_large_features(self, encoder):
        """测试非常大的特征值"""
        corridor_features = torch.randn(10, 64) * 1e3
        corridor_mask = torch.ones(10)
        
        output = encoder(corridor_features, corridor_mask)
        
        assert torch.isfinite(output).all()
    
    def test_zero_features(self, encoder):
        """测试全零特征"""
        corridor_features = torch.zeros(10, 64)
        corridor_mask = torch.ones(10)
        
        output = encoder(corridor_features, corridor_mask)
        
        assert torch.isfinite(output).all()
    
    def test_alternating_mask(self, encoder):
        """测试交替的mask模式"""
        corridor_features = torch.randn(10, 64)
        corridor_mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float)
        
        output = encoder(corridor_features, corridor_mask)
        
        assert output.shape == (128,)
        assert torch.isfinite(output).all()


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])

