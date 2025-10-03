"""
PointNet编码器的单元测试
"""

import pytest
import torch
from agsac.models.encoders.pointnet import (
    PointNet,
    PointNetEncoder,
    AdaptivePointNetEncoder,
    create_pointnet_encoder
)


class TestPointNet:
    """测试基础PointNet"""
    
    @pytest.fixture
    def pointnet(self):
        """创建测试用的PointNet"""
        model = PointNet(input_dim=2, feature_dim=64, hidden_dims=[64, 128, 256])
        model.eval()  # 设置为评估模式，避免BatchNorm的batch_size=1问题
        return model
    
    def test_forward_single_pointcloud(self, pointnet):
        """测试单个点云的前向传播"""
        points = torch.randn(10, 2)
        features = pointnet(points)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_forward_batch(self, pointnet):
        """测试批量点云"""
        batch_points = torch.randn(4, 10, 2)
        batch_features = pointnet(batch_points)
        
        assert batch_features.shape == (4, 64)
        assert torch.isfinite(batch_features).all()
    
    def test_permutation_invariance(self, pointnet):
        """测试置换不变性"""
        points = torch.randn(10, 2)
        
        # 原始顺序
        features1 = pointnet(points)
        
        # 随机打乱
        perm = torch.randperm(10)
        shuffled_points = points[perm]
        features2 = pointnet(shuffled_points)
        
        # 应该得到相同的特征
        torch.testing.assert_close(features1, features2, atol=1e-5, rtol=1e-5)
    
    def test_different_num_points(self, pointnet):
        """测试不同数量的点"""
        # 少量点
        few_points = torch.randn(5, 2)
        features_few = pointnet(few_points)
        assert features_few.shape == (64,)
        
        # 大量点
        many_points = torch.randn(100, 2)
        features_many = pointnet(many_points)
        assert features_many.shape == (64,)
    
    def test_gradient_flow(self, pointnet):
        """测试梯度流"""
        points = torch.randn(10, 2, requires_grad=True)
        features = pointnet(points)
        
        # 反向传播
        loss = features.sum()
        loss.backward()
        
        # 检查梯度
        assert points.grad is not None
        assert torch.isfinite(points.grad).all()
    
    def test_output_consistency(self, pointnet):
        """测试输出一致性"""
        points = torch.randn(10, 2)
        
        # 多次前向传播应该得到相同结果
        features1 = pointnet(points)
        features2 = pointnet(points)
        
        torch.testing.assert_close(features1, features2)


class TestPointNetEncoder:
    """测试增强版PointNetEncoder"""
    
    @pytest.fixture
    def encoder(self):
        """创建测试用的编码器"""
        model = PointNetEncoder(feature_dim=64, use_relative_coords=True)
        model.eval()  # 设置为评估模式
        return model
    
    def test_forward_with_reference(self, encoder):
        """测试使用参考点的前向传播"""
        vertices = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        reference = torch.zeros(2)
        
        features = encoder(vertices, reference)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_forward_without_reference(self, encoder):
        """测试不提供参考点"""
        vertices = torch.randn(5, 2)
        features = encoder(vertices)
        
        assert features.shape == (64,)
    
    def test_batch_processing(self, encoder):
        """测试批量处理"""
        batch_vertices = torch.randn(3, 8, 2)
        batch_reference = torch.randn(3, 2)
        
        batch_features = encoder(batch_vertices, batch_reference)
        
        assert batch_features.shape == (3, 64)
    
    def test_relative_features_computation(self, encoder):
        """测试相对特征计算"""
        # 正方形，质心在(0.5, 0.5)
        vertices = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        reference = torch.zeros(2)
        
        # 计算相对特征
        relative_features = encoder._compute_relative_features(vertices, reference)
        
        # 检查形状（原始2 + 相对参考2 + 相对质心2 = 6）
        assert relative_features.shape == (4, 6)
        
        # 检查相对质心的坐标
        # 质心是(0.5, 0.5)，所以相对质心应该是对称的
        relative_to_centroid = relative_features[:, 4:]  # 最后2维
        assert torch.allclose(relative_to_centroid.mean(dim=0), torch.zeros(2), atol=1e-5)
    
    def test_permutation_invariance(self, encoder):
        """测试置换不变性"""
        vertices = torch.randn(6, 2)
        reference = torch.zeros(2)
        
        features1 = encoder(vertices, reference)
        
        # 打乱顺序
        perm = torch.randperm(6)
        shuffled_vertices = vertices[perm]
        features2 = encoder(shuffled_vertices, reference)
        
        torch.testing.assert_close(features1, features2, atol=1e-5, rtol=1e-5)
    
    def test_different_shapes(self, encoder):
        """测试不同形状的多边形"""
        # 三角形
        triangle = torch.randn(3, 2)
        features_tri = encoder(triangle)
        assert features_tri.shape == (64,)
        
        # 八边形
        octagon = torch.randn(8, 2)
        features_oct = encoder(octagon)
        assert features_oct.shape == (64,)


class TestAdaptivePointNetEncoder:
    """测试自适应编码器"""
    
    @pytest.fixture
    def adaptive_encoder(self):
        """创建自适应编码器"""
        model = AdaptivePointNetEncoder(
            feature_dim=64,
            min_points=3,
            max_points=50
        )
        model.eval()  # 设置为评估模式
        return model
    
    def test_handle_few_points(self, adaptive_encoder):
        """测试处理少量点"""
        # 只有2个点（少于min_points=3）
        few_points = torch.randn(2, 2)
        features = adaptive_encoder(few_points)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_handle_many_points(self, adaptive_encoder):
        """测试处理大量点"""
        # 100个点（超过max_points=50）
        many_points = torch.randn(100, 2)
        features = adaptive_encoder(many_points)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_handle_normal_points(self, adaptive_encoder):
        """测试正常数量的点"""
        normal_points = torch.randn(10, 2)
        features = adaptive_encoder(normal_points)
        
        assert features.shape == (64,)
    
    def test_extreme_cases_adjustment(self, adaptive_encoder):
        """测试极端情况的调整"""
        # 测试调整逻辑
        # 少量点
        few = torch.randn(2, 2)
        adjusted_few = adaptive_encoder._handle_extreme_cases(few)
        assert adjusted_few.size(0) >= adaptive_encoder.min_points
        
        # 大量点
        many = torch.randn(100, 2)
        adjusted_many = adaptive_encoder._handle_extreme_cases(many)
        assert adjusted_many.size(0) <= adaptive_encoder.max_points


class TestFactoryFunction:
    """测试工厂函数"""
    
    def test_create_basic_encoder(self):
        """测试创建基础编码器"""
        encoder = create_pointnet_encoder(feature_dim=64, encoder_type='basic')
        assert isinstance(encoder, PointNet)
    
    def test_create_enhanced_encoder(self):
        """测试创建增强编码器"""
        encoder = create_pointnet_encoder(feature_dim=64, encoder_type='enhanced')
        assert isinstance(encoder, PointNetEncoder)
    
    def test_create_adaptive_encoder(self):
        """测试创建自适应编码器"""
        encoder = create_pointnet_encoder(feature_dim=64, encoder_type='adaptive')
        assert isinstance(encoder, AdaptivePointNetEncoder)
    
    def test_invalid_encoder_type(self):
        """测试无效的编码器类型"""
        with pytest.raises(ValueError):
            create_pointnet_encoder(feature_dim=64, encoder_type='invalid')


class TestRealWorldScenarios:
    """测试实际应用场景"""
    
    def test_corridor_encoding(self):
        """测试通路编码场景"""
        encoder = PointNetEncoder(feature_dim=64, use_relative_coords=True)
        encoder.eval()
        
        # 模拟多个通路
        corridors = [
            torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]),  # 矩形
            torch.tensor([[3.0, 0.0], [5.0, 0.0], [4.0, 2.0]]),               # 三角形
            torch.tensor([[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0], [0.5, 3.0]])  # 五边形
        ]
        
        robot_pos = torch.tensor([1.0, 1.0])
        
        # 编码所有通路
        corridor_features = []
        for corridor in corridors:
            feat = encoder(corridor, robot_pos)
            corridor_features.append(feat)
            assert feat.shape == (64,)
        
        # 所有特征应该不同
        for i in range(len(corridor_features)):
            for j in range(i + 1, len(corridor_features)):
                assert not torch.allclose(corridor_features[i], corridor_features[j])
    
    def test_variable_corridor_sizes(self):
        """测试可变大小的通路"""
        encoder = PointNetEncoder(feature_dim=64)
        encoder.eval()
        
        # 不同顶点数的多边形
        sizes = [3, 4, 5, 6, 8, 10, 20]
        
        for size in sizes:
            vertices = torch.randn(size, 2)
            features = encoder(vertices)
            assert features.shape == (64,), f"Failed for size {size}"
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        encoder = PointNetEncoder(feature_dim=64)
        encoder.eval()
        
        # 非常小的坐标
        tiny_vertices = torch.randn(5, 2) * 1e-6
        features_tiny = encoder(tiny_vertices)
        assert torch.isfinite(features_tiny).all()
        
        # 非常大的坐标
        huge_vertices = torch.randn(5, 2) * 1e6
        features_huge = encoder(huge_vertices)
        assert torch.isfinite(features_huge).all()
    
    def test_colinear_points(self):
        """测试共线点"""
        encoder = PointNetEncoder(feature_dim=64)
        encoder.eval()
        
        # 共线点（在x轴上）
        colinear = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0]
        ])
        
        features = encoder(colinear)
        assert features.shape == (64,)
        assert torch.isfinite(features).all()


class TestParameterCount:
    """测试参数量"""
    
    def test_parameter_budget(self):
        """测试参数量在预算内"""
        encoder = PointNetEncoder(feature_dim=64)
        
        total_params = sum(p.numel() for p in encoder.parameters())
        
        # PointNet编码器应该较小（<150K参数）
        # 实际约117K，这是合理的（用于走廊几何编码）
        assert total_params < 150000, f"Too many parameters: {total_params}"
        
        print(f"PointNet参数量: {total_params:,}")


class TestEdgeCases:
    """测试边界情况"""
    
    def test_single_point(self):
        """测试单点（退化的多边形）"""
        encoder = AdaptivePointNetEncoder(feature_dim=64, min_points=3)
        encoder.eval()
        
        single_point = torch.randn(1, 2)
        features = encoder(single_point)
        
        assert features.shape == (64,)
    
    def test_duplicate_points(self):
        """测试重复点"""
        encoder = PointNetEncoder(feature_dim=64)
        encoder.eval()
        
        # 所有点相同
        duplicate = torch.ones(5, 2)
        features = encoder(duplicate)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_zero_points(self):
        """测试全零点"""
        encoder = PointNetEncoder(feature_dim=64)
        encoder.eval()
        
        zeros = torch.zeros(4, 2)
        features = encoder(zeros)
        
        assert features.shape == (64,)
    
    def test_nan_handling(self):
        """测试NaN输入（应该被检测到）"""
        encoder = PointNetEncoder(feature_dim=64)
        encoder.eval()
        
        # 包含NaN的输入
        vertices = torch.randn(5, 2)
        vertices[2, 0] = float('nan')
        
        # 前向传播会产生NaN
        features = encoder(vertices)
        
        # 输出应该包含NaN（这是预期的，因为输入有问题）
        assert not torch.isfinite(features).all()


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])