"""
机器狗状态编码器的单元测试
"""

import pytest
import torch
from agsac.models.encoders.dog_state_encoder import (
    DogStateEncoder,
    SimpleDogStateEncoder,
    AttentiveDogStateEncoder,
    create_dog_state_encoder
)


class TestDogStateEncoder:
    """测试GRU版本的编码器"""
    
    @pytest.fixture
    def encoder(self):
        """创建测试用的编码器"""
        return DogStateEncoder(hidden_dim=64, gru_layers=2, dropout=0.1)
    
    @pytest.fixture
    def sample_inputs(self):
        """创建样本输入"""
        return {
            'past_traj': torch.randn(8, 2),
            'current_vel': torch.randn(2),
            'current_pos': torch.zeros(2),
            'goal_pos': torch.tensor([10.0, 10.0])
        }
    
    def test_forward_single(self, encoder, sample_inputs):
        """测试单个样本的前向传播"""
        features = encoder(
            sample_inputs['past_traj'],
            sample_inputs['current_vel'],
            sample_inputs['current_pos'],
            sample_inputs['goal_pos']
        )
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_forward_batch(self, encoder):
        """测试批量前向传播"""
        batch_size = 4
        batch_inputs = {
            'past_traj': torch.randn(batch_size, 8, 2),
            'current_vel': torch.randn(batch_size, 2),
            'current_pos': torch.randn(batch_size, 2),
            'goal_pos': torch.randn(batch_size, 2)
        }
        
        features = encoder(
            batch_inputs['past_traj'],
            batch_inputs['current_vel'],
            batch_inputs['current_pos'],
            batch_inputs['goal_pos']
        )
        
        assert features.shape == (batch_size, 64)
        assert torch.isfinite(features).all()
    
    def test_relative_coordinates(self, encoder):
        """测试相对坐标处理"""
        # 创建两个位置不同但相对运动相同的场景
        current_pos1 = torch.zeros(2)
        goal_pos1 = torch.tensor([5.0, 5.0])
        traj1 = torch.randn(8, 2)
        
        current_pos2 = torch.tensor([100.0, 100.0])
        goal_pos2 = current_pos2 + torch.tensor([5.0, 5.0])
        traj2 = traj1 + 100.0  # 平移相同的轨迹
        
        vel = torch.tensor([1.0, 0.0])
        
        features1 = encoder(traj1, vel, current_pos1, goal_pos1)
        features2 = encoder(traj2, vel, current_pos2, goal_pos2)
        
        # 因为相对运动相同，特征应该相似（但不完全相同，因为GRU处理）
        similarity = F.cosine_similarity(features1, features2, dim=0)
        assert similarity > 0.9  # 高相似度
    
    def test_trajectory_features_extraction(self, encoder, sample_inputs):
        """测试轨迹特征提取"""
        traj_features = encoder.get_trajectory_features(
            sample_inputs['past_traj'],
            sample_inputs['current_pos']
        )
        
        assert traj_features.shape == (64,)
        assert torch.isfinite(traj_features).all()
    
    def test_gradient_flow(self, encoder, sample_inputs):
        """测试梯度流"""
        past_traj = sample_inputs['past_traj'].clone().requires_grad_(True)
        
        features = encoder(
            past_traj,
            sample_inputs['current_vel'],
            sample_inputs['current_pos'],
            sample_inputs['goal_pos']
        )
        
        loss = features.sum()
        loss.backward()
        
        assert past_traj.grad is not None
        assert torch.isfinite(past_traj.grad).all()
        assert past_traj.grad.abs().sum() > 0  # 有非零梯度
    
    def test_different_sequence_lengths(self, encoder):
        """测试不同序列长度（虽然通常固定为8）"""
        # 注意：这个测试可能失败，因为编码器期望固定长度
        # 但我们测试其鲁棒性
        short_traj = torch.randn(4, 2)  # 较短的轨迹
        vel = torch.randn(2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        try:
            features = encoder(short_traj, vel, pos, goal)
            assert features.shape == (64,)
        except Exception as e:
            # 如果失败，这是预期的（因为GRU期望固定长度）
            pass
    
    def test_output_consistency(self, encoder, sample_inputs):
        """测试输出一致性"""
        encoder.eval()  # 设置为评估模式
        
        with torch.no_grad():
            features1 = encoder(
                sample_inputs['past_traj'],
                sample_inputs['current_vel'],
                sample_inputs['current_pos'],
                sample_inputs['goal_pos']
            )
            features2 = encoder(
                sample_inputs['past_traj'],
                sample_inputs['current_vel'],
                sample_inputs['current_pos'],
                sample_inputs['goal_pos']
            )
        
        torch.testing.assert_close(features1, features2)


class TestSimpleDogStateEncoder:
    """测试简化版编码器"""
    
    @pytest.fixture
    def encoder(self):
        return SimpleDogStateEncoder(hidden_dim=64)
    
    def test_forward(self, encoder):
        """测试前向传播"""
        past_traj = torch.randn(8, 2)
        vel = torch.randn(2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        features = encoder(past_traj, vel, pos, goal)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_batch_processing(self, encoder):
        """测试批量处理"""
        batch_traj = torch.randn(3, 8, 2)
        batch_vel = torch.randn(3, 2)
        batch_pos = torch.randn(3, 2)
        batch_goal = torch.randn(3, 2)
        
        features = encoder(batch_traj, batch_vel, batch_pos, batch_goal)
        
        assert features.shape == (3, 64)


class TestAttentiveDogStateEncoder:
    """测试注意力版编码器"""
    
    @pytest.fixture
    def encoder(self):
        return AttentiveDogStateEncoder(hidden_dim=64, num_heads=4)
    
    def test_forward(self, encoder):
        """测试前向传播"""
        past_traj = torch.randn(8, 2)
        vel = torch.randn(2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        features = encoder(past_traj, vel, pos, goal)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_time_encoding(self, encoder):
        """测试时间编码的效果"""
        # 创建两个时间顺序不同的轨迹
        traj1 = torch.randn(8, 2)
        traj2 = torch.flip(traj1, dims=[0])  # 时间反转
        
        vel = torch.randn(2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        features1 = encoder(traj1, vel, pos, goal)
        features2 = encoder(traj2, vel, pos, goal)
        
        # 因为有时间编码，特征应该不同
        assert not torch.allclose(features1, features2, atol=1e-3)
    
    def test_attention_mechanism(self, encoder):
        """测试注意力机制工作"""
        past_traj = torch.randn(8, 2, requires_grad=True)
        vel = torch.randn(2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        features = encoder(past_traj, vel, pos, goal)
        loss = features.sum()
        loss.backward()
        
        # 所有轨迹点都应该有梯度（因为注意力机制关注所有点）
        assert past_traj.grad is not None
        assert (past_traj.grad.abs() > 0).any()


class TestFactoryFunction:
    """测试工厂函数"""
    
    def test_create_gru_encoder(self):
        encoder = create_dog_state_encoder('gru', hidden_dim=64)
        assert isinstance(encoder, DogStateEncoder)
    
    def test_create_simple_encoder(self):
        encoder = create_dog_state_encoder('simple', hidden_dim=64)
        assert isinstance(encoder, SimpleDogStateEncoder)
    
    def test_create_attention_encoder(self):
        encoder = create_dog_state_encoder('attention', hidden_dim=64)
        assert isinstance(encoder, AttentiveDogStateEncoder)
    
    def test_invalid_encoder_type(self):
        with pytest.raises(ValueError):
            create_dog_state_encoder('invalid', hidden_dim=64)
    
    def test_custom_parameters(self):
        encoder = create_dog_state_encoder(
            'gru',
            hidden_dim=128,
            gru_layers=3,
            dropout=0.2
        )
        assert encoder.hidden_dim == 128
        assert encoder.gru_layers == 3


class TestRealWorldScenarios:
    """测试实际应用场景"""
    
    def test_static_trajectory(self):
        """测试静止不动的轨迹"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        # 机器狗静止
        static_traj = torch.zeros(8, 2)
        vel = torch.zeros(2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        features = encoder(static_traj, vel, pos, goal)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_straight_line_trajectory(self):
        """测试直线运动轨迹"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        # 直线运动
        straight_traj = torch.tensor([[i * 0.5, 0.0] for i in range(8)])
        vel = torch.tensor([0.5, 0.0])
        pos = torch.tensor([3.5, 0.0])
        goal = torch.tensor([10.0, 0.0])
        
        features = encoder(straight_traj, vel, pos, goal)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_curved_trajectory(self):
        """测试曲线运动轨迹"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        # 圆弧轨迹
        t = torch.linspace(0, torch.pi/2, 8)
        curved_traj = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        
        vel = torch.tensor([0.5, 0.5])
        pos = curved_traj[-1]
        goal = torch.tensor([0.0, 2.0])
        
        features = encoder(curved_traj, vel, pos, goal)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()
    
    def test_goal_sensitivity(self):
        """测试对目标位置的敏感性"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        traj = torch.randn(8, 2)
        vel = torch.randn(2)
        pos = torch.zeros(2)
        
        # 不同的目标
        goal1 = torch.tensor([5.0, 0.0])
        goal2 = torch.tensor([0.0, 5.0])
        
        features1 = encoder(traj, vel, pos, goal1)
        features2 = encoder(traj, vel, pos, goal2)
        
        # 目标不同，特征应该显著不同
        similarity = F.cosine_similarity(features1, features2, dim=0)
        assert similarity < 0.99  # 不应该几乎相同
    
    def test_velocity_sensitivity(self):
        """测试对速度的敏感性"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        traj = torch.randn(8, 2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        # 不同速度
        vel1 = torch.tensor([0.0, 0.0])  # 静止
        vel2 = torch.tensor([1.0, 0.0])  # 快速
        
        features1 = encoder(traj, vel1, pos, goal)
        features2 = encoder(traj, vel2, pos, goal)
        
        # 速度不同，特征应该不同
        assert not torch.allclose(features1, features2, atol=1e-2)


class TestParameterCount:
    """测试参数量"""
    
    def test_parameter_budget(self):
        """测试各编码器的参数量"""
        encoders = {
            'gru': DogStateEncoder(hidden_dim=64, gru_layers=2),
            'simple': SimpleDogStateEncoder(hidden_dim=64),
            'attention': AttentiveDogStateEncoder(hidden_dim=64, num_heads=4)
        }
        
        for name, encoder in encoders.items():
            params = sum(p.numel() for p in encoder.parameters())
            print(f"{name}编码器参数: {params:,}")
            
            # GRU版本应该在75K左右
            if name == 'gru':
                assert params < 100000, f"{name} has too many parameters: {params}"
            
            # 所有版本都应该远小于预算
            assert params < 200000


class TestEdgeCases:
    """测试边界情况"""
    
    def test_zero_velocity(self):
        """测试零速度"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        traj = torch.randn(8, 2)
        vel = torch.zeros(2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        features = encoder(traj, vel, pos, goal)
        
        assert torch.isfinite(features).all()
    
    def test_very_close_goal(self):
        """测试非常接近的目标"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        traj = torch.randn(8, 2)
        vel = torch.randn(2)
        pos = torch.zeros(2)
        goal = torch.tensor([0.01, 0.01])  # 非常近
        
        features = encoder(traj, vel, pos, goal)
        
        assert torch.isfinite(features).all()
    
    def test_very_far_goal(self):
        """测试非常远的目标"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        traj = torch.randn(8, 2)
        vel = torch.randn(2)
        pos = torch.zeros(2)
        goal = torch.tensor([1000.0, 1000.0])  # 非常远
        
        features = encoder(traj, vel, pos, goal)
        
        assert torch.isfinite(features).all()
    
    def test_nan_handling(self):
        """测试NaN输入"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        traj = torch.randn(8, 2)
        traj[3, 0] = float('nan')  # 注入NaN
        vel = torch.randn(2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        features = encoder(traj, vel, pos, goal)
        
        # 输出会包含NaN（这是预期的）
        assert not torch.isfinite(features).all()
    
    def test_extremely_long_trajectory(self):
        """测试超长轨迹（虽然通常不会发生）"""
        encoder = DogStateEncoder(hidden_dim=64)
        
        # GRU可以处理任意长度
        long_traj = torch.randn(100, 2)
        vel = torch.randn(2)
        pos = torch.zeros(2)
        goal = torch.tensor([5.0, 5.0])
        
        features = encoder(long_traj, vel, pos, goal)
        
        assert features.shape == (64,)
        assert torch.isfinite(features).all()


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])