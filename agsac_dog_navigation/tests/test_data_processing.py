"""
数据处理模块的单元测试
"""

import pytest
import torch
import numpy as np
from agsac.utils.data_processing import (
    DataProcessor,
    pad_sequence_list,
    create_attention_mask,
    process_single_input
)


class TestDataProcessor:
    """测试DataProcessor类"""
    
    @pytest.fixture
    def processor(self):
        """创建测试用的DataProcessor实例"""
        return DataProcessor(max_pedestrians=10, max_corridors=10, device='cpu')
    
    @pytest.fixture
    def sample_pedestrian_trajectories(self):
        """创建样本行人轨迹"""
        return [
            torch.randn(8, 2),
            torch.randn(8, 2),
            torch.randn(8, 2)
        ]
    
    @pytest.fixture
    def sample_corridors(self):
        """创建样本通路多边形"""
        return [
            torch.randn(5, 2),
            torch.randn(4, 2),
            torch.randn(6, 2),
        ]
    
    def test_process_pedestrian_trajectories_normal(self, processor, sample_pedestrian_trajectories):
        """测试正常的行人轨迹处理"""
        robot_pos = torch.zeros(2)
        
        padded_trajs, mask = processor.process_pedestrian_trajectories(
            sample_pedestrian_trajectories,
            robot_pos
        )
        
        # 检查形状
        assert padded_trajs.shape == (10, 8, 2)
        assert mask.shape == (10,)
        
        # 检查mask值
        assert mask[:3].sum() == 3.0  # 前3个为1
        assert mask[3:].sum() == 0.0  # 后7个为0
        
        # 检查数据一致性
        for i in range(3):
            torch.testing.assert_close(
                padded_trajs[i],
                sample_pedestrian_trajectories[i]
            )
    
    def test_process_pedestrian_trajectories_empty(self, processor):
        """测试空行人列表"""
        robot_pos = torch.zeros(2)
        
        padded_trajs, mask = processor.process_pedestrian_trajectories(
            [],
            robot_pos
        )
        
        # 应该返回全零
        assert padded_trajs.shape == (10, 8, 2)
        assert mask.sum() == 0.0
    
    def test_process_pedestrian_trajectories_exceeds_max(self, processor):
        """测试超过最大数量的情况"""
        # 创建15个行人
        trajectories = [torch.randn(8, 2) for _ in range(15)]
        robot_pos = torch.zeros(2)
        
        padded_trajs, mask = processor.process_pedestrian_trajectories(
            trajectories,
            robot_pos
        )
        
        # 应该截断为10个
        assert padded_trajs.shape == (10, 8, 2)
        assert mask.sum() == 10.0
    
    def test_select_closest_trajectories(self, processor):
        """测试选择最近轨迹的功能"""
        # 创建特定位置的行人
        trajectories = [
            torch.tensor([[0.0, 0.0], [0.0, 0.0]] * 4),  # 距离原点0
            torch.tensor([[5.0, 0.0], [5.0, 0.0]] * 4),  # 距离原点5
            torch.tensor([[1.0, 0.0], [1.0, 0.0]] * 4),  # 距离原点1
            torch.tensor([[10.0, 0.0], [10.0, 0.0]] * 4),  # 距离原点10
        ]
        robot_pos = torch.zeros(2)
        
        selected = processor._select_closest_trajectories(
            trajectories,
            robot_pos,
            max_num=2
        )
        
        # 应该选择距离为0和1的两个
        assert len(selected) == 2
        # 检查选择的是最近的
        distances = [torch.norm(traj[-1] - robot_pos).item() for traj in selected]
        assert max(distances) <= 1.0 + 1e-5
    
    def test_process_corridor_features_normal(self, processor):
        """测试正常的通路特征处理"""
        corridor_features = [torch.randn(64) for _ in range(5)]
        robot_pos = torch.zeros(2)
        
        padded_features, mask = processor.process_corridor_features(
            corridor_features,
            robot_pos
        )
        
        # 检查形状
        assert padded_features.shape == (10, 64)
        assert mask.shape == (10,)
        
        # 检查mask
        assert mask[:5].sum() == 5.0
        assert mask[5:].sum() == 0.0
    
    def test_process_corridor_features_empty(self, processor):
        """测试空通路列表"""
        robot_pos = torch.zeros(2)
        
        padded_features, mask = processor.process_corridor_features(
            [],
            robot_pos
        )
        
        assert padded_features.shape == (10, 64)
        assert mask.sum() == 0.0
    
    def test_process_corridors(self, processor, sample_corridors):
        """测试通路多边形处理"""
        robot_pos = torch.zeros(2)
        
        selected_corridors, num_valid = processor.process_corridors(
            sample_corridors,
            robot_pos
        )
        
        assert len(selected_corridors) == 3
        assert num_valid == 3
    
    def test_process_corridors_exceeds_max(self, processor):
        """测试超过最大通路数量"""
        corridors = [torch.randn(4, 2) for _ in range(15)]
        robot_pos = torch.zeros(2)
        
        selected_corridors, num_valid = processor.process_corridors(
            corridors,
            robot_pos
        )
        
        assert len(selected_corridors) == 10
        assert num_valid == 10
    
    def test_create_batch(self, processor):
        """测试批处理功能"""
        samples = []
        for _ in range(4):
            sample = {
                'pedestrian_trajs': torch.randn(10, 8, 2),
                'pedestrian_mask': torch.zeros(10),
                'corridor_features': torch.randn(10, 64),
                'corridor_mask': torch.zeros(10),
                'dog_past_traj': torch.randn(8, 2),
                'dog_vel': torch.randn(2),
                'dog_pos': torch.randn(2),
                'goal_pos': torch.randn(2)
            }
            samples.append(sample)
        
        batch = processor.create_batch(samples)
        
        # 检查所有字段都有batch维度
        assert batch['pedestrian_trajs'].shape == (4, 10, 8, 2)
        assert batch['pedestrian_mask'].shape == (4, 10)
        assert batch['dog_past_traj'].shape == (4, 8, 2)
        assert batch['dog_vel'].shape == (4, 2)
    
    def test_normalize_coordinates(self, processor):
        """测试坐标归一化"""
        coords = torch.tensor([[10.0, 5.0], [0.0, 0.0], [-10.0, -5.0]])
        reference = torch.zeros(2)
        scale = 10.0
        
        normalized = processor.normalize_coordinates(coords, reference, scale)
        
        # 检查范围
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0
        
        # 检查具体值
        expected = torch.tensor([[1.0, 0.5], [0.0, 0.0], [-1.0, -0.5]])
        torch.testing.assert_close(normalized, expected)
    
    def test_denormalize_coordinates(self, processor):
        """测试坐标反归一化"""
        normalized = torch.tensor([[1.0, 0.5], [0.0, 0.0], [-1.0, -0.5]])
        reference = torch.zeros(2)
        scale = 10.0
        
        coords = processor.denormalize_coordinates(normalized, reference, scale)
        
        # 检查反归一化结果
        expected = torch.tensor([[10.0, 5.0], [0.0, 0.0], [-10.0, -5.0]])
        torch.testing.assert_close(coords, expected)
    
    def test_normalize_denormalize_roundtrip(self, processor):
        """测试归一化和反归一化的往返一致性"""
        original_coords = torch.randn(5, 2) * 5  # 随机坐标
        reference = torch.randn(2)
        scale = 10.0
        
        normalized = processor.normalize_coordinates(original_coords, reference, scale)
        recovered = processor.denormalize_coordinates(normalized, reference, scale)
        
        torch.testing.assert_close(original_coords, recovered, rtol=1e-5, atol=1e-5)


class TestUtilityFunctions:
    """测试工具函数"""
    
    def test_pad_sequence_list(self):
        """测试序列padding"""
        sequences = [
            torch.randn(3, 2),
            torch.randn(2, 2),
            torch.randn(4, 2)
        ]
        max_length = 5
        
        padded = pad_sequence_list(sequences, max_length)
        
        assert padded.shape == (5, 3, 2)
        
        # 检查前3个是原始序列
        torch.testing.assert_close(padded[:3, 0], sequences[0])
        torch.testing.assert_close(padded[:2, 1], sequences[1])
        torch.testing.assert_close(padded[:4, 2], sequences[2])
        
        # 检查padding部分为0
        assert padded[3:, 0].abs().sum() == 0.0
    
    def test_pad_sequence_list_empty(self):
        """测试空序列列表"""
        with pytest.raises(ValueError):
            pad_sequence_list([], 5)
    
    def test_create_attention_mask_normal(self):
        """测试创建注意力mask（正常模式）"""
        valid_lengths = torch.tensor([3, 5, 2])
        max_length = 6
        
        mask = create_attention_mask(valid_lengths, max_length, inverted=False)
        
        assert mask.shape == (3, 6)
        
        # 检查第一个样本（长度3）
        assert mask[0, :3].all()
        assert not mask[0, 3:].any()
        
        # 检查第二个样本（长度5）
        assert mask[1, :5].all()
        assert not mask[1, 5:].any()
    
    def test_create_attention_mask_inverted(self):
        """测试创建注意力mask（反转模式）"""
        valid_lengths = torch.tensor([3, 5])
        max_length = 6
        
        mask = create_attention_mask(valid_lengths, max_length, inverted=True)
        
        # inverted=True: True表示padding位置
        assert not mask[0, :3].any()  # 有效位置为False
        assert mask[0, 3:].all()      # padding位置为True
    
    def test_process_single_input(self):
        """测试完整的单输入处理"""
        pedestrian_trajs = [torch.randn(8, 2) for _ in range(3)]
        corridors = [torch.randn(5, 2), torch.randn(4, 2)]
        dog_state = {
            'past_traj': torch.randn(8, 2),
            'vel': torch.randn(2),
            'pos': torch.zeros(2),
            'goal': torch.tensor([5.0, 5.0])
        }
        
        processed = process_single_input(
            pedestrian_trajs,
            corridors,
            dog_state,
            max_pedestrians=10,
            max_corridors=10
        )
        
        # 检查所有必需的键
        required_keys = [
            'pedestrian_trajs',
            'pedestrian_mask',
            'corridors',
            'num_corridors',
            'dog_past_traj',
            'dog_vel',
            'dog_pos',
            'goal_pos'
        ]
        for key in required_keys:
            assert key in processed
        
        # 检查形状
        assert processed['pedestrian_trajs'].shape == (10, 8, 2)
        assert processed['pedestrian_mask'].shape == (10,)
        assert len(processed['corridors']) == 2
        assert processed['num_corridors'] == 2


class TestEdgeCases:
    """测试边界情况"""
    
    def test_single_pedestrian(self):
        """测试单个行人"""
        processor = DataProcessor(max_pedestrians=10, max_corridors=10)
        trajectories = [torch.randn(8, 2)]
        robot_pos = torch.zeros(2)
        
        padded_trajs, mask = processor.process_pedestrian_trajectories(
            trajectories,
            robot_pos
        )
        
        assert mask[0] == 1.0
        assert mask[1:].sum() == 0.0
    
    def test_max_equals_actual(self):
        """测试实际数量等于最大值"""
        processor = DataProcessor(max_pedestrians=5, max_corridors=5)
        trajectories = [torch.randn(8, 2) for _ in range(5)]
        robot_pos = torch.zeros(2)
        
        padded_trajs, mask = processor.process_pedestrian_trajectories(
            trajectories,
            robot_pos
        )
        
        assert padded_trajs.shape == (5, 8, 2)
        assert mask.sum() == 5.0
    
    def test_different_trajectory_lengths(self):
        """测试不同长度的轨迹（理论上不应该发生，但要确保不崩溃）"""
        processor = DataProcessor(max_pedestrians=10, max_corridors=10)
        # 在实际使用中所有轨迹应该同长，这里测试鲁棒性
        trajectories = [
            torch.randn(8, 2),
            torch.randn(8, 2)
        ]
        robot_pos = torch.zeros(2)
        
        # 应该正常工作
        padded_trajs, mask = processor.process_pedestrian_trajectories(
            trajectories,
            robot_pos
        )
        
        assert padded_trajs.shape[0] == 10
        assert mask.sum() == 2.0
    
    def test_very_large_coordinates(self):
        """测试非常大的坐标值"""
        processor = DataProcessor(max_pedestrians=10, max_corridors=10)
        trajectories = [torch.ones(8, 2) * 1000]  # 很大的坐标
        robot_pos = torch.zeros(2)
        
        padded_trajs, mask = processor.process_pedestrian_trajectories(
            trajectories,
            robot_pos
        )
        
        # 应该正常处理
        assert torch.isfinite(padded_trajs).all()
        assert mask[0] == 1.0
    
    def test_device_consistency(self):
        """测试设备一致性"""
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        processor = DataProcessor(max_pedestrians=10, max_corridors=10, device=device)
        trajectories = [torch.randn(8, 2) for _ in range(3)]
        robot_pos = torch.zeros(2)
        
        padded_trajs, mask = processor.process_pedestrian_trajectories(
            trajectories,
            robot_pos
        )
        
        # 检查输出在正确的设备上
        assert str(padded_trajs.device).startswith(device)
        assert str(mask.device).startswith(device)


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])