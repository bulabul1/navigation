"""
SequenceReplayBuffer单元测试
"""

import pytest
import torch
import random
from agsac.training import SequenceReplayBuffer


def create_dummy_episode(length: int, hidden_dim: int = 128):
    """创建测试用的episode数据"""
    return {
        'fused_states': [torch.randn(64) for _ in range(length)],
        'actions': [torch.randn(22) for _ in range(length)],
        'rewards': [random.random() for _ in range(length)],
        'dones': [False] * (length - 1) + [True],
        'hidden_states': [
            {
                'actor': (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim)),
                'critic1': (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim)),
                'critic2': (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
            }
            for _ in range(length)
        ],
        'episode_return': random.uniform(50, 150),
        'episode_length': length
    }


def test_buffer_initialization():
    """测试buffer初始化"""
    buffer = SequenceReplayBuffer(capacity=100, seq_len=16, burn_in=4)
    
    assert buffer.capacity == 100
    assert buffer.seq_len == 16
    assert buffer.burn_in == 4
    assert len(buffer) == 0
    assert buffer.size() == 0


def test_add_episode():
    """测试添加episode"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16)
    
    # 添加一个episode
    episode = create_dummy_episode(50)
    buffer.add_episode(episode)
    
    assert len(buffer) == 1
    
    # 添加多个episodes
    for _ in range(5):
        buffer.add_episode(create_dummy_episode(40))
    
    assert len(buffer) == 6


def test_skip_short_episode():
    """测试跳过太短的episode"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16)
    
    # 添加一个太短的episode
    short_episode = create_dummy_episode(10)
    buffer.add_episode(short_episode)
    
    # 应该被跳过
    assert len(buffer) == 0


def test_sample_segments():
    """测试采样segments"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16, burn_in=4)
    
    # 添加episodes
    for _ in range(5):
        buffer.add_episode(create_dummy_episode(50))
    
    # 采样
    batch_size = 4
    segments = buffer.sample(batch_size)
    
    assert len(segments) == batch_size
    
    # 检查第一个segment
    seg = segments[0]
    assert 'states' in seg
    assert 'actions' in seg
    assert 'rewards' in seg
    assert 'next_states' in seg
    assert 'dones' in seg
    assert 'init_hidden' in seg
    
    # 检查shape
    assert seg['states'].shape == (16, 64)
    assert seg['actions'].shape == (16, 22)
    assert seg['rewards'].shape == (16,)
    assert seg['next_states'].shape == (16, 64)
    assert seg['dones'].shape == (16,)


def test_burn_in():
    """测试burn-in功能"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16, burn_in=4)
    
    # 添加足够长的episode
    buffer.add_episode(create_dummy_episode(60))
    
    # 采样
    segments = buffer.sample(8)
    
    # 检查burn-in（注意：不是所有segment都一定有burn-in）
    has_burn_in = any('burn_in' in seg for seg in segments)
    
    if has_burn_in:
        for seg in segments:
            if 'burn_in' in seg:
                assert seg['burn_in']['states'].shape == (4, 64)
                assert seg['burn_in']['actions'].shape == (4, 22)


def test_buffer_capacity():
    """测试buffer容量限制"""
    buffer = SequenceReplayBuffer(capacity=5, seq_len=16)
    
    # 添加超过容量的episodes
    for i in range(10):
        buffer.add_episode(create_dummy_episode(50))
    
    # 应该只保留最新的5个
    assert len(buffer) == 5


def test_stats():
    """测试统计信息"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16)
    
    # 空buffer
    stats = buffer.get_stats()
    assert stats['num_episodes'] == 0
    assert stats['total_steps'] == 0
    
    # 添加episodes
    for _ in range(3):
        buffer.add_episode(create_dummy_episode(40))
    
    stats = buffer.get_stats()
    assert stats['num_episodes'] == 3
    assert stats['total_steps'] == 120  # 3 * 40
    assert stats['avg_episode_length'] == 40.0
    assert 'avg_episode_return' in stats
    assert 'capacity_usage' in stats


def test_clear_buffer():
    """测试清空buffer"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16)
    
    # 添加episodes
    for _ in range(5):
        buffer.add_episode(create_dummy_episode(40))
    
    assert len(buffer) == 5
    
    # 清空
    buffer.clear()
    assert len(buffer) == 0
    
    stats = buffer.get_stats()
    assert stats['num_episodes'] == 0


def test_batch_consistency():
    """测试batch一致性"""
    buffer = SequenceReplayBuffer(capacity=20, seq_len=16)
    
    # 添加episodes
    for _ in range(10):
        buffer.add_episode(create_dummy_episode(50))
    
    # 多次采样
    for _ in range(5):
        segments = buffer.sample(8)
        assert len(segments) == 8
        
        for seg in segments:
            assert seg['states'].shape[0] == 16
            assert seg['actions'].shape[0] == 16
            assert seg['rewards'].shape[0] == 16


def test_segment_temporal_consistency():
    """测试segment的时序一致性"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16)
    
    # 创建一个可追踪的episode
    episode_length = 50
    episode = {
        'fused_states': [torch.ones(64) * i for i in range(episode_length)],
        'actions': [torch.ones(22) * i for i in range(episode_length)],
        'rewards': [float(i) for i in range(episode_length)],
        'dones': [False] * (episode_length - 1) + [True],
        'hidden_states': [
            {
                'actor': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128)),
                'critic1': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128)),
                'critic2': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
            }
            for _ in range(episode_length)
        ],
        'episode_return': 100.0,
        'episode_length': episode_length
    }
    
    buffer.add_episode(episode)
    
    # 采样一个segment
    segments = buffer.sample(1)
    seg = segments[0]
    
    # 验证next_state确实是下一个时间步的state
    for t in range(15):
        # state[t+1]应该等于next_state[t]
        assert torch.allclose(seg['states'][t+1], seg['next_states'][t])


def test_hidden_state_structure():
    """测试hidden state结构"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16)
    
    buffer.add_episode(create_dummy_episode(40))
    
    segments = buffer.sample(2)
    
    for seg in segments:
        hidden = seg['init_hidden']
        assert 'actor' in hidden
        assert 'critic1' in hidden
        assert 'critic2' in hidden
        
        # 检查hidden state是tuple of (h, c)
        assert isinstance(hidden['actor'], tuple)
        assert len(hidden['actor']) == 2


def test_device_placement():
    """测试设备放置"""
    if torch.cuda.is_available():
        buffer_cuda = SequenceReplayBuffer(capacity=10, seq_len=16, device='cuda')
        buffer_cuda.add_episode(create_dummy_episode(40))
        segments = buffer_cuda.sample(2)
        
        assert segments[0]['states'].device.type == 'cuda'
        assert segments[0]['actions'].device.type == 'cuda'
    
    # CPU测试
    buffer_cpu = SequenceReplayBuffer(capacity=10, seq_len=16, device='cpu')
    buffer_cpu.add_episode(create_dummy_episode(40))
    segments = buffer_cpu.sample(2)
    
    assert segments[0]['states'].device.type == 'cpu'
    assert segments[0]['actions'].device.type == 'cpu'


def test_sample_from_empty_buffer():
    """测试从空buffer采样（应该抛出异常）"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16)
    
    with pytest.raises(ValueError, match="Buffer is empty"):
        buffer.sample(4)


def test_episode_return_tracking():
    """测试episode return追踪"""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=16)
    
    returns = [50.0, 100.0, 75.0]
    for ret in returns:
        episode = create_dummy_episode(40)
        episode['episode_return'] = ret
        buffer.add_episode(episode)
    
    stats = buffer.get_stats()
    expected_avg = sum(returns) / len(returns)
    assert abs(stats['avg_episode_return'] - expected_avg) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

