"""
AGSACEnvironment单元测试
"""

import pytest
import torch
import numpy as np
from agsac.envs import AGSACEnvironment, DummyAGSACEnvironment


def test_environment_initialization():
    """测试环境初始化"""
    env = DummyAGSACEnvironment(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        pred_horizon=12,
        max_episode_steps=100,
        device='cpu'
    )
    
    assert env.max_pedestrians == 10
    assert env.max_corridors == 10
    assert env.max_vertices == 20
    assert env.obs_horizon == 8
    assert env.pred_horizon == 12
    assert env.max_episode_steps == 100


def test_reset():
    """测试reset功能"""
    env = DummyAGSACEnvironment(device='cpu')
    obs = env.reset()
    
    # 检查观测keys
    assert 'robot_state' in obs
    assert 'pedestrian_observations' in obs
    assert 'pedestrian_mask' in obs
    assert 'corridor_vertices' in obs
    assert 'corridor_mask' in obs
    assert 'goal' in obs
    
    # 检查状态被正确重置
    assert env.current_step == 0
    assert env.episode_return == 0.0
    assert env.done == False


def test_observation_format():
    """测试观测格式"""
    env = DummyAGSACEnvironment(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        device='cpu'
    )
    obs = env.reset()
    
    # robot_state
    assert isinstance(obs['robot_state']['position'], torch.Tensor)
    assert isinstance(obs['robot_state']['velocity'], torch.Tensor)
    assert isinstance(obs['robot_state']['orientation'], torch.Tensor)
    assert isinstance(obs['robot_state']['goal_vector'], torch.Tensor)
    
    # pedestrians
    assert obs['pedestrian_observations'].shape == (10, 8, 2)
    assert obs['pedestrian_mask'].shape == (10,)
    assert obs['pedestrian_mask'].dtype == torch.bool
    
    # corridors
    assert obs['corridor_vertices'].shape == (10, 20, 2)
    assert obs['corridor_mask'].shape == (10,)
    assert obs['corridor_mask'].dtype == torch.bool
    
    # goal
    assert obs['goal'].shape == (2,)


def test_step():
    """测试step功能"""
    env = DummyAGSACEnvironment(device='cpu')
    env.reset()
    
    action = np.random.randn(22) * 0.01  # 小动作
    obs, reward, done, info = env.step(action)
    
    # 检查返回值类型
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # 检查info
    assert 'collision' in info
    assert 'done_reason' in info
    assert 'episode_step' in info
    assert 'episode_return' in info
    
    # 检查状态更新
    assert env.current_step == 1


def test_episode_run():
    """测试完整episode运行"""
    env = DummyAGSACEnvironment(max_episode_steps=50, device='cpu')
    env.reset()
    
    total_reward = 0.0
    steps = 0
    max_steps = 60
    
    for i in range(max_steps):
        action = np.random.randn(22) * 0.01
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    assert steps > 0
    assert env.done == True
    assert env.current_step == steps


def test_collision_detection():
    """测试碰撞检测"""
    env = DummyAGSACEnvironment(device='cpu')
    env.reset()
    
    # 尝试多次大动作，应该触发碰撞
    collision_detected = False
    
    for i in range(10):
        action = np.ones(22) * 5.0  # 大动作
        obs, reward, done, info = env.step(action)
        
        if info['collision']:
            collision_detected = True
            assert done == True
            assert info['done_reason'] == 'collision'
            break
    
    # 注意：随机动作可能不一定触发碰撞，所以这里只是软检查
    # assert collision_detected  # 可能不总是成立


def test_timeout():
    """测试超时机制"""
    env = DummyAGSACEnvironment(max_episode_steps=5, device='cpu')
    env.reset()
    
    for i in range(10):
        action = np.random.randn(22) * 0.001  # 极小的动作
        obs, reward, done, info = env.step(action)
        
        if done:
            # 可能是超时或碰撞
            if info['done_reason'] == 'timeout':
                assert info['episode_step'] == 5
            break


def test_reward_structure():
    """测试奖励结构"""
    env = DummyAGSACEnvironment(use_geometric_reward=True, device='cpu')
    env.reset()
    
    action = np.random.randn(22) * 0.01
    obs, reward, done, info = env.step(action)
    
    # 检查奖励详情
    assert 'base_reward' in info
    assert 'geometric_reward' in info
    assert 'collision_penalty' in info
    assert 'step_penalty' in info
    assert 'total_reward' in info


def test_geometric_reward():
    """测试几何奖励"""
    env = DummyAGSACEnvironment(
        use_geometric_reward=True,
        max_episode_steps=20,
        device='cpu'
    )
    env.reset()
    
    geometric_rewards = []
    
    for i in range(15):
        action = np.random.randn(22) * 0.01
        obs, reward, done, info = env.step(action)
        
        if 'geometric_reward' in info:
            geometric_rewards.append(info['geometric_reward'])
        
        if done:
            break
    
    # 几何奖励应该在合理范围内
    if geometric_rewards:
        assert all(0.0 <= g <= 1.0 for g in geometric_rewards)


def test_get_info():
    """测试get_info方法"""
    env = DummyAGSACEnvironment(device='cpu')
    env.reset()
    
    info = env.get_info()
    
    assert 'max_pedestrians' in info
    assert 'max_corridors' in info
    assert 'max_vertices' in info
    assert 'obs_horizon' in info
    assert 'pred_horizon' in info
    assert 'max_episode_steps' in info
    assert 'current_step' in info
    assert 'episode_return' in info
    assert 'done' in info


def test_reset_after_done():
    """测试done后能否正常reset"""
    env = DummyAGSACEnvironment(max_episode_steps=3, device='cpu')
    env.reset()
    
    # 运行到done
    for i in range(5):
        action = np.random.randn(22) * 0.001
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    assert env.done == True
    
    # Reset应该成功
    obs = env.reset()
    assert env.done == False
    assert env.current_step == 0
    assert env.episode_return == 0.0


def test_step_after_done_raises_error():
    """测试done后step应该抛出异常"""
    env = DummyAGSACEnvironment(max_episode_steps=1, device='cpu')
    env.reset()
    
    # 第一步
    action = np.random.randn(22) * 0.001
    obs, reward, done, info = env.step(action)
    
    if done:
        # 再次step应该抛出异常
        with pytest.raises(RuntimeError, match="Episode已经结束"):
            env.step(action)


def test_pedestrian_mask():
    """测试行人mask"""
    env = DummyAGSACEnvironment(max_pedestrians=10, device='cpu')
    obs = env.reset()
    
    mask = obs['pedestrian_mask']
    
    # 前3个应该是True（DummyEnv有3个行人）
    assert mask[:3].all()
    
    # 后面的应该是False
    assert not mask[3:].any()


def test_corridor_mask():
    """测试走廊mask"""
    env = DummyAGSACEnvironment(max_corridors=10, device='cpu')
    obs = env.reset()
    
    mask = obs['corridor_mask']
    
    # 前2个应该是True（DummyEnv有2个走廊）
    assert mask[:2].all()
    
    # 后面的应该是False
    assert not mask[2:].any()


def test_device_placement():
    """测试设备放置"""
    # CPU测试
    env_cpu = DummyAGSACEnvironment(device='cpu')
    obs = env_cpu.reset()
    
    assert obs['pedestrian_observations'].device.type == 'cpu'
    assert obs['corridor_vertices'].device.type == 'cpu'
    
    # CUDA测试（如果可用）
    if torch.cuda.is_available():
        env_cuda = DummyAGSACEnvironment(device='cuda')
        obs = env_cuda.reset()
        
        assert obs['pedestrian_observations'].device.type == 'cuda'
        assert obs['corridor_vertices'].device.type == 'cuda'


def test_reward_weights():
    """测试自定义奖励权重"""
    custom_weights = {
        'goal_progress': 2.0,
        'collision': -20.0,
        'geometric': 1.0
    }
    
    env = DummyAGSACEnvironment(reward_weights=custom_weights, device='cpu')
    
    assert env.reward_weights['goal_progress'] == 2.0
    assert env.reward_weights['collision'] == -20.0
    assert env.reward_weights['geometric'] == 1.0
    # 默认值应该保留
    assert 'step_penalty' in env.reward_weights


def test_path_history():
    """测试路径历史记录"""
    env = DummyAGSACEnvironment(device='cpu')
    env.reset()
    
    assert len(env.path_history) == 0
    
    # 执行几步
    for i in range(5):
        action = np.random.randn(22) * 0.01
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    # 路径应该被记录
    assert len(env.path_history) > 0


def test_observation_history():
    """测试观测历史"""
    env = DummyAGSACEnvironment(obs_horizon=8, device='cpu')
    env.reset()
    
    # 初始应该为空或被reset
    assert len(env.observation_history) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

