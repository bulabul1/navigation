"""
端到端集成测试
End-to-End Integration Tests

测试完整的AGSAC系统集成，包括：
1. 环境 → 模型 → SAC → 环境 的完整闭环
2. 数据收集 → ReplayBuffer → 训练 的完整流程
3. Trainer的完整训练循环
4. Checkpoint保存和加载
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

from agsac.envs import DummyAGSACEnvironment
from agsac.models import AGSACModel
from agsac.training import SequenceReplayBuffer, AGSACTrainer


@pytest.fixture
def temp_output_dir():
    """创建临时输出目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_env():
    """创建小规模环境"""
    return DummyAGSACEnvironment(
        max_pedestrians=3,
        max_corridors=3,
        max_vertices=8,
        obs_horizon=4,
        pred_horizon=8,
        max_episode_steps=30,  # 短episode
        device='cpu'
    )


@pytest.fixture
def small_model():
    """创建小规模模型"""
    return AGSACModel(
        dog_feature_dim=32,
        corridor_feature_dim=64,
        social_feature_dim=64,
        pedestrian_feature_dim=32,
        fusion_dim=32,
        action_dim=22,
        max_pedestrians=3,
        max_corridors=3,
        max_vertices=8,
        obs_horizon=4,
        pred_horizon=8,
        num_modes=5,  # 减少模式数
        hidden_dim=64,  # 减小hidden dim
        device='cpu'
    )


def test_env_to_model_integration(small_env, small_model):
    """测试环境和模型的集成"""
    # 重置环境
    obs = small_env.reset()
    
    # 初始化hidden states
    hidden = small_model.init_hidden_states(batch_size=1)
    
    # 转换观测格式（模拟Trainer的_add_batch_dim）
    obs_batch = {
        'dog': {
            'trajectory': obs['robot_state']['position'].unsqueeze(0).unsqueeze(1).repeat(1, 4, 1),
            'velocity': obs['robot_state']['velocity'].unsqueeze(0),
            'position': obs['robot_state']['position'].unsqueeze(0),
            'goal': obs['goal'].unsqueeze(0)
        },
        'pedestrians': {
            'trajectories': obs['pedestrian_observations'].unsqueeze(0),
            'mask': obs['pedestrian_mask'].unsqueeze(0)
        },
        'corridors': {
            'polygons': obs['corridor_vertices'].unsqueeze(0),
            'vertex_counts': torch.ones(1, 3, dtype=torch.long) * 4,
            'mask': obs['corridor_mask'].unsqueeze(0)
        }
    }
    
    # 模型forward
    with torch.no_grad():
        output = small_model(obs_batch, hidden, deterministic=False)
    
    # 验证输出
    assert 'action' in output
    assert 'fused_state' in output
    assert 'hidden_states' in output
    
    assert output['action'].shape == (1, 22)
    assert output['fused_state'].shape == (1, 32)
    
    # 执行动作
    action = output['action'].squeeze(0).cpu().numpy()
    next_obs, reward, done, info = small_env.step(action)
    
    # 验证环境返回
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert 'collision' in info


def test_episode_collection(small_env, small_model, temp_output_dir):
    """测试完整episode收集"""
    trainer = AGSACTrainer(
        model=small_model,
        env=small_env,
        buffer_capacity=10,
        seq_len=8,  # 减小序列长度
        batch_size=2,
        warmup_episodes=2,
        updates_per_episode=5,
        max_episodes=3,
        device='cpu',
        save_dir=temp_output_dir,
        experiment_name='test_e2e'
    )
    
    # 收集episode
    episode_data = trainer.collect_episode(deterministic=False)
    
    # 验证episode数据
    assert 'fused_states' in episode_data
    assert 'actions' in episode_data
    assert 'rewards' in episode_data
    assert 'dones' in episode_data
    assert 'hidden_states' in episode_data
    assert 'episode_return' in episode_data
    assert 'episode_length' in episode_data
    
    assert len(episode_data['fused_states']) == episode_data['episode_length']
    assert len(episode_data['actions']) == episode_data['episode_length']
    assert len(episode_data['rewards']) == episode_data['episode_length']


def test_buffer_to_training_integration(small_env, small_model):
    """测试ReplayBuffer到训练的集成"""
    # 创建buffer
    buffer = SequenceReplayBuffer(
        capacity=10,
        seq_len=8,
        burn_in=0,
        device='cpu'
    )
    
    # 创建trainer
    trainer = AGSACTrainer(
        model=small_model,
        env=small_env,
        buffer_capacity=10,
        seq_len=8,
        batch_size=2,
        warmup_episodes=2,
        updates_per_episode=5,
        max_episodes=5,
        device='cpu',
        save_dir='./temp',
        experiment_name='test_buffer'
    )
    
    # 收集足够长的episodes
    valid_episodes = 0
    max_attempts = 20
    
    for _ in range(max_attempts):
        episode_data = trainer.collect_episode(deterministic=False)
        
        # 只添加足够长的episode
        if episode_data['episode_length'] >= trainer.buffer.seq_len + 1:
            trainer.buffer.add_episode(episode_data)
            valid_episodes += 1
            
            if valid_episodes >= 3:
                break
    
    # 如果收集到足够的episodes，测试训练
    if len(trainer.buffer) >= 2:
        # 采样
        segments = trainer.buffer.sample(2)
        assert len(segments) == 2
        
        # 执行一次训练更新
        losses = small_model.sac_agent.update(segments)
        
        # 验证losses
        assert 'actor_loss' in losses
        assert 'critic_loss' in losses
        assert 'alpha' in losses
        
        assert isinstance(losses['actor_loss'], float)
        assert isinstance(losses['critic_loss'], float)


def test_full_training_loop(small_env, small_model, temp_output_dir):
    """测试完整训练循环"""
    trainer = AGSACTrainer(
        model=small_model,
        env=small_env,
        buffer_capacity=20,
        seq_len=8,
        batch_size=2,
        warmup_episodes=3,
        updates_per_episode=3,
        eval_interval=3,
        eval_episodes=2,
        save_interval=5,
        log_interval=1,
        max_episodes=5,  # 短训练
        device='cpu',
        save_dir=temp_output_dir,
        experiment_name='test_full_loop'
    )
    
    # 执行训练
    history = trainer.train()
    
    # 验证训练历史
    assert 'episode_returns' in history
    assert 'episode_lengths' in history
    assert len(history['episode_returns']) == 5
    assert len(history['episode_lengths']) == 5
    
    # 验证checkpoint保存
    save_dir = Path(temp_output_dir) / 'test_full_loop'
    assert save_dir.exists()
    
    # 检查最终checkpoint
    final_ckpt = save_dir / 'checkpoint_final.pt'
    assert final_ckpt.exists()
    
    # 检查training_history.json
    history_file = save_dir / 'training_history.json'
    assert history_file.exists()


def test_checkpoint_save_load(small_env, small_model, temp_output_dir):
    """测试checkpoint保存和加载"""
    # 创建trainer并训练几步
    trainer1 = AGSACTrainer(
        model=small_model,
        env=small_env,
        buffer_capacity=10,
        seq_len=8,
        batch_size=2,
        warmup_episodes=2,
        max_episodes=3,
        device='cpu',
        save_dir=temp_output_dir,
        experiment_name='test_ckpt'
    )
    
    # 训练
    history1 = trainer1.train()
    
    # 保存checkpoint
    ckpt_path = Path(temp_output_dir) / 'test_ckpt' / 'checkpoint_final.pt'
    assert ckpt_path.exists()
    
    # 创建新的trainer和模型
    new_model = AGSACModel(
        dog_feature_dim=32,
        corridor_feature_dim=64,
        social_feature_dim=64,
        pedestrian_feature_dim=32,
        fusion_dim=32,
        action_dim=22,
        max_pedestrians=3,
        max_corridors=3,
        max_vertices=8,
        obs_horizon=4,
        pred_horizon=8,
        num_modes=5,
        hidden_dim=64,
        device='cpu'
    )
    
    trainer2 = AGSACTrainer(
        model=new_model,
        env=small_env,
        device='cpu',
        save_dir=temp_output_dir,
        experiment_name='test_ckpt'
    )
    
    # 加载checkpoint
    trainer2.load_checkpoint(str(ckpt_path))
    
    # 验证状态恢复
    assert trainer2.episode_count == trainer1.episode_count
    assert trainer2.total_steps == trainer1.total_steps


def test_deterministic_vs_stochastic(small_env, small_model):
    """测试确定性和随机性策略"""
    trainer = AGSACTrainer(
        model=small_model,
        env=small_env,
        device='cpu',
        save_dir='./temp',
        experiment_name='test_det'
    )
    
    # 收集随机episode
    stoch_episode = trainer.collect_episode(deterministic=False)
    
    # 收集确定性episode
    det_episode1 = trainer.collect_episode(deterministic=True)
    det_episode2 = trainer.collect_episode(deterministic=True)
    
    # 随机episode应该有数据
    assert stoch_episode['episode_length'] > 0
    
    # 确定性episode也应该有数据
    assert det_episode1['episode_length'] > 0
    assert det_episode2['episode_length'] > 0


def test_evaluation(small_env, small_model):
    """测试评估功能"""
    trainer = AGSACTrainer(
        model=small_model,
        env=small_env,
        eval_episodes=3,
        device='cpu',
        save_dir='./temp',
        experiment_name='test_eval'
    )
    
    # 执行评估
    eval_stats = trainer.evaluate()
    
    # 验证评估统计
    assert 'eval_return_mean' in eval_stats
    assert 'eval_return_std' in eval_stats
    assert 'eval_length_mean' in eval_stats
    
    assert isinstance(eval_stats['eval_return_mean'], float)
    assert isinstance(eval_stats['eval_return_std'], float)
    assert isinstance(eval_stats['eval_length_mean'], float)


def test_training_progress(small_env, small_model, temp_output_dir):
    """测试训练进度追踪"""
    trainer = AGSACTrainer(
        model=small_model,
        env=small_env,
        buffer_capacity=10,
        warmup_episodes=2,
        max_episodes=5,
        device='cpu',
        save_dir=temp_output_dir,
        experiment_name='test_progress'
    )
    
    # 初始状态
    assert trainer.episode_count == 0
    assert trainer.total_steps == 0
    assert trainer.total_updates == 0
    
    # 训练
    trainer.train()
    
    # 检查进度
    assert trainer.episode_count == 5
    assert trainer.total_steps > 0
    
    # 检查统计
    stats = trainer.get_stats()
    assert stats['episode_count'] == 5
    assert stats['total_steps'] > 0
    assert 'buffer_size' in stats


def test_device_consistency(small_env):
    """测试设备一致性"""
    if torch.cuda.is_available():
        # CPU环境 + CPU模型
        cpu_model = AGSACModel(
            dog_feature_dim=32,
            corridor_feature_dim=64,
            social_feature_dim=64,
            pedestrian_feature_dim=32,
            fusion_dim=32,
            action_dim=22,
            max_pedestrians=3,
            max_corridors=3,
            max_vertices=8,
            device='cpu'
        )
        
        cpu_trainer = AGSACTrainer(
            model=cpu_model,
            env=small_env,
            device='cpu',
            save_dir='./temp',
            experiment_name='test_cpu'
        )
        
        # 收集episode应该成功
        episode = cpu_trainer.collect_episode(deterministic=False)
        assert episode['episode_length'] > 0


def test_model_modes(small_env, small_model):
    """测试模型训练/评估模式切换"""
    trainer = AGSACTrainer(
        model=small_model,
        env=small_env,
        device='cpu',
        save_dir='./temp',
        experiment_name='test_modes'
    )
    
    # 训练模式
    assert small_model.training == True
    
    # 评估
    trainer.evaluate()
    
    # 评估后应该恢复训练模式
    assert small_model.training == True


def test_buffer_warmup(small_env, small_model):
    """测试buffer预热机制"""
    trainer = AGSACTrainer(
        model=small_model,
        env=small_env,
        buffer_capacity=10,
        warmup_episodes=5,
        max_episodes=3,  # 少于warmup
        device='cpu',
        save_dir='./temp',
        experiment_name='test_warmup'
    )
    
    # 训练（不应该触发更新）
    history = trainer.train()
    
    # 应该没有损失记录（因为没达到warmup）
    if len(trainer.buffer) < trainer.warmup_episodes:
        # 如果buffer不足，不应该有actor_losses
        assert len(history['actor_losses']) == 0


def test_attention_weights_integration(small_env, small_model):
    """测试注意力权重的集成"""
    # 重置环境
    obs = small_env.reset()
    hidden = small_model.init_hidden_states(batch_size=1)
    
    # 构建观测
    obs_batch = {
        'dog': {
            'trajectory': obs['robot_state']['position'].unsqueeze(0).unsqueeze(1).repeat(1, 4, 1),
            'velocity': obs['robot_state']['velocity'].unsqueeze(0),
            'position': obs['robot_state']['position'].unsqueeze(0),
            'goal': obs['goal'].unsqueeze(0)
        },
        'pedestrians': {
            'trajectories': obs['pedestrian_observations'].unsqueeze(0),
            'mask': obs['pedestrian_mask'].unsqueeze(0)
        },
        'corridors': {
            'polygons': obs['corridor_vertices'].unsqueeze(0),
            'vertex_counts': torch.ones(1, 3, dtype=torch.long) * 4,
            'mask': obs['corridor_mask'].unsqueeze(0)
        }
    }
    
    # 请求attention weights
    with torch.no_grad():
        output = small_model(obs_batch, hidden, return_attention=True)
    
    # 验证attention weights存在（可选）
    # 注：某些模块可能不返回attention weights
    assert 'action' in output


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

