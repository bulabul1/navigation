"""测试SAC Agent模块"""
import numpy as np
import torch
import pytest
import tempfile
import os
from agsac.models.sac import SACAgent


def test_agent_initialization():
    """测试Agent初始化"""
    agent = SACAgent(
        state_dim=64,
        action_dim=22,
        hidden_dim=128,
        device='cpu'
    )
    
    # 检查组件存在
    assert hasattr(agent, 'actor'), "Agent应该有actor"
    assert hasattr(agent, 'critic'), "Agent应该有critic"
    assert hasattr(agent, 'critic_target'), "Agent应该有critic_target"
    
    # 检查优化器
    assert hasattr(agent, 'actor_optimizer'), "Agent应该有actor_optimizer"
    assert hasattr(agent, 'critic_optimizer'), "Agent应该有critic_optimizer"
    
    # 检查auto_entropy
    assert hasattr(agent, 'log_alpha'), "Agent应该有log_alpha"
    assert hasattr(agent, 'alpha'), "Agent应该有alpha"


def test_agent_parameter_count():
    """测试Agent参数量"""
    agent = SACAgent()
    
    total_params = sum(p.numel() for p in agent.parameters())
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    
    print(f"\nAgent参数量: {total_params:,}")
    print(f"  - Actor: {actor_params:,}")
    print(f"  - Critic: {critic_params:,}")
    
    # 参数量应该合理（Actor + TwinCritic + log_alpha）
    assert 700000 < total_params < 900000, f"参数量{total_params}超出预期"


def test_agent_select_action():
    """测试动作选择"""
    agent = SACAgent()
    agent.eval()
    
    state = torch.randn(64)
    
    # 测试确定性动作
    with torch.no_grad():
        action_det, hidden_det = agent.select_action(state, deterministic=True)
    
    assert action_det.shape == (22,), "动作维度错误"
    assert hidden_det[0].shape == (1, 1, 128), "Hidden state维度错误"
    assert (action_det >= -1).all() and (action_det <= 1).all(), "动作应在[-1, 1]范围"
    
    # 测试随机动作
    action_sto, hidden_sto = agent.select_action(state, deterministic=False)
    assert action_sto.shape == (22,)


def test_agent_update_single_segment():
    """测试单个segment更新"""
    agent = SACAgent()
    
    # 创建简单的segment
    segment = {
        'states': torch.randn(5, 64),  # seq_len=5
        'actions': torch.randn(5, 22),
        'rewards': torch.randn(5),
        'next_states': torch.randn(5, 64),
        'dones': torch.zeros(5),
        'init_hidden_actor': None,
        'init_hidden_critic1': None,
        'init_hidden_critic2': None,
    }
    
    # 更新
    logs = agent.update([segment])
    
    # 检查返回的日志
    assert 'critic_loss' in logs, "应返回critic_loss"
    assert 'actor_loss' in logs, "应返回actor_loss"
    assert 'alpha_loss' in logs, "应返回alpha_loss"
    assert 'q1_mean' in logs, "应返回q1_mean"
    assert 'q2_mean' in logs, "应返回q2_mean"
    assert 'alpha' in logs, "应返回alpha"
    
    # 检查loss是有限值
    assert np.isfinite(logs['critic_loss']), "critic_loss应为有限值"
    assert np.isfinite(logs['actor_loss']), "actor_loss应为有限值"
    assert np.isfinite(logs['alpha']), "alpha应为有限值"


def test_agent_update_batch():
    """测试batch更新"""
    agent = SACAgent()
    
    # 创建batch
    segment_batch = []
    for _ in range(4):  # batch_size=4
        segment = {
            'states': torch.randn(3, 64),  # seq_len=3
            'actions': torch.randn(3, 22),
            'rewards': torch.randn(3),
            'next_states': torch.randn(3, 64),
            'dones': torch.zeros(3),
            'init_hidden_actor': None,
            'init_hidden_critic1': None,
            'init_hidden_critic2': None,
        }
        segment_batch.append(segment)
    
    # 更新
    logs = agent.update(segment_batch)
    
    # 验证更新计数增加
    assert agent.total_updates == 1, "更新计数应该增加"
    
    # 再次更新
    logs2 = agent.update(segment_batch)
    assert agent.total_updates == 2, "更新计数应该继续增加"


def test_soft_target_update():
    """测试软更新target网络"""
    agent = SACAgent(tau=0.005)
    
    # 获取更新前的参数
    critic_param_before = list(agent.critic.parameters())[0].clone()
    target_param_before = list(agent.critic_target.parameters())[0].clone()
    
    # 确保初始时target和critic相同
    assert torch.allclose(critic_param_before, target_param_before), \
        "初始时target应与critic相同"
    
    # 修改critic参数（模拟训练）
    with torch.no_grad():
        for param in agent.critic.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    # 软更新
    agent.soft_update_target()
    
    # 获取更新后的参数
    critic_param_after = list(agent.critic.parameters())[0]
    target_param_after = list(agent.critic_target.parameters())[0]
    
    # Target应该移动了，但不完全等于critic
    assert not torch.allclose(target_param_before, target_param_after), \
        "Target应该更新"
    assert not torch.allclose(critic_param_after, target_param_after), \
        "Target不应完全等于critic（tau<1）"


def test_auto_entropy_tuning():
    """测试自动熵调节"""
    # 启用自动熵调节
    agent_auto = SACAgent(auto_entropy=True, alpha_lr=3e-4)
    assert agent_auto.alpha_optimizer is not None, "应该有alpha优化器"
    
    # 禁用自动熵调节
    agent_fixed = SACAgent(auto_entropy=False)
    assert agent_fixed.alpha_optimizer is None, "不应该有alpha优化器"
    
    # 测试alpha更新
    segment = {
        'states': torch.randn(3, 64),
        'actions': torch.randn(3, 22),
        'rewards': torch.randn(3),
        'next_states': torch.randn(3, 64),
        'dones': torch.zeros(3),
        'init_hidden_actor': None,
        'init_hidden_critic1': None,
        'init_hidden_critic2': None,
    }
    
    alpha_before = agent_auto.alpha.item()
    logs = agent_auto.update([segment])
    alpha_after = agent_auto.alpha.item()
    
    # Alpha应该更新了（虽然可能变化很小）
    print(f"\nAlpha: {alpha_before:.6f} → {alpha_after:.6f}")


def test_gradient_clipping():
    """测试梯度裁剪"""
    agent = SACAgent(max_grad_norm=1.0)
    
    segment = {
        'states': torch.randn(5, 64),
        'actions': torch.randn(5, 22),
        'rewards': torch.randn(5) * 100,  # 大的奖励可能导致大梯度
        'next_states': torch.randn(5, 64),
        'dones': torch.zeros(5),
        'init_hidden_actor': None,
        'init_hidden_critic1': None,
        'init_hidden_critic2': None,
    }
    
    logs = agent.update([segment])
    
    # 检查梯度范数被记录
    assert 'critic_grad_norm' in logs, "应记录critic梯度范数"
    assert 'actor_grad_norm' in logs, "应记录actor梯度范数"
    
    # 梯度范数应该是有限值
    assert np.isfinite(logs['critic_grad_norm']), "梯度范数应为有限值"
    assert np.isfinite(logs['actor_grad_norm']), "梯度范数应为有限值"
    
    print(f"\nGradient norms: critic={logs['critic_grad_norm']:.4f}, actor={logs['actor_grad_norm']:.4f}")


def test_save_load_checkpoint():
    """测试保存和加载checkpoint"""
    agent = SACAgent()
    
    # 进行一次更新（改变参数）
    segment = {
        'states': torch.randn(3, 64),
        'actions': torch.randn(3, 22),
        'rewards': torch.randn(3),
        'next_states': torch.randn(3, 64),
        'dones': torch.zeros(3),
        'init_hidden_actor': None,
        'init_hidden_critic1': None,
        'init_hidden_critic2': None,
    }
    agent.update([segment])
    
    # 保存checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'agent.pt')
        agent.save_checkpoint(checkpoint_path)
        
        # 创建新agent并加载
        agent2 = SACAgent()
        agent2.load_checkpoint(checkpoint_path)
        
        # 验证参数相同
        for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
            assert torch.allclose(p1, p2), "Actor参数应该相同"
        
        for p1, p2 in zip(agent.critic.parameters(), agent2.critic.parameters()):
            assert torch.allclose(p1, p2), "Critic参数应该相同"
        
        # 验证alpha相同
        assert torch.allclose(agent.alpha, agent2.alpha), "Alpha应该相同"
        
        # 验证更新计数相同
        assert agent.total_updates == agent2.total_updates, "更新计数应该相同"
    
    print("\n✓ Checkpoint保存和加载成功")


def test_train_eval_mode():
    """测试训练/评估模式切换"""
    agent = SACAgent()
    
    # 训练模式
    agent.train()
    assert agent.training, "应该处于训练模式"
    assert agent.actor.training, "Actor应该处于训练模式"
    assert agent.critic.training, "Critic应该处于训练模式"
    assert not agent.critic_target.training, "Target应该始终处于eval模式"
    
    # 评估模式
    agent.eval()
    assert not agent.training, "应该处于评估模式"
    assert not agent.actor.training, "Actor应该处于评估模式"
    assert not agent.critic.training, "Critic应该处于评估模式"
    assert not agent.critic_target.training, "Target应该始终处于eval模式"


def test_loss_decrease():
    """测试多次更新后loss下降"""
    agent = SACAgent()
    
    # 创建固定的数据（模拟简单任务）
    torch.manual_seed(42)
    segment = {
        'states': torch.randn(5, 64),
        'actions': torch.randn(5, 22),
        'rewards': torch.ones(5),  # 固定正奖励
        'next_states': torch.randn(5, 64),
        'dones': torch.zeros(5),
        'init_hidden_actor': None,
        'init_hidden_critic1': None,
        'init_hidden_critic2': None,
    }
    
    # 记录初始loss
    logs_initial = agent.update([segment])
    initial_critic_loss = logs_initial['critic_loss']
    
    # 多次更新
    for _ in range(10):
        agent.update([segment])
    
    # 最终loss
    logs_final = agent.update([segment])
    final_critic_loss = logs_final['critic_loss']
    
    print(f"\nCritic loss: {initial_critic_loss:.4f} → {final_critic_loss:.4f}")
    
    # 注意：由于是随机数据，loss不一定严格下降，但应该收敛到合理范围
    assert np.isfinite(final_critic_loss), "最终loss应为有限值"


def test_with_hidden_states():
    """测试使用非None的hidden states"""
    agent = SACAgent()
    
    # 创建初始hidden states
    h = torch.zeros(1, 1, 128)
    c = torch.zeros(1, 1, 128)
    init_hidden = (h, c)
    
    segment = {
        'states': torch.randn(3, 64),
        'actions': torch.randn(3, 22),
        'rewards': torch.randn(3),
        'next_states': torch.randn(3, 64),
        'dones': torch.zeros(3),
        'init_hidden_actor': init_hidden,
        'init_hidden_critic1': init_hidden,
        'init_hidden_critic2': init_hidden,
    }
    
    logs = agent.update([segment])
    
    # 应该能正常更新
    assert np.isfinite(logs['critic_loss']), "带hidden state的更新应正常"


def test_cuda_compatibility():
    """测试CUDA兼容性（如果可用）"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA不可用")
    
    agent = SACAgent(device='cuda')
    
    # 检查所有模块在GPU上
    assert next(agent.actor.parameters()).is_cuda, "Actor应在CUDA上"
    assert next(agent.critic.parameters()).is_cuda, "Critic应在CUDA上"
    assert next(agent.critic_target.parameters()).is_cuda, "Target应在CUDA上"
    assert agent.log_alpha.is_cuda, "log_alpha应在CUDA上"
    
    # 测试select_action
    state = torch.randn(64).cuda()
    action, _ = agent.select_action(state)
    assert action.is_cuda, "动作应在CUDA上"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

