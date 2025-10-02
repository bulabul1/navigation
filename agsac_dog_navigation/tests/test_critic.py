"""测试Hybrid SAC Critic模块"""
import numpy as np
import torch
import pytest
from agsac.models.sac import HybridCritic, TwinCritic


def test_critic_basic_forward():
    """测试单Critic基础前向传播"""
    critic = HybridCritic(state_dim=64, action_dim=22, hidden_dim=128)
    
    state = torch.randn(64)
    action = torch.randn(22)
    q_value, hidden = critic(state, action)
    
    # 检查输出维度
    assert q_value.shape == (), f"Q值维度错误，期望()，实际{q_value.shape}"
    assert hidden[0].shape == (1, 1, 128), "Hidden state h维度错误"
    assert hidden[1].shape == (1, 1, 128), "Hidden state c维度错误"


def test_critic_q_value_validity():
    """测试Q值有效性"""
    critic = HybridCritic()
    
    state = torch.randn(64)
    action = torch.randn(22)
    q_value, _ = critic(state, action)
    
    # Q值应该是有限值
    assert torch.isfinite(q_value), "Q值应为有限值（不是nan或inf）"


def test_critic_hidden_state_propagation():
    """测试hidden state的传递和更新"""
    critic = HybridCritic()
    
    state1 = torch.randn(64)
    action1 = torch.randn(22)
    state2 = torch.randn(64)
    action2 = torch.randn(22)
    
    # 第一步：从None初始化
    _, hidden1 = critic(state1, action1, hidden_state=None)
    
    # 第二步：传递hidden state
    _, hidden2 = critic(state2, action2, hidden_state=hidden1)
    
    # Hidden state应该被更新
    assert not torch.allclose(hidden1[0], hidden2[0], atol=1e-6), \
        "Hidden state应该随时间更新"


def test_critic_batch_processing():
    """测试批量处理"""
    critic = HybridCritic()
    batch_size = 16
    
    batch_state = torch.randn(batch_size, 64)
    batch_action = torch.randn(batch_size, 22)
    batch_q, batch_hidden = critic(batch_state, batch_action)
    
    assert batch_q.shape == (batch_size,), "批量Q值维度错误"
    assert batch_hidden[0].shape == (1, batch_size, 128), "批量hidden h维度错误"
    assert batch_hidden[1].shape == (1, batch_size, 128), "批量hidden c维度错误"


def test_critic_gradient_flow():
    """测试梯度传播"""
    critic = HybridCritic()
    
    state = torch.randn(64, requires_grad=True)
    action = torch.randn(22, requires_grad=True)
    q_value, _ = critic(state, action)
    
    # 计算损失并反向传播
    loss = q_value
    loss.backward()
    
    # 检查梯度
    assert state.grad is not None, "状态梯度不应为None"
    assert action.grad is not None, "动作梯度不应为None"
    assert torch.isfinite(state.grad).all(), "梯度应为有限值"
    assert torch.isfinite(action.grad).all(), "梯度应为有限值"


def test_critic_parameter_count():
    """测试单Critic参数量"""
    critic = HybridCritic(state_dim=64, action_dim=22, hidden_dim=128)
    
    total_params = sum(p.numel() for p in critic.parameters())
    trainable_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    
    print(f"\n单Critic参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 参数量应该在预期范围内（约160K）
    assert 150000 < total_params < 170000, f"参数量{total_params}超出预期范围"
    assert trainable_params == total_params, "所有参数应可训练"


def test_twin_critic_basic_forward():
    """测试TwinCritic基础功能"""
    twin_critic = TwinCritic(state_dim=64, action_dim=22, hidden_dim=128)
    
    state = torch.randn(64)
    action = torch.randn(22)
    q1, q2, hidden1, hidden2 = twin_critic(state, action)
    
    # 检查输出维度
    assert q1.shape == (), "Q1维度错误"
    assert q2.shape == (), "Q2维度错误"
    assert hidden1[0].shape == (1, 1, 128), "Hidden1维度错误"
    assert hidden2[0].shape == (1, 1, 128), "Hidden2维度错误"


def test_twin_critic_independence():
    """测试双Critic的独立性"""
    twin_critic = TwinCritic()
    
    state = torch.randn(64)
    action = torch.randn(22)
    
    # 多次测试
    for _ in range(5):
        q1, q2, _, _ = twin_critic(state, action)
        # 两个Critic应该产生不同的Q值（随机初始化）
        assert not torch.allclose(q1, q2, atol=1e-3), \
            f"两个Critic应该独立（Q1={q1.item():.3f}, Q2={q2.item():.3f}）"


def test_twin_critic_batch_processing():
    """测试TwinCritic批量处理"""
    twin_critic = TwinCritic()
    batch_size = 16
    
    batch_state = torch.randn(batch_size, 64)
    batch_action = torch.randn(batch_size, 22)
    batch_q1, batch_q2, batch_h1, batch_h2 = twin_critic(batch_state, batch_action)
    
    assert batch_q1.shape == (batch_size,), "批量Q1维度错误"
    assert batch_q2.shape == (batch_size,), "批量Q2维度错误"
    assert batch_h1[0].shape == (1, batch_size, 128), "批量hidden1维度错误"
    assert batch_h2[0].shape == (1, batch_size, 128), "批量hidden2维度错误"


def test_twin_critic_parameter_count():
    """测试TwinCritic参数量"""
    twin_critic = TwinCritic(state_dim=64, action_dim=22, hidden_dim=128)
    
    total_params = sum(p.numel() for p in twin_critic.parameters())
    trainable_params = sum(p.numel() for p in twin_critic.parameters() if p.requires_grad)
    
    print(f"\nTwinCritic参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 参数量应该约为单Critic的2倍（约320K）
    assert 300000 < total_params < 340000, f"参数量{total_params}超出预期范围"
    assert trainable_params == total_params, "所有参数应可训练"


def test_twin_critic_q1_q2_methods():
    """测试q1和q2单独调用方法"""
    twin_critic = TwinCritic()
    
    state = torch.randn(64)
    action = torch.randn(22)
    
    # 测试单独调用
    q1_only, h1_only = twin_critic.q1(state, action)
    q2_only, h2_only = twin_critic.q2(state, action)
    
    assert q1_only.shape == (), "q1方法输出维度错误"
    assert q2_only.shape == (), "q2方法输出维度错误"
    assert h1_only[0].shape == (1, 1, 128), "q1方法hidden维度错误"
    assert h2_only[0].shape == (1, 1, 128), "q2方法hidden维度错误"


def test_critic_action_preprocessing():
    """测试不同形状的action输入"""
    critic = HybridCritic()
    
    state = torch.randn(64)
    
    # 测试扁平action
    action_flat = torch.randn(22)
    q_flat, _ = critic(state, action_flat)
    assert q_flat.shape == ()
    
    # 测试2D action（需要手动flatten）
    action_2d = torch.randn(11, 2)
    q_2d, _ = critic(state, action_2d.view(-1))
    assert q_2d.shape == ()


def test_critic_deterministic_output():
    """测试确定性输出（eval模式）"""
    critic = HybridCritic()
    critic.eval()
    
    state = torch.randn(64)
    action = torch.randn(22)
    
    with torch.no_grad():
        q1, _ = critic(state, action)
        q2, _ = critic(state, action)
    
    # eval模式下，相同输入应产生相同输出
    assert torch.allclose(q1, q2), "eval模式下相同输入应产生相同输出"


def test_critic_sequence_processing():
    """测试序列处理（模拟episode）"""
    critic = HybridCritic()
    critic.eval()
    
    # 模拟一个episode的5步
    sequence_length = 5
    hidden = None
    q_values = []
    
    with torch.no_grad():
        for t in range(sequence_length):
            state = torch.randn(64)
            action = torch.randn(22)
            q_t, hidden = critic(state, action, hidden_state=hidden)
            q_values.append(q_t.item())
    
    # 每一步都应该产生有效Q值
    assert len(q_values) == sequence_length
    for i, q in enumerate(q_values):
        assert np.isfinite(q), f"步骤{i}的Q值应为有限值"


def test_critic_different_initializations():
    """测试不同的初始化方式"""
    # Orthogonal初始化
    critic_orth = HybridCritic(init_type='orthogonal')
    state = torch.randn(64)
    action = torch.randn(22)
    q_orth, _ = critic_orth(state, action)
    assert torch.isfinite(q_orth)
    
    # Xavier初始化
    critic_xavier = HybridCritic(init_type='xavier')
    q_xavier, _ = critic_xavier(state, action)
    assert torch.isfinite(q_xavier)
    
    # 两种初始化应产生不同的输出
    assert not torch.allclose(q_orth, q_xavier), \
        "不同初始化应产生不同输出"


def test_critic_gradient_magnitude():
    """测试梯度幅度合理性"""
    critic = HybridCritic()
    
    state = torch.randn(64, requires_grad=True)
    action = torch.randn(22, requires_grad=True)
    q_value, _ = critic(state, action)
    
    loss = q_value ** 2
    loss.backward()
    
    # 检查梯度幅度
    state_grad_norm = state.grad.norm().item()
    action_grad_norm = action.grad.norm().item()
    
    print(f"\n梯度范数: state={state_grad_norm:.4f}, action={action_grad_norm:.4f}")
    
    # 梯度应该有合理的幅度（不应该过大或过小）
    assert 0.001 < state_grad_norm < 100, "状态梯度幅度异常"
    assert 0.001 < action_grad_norm < 100, "动作梯度幅度异常"


def test_twin_critic_target_network_copy():
    """测试创建target网络（硬拷贝）"""
    twin_critic = TwinCritic()
    
    # 创建target网络（用于SAC）
    twin_critic_target = TwinCritic()
    twin_critic_target.load_state_dict(twin_critic.state_dict())
    
    # 测试参数是否完全一致
    for p1, p2 in zip(twin_critic.parameters(), twin_critic_target.parameters()):
        assert torch.allclose(p1, p2), "Target网络参数应与源网络一致"
    
    print("\n✓ Target网络拷贝成功")


def test_critic_cuda_compatibility():
    """测试CUDA兼容性（如果可用）"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA不可用")
    
    critic = HybridCritic().cuda()
    
    state = torch.randn(64).cuda()
    action = torch.randn(22).cuda()
    q_value, hidden = critic(state, action)
    
    # 检查输出在GPU上
    assert q_value.is_cuda, "Q值应在CUDA设备上"
    assert hidden[0].is_cuda and hidden[1].is_cuda, "Hidden state应在CUDA设备上"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

