"""测试Hybrid SAC Actor模块"""
import torch
import pytest
from agsac.models.sac import HybridActor


def test_actor_basic_forward():
    """测试基础前向传播"""
    actor = HybridActor(state_dim=64, action_dim=22, hidden_dim=128)
    
    state = torch.randn(64)
    action, log_prob, hidden = actor(state)
    
    # 检查输出维度
    assert action.shape == (22,), f"动作维度错误，期望(22,)，实际{action.shape}"
    assert log_prob.shape == (), f"log_prob维度错误，期望()，实际{log_prob.shape}"
    assert hidden[0].shape == (1, 1, 128), "Hidden state h维度错误"
    assert hidden[1].shape == (1, 1, 128), "Hidden state c维度错误"


def test_actor_action_bounds():
    """测试动作范围约束"""
    actor = HybridActor()
    
    # 测试多次采样
    for _ in range(10):
        state = torch.randn(64)
        action, _, _ = actor(state)
        
        assert (action >= -1).all(), f"动作应 >= -1，实际最小值{action.min().item()}"
        assert (action <= 1).all(), f"动作应 <= 1，实际最大值{action.max().item()}"


def test_actor_log_prob_validity():
    """测试log probability的有效性"""
    actor = HybridActor()
    
    state = torch.randn(64)
    _, log_prob, _ = actor(state)
    
    # Log probability应该是负数（概率密度的对数）
    assert log_prob <= 0, f"log_prob应为负数，实际{log_prob.item()}"
    # 应该是有限值（不是nan或inf）
    assert torch.isfinite(log_prob), "log_prob应为有限值"


def test_actor_hidden_state_propagation():
    """测试hidden state的传递和更新"""
    actor = HybridActor()
    
    state1 = torch.randn(64)
    state2 = torch.randn(64)
    
    # 第一步：从None初始化
    _, _, hidden1 = actor(state1, hidden_state=None)
    
    # 第二步：传递hidden state
    _, _, hidden2 = actor(state2, hidden_state=hidden1)
    
    # Hidden state应该被更新
    assert not torch.allclose(hidden1[0], hidden2[0], atol=1e-6), \
        "Hidden state应该随时间更新"
    
    # 测试从None初始化应该得到零向量
    h_init, c_init = actor._init_hidden(1, torch.device('cpu'))
    assert torch.allclose(h_init, torch.zeros_like(h_init)), "h应初始化为零"
    assert torch.allclose(c_init, torch.zeros_like(c_init)), "c应初始化为零"


def test_actor_batch_processing():
    """测试批量处理"""
    actor = HybridActor()
    batch_size = 16
    
    batch_state = torch.randn(batch_size, 64)
    batch_action, batch_log_prob, batch_hidden = actor(batch_state)
    
    assert batch_action.shape == (batch_size, 22), "批量动作维度错误"
    assert batch_log_prob.shape == (batch_size,), "批量log_prob维度错误"
    assert batch_hidden[0].shape == (1, batch_size, 128), "批量hidden h维度错误"
    assert batch_hidden[1].shape == (1, batch_size, 128), "批量hidden c维度错误"


def test_actor_deterministic_mode():
    """测试确定性模式"""
    actor = HybridActor()
    actor.eval()
    
    state = torch.randn(64)
    
    with torch.no_grad():
        action1, log_prob1, _ = actor(state, deterministic=True)
        action2, log_prob2, _ = actor(state, deterministic=True)
    
    # 确定性模式下，相同输入应产生相同输出
    assert torch.allclose(action1, action2), "确定性模式应产生相同输出"
    
    # 确定性模式下log_prob应为0
    assert log_prob1 == 0 and log_prob2 == 0, "确定性模式log_prob应为0"


def test_actor_stochastic_mode():
    """测试随机模式"""
    actor = HybridActor()
    actor.train()
    
    state = torch.randn(64)
    
    # 多次采样应产生不同结果
    actions = []
    for _ in range(5):
        action, _, _ = actor(state, deterministic=False)
        actions.append(action)
    
    # 检查至少有一些差异
    all_same = all(torch.allclose(actions[0], a) for a in actions[1:])
    assert not all_same, "随机模式应产生不同输出"


def test_actor_gradient_flow():
    """测试梯度传播"""
    actor = HybridActor()
    
    state = torch.randn(64, requires_grad=True)
    action, log_prob, _ = actor(state)
    
    # 计算损失并反向传播
    loss = action.sum() + log_prob
    loss.backward()
    
    # 检查梯度
    assert state.grad is not None, "状态梯度不应为None"
    assert torch.isfinite(state.grad).all(), "梯度应为有限值"
    
    # 检查网络参数也有梯度
    for name, param in actor.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"参数{name}的梯度为None"
            assert torch.isfinite(param.grad).all(), f"参数{name}的梯度包含非有限值"


def test_actor_parameter_count():
    """测试参数量"""
    actor = HybridActor(state_dim=64, action_dim=22, hidden_dim=128)
    
    total_params = sum(p.numel() for p in actor.parameters())
    trainable_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    
    print(f"\nActor参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 参数量应该在预期范围内（约146K）
    assert 140000 < total_params < 150000, f"参数量{total_params}超出预期范围"
    assert trainable_params == total_params, "所有参数应可训练"


def test_actor_select_action_interface():
    """测试select_action接口"""
    actor = HybridActor()
    
    state = torch.randn(64)
    
    # 测试确定性选择
    action_det, hidden_det = actor.select_action(state, deterministic=True)
    assert action_det.shape == (22,), "select_action输出维度错误"
    assert hidden_det[0].shape == (1, 1, 128), "select_action hidden维度错误"
    
    # 测试随机选择
    action_sto, hidden_sto = actor.select_action(state, deterministic=False)
    assert action_sto.shape == (22,), "select_action随机模式输出维度错误"


def test_actor_evaluate_actions_interface():
    """测试evaluate_actions接口"""
    actor = HybridActor()
    
    batch_state = torch.randn(4, 64)
    action, log_prob, entropy, hidden = actor.evaluate_actions(batch_state)
    
    # 检查输出维度
    assert action.shape == (4, 22), "evaluate_actions动作维度错误"
    assert log_prob.shape == (4,), "evaluate_actions log_prob维度错误"
    assert entropy.shape == (4,), "evaluate_actions熵维度错误"
    assert hidden[0].shape == (1, 4, 128), "evaluate_actions hidden维度错误"
    
    # 熵应该是正数
    assert (entropy > 0).all(), "策略熵应为正数"


def test_actor_sequence_processing():
    """测试序列处理（模拟episode）"""
    actor = HybridActor()
    actor.eval()
    
    # 模拟一个episode的5步
    sequence_length = 5
    hidden = None
    actions = []
    
    with torch.no_grad():
        for t in range(sequence_length):
            state = torch.randn(64)
            action, _, hidden = actor(state, hidden_state=hidden, deterministic=True)
            actions.append(action)
    
    # 每一步都应该产生有效动作
    assert len(actions) == sequence_length
    for i, action in enumerate(actions):
        assert action.shape == (22,), f"步骤{i}的动作维度错误"
        assert (action >= -1).all() and (action <= 1).all(), \
            f"步骤{i}的动作超出范围"


def test_actor_different_initializations():
    """测试不同的初始化方式"""
    # Orthogonal初始化
    actor_orth = HybridActor(init_type='orthogonal')
    state = torch.randn(64)
    action_orth, _, _ = actor_orth(state)
    assert action_orth.shape == (22,)
    
    # Xavier初始化
    actor_xavier = HybridActor(init_type='xavier')
    action_xavier, _, _ = actor_xavier(state)
    assert action_xavier.shape == (22,)
    
    # 两种初始化应产生不同的输出
    assert not torch.allclose(action_orth, action_xavier), \
        "不同初始化应产生不同输出"


def test_actor_entropy_decreases():
    """测试熵的合理性"""
    actor = HybridActor()
    
    # 测试多个样本
    batch_state = torch.randn(10, 64)
    _, _, entropy, _ = actor.evaluate_actions(batch_state)
    
    # 熵应该是正数且有合理范围
    assert (entropy > 0).all(), "熵应为正数"
    assert (entropy < 100).all(), "熵不应过大"
    
    # 平均熵应该在合理范围内
    mean_entropy = entropy.mean()
    print(f"\n平均熵: {mean_entropy.item():.3f}")
    assert 1 < mean_entropy < 50, f"平均熵{mean_entropy.item()}超出合理范围"


def test_actor_cuda_compatibility():
    """测试CUDA兼容性（如果可用）"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA不可用")
    
    actor = HybridActor().cuda()
    
    state = torch.randn(64).cuda()
    action, log_prob, hidden = actor(state)
    
    # 检查输出在GPU上
    assert action.is_cuda, "动作应在CUDA设备上"
    assert log_prob.is_cuda, "log_prob应在CUDA设备上"
    assert hidden[0].is_cuda and hidden[1].is_cuda, "Hidden state应在CUDA设备上"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

