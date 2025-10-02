"""
Hybrid SAC Critic模块
评估状态-动作对的Q值，支持LSTM时序记忆
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HybridCritic(nn.Module):
    """
    混合SAC Critic网络（单个Q网络）
    
    设计思路：
    - 拼接state和action作为输入
    - 使用LSTM捕捉时序依赖
    - 输出单个Q值估计
    
    输入:
        state: (64,) 或 (batch, 64) 融合后的状态
        action: (22,) 或 (batch, 22) 扁平格式的动作
        hidden_state: Optional[(h, c)] LSTM隐藏状态
    
    输出:
        q_value: () 或 (batch,) Q值估计
        new_hidden_state: (h, c) 更新后的隐藏状态
    
    参数量: ~160K
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 22,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        init_type: str = 'orthogonal'
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: LSTM隐藏层维度
            lstm_layers: LSTM层数
            dropout: Dropout比例
            init_type: 参数初始化方式
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        
        # 输入维度：state + action
        input_dim = state_dim + action_dim  # 64 + 22 = 86
        
        # 1. 预处理层：将state-action对投影到LSTM输入维度
        self.pre_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 2. LSTM层：捕捉时序依赖
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=False  # 输入格式: (seq_len, batch, hidden_dim)
        )
        
        # 3. Q值头：输出Q值估计（使用两层MLP提升表达能力）
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # 4. 参数初始化
        self._initialize_weights(init_type)
    
    def _initialize_weights(self, init_type: str):
        """初始化网络参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化LSTM隐藏状态为零向量"""
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            state: (64,) 或 (batch, 64) 状态特征
            action: (22,) 或 (batch, 22) 动作
            hidden_state: Optional[(h, c)] LSTM隐藏状态
        
        Returns:
            q_value: () 或 (batch,) Q值
            new_hidden_state: (h, c) 新的隐藏状态
        """
        # 处理输入维度
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, 64)
            action = action.unsqueeze(0)  # (1, 22)
            single_input = True
        else:
            single_input = False
        
        batch_size = state.size(0)
        device = state.device
        
        # 确保action是扁平的
        if action.dim() > 2:
            # (batch, 11, 2) → (batch, 22)
            action = action.view(batch_size, -1)
        
        # 1. 初始化hidden_state（如果需要）
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, device)
        
        # 2. 拼接state和action
        x = torch.cat([state, action], dim=-1)  # (batch, 86)
        
        # 3. 预处理
        x = self.pre_fc(x)  # (batch, 128)
        
        # 4. LSTM前向传播
        # 需要reshape: (seq_len=1, batch, hidden_dim)
        x = x.unsqueeze(0)  # (1, batch, 128)
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(0)  # (batch, 128)
        
        # 5. Q值头
        q_value = self.q_head(lstm_out)  # (batch, 1)
        q_value = q_value.squeeze(-1)  # (batch,)
        
        # 6. 处理输出维度
        if single_input:
            q_value = q_value.squeeze(0)  # ()
        
        return q_value, new_hidden_state


class TwinCritic(nn.Module):
    """
    双Critic网络（用于Double Q-learning）
    
    包含两个独立的HybridCritic网络，
    用于减轻Q值过估计问题
    
    参数量: ~320K (两个Critic)
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 22,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        init_type: str = 'orthogonal'
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: LSTM隐藏层维度
            lstm_layers: LSTM层数
            dropout: Dropout比例
            init_type: 参数初始化方式
        """
        super().__init__()
        
        # 创建两个独立的Critic网络
        self.critic1 = HybridCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            init_type=init_type
        )
        
        self.critic2 = HybridCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            init_type=init_type
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden_state1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_state2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播（双Critic）
        
        Args:
            state: 状态
            action: 动作
            hidden_state1: Critic1的隐藏状态
            hidden_state2: Critic2的隐藏状态
        
        Returns:
            q1: Critic1的Q值
            q2: Critic2的Q值
            new_hidden1: Critic1的新隐藏状态
            new_hidden2: Critic2的新隐藏状态
        """
        q1, new_hidden1 = self.critic1(state, action, hidden_state1)
        q2, new_hidden2 = self.critic2(state, action, hidden_state2)
        
        return q1, q2, new_hidden1, new_hidden2
    
    def q1(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """只使用Critic1（用于Actor更新）"""
        return self.critic1(state, action, hidden_state)
    
    def q2(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """只使用Critic2"""
        return self.critic2(state, action, hidden_state)


if __name__ == '__main__':
    """单元测试"""
    print("测试Hybrid SAC Critic模块...")
    
    # 测试1: 单Critic基础功能
    print("\n1. 单Critic基础功能测试")
    critic = HybridCritic(
        state_dim=64,
        action_dim=22,
        hidden_dim=128,
        lstm_layers=1
    )
    
    state = torch.randn(64)
    action = torch.randn(22)
    q_value, hidden = critic(state, action)
    
    print(f"输入: state{state.shape}, action{action.shape}")
    print(f"输出: q_value{q_value.shape}")
    print(f"Hidden state: h{hidden[0].shape}, c{hidden[1].shape}")
    assert q_value.shape == (), f"Expected (), got {q_value.shape}"
    assert hidden[0].shape == (1, 1, 128)
    print(f"Q值: {q_value.item():.3f}")
    print("✓ 单Critic基础功能正常")
    
    # 测试2: Q值有效性
    print("\n2. Q值有效性测试")
    assert torch.isfinite(q_value), "Q值应为有限值"
    print("✓ Q值有效")
    
    # 测试3: Hidden state传递
    print("\n3. Hidden state传递测试")
    state2 = torch.randn(64)
    action2 = torch.randn(22)
    q_value2, hidden2 = critic(state2, action2, hidden)
    
    assert not torch.allclose(hidden[0], hidden2[0]), "Hidden state应该更新"
    print(f"Hidden变化: {(hidden2[0] - hidden[0]).abs().mean().item():.6f}")
    print("✓ Hidden state传递正常")
    
    # 测试4: 批量处理
    print("\n4. 批量处理测试")
    batch_state = torch.randn(4, 64)
    batch_action = torch.randn(4, 22)
    batch_q, batch_hidden = critic(batch_state, batch_action)
    
    print(f"批量输入: state{batch_state.shape}, action{batch_action.shape}")
    print(f"批量输出: q{batch_q.shape}")
    assert batch_q.shape == (4,)
    assert batch_hidden[0].shape == (1, 4, 128)
    print("✓ 批量处理正常")
    
    # 测试5: 梯度流
    print("\n5. 梯度流测试")
    state_grad = torch.randn(64, requires_grad=True)
    action_grad = torch.randn(22, requires_grad=True)
    q_grad, _ = critic(state_grad, action_grad)
    loss = q_grad
    loss.backward()
    
    assert state_grad.grad is not None, "状态梯度不应为None"
    assert action_grad.grad is not None, "动作梯度不应为None"
    assert torch.isfinite(state_grad.grad).all(), "梯度应为有限值"
    print("✓ 梯度流正常")
    
    # 测试6: 单Critic参数量
    print("\n6. 单Critic参数量统计")
    total_params = sum(p.numel() for p in critic.parameters())
    trainable_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 详细分解
    pre_fc_params = sum(p.numel() for p in critic.pre_fc.parameters())
    lstm_params = sum(p.numel() for p in critic.lstm.parameters())
    q_head_params = sum(p.numel() for p in critic.q_head.parameters())
    
    print(f"  - PreFC: {pre_fc_params:,}")
    print(f"  - LSTM: {lstm_params:,}")
    print(f"  - QHead: {q_head_params:,}")
    
    assert 150000 < total_params < 170000, f"参数量应在150K-170K之间，实际{total_params}"
    print("✓ 参数量符合预期")
    
    # 测试7: TwinCritic基础功能
    print("\n7. TwinCritic基础功能测试")
    twin_critic = TwinCritic(
        state_dim=64,
        action_dim=22,
        hidden_dim=128
    )
    
    state = torch.randn(64)
    action = torch.randn(22)
    q1, q2, hidden1, hidden2 = twin_critic(state, action)
    
    print(f"Q1: {q1.item():.3f}, Q2: {q2.item():.3f}")
    assert q1.shape == () and q2.shape == ()
    assert hidden1[0].shape == (1, 1, 128)
    assert hidden2[0].shape == (1, 1, 128)
    print("✓ TwinCritic基础功能正常")
    
    # 测试8: 双Critic独立性
    print("\n8. 双Critic独立性测试")
    # 两个Critic应该产生不同的Q值（随机初始化）
    assert not torch.allclose(q1, q2, atol=1e-3), "两个Critic应该独立（不同的Q值）"
    print(f"Q值差异: {(q1 - q2).abs().item():.3f}")
    print("✓ 双Critic独立")
    
    # 测试9: TwinCritic批量处理
    print("\n9. TwinCritic批量处理测试")
    batch_state = torch.randn(4, 64)
    batch_action = torch.randn(4, 22)
    batch_q1, batch_q2, batch_h1, batch_h2 = twin_critic(batch_state, batch_action)
    
    assert batch_q1.shape == (4,) and batch_q2.shape == (4,)
    assert batch_h1[0].shape == (1, 4, 128)
    assert batch_h2[0].shape == (1, 4, 128)
    print("✓ TwinCritic批量处理正常")
    
    # 测试10: TwinCritic参数量
    print("\n10. TwinCritic参数量统计")
    twin_total = sum(p.numel() for p in twin_critic.parameters())
    twin_trainable = sum(p.numel() for p in twin_critic.parameters() if p.requires_grad)
    
    print(f"总参数量: {twin_total:,}")
    print(f"可训练参数: {twin_trainable:,}")
    print(f"单Critic参数: {total_params:,}")
    print(f"双Critic应约为单Critic的2倍: {twin_total / total_params:.2f}x")
    
    assert twin_total > total_params * 1.9, "TwinCritic应约为单Critic的2倍"
    print("✓ TwinCritic参数量正常")
    
    # 测试11: q1和q2单独调用
    print("\n11. q1/q2单独调用测试")
    q1_only, h1_only = twin_critic.q1(state, action)
    q2_only, h2_only = twin_critic.q2(state, action)
    
    assert q1_only.shape == () and q2_only.shape == ()
    print(f"Q1 only: {q1_only.item():.3f}")
    print(f"Q2 only: {q2_only.item():.3f}")
    print("✓ 单独调用正常")
    
    # 测试12: Action预处理
    print("\n12. Action预处理测试")
    # 测试不同形状的action输入
    action_flat = torch.randn(22)
    action_2d = torch.randn(11, 2)
    
    q_flat, _ = critic(state, action_flat)
    q_2d, _ = critic(state, action_2d.view(-1))
    
    print(f"扁平action输入: {action_flat.shape} → Q={q_flat.item():.3f}")
    print(f"2D action输入: {action_2d.shape} → Q={q_2d.item():.3f}")
    print("✓ Action预处理正常")
    
    # 测试13: 确定性输出（eval模式）
    print("\n13. 确定性输出测试（eval模式）")
    critic.eval()
    twin_critic.eval()
    
    with torch.no_grad():
        q_eval1, _ = critic(state, action)
        q_eval2, _ = critic(state, action)
    
    assert torch.allclose(q_eval1, q_eval2), "eval模式下相同输入应产生相同输出"
    print("✓ 确定性输出正常")
    
    # 测试14: 序列处理模拟
    print("\n14. 序列处理模拟测试")
    critic.train()
    sequence_length = 5
    hidden = None
    q_values = []
    
    for t in range(sequence_length):
        state_t = torch.randn(64)
        action_t = torch.randn(22)
        q_t, hidden = critic(state_t, action_t, hidden)
        q_values.append(q_t.item())
    
    print(f"序列Q值: {[f'{q:.2f}' for q in q_values]}")
    assert len(q_values) == sequence_length
    print("✓ 序列处理正常")
    
    print("\n✅ Hybrid SAC Critic测试全部通过！")

