"""
Hybrid SAC Actor模块
输出连续动作（路径点），支持LSTM时序记忆
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional


class HybridActor(nn.Module):
    """
    混合SAC Actor网络
    
    设计思路：
    - 使用LSTM捕捉时序依赖
    - 重参数化技巧实现可微采样
    - Tanh变换约束动作空间到[-1, 1]
    - 数值稳定的log_prob计算
    
    输入:
        state: (64,) 或 (batch, 64) 融合后的状态
        hidden_state: Optional[(h, c)] LSTM隐藏状态
            - None: Episode开始，自动初始化零向量
            - (h, c): 传递上一步的隐藏状态
    
    输出:
        action: (22,) 或 (batch, 22) 扁平格式的路径点
        log_prob: () 或 (batch,) 对数概率
        new_hidden_state: (h, c) 更新后的隐藏状态
    
    参数量: ~81K
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 22,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_type: str = 'orthogonal'
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度（11个点×2维=22）
            hidden_dim: LSTM隐藏层维度
            lstm_layers: LSTM层数
            log_std_min: log标准差下限
            log_std_max: log标准差上限
            init_type: 参数初始化方式 ('orthogonal', 'xavier')
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 1. 预处理层：将状态投影到LSTM输入维度
        self.pre_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. LSTM层：捕捉时序依赖
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=False  # 输入格式: (seq_len, batch, hidden_dim)
        )
        
        # 3. 策略头：输出均值和log标准差
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
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
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            state: (64,) 或 (batch, 64) 状态特征
            hidden_state: Optional[(h, c)] LSTM隐藏状态
            deterministic: 是否使用确定性策略（返回均值）
        
        Returns:
            action: (22,) 或 (batch, 22) 动作
            log_prob: () 或 (batch,) 对数概率
            new_hidden_state: (h, c) 新的隐藏状态
        """
        # 处理输入维度
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, 64)
            single_input = True
        else:
            single_input = False
        
        batch_size = state.size(0)
        device = state.device
        
        # 1. 初始化hidden_state（如果需要）
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, device)
        
        # 2. 预处理
        x = self.pre_fc(state)  # (batch, 128)
        
        # 3. LSTM前向传播
        # 需要reshape: (seq_len=1, batch, hidden_dim)
        x = x.unsqueeze(0)  # (1, batch, 128)
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(0)  # (batch, 128)
        
        # 4. 计算均值和log标准差
        mean = self.mean_head(lstm_out)  # (batch, 22)
        log_std = self.log_std_head(lstm_out)  # (batch, 22)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        # 5. 采样或返回均值
        if deterministic:
            # 确定性策略：直接返回均值
            action = torch.tanh(mean)
            log_prob = torch.zeros(batch_size, 1, device=device)
        else:
            # 随机策略：重参数化采样
            normal = Normal(mean, std)
            x_t = normal.rsample()  # 可微采样
            action = torch.tanh(x_t)  # 压缩到[-1, 1]
            
            # 计算log_prob（数值稳定版本）
            log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)  # (batch, 1)
            
            # 修正tanh变换的雅可比行列式
            # log|det(d tanh(x)/dx)| = log(1 - tanh^2(x))
            # 数值稳定版本: log(1 - tanh^2(x)) = 2*(log(2) - x - softplus(-2x))
            log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(
                dim=-1, keepdim=True
            )
        
        # 6. 处理输出维度
        if single_input:
            action = action.squeeze(0)  # (22,)
            log_prob = log_prob.squeeze()  # ()
        else:
            log_prob = log_prob.squeeze(-1)  # (batch,)
        
        return action, log_prob, new_hidden_state
    
    def select_action(
        self,
        state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        推理接口：选择动作（不返回log_prob）
        
        Args:
            state: 状态
            hidden_state: LSTM隐藏状态
            deterministic: 是否确定性输出
        
        Returns:
            action: 动作
            new_hidden_state: 新的隐藏状态
        """
        action, _, new_hidden_state = self.forward(
            state, hidden_state, deterministic
        )
        return action, new_hidden_state
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        评估动作：返回动作、log_prob、熵和新隐藏状态
        用于Actor更新时重新计算log_prob
        
        Returns:
            action: (batch, 22)
            log_prob: (batch,)
            entropy: (batch,) 策略熵
            new_hidden_state: (h, c)
        """
        # 处理输入
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.size(0)
        device = state.device
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, device)
        
        # 前向传播
        x = self.pre_fc(state)
        x = x.unsqueeze(0)
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(0)
        
        mean = self.mean_head(lstm_out)
        log_std = self.log_std_head(lstm_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        # 采样
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # 计算log_prob
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(
            dim=-1, keepdim=True
        )
        log_prob = log_prob.squeeze(-1)
        
        # 计算熵
        entropy = normal.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, new_hidden_state


if __name__ == '__main__':
    """单元测试"""
    print("测试Hybrid SAC Actor模块...")
    
    # 测试1: 基础功能
    print("\n1. 基础功能测试")
    actor = HybridActor(
        state_dim=64,
        action_dim=22,
        hidden_dim=128,
        lstm_layers=1
    )
    
    state = torch.randn(64)
    action, log_prob, hidden = actor(state)
    
    print(f"输入: state{state.shape}")
    print(f"输出: action{action.shape}, log_prob{log_prob.shape}")
    print(f"Hidden state: h{hidden[0].shape}, c{hidden[1].shape}")
    assert action.shape == (22,), f"Expected (22,), got {action.shape}"
    assert log_prob.shape == (), f"Expected (), got {log_prob.shape}"
    assert hidden[0].shape == (1, 1, 128)
    print("✓ 基础功能正常")
    
    # 测试2: 动作范围约束
    print("\n2. 动作范围约束测试")
    assert (action >= -1).all() and (action <= 1).all(), "动作应在[-1, 1]范围内"
    print(f"动作范围: [{action.min().item():.3f}, {action.max().item():.3f}]")
    print("✓ 动作约束正常")
    
    # 测试3: Log prob范围
    print("\n3. Log probability测试")
    assert log_prob <= 0, f"log_prob应为负数，实际: {log_prob.item()}"
    assert torch.isfinite(log_prob), "log_prob应为有限值"
    print(f"Log prob: {log_prob.item():.3f}")
    print("✓ Log prob正常")
    
    # 测试4: Hidden state传递
    print("\n4. Hidden state传递测试")
    state2 = torch.randn(64)
    action2, log_prob2, hidden2 = actor(state2, hidden)
    
    # 验证hidden确实改变了
    assert not torch.allclose(hidden[0], hidden2[0]), "Hidden state应该更新"
    print(f"Hidden变化: {(hidden2[0] - hidden[0]).abs().mean().item():.6f}")
    print("✓ Hidden state传递正常")
    
    # 测试5: 批量处理
    print("\n5. 批量处理测试")
    batch_state = torch.randn(4, 64)
    batch_action, batch_log_prob, batch_hidden = actor(batch_state)
    
    print(f"批量输入: {batch_state.shape}")
    print(f"批量输出: action{batch_action.shape}, log_prob{batch_log_prob.shape}")
    assert batch_action.shape == (4, 22)
    assert batch_log_prob.shape == (4,)
    assert batch_hidden[0].shape == (1, 4, 128)
    print("✓ 批量处理正常")
    
    # 测试6: 确定性模式
    print("\n6. 确定性模式测试")
    actor.eval()
    with torch.no_grad():
        det_action1, det_log_prob1, _ = actor(state, deterministic=True)
        det_action2, det_log_prob2, _ = actor(state, deterministic=True)
    
    assert torch.allclose(det_action1, det_action2), "确定性模式应产生相同输出"
    assert det_log_prob1 == 0 and det_log_prob2 == 0, "确定性模式log_prob应为0"
    print("✓ 确定性模式正常")
    
    # 测试7: 随机性模式
    print("\n7. 随机性模式测试")
    actor.train()
    action_rand1, _, _ = actor(state)
    action_rand2, _, _ = actor(state)
    
    assert not torch.allclose(action_rand1, action_rand2), "随机模式应产生不同输出"
    print(f"动作差异: {(action_rand1 - action_rand2).abs().mean().item():.6f}")
    print("✓ 随机性模式正常")
    
    # 测试8: 梯度流
    print("\n8. 梯度流测试")
    state_grad = torch.randn(64, requires_grad=True)
    action_grad, log_prob_grad, _ = actor(state_grad)
    loss = action_grad.sum() + log_prob_grad
    loss.backward()
    
    assert state_grad.grad is not None, "梯度应传播到输入"
    assert torch.isfinite(state_grad.grad).all(), "梯度应为有限值"
    print("✓ 梯度流正常")
    
    # 测试9: 参数量统计
    print("\n9. 参数量统计")
    total_params = sum(p.numel() for p in actor.parameters())
    trainable_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 详细分解
    pre_fc_params = sum(p.numel() for p in actor.pre_fc.parameters())
    lstm_params = sum(p.numel() for p in actor.lstm.parameters())
    mean_params = sum(p.numel() for p in actor.mean_head.parameters())
    std_params = sum(p.numel() for p in actor.log_std_head.parameters())
    
    print(f"  - PreFC: {pre_fc_params:,}")
    print(f"  - LSTM: {lstm_params:,}")
    print(f"  - MeanHead: {mean_params:,}")
    print(f"  - LogStdHead: {std_params:,}")
    
    assert 140000 < total_params < 150000, f"参数量应在140K-150K之间，实际{total_params}"
    print("✓ 参数量符合预期（约146K）")
    
    # 测试10: select_action接口
    print("\n10. select_action接口测试")
    action_select, hidden_select = actor.select_action(state, deterministic=False)
    assert action_select.shape == (22,)
    assert hidden_select[0].shape == (1, 1, 128)
    print("✓ select_action接口正常")
    
    # 测试11: evaluate_actions接口
    print("\n11. evaluate_actions接口测试")
    batch_state = torch.randn(4, 64)
    action_eval, log_prob_eval, entropy_eval, hidden_eval = actor.evaluate_actions(batch_state)
    
    assert action_eval.shape == (4, 22)
    assert log_prob_eval.shape == (4,)
    assert entropy_eval.shape == (4,)
    assert entropy_eval.mean() > 0, "熵应为正数"
    print(f"平均熵: {entropy_eval.mean().item():.3f}")
    print("✓ evaluate_actions接口正常")
    
    print("\n✅ Hybrid SAC Actor测试全部通过！")

