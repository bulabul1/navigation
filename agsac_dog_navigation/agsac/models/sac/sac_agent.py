"""
SAC Agent主控制器
整合Actor和Critic，实现完整的SAC训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import copy

from .actor import HybridActor
from .critic import TwinCritic


class SACAgent(nn.Module):
    """
    Soft Actor-Critic Agent
    
    组件：
    - actor: 策略网络（HybridActor）
    - critic: 双Q网络（TwinCritic）
    - critic_target: 目标Q网络
    - log_alpha: 熵系数（可学习或固定）
    
    功能：
    - select_action(): 推理/数据收集
    - update(): 训练更新（支持序列片段）
    - soft_update_target(): 软更新目标网络
    - save_checkpoint()/load_checkpoint(): 模型保存加载
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 22,
        hidden_dim: int = 128,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_entropy: bool = True,
        target_entropy: Optional[float] = None,
        max_grad_norm: float = 1.0,
        device: str = 'cpu'
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: LSTM隐藏层维度
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            alpha_lr: 熵系数学习率
            gamma: 折扣因子
            tau: 软更新系数
            auto_entropy: 是否自动调节熵系数
            target_entropy: 目标熵（默认为-action_dim）
            max_grad_norm: 梯度裁剪阈值
            device: 设备
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy = auto_entropy
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # 目标熵（默认为-action_dim）
        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy
        
        # 创建网络
        self.actor = HybridActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.critic = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.critic_target = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # 硬拷贝critic参数到critic_target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 冻结target网络（不需要梯度）
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )
        
        # 熵系数
        if auto_entropy:
            # 可学习的log(alpha)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            # 固定alpha
            self.log_alpha = torch.log(torch.tensor(0.2, device=self.device))
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = None
        
        # 训练步数
        self.total_updates = 0
    
    def select_action(
        self,
        state: torch.Tensor,
        hidden_actor: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        选择动作（推理/数据收集）
        
        Args:
            state: (64,) 状态
            hidden_actor: Actor的隐藏状态
            deterministic: 是否确定性策略
        
        Returns:
            action: (22,) 动作
            new_hidden_actor: 新的隐藏状态
        """
        with torch.no_grad():
            action, new_hidden_actor = self.actor.select_action(
                state, hidden_actor, deterministic
            )
        return action, new_hidden_actor
    
    def update(self, segment_batch: List[Dict]) -> Dict[str, float]:
        """
        使用序列片段batch更新网络
        
        Args:
            segment_batch: List[dict]，每个dict包含:
                - states: (seq_len, 64) 状态序列
                - actions: (seq_len, 22) 动作序列
                - rewards: (seq_len,) 奖励序列
                - next_states: (seq_len, 64) 下一状态序列
                - dones: (seq_len,) 终止标志序列
                - init_hidden_actor: (h, c) Actor初始隐藏状态
                - init_hidden_critic1: (h, c) Critic1初始隐藏状态
                - init_hidden_critic2: (h, c) Critic2初始隐藏状态
        
        Returns:
            logs: dict，包含损失和指标信息
        """
        batch_size = len(segment_batch)
        
        # 累积损失
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        
        # 累积指标
        total_q1_mean = 0.0
        total_q2_mean = 0.0
        total_log_prob_mean = 0.0
        
        # 更新alpha（如果启用自动熵调节）
        if self.auto_entropy:
            self.alpha = self.log_alpha.exp()
        
        # ==================== 1. 更新Critic ====================
        self.critic_optimizer.zero_grad()
        
        for segment in segment_batch:
            states = segment['states'].to(self.device)  # (seq_len, 64)
            actions = segment['actions'].to(self.device)  # (seq_len, 22)
            rewards = segment['rewards'].to(self.device)  # (seq_len,)
            next_states = segment['next_states'].to(self.device)  # (seq_len, 64)
            dones = segment['dones'].to(self.device)  # (seq_len,)
            
            seq_len = states.shape[0]
            
            # 初始化隐藏状态
            # Critic更新时hidden=None（根据设计决策）
            h_c1 = None
            h_c2 = None
            h_actor = None
            
            # 逐步计算target Q和current Q
            for t in range(seq_len):
                # 计算target Q（使用target网络和target actor）
                with torch.no_grad():
                    # 使用Actor采样下一个动作
                    next_action, next_log_prob, h_actor = self.actor(
                        next_states[t], h_actor, deterministic=False
                    )
                    
                    # 使用Target Critic评估
                    next_q1, next_q2, _, _ = self.critic_target(
                        next_states[t], next_action, None, None
                    )
                    
                    # 取较小的Q值（Double Q-learning）
                    min_next_q = torch.min(next_q1, next_q2)
                    
                    # 计算target Q（带熵正则化）
                    target_q = rewards[t] + self.gamma * (1 - dones[t]) * (
                        min_next_q - self.alpha * next_log_prob
                    )
                
                # 计算current Q
                curr_q1, curr_q2, h_c1, h_c2 = self.critic(
                    states[t], actions[t], h_c1, h_c2
                )
                
                # 确保维度一致（target_q和curr_q都应该是标量）
                if target_q.dim() > 0:
                    target_q = target_q.squeeze()
                if curr_q1.dim() > 0:
                    curr_q1 = curr_q1.squeeze()
                if curr_q2.dim() > 0:
                    curr_q2 = curr_q2.squeeze()
                
                # 累积Critic损失（MSE）
                total_critic_loss += F.mse_loss(curr_q1, target_q)
                total_critic_loss += F.mse_loss(curr_q2, target_q)
                
                # 记录指标
                total_q1_mean += curr_q1.item()
                total_q2_mean += curr_q2.item()
        
        # 平均Critic损失
        critic_loss = total_critic_loss / batch_size
        
        # 反向传播和梯度裁剪
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.critic_optimizer.step()
        
        # ==================== 2. 更新Actor ====================
        self.actor_optimizer.zero_grad()
        
        for segment in segment_batch:
            states = segment['states'].to(self.device)
            seq_len = states.shape[0]
            
            # 重置隐藏状态
            h_actor = segment['init_hidden_actor']
            if h_actor is not None:
                h_actor = (h_actor[0].to(self.device), h_actor[1].to(self.device))
            
            # 逐步重新采样动作并计算loss
            for t in range(seq_len):
                # 重新采样动作
                new_action, log_prob, h_actor = self.actor(
                    states[t], h_actor, deterministic=False
                )
                
                # 使用Critic1评估（hidden=None）
                q_value, _ = self.critic.q1(states[t], new_action, None)
                
                # Actor loss（最大化Q - α*entropy）
                # 等价于最小化 α*log_prob - Q
                total_actor_loss += (self.alpha * log_prob - q_value)
                
                # 记录log_prob
                total_log_prob_mean += log_prob.item()
        
        # 平均Actor损失
        actor_loss = total_actor_loss / batch_size
        
        # 反向传播和梯度裁剪
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.max_grad_norm
        )
        self.actor_optimizer.step()
        
        # ==================== 3. 更新Alpha（自动熵调节）====================
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy and self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
            
            # 重新计算log_prob（需要detach）
            total_log_prob_for_alpha = 0.0
            for segment in segment_batch:
                states = segment['states'].to(self.device)
                seq_len = states.shape[0]
                h_actor = segment['init_hidden_actor']
                if h_actor is not None:
                    h_actor = (h_actor[0].to(self.device), h_actor[1].to(self.device))
                
                for t in range(seq_len):
                    _, log_prob, h_actor = self.actor(
                        states[t], h_actor, deterministic=False
                    )
                    total_log_prob_for_alpha += log_prob.detach().item()
            
            # Alpha loss（最小化 -log_alpha * (log_prob + target_entropy)）
            # 计算总样本数（所有segment的seq_len之和）
            total_samples = sum(segment['states'].shape[0] for segment in segment_batch)
            avg_log_prob = total_log_prob_for_alpha / total_samples
            alpha_loss = -(
                self.log_alpha * (avg_log_prob + self.target_entropy)
            )
            
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # 更新alpha
            self.alpha = self.log_alpha.exp()
            
            total_alpha_loss = alpha_loss.item()
        else:
            total_alpha_loss = 0.0
        
        # ==================== 4. 软更新Target网络 ====================
        self.soft_update_target()
        
        # 更新计数
        self.total_updates += 1
        
        # 返回日志
        logs = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': total_alpha_loss,
            'q1_mean': total_q1_mean / (batch_size * seq_len),
            'q2_mean': total_q2_mean / (batch_size * seq_len),
            'log_prob_mean': total_log_prob_mean / (batch_size * seq_len),
            'alpha': self.alpha.item(),
            'critic_grad_norm': critic_grad_norm.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'total_updates': self.total_updates
        }
        
        return logs
    
    def soft_update_target(self):
        """软更新target网络"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save_checkpoint(self, filepath: str):
        """保存checkpoint"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'total_updates': self.total_updates,
        }
        
        if self.alpha_optimizer is not None:
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """加载checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()
        self.total_updates = checkpoint['total_updates']
        
        if self.alpha_optimizer is not None and 'alpha_optimizer_state_dict' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        super().train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        # target网络始终保持eval模式
        self.critic_target.eval()
        return self
    
    def eval(self):
        """设置评估模式"""
        return self.train(False)


if __name__ == '__main__':
    """简单测试"""
    print("测试SAC Agent...")
    
    # 创建Agent
    agent = SACAgent(
        state_dim=64,
        action_dim=22,
        hidden_dim=128,
        device='cpu'
    )
    
    print(f"✓ Agent创建成功")
    print(f"  - Actor参数: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  - Critic参数: {sum(p.numel() for p in agent.critic.parameters()):,}")
    print(f"  - 总参数: {sum(p.numel() for p in agent.parameters()):,}")
    
    # 测试select_action
    state = torch.randn(64)
    action, hidden = agent.select_action(state, deterministic=False)
    print(f"\n✓ select_action测试通过")
    print(f"  - Action shape: {action.shape}")
    print(f"  - Hidden shape: h{hidden[0].shape}, c{hidden[1].shape}")
    
    # 测试update（创建简单的segment batch）
    segment_batch = []
    for _ in range(2):  # batch_size=2
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
        segment_batch.append(segment)
    
    logs = agent.update(segment_batch)
    
    print(f"\n✓ update测试通过")
    print(f"  - Critic loss: {logs['critic_loss']:.4f}")
    print(f"  - Actor loss: {logs['actor_loss']:.4f}")
    print(f"  - Alpha: {logs['alpha']:.4f}")
    print(f"  - Q1 mean: {logs['q1_mean']:.4f}")
    print(f"  - Q2 mean: {logs['q2_mean']:.4f}")
    
    print("\n✅ SAC Agent基础测试全部通过！")

