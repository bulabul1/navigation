"""
机器狗状态编码器
使用GRU处理历史轨迹，MLP处理速度和目标位置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DogStateEncoder(nn.Module):
    """
    机器狗状态编码器
    
    输入：
        - 历史轨迹 (8, 2)
        - 当前速度 (2,)
        - 当前位置 (2,)
        - 目标位置 (2,)
    
    输出：
        - 状态特征 (64,)
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        gru_layers: int = 2,
        vel_hidden_dim: int = 32,
        goal_hidden_dim: int = 32,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: 输出特征维度
            gru_layers: GRU层数
            vel_hidden_dim: 速度编码的隐藏维度
            goal_hidden_dim: 目标编码的隐藏维度
            dropout: Dropout比例
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        
        # GRU处理历史轨迹序列
        self.trajectory_gru = nn.GRU(
            input_size=2,  # (x, y)坐标
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        
        # 速度编码器
        self.vel_encoder = nn.Sequential(
            nn.Linear(2, vel_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(vel_hidden_dim, vel_hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 目标编码器（编码相对目标）
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, goal_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(goal_hidden_dim, goal_hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 融合所有特征
        fusion_input_dim = hidden_dim + vel_hidden_dim + goal_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 层归一化（可选，提高训练稳定性）
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        past_trajectory: torch.Tensor,
        current_velocity: torch.Tensor,
        current_position: torch.Tensor,
        goal_position: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            past_trajectory: Tensor(seq_len, 2) 或 Tensor(batch, seq_len, 2)
                历史轨迹，seq_len通常为8
            current_velocity: Tensor(2,) 或 Tensor(batch, 2)
                当前速度 [vx, vy]
            current_position: Tensor(2,) 或 Tensor(batch, 2)
                当前位置 [x, y]
            goal_position: Tensor(2,) 或 Tensor(batch, 2)
                目标位置 [x, y]
        
        Returns:
            state_features: Tensor(hidden_dim,) 或 Tensor(batch, hidden_dim)
                机器狗状态特征
        """
        # 处理输入维度
        if past_trajectory.dim() == 2:
            # 单个样本: (seq_len, 2)
            past_trajectory = past_trajectory.unsqueeze(0)  # (1, seq_len, 2)
            current_velocity = current_velocity.unsqueeze(0)  # (1, 2)
            current_position = current_position.unsqueeze(0)  # (1, 2)
            goal_position = goal_position.unsqueeze(0)  # (1, 2)
            single_input = True
        else:
            single_input = False
        
        batch_size = past_trajectory.size(0)
        
        # 1. GRU处理历史轨迹
        # 可以选择归一化轨迹（相对于当前位置）
        relative_trajectory = past_trajectory - current_position.unsqueeze(1)
        
        # GRU前向传播
        # gru_out: (batch, seq_len, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        gru_out, h_n = self.trajectory_gru(relative_trajectory)
        
        # 取最后一层的隐藏状态
        traj_feature = h_n[-1]  # (batch, hidden_dim)
        
        # 2. 编码当前速度
        vel_feature = self.vel_encoder(current_velocity)  # (batch, vel_hidden_dim)
        
        # 3. 编码相对目标
        relative_goal = goal_position - current_position  # (batch, 2)
        goal_feature = self.goal_encoder(relative_goal)  # (batch, goal_hidden_dim)
        
        # 4. 融合所有特征
        combined = torch.cat([
            traj_feature,   # (batch, hidden_dim)
            vel_feature,    # (batch, vel_hidden_dim)
            goal_feature    # (batch, goal_hidden_dim)
        ], dim=-1)  # (batch, fusion_input_dim)
        
        fused_features = self.fusion(combined)  # (batch, hidden_dim)
        
        # 5. 层归一化
        output_features = self.layer_norm(fused_features)  # (batch, hidden_dim)
        
        # 如果输入是单个样本，返回单个特征向量
        if single_input:
            output_features = output_features.squeeze(0)
        
        return output_features
    
    def get_trajectory_features(
        self,
        past_trajectory: torch.Tensor,
        current_position: torch.Tensor
    ) -> torch.Tensor:
        """
        仅提取轨迹特征（用于分析）
        
        Args:
            past_trajectory: Tensor(seq_len, 2) 或 Tensor(batch, seq_len, 2)
            current_position: Tensor(2,) 或 Tensor(batch, 2)
        
        Returns:
            traj_features: Tensor(hidden_dim,) 或 Tensor(batch, hidden_dim)
        """
        if past_trajectory.dim() == 2:
            past_trajectory = past_trajectory.unsqueeze(0)
            current_position = current_position.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        relative_trajectory = past_trajectory - current_position.unsqueeze(1)
        _, h_n = self.trajectory_gru(relative_trajectory)
        traj_feature = h_n[-1]
        
        if single_input:
            traj_feature = traj_feature.squeeze(0)
        
        return traj_feature


class SimpleDogStateEncoder(nn.Module):
    """
    简化版机器狗状态编码器
    
    使用更简单的架构，适合快速原型和测试
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 轨迹编码：直接展平+MLP（无GRU）
        self.traj_encoder = nn.Sequential(
            nn.Linear(8 * 2, hidden_dim),  # 8个时间步 * 2维坐标
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 速度和目标编码
        self.vel_goal_encoder = nn.Sequential(
            nn.Linear(4, 32),  # vel(2) + relative_goal(2)
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        past_trajectory: torch.Tensor,
        current_velocity: torch.Tensor,
        current_position: torch.Tensor,
        goal_position: torch.Tensor
    ) -> torch.Tensor:
        """简化版前向传播"""
        if past_trajectory.dim() == 2:
            past_trajectory = past_trajectory.unsqueeze(0)
            current_velocity = current_velocity.unsqueeze(0)
            current_position = current_position.unsqueeze(0)
            goal_position = goal_position.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # 轨迹展平
        traj_flat = past_trajectory.reshape(past_trajectory.size(0), -1)
        traj_feature = self.traj_encoder(traj_flat)
        
        # 速度和相对目标
        relative_goal = goal_position - current_position
        vel_goal = torch.cat([current_velocity, relative_goal], dim=-1)
        vel_goal_feature = self.vel_goal_encoder(vel_goal)
        
        # 融合
        combined = torch.cat([traj_feature, vel_goal_feature], dim=-1)
        output = self.fusion(combined)
        
        if single_input:
            output = output.squeeze(0)
        
        return output


class AttentiveDogStateEncoder(nn.Module):
    """
    带注意力机制的机器狗状态编码器
    
    使用自注意力处理历史轨迹
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 轨迹点嵌入
        self.point_embedding = nn.Linear(2, hidden_dim)
        
        # 位置编码（时间编码）
        self.time_encoding = nn.Parameter(torch.randn(1, 8, hidden_dim))
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 速度和目标编码
        self.vel_encoder = nn.Linear(2, 32)
        self.goal_encoder = nn.Linear(2, 32)
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(
        self,
        past_trajectory: torch.Tensor,
        current_velocity: torch.Tensor,
        current_position: torch.Tensor,
        goal_position: torch.Tensor
    ) -> torch.Tensor:
        """注意力版前向传播"""
        if past_trajectory.dim() == 2:
            past_trajectory = past_trajectory.unsqueeze(0)
            current_velocity = current_velocity.unsqueeze(0)
            current_position = current_position.unsqueeze(0)
            goal_position = goal_position.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = past_trajectory.size(0)
        
        # 轨迹点嵌入
        embedded_traj = self.point_embedding(past_trajectory)  # (batch, 8, hidden_dim)
        
        # 添加时间编码
        embedded_traj = embedded_traj + self.time_encoding
        
        # 自注意力
        attn_out, _ = self.self_attention(
            embedded_traj,
            embedded_traj,
            embedded_traj
        )  # (batch, 8, hidden_dim)
        
        # 平均池化
        traj_feature = attn_out.mean(dim=1)  # (batch, hidden_dim)
        
        # 速度和目标
        vel_feature = self.vel_encoder(current_velocity)
        relative_goal = goal_position - current_position
        goal_feature = self.goal_encoder(relative_goal)
        
        # 融合
        combined = torch.cat([traj_feature, vel_feature, goal_feature], dim=-1)
        output = self.fusion(combined)
        
        if single_input:
            output = output.squeeze(0)
        
        return output


# 工厂函数
def create_dog_state_encoder(
    encoder_type: str = 'gru',
    hidden_dim: int = 64,
    **kwargs
) -> nn.Module:
    """
    创建机器狗状态编码器的工厂函数
    
    Args:
        encoder_type: 编码器类型
            - 'gru': GRU版本（推荐）
            - 'simple': 简化版
            - 'attention': 注意力版
        hidden_dim: 输出特征维度
        **kwargs: 其他参数
    
    Returns:
        encoder: 机器狗状态编码器
    """
    if encoder_type == 'gru':
        return DogStateEncoder(hidden_dim=hidden_dim, **kwargs)
    elif encoder_type == 'simple':
        return SimpleDogStateEncoder(hidden_dim=hidden_dim)
    elif encoder_type == 'attention':
        return AttentiveDogStateEncoder(hidden_dim=hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == '__main__':
    """简单测试"""
    print("测试机器狗状态编码器...")
    
    # 测试1: GRU版本
    print("\n1. GRU版本")
    encoder = DogStateEncoder(hidden_dim=64, gru_layers=2)
    
    # 单个样本
    past_traj = torch.randn(8, 2)
    current_vel = torch.randn(2)
    current_pos = torch.zeros(2)
    goal_pos = torch.tensor([10.0, 10.0])
    
    features = encoder(past_traj, current_vel, current_pos, goal_pos)
    print(f"输入: 轨迹{past_traj.shape}, 速度{current_vel.shape}")
    print(f"输出: {features.shape}")
    
    # 批量样本
    batch_traj = torch.randn(4, 8, 2)
    batch_vel = torch.randn(4, 2)
    batch_pos = torch.randn(4, 2)
    batch_goal = torch.randn(4, 2)
    
    batch_features = encoder(batch_traj, batch_vel, batch_pos, batch_goal)
    print(f"批量输入: {batch_traj.shape}")
    print(f"批量输出: {batch_features.shape}")
    
    # 测试2: 简化版本
    print("\n2. 简化版本")
    simple_encoder = SimpleDogStateEncoder(hidden_dim=64)
    simple_features = simple_encoder(past_traj, current_vel, current_pos, goal_pos)
    print(f"简化版输出: {simple_features.shape}")
    
    # 测试3: 注意力版本
    print("\n3. 注意力版本")
    attn_encoder = AttentiveDogStateEncoder(hidden_dim=64, num_heads=4)
    attn_features = attn_encoder(past_traj, current_vel, current_pos, goal_pos)
    print(f"注意力版输出: {attn_features.shape}")
    
    # 测试4: 工厂函数
    print("\n4. 工厂函数")
    encoders = {
        'gru': create_dog_state_encoder('gru', hidden_dim=64),
        'simple': create_dog_state_encoder('simple', hidden_dim=64),
        'attention': create_dog_state_encoder('attention', hidden_dim=64)
    }
    
    for name, enc in encoders.items():
        feat = enc(past_traj, current_vel, current_pos, goal_pos)
        print(f"{name}: {feat.shape}")
    
    # 测试5: 梯度流
    print("\n5. 梯度流测试")
    past_traj_grad = torch.randn(8, 2, requires_grad=True)
    features_grad = encoder(past_traj_grad, current_vel, current_pos, goal_pos)
    loss = features_grad.sum()
    loss.backward()
    print(f"梯度形状: {past_traj_grad.grad.shape}")
    print(f"梯度是否有限: {torch.isfinite(past_traj_grad.grad).all()}")
    
    # 测试6: 参数量统计
    print("\n6. 参数量统计")
    for name, enc in encoders.items():
        params = sum(p.numel() for p in enc.parameters())
        print(f"{name}编码器参数: {params:,}")
    
    print("\n✓ 机器狗状态编码器测试通过！")