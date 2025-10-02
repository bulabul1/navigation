"""
SocialCircle社交交互表示
基于论文将社交空间划分为多个扇区，编码每个扇区内的邻居信息
用于行人轨迹预测的社交上下文编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SocialCircle(nn.Module):
    """
    SocialCircle社交交互编码器
    
    将目标行人周围的空间划分为多个扇区（类似雷达图），
    编码每个扇区内的邻居信息，形成社交上下文表示
    
    输入：
        - target_trajectory: (8, 2) 目标行人的历史轨迹
        - neighbor_trajectories: (N_neighbors, 8, 2) 邻居行人的历史轨迹
        - relative_angles: (N_neighbors,) 邻居相对于目标的角度
    
    输出：
        - social_features: (feature_dim,) 社交上下文特征
    """
    
    def __init__(
        self,
        num_sectors: int = 8,
        feature_dim: int = 64,
        trajectory_dim: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_distance: float = 10.0
    ):
        """
        Args:
            num_sectors: 扇区数量（将360度划分为几份）
            feature_dim: 输出特征维度
            trajectory_dim: 轨迹点的维度（2D为2）
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
            max_distance: 最大社交距离（超过此距离的邻居影响较小）
        """
        super().__init__()
        
        self.num_sectors = num_sectors
        self.feature_dim = feature_dim
        self.trajectory_dim = trajectory_dim
        self.hidden_dim = hidden_dim
        self.max_distance = max_distance
        
        # 目标行人轨迹编码器（GRU）
        self.target_encoder = nn.GRU(
            input_size=trajectory_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 邻居轨迹编码器（GRU，共享权重）
        self.neighbor_encoder = nn.GRU(
            input_size=trajectory_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
        # 扇区特征编码器
        # 每个扇区包含：邻居数量、平均距离、平均速度、编码后的轨迹特征
        sector_input_dim = 1 + 1 + 2 + hidden_dim // 2  # count + dist + vel(2) + traj_feat
        self.sector_encoder = nn.Sequential(
            nn.Linear(sector_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 融合目标和扇区特征
        fusion_input_dim = hidden_dim + num_sectors * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def _compute_sector_assignment(
        self,
        relative_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        将邻居分配到对应的扇区
        
        Args:
            relative_angles: (N_neighbors,) 相对角度，范围[0, 2π]
        
        Returns:
            sector_ids: (N_neighbors,) 扇区ID，范围[0, num_sectors-1]
        """
        # 每个扇区的角度范围
        sector_size = 2 * math.pi / self.num_sectors
        
        # 计算扇区ID
        sector_ids = (relative_angles / sector_size).long()
        sector_ids = torch.clamp(sector_ids, 0, self.num_sectors - 1)
        
        return sector_ids
    
    def _aggregate_sector_features(
        self,
        neighbor_features: torch.Tensor,
        neighbor_distances: torch.Tensor,
        neighbor_velocities: torch.Tensor,
        sector_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        聚合每个扇区内的邻居特征
        
        Args:
            neighbor_features: (N_neighbors, hidden_dim//2) 邻居轨迹特征
            neighbor_distances: (N_neighbors,) 邻居距离
            neighbor_velocities: (N_neighbors, 2) 邻居速度
            sector_ids: (N_neighbors,) 扇区ID
        
        Returns:
            sector_features: (num_sectors, sector_input_dim) 每个扇区的聚合特征
        """
        device = neighbor_features.device
        num_neighbors = neighbor_features.size(0)
        
        # 初始化扇区特征
        sector_input_dim = 1 + 1 + 2 + self.hidden_dim // 2
        sector_features = torch.zeros(
            self.num_sectors, 
            sector_input_dim,
            device=device
        )
        
        if num_neighbors == 0:
            return sector_features
        
        # 对每个扇区聚合特征
        for sector_id in range(self.num_sectors):
            # 找到属于该扇区的邻居
            mask = (sector_ids == sector_id)
            
            if mask.sum() == 0:
                # 该扇区没有邻居，保持零特征
                continue
            
            # 提取该扇区的邻居信息
            sector_neighbor_features = neighbor_features[mask]  # (n, hidden_dim//2)
            sector_distances = neighbor_distances[mask]  # (n,)
            sector_velocities = neighbor_velocities[mask]  # (n, 2)
            
            # 聚合统计信息
            count = mask.sum().float()  # 邻居数量
            avg_distance = sector_distances.mean()  # 平均距离
            avg_velocity = sector_velocities.mean(dim=0)  # 平均速度 (2,)
            
            # 加权平均轨迹特征（距离越近权重越大）
            weights = torch.exp(-sector_distances / self.max_distance)
            weights = weights / (weights.sum() + 1e-8)
            weighted_features = (sector_neighbor_features * weights.unsqueeze(-1)).sum(dim=0)
            
            # 组合扇区特征
            sector_features[sector_id] = torch.cat([
                count.unsqueeze(0),              # (1,)
                avg_distance.unsqueeze(0),       # (1,)
                avg_velocity,                    # (2,)
                weighted_features                # (hidden_dim//2,)
            ])
        
        return sector_features
    
    def forward(
        self,
        target_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        relative_angles: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            target_trajectory: Tensor(8, 2) 或 Tensor(batch, 8, 2)
                目标行人的历史轨迹
            neighbor_trajectories: Tensor(N_neighbors, 8, 2) 或 Tensor(batch, N_neighbors, 8, 2)
                邻居行人的历史轨迹
            relative_angles: Tensor(N_neighbors,) 或 Tensor(batch, N_neighbors)
                邻居相对于目标的角度，范围[0, 2π]
            neighbor_mask: Tensor(N_neighbors,) 或 Tensor(batch, N_neighbors)
                邻居有效性mask，1表示有效，0表示padding
        
        Returns:
            social_features: Tensor(feature_dim,) 或 Tensor(batch, feature_dim)
                社交上下文特征
        """
        # 处理输入维度
        if target_trajectory.dim() == 2:
            # 单个样本
            target_trajectory = target_trajectory.unsqueeze(0)  # (1, 8, 2)
            neighbor_trajectories = neighbor_trajectories.unsqueeze(0)  # (1, N, 8, 2)
            relative_angles = relative_angles.unsqueeze(0)  # (1, N)
            if neighbor_mask is not None:
                neighbor_mask = neighbor_mask.unsqueeze(0)  # (1, N)
            single_input = True
        else:
            single_input = False
        
        batch_size = target_trajectory.size(0)
        num_neighbors = neighbor_trajectories.size(1)
        
        # 创建默认mask（如果未提供）
        if neighbor_mask is None:
            neighbor_mask = torch.ones(batch_size, num_neighbors, device=target_trajectory.device)
        
        # 1. 编码目标行人轨迹
        target_out, target_h = self.target_encoder(target_trajectory)
        target_feature = target_h.squeeze(0)  # (batch, hidden_dim)
        
        # 2. 批量编码所有邻居的轨迹
        if num_neighbors > 0:
            # 重塑邻居轨迹: (batch, N, 8, 2) -> (batch*N, 8, 2)
            neighbor_traj_flat = neighbor_trajectories.reshape(-1, 8, self.trajectory_dim)
            
            # 通过GRU编码
            neighbor_out, neighbor_h = self.neighbor_encoder(neighbor_traj_flat)
            neighbor_features = neighbor_h.squeeze(0)  # (batch*N, hidden_dim//2)
            
            # 重塑回: (batch, N, hidden_dim//2)
            neighbor_features = neighbor_features.reshape(batch_size, num_neighbors, -1)
        else:
            # 没有邻居，创建空特征
            neighbor_features = torch.zeros(
                batch_size, 0, self.hidden_dim // 2,
                device=target_trajectory.device
            )
        
        # 3. 计算邻居距离和速度
        # 使用轨迹的最后一个点计算当前状态
        target_pos = target_trajectory[:, -1, :]  # (batch, 2)
        neighbor_pos = neighbor_trajectories[:, :, -1, :]  # (batch, N, 2)
        
        # 距离
        neighbor_distances = torch.norm(
            neighbor_pos - target_pos.unsqueeze(1), 
            dim=-1
        )  # (batch, N)
        
        # 速度（最后两个点的差）
        neighbor_velocities = neighbor_trajectories[:, :, -1, :] - neighbor_trajectories[:, :, -2, :]
        # (batch, N, 2)
        
        # 4. 对batch中的每个样本处理扇区特征
        all_sector_features = []
        
        for i in range(batch_size):
            # 获取该样本的有效邻居
            valid_mask = neighbor_mask[i] > 0
            
            if valid_mask.sum() == 0:
                # 没有有效邻居，使用零特征
                sector_feats = torch.zeros(
                    self.num_sectors,
                    1 + 1 + 2 + self.hidden_dim // 2,
                    device=target_trajectory.device
                )
            else:
                # 提取有效邻居的信息
                valid_features = neighbor_features[i][valid_mask]
                valid_distances = neighbor_distances[i][valid_mask]
                valid_velocities = neighbor_velocities[i][valid_mask]
                valid_angles = relative_angles[i][valid_mask]
                
                # 计算扇区分配
                sector_ids = self._compute_sector_assignment(valid_angles)
                
                # 聚合扇区特征
                sector_feats = self._aggregate_sector_features(
                    valid_features,
                    valid_distances,
                    valid_velocities,
                    sector_ids
                )
            
            all_sector_features.append(sector_feats)
        
        # Stack: (batch, num_sectors, sector_input_dim)
        all_sector_features = torch.stack(all_sector_features)
        
        # 5. 编码每个扇区的特征
        # 重塑: (batch*num_sectors, sector_input_dim)
        sector_feats_flat = all_sector_features.reshape(-1, all_sector_features.size(-1))
        
        # 通过MLP编码
        encoded_sectors = self.sector_encoder(sector_feats_flat)  # (batch*num_sectors, hidden_dim)
        
        # 重塑回: (batch, num_sectors, hidden_dim)
        encoded_sectors = encoded_sectors.reshape(batch_size, self.num_sectors, self.hidden_dim)
        
        # 展平扇区特征
        encoded_sectors_flat = encoded_sectors.reshape(batch_size, -1)  # (batch, num_sectors*hidden_dim)
        
        # 6. 融合目标特征和扇区特征
        combined = torch.cat([target_feature, encoded_sectors_flat], dim=-1)
        
        # 通过融合网络
        social_features = self.fusion(combined)  # (batch, feature_dim)
        
        # 层归一化
        social_features = self.layer_norm(social_features)
        
        # 如果输入是单个样本，返回单个特征向量
        if single_input:
            social_features = social_features.squeeze(0)
        
        return social_features
    
    def visualize_sectors(
        self,
        target_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        relative_angles: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        可视化社交扇区信息（用于调试和分析）
        
        Returns:
            dict: 包含扇区统计信息的字典
        """
        if target_trajectory.dim() == 2:
            target_trajectory = target_trajectory.unsqueeze(0)
            neighbor_trajectories = neighbor_trajectories.unsqueeze(0)
            relative_angles = relative_angles.unsqueeze(0)
            if neighbor_mask is not None:
                neighbor_mask = neighbor_mask.unsqueeze(0)
        
        if neighbor_mask is None:
            neighbor_mask = torch.ones(
                neighbor_trajectories.size(0),
                neighbor_trajectories.size(1),
                device=target_trajectory.device
            )
        
        # 只处理第一个样本
        valid_mask = neighbor_mask[0] > 0
        valid_angles = relative_angles[0][valid_mask]
        
        if valid_mask.sum() == 0:
            return {
                'sector_counts': [0] * self.num_sectors,
                'sector_angles': [],
                'num_sectors': self.num_sectors
            }
        
        sector_ids = self._compute_sector_assignment(valid_angles)
        
        # 统计每个扇区的邻居数量
        sector_counts = []
        for sector_id in range(self.num_sectors):
            count = (sector_ids == sector_id).sum().item()
            sector_counts.append(count)
        
        return {
            'sector_counts': sector_counts,
            'sector_angles': valid_angles.cpu().numpy().tolist(),
            'num_sectors': self.num_sectors,
            'sector_ids': sector_ids.cpu().numpy().tolist()
        }


class SimplifiedSocialCircle(nn.Module):
    """
    简化版SocialCircle
    
    不使用扇区划分，直接聚合所有邻居的特征
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 目标轨迹编码
        self.target_encoder = nn.GRU(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 邻居轨迹编码
        self.neighbor_encoder = nn.GRU(
            input_size=2,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(
        self,
        target_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        relative_angles: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """简化版前向传播"""
        if target_trajectory.dim() == 2:
            target_trajectory = target_trajectory.unsqueeze(0)
            neighbor_trajectories = neighbor_trajectories.unsqueeze(0)
            if neighbor_mask is not None:
                neighbor_mask = neighbor_mask.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = target_trajectory.size(0)
        num_neighbors = neighbor_trajectories.size(1)
        
        if neighbor_mask is None:
            neighbor_mask = torch.ones(batch_size, num_neighbors, device=target_trajectory.device)
        
        # 编码目标
        _, target_h = self.target_encoder(target_trajectory)
        target_feature = target_h.squeeze(0)
        
        # 编码邻居
        neighbor_traj_flat = neighbor_trajectories.reshape(-1, 8, 2)
        _, neighbor_h = self.neighbor_encoder(neighbor_traj_flat)
        neighbor_features = neighbor_h.squeeze(0).reshape(batch_size, num_neighbors, -1)
        
        # 平均池化（考虑mask）
        mask_expanded = neighbor_mask.unsqueeze(-1)
        masked_features = neighbor_features * mask_expanded
        num_valid = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        neighbor_aggregated = masked_features.sum(dim=1) / num_valid
        
        # 融合
        combined = torch.cat([target_feature, neighbor_aggregated], dim=-1)
        output = self.fusion(combined)
        
        if single_input:
            output = output.squeeze(0)
        
        return output


# 工厂函数
def create_social_circle(
    encoder_type: str = 'full',
    feature_dim: int = 64,
    **kwargs
) -> nn.Module:
    """
    创建SocialCircle编码器的工厂函数
    
    Args:
        encoder_type: 编码器类型
            - 'full': 完整版（扇区划分）
            - 'simplified': 简化版
        feature_dim: 输出特征维度
        **kwargs: 其他参数
    
    Returns:
        encoder: SocialCircle编码器
    """
    if encoder_type == 'full':
        return SocialCircle(feature_dim=feature_dim, **kwargs)
    elif encoder_type == 'simplified':
        return SimplifiedSocialCircle(feature_dim=feature_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == '__main__':
    """简单测试"""
    print("测试SocialCircle编码器...")
    
    # 测试1: 基础功能
    print("\n1. 基础功能测试")
    social_circle = SocialCircle(
        num_sectors=8,
        feature_dim=64,
        hidden_dim=128
    )
    
    # 创建测试数据
    target_traj = torch.randn(8, 2)  # 目标行人轨迹
    neighbor_trajs = torch.randn(5, 8, 2)  # 5个邻居
    
    # 计算相对角度（随机）
    angles = torch.rand(5) * 2 * math.pi  # [0, 2π]
    
    # 前向传播
    social_features = social_circle(target_traj, neighbor_trajs, angles)
    print(f"输入: 目标{target_traj.shape}, 邻居{neighbor_trajs.shape}")
    print(f"输出: {social_features.shape}")
    assert social_features.shape == (64,)
    
    # 测试2: 批量处理
    print("\n2. 批量处理测试")
    batch_target = torch.randn(4, 8, 2)
    batch_neighbors = torch.randn(4, 10, 8, 2)
    batch_angles = torch.rand(4, 10) * 2 * math.pi
    batch_mask = torch.rand(4, 10) > 0.3  # 随机mask
    batch_mask = batch_mask.float()
    
    batch_output = social_circle(
        batch_target,
        batch_neighbors,
        batch_angles,
        batch_mask
    )
    print(f"批量输入: {batch_target.shape}")
    print(f"批量输出: {batch_output.shape}")
    assert batch_output.shape == (4, 64)
    
    # 测试3: 无邻居情况
    print("\n3. 无邻居情况测试")
    empty_neighbors = torch.randn(0, 8, 2)
    empty_angles = torch.tensor([])
    
    # 使用批量版本（因为单个样本需要至少有形状）
    no_neighbor_target = target_traj.unsqueeze(0)  # (1, 8, 2)
    no_neighbor_neighbors = torch.randn(1, 0, 8, 2)  # (1, 0, 8, 2)
    no_neighbor_angles = torch.zeros(1, 0)  # (1, 0)
    
    no_neighbor_output = social_circle(
        no_neighbor_target,
        no_neighbor_neighbors,
        no_neighbor_angles
    )
    print(f"无邻居输出: {no_neighbor_output.shape}")
    assert torch.isfinite(no_neighbor_output).all()
    
    # 测试4: 扇区可视化
    print("\n4. 扇区可视化测试")
    sector_info = social_circle.visualize_sectors(
        target_traj,
        neighbor_trajs,
        angles
    )
    print(f"扇区数量: {sector_info['num_sectors']}")
    print(f"每个扇区的邻居数: {sector_info['sector_counts']}")
    print(f"总邻居数: {sum(sector_info['sector_counts'])}")
    
    # 测试5: 梯度流
    print("\n5. 梯度流测试")
    target_traj_grad = torch.randn(8, 2, requires_grad=True)
    neighbor_trajs_grad = torch.randn(5, 8, 2, requires_grad=True)
    
    output_grad = social_circle(target_traj_grad, neighbor_trajs_grad, angles)
    loss = output_grad.sum()
    loss.backward()
    
    assert target_traj_grad.grad is not None
    assert neighbor_trajs_grad.grad is not None
    assert torch.isfinite(target_traj_grad.grad).all()
    print("✓ 梯度流正常")
    
    # 测试6: 简化版
    print("\n6. 简化版测试")
    simplified = SimplifiedSocialCircle(feature_dim=64, hidden_dim=128)
    
    simplified_output = simplified(target_traj, neighbor_trajs, angles)
    print(f"简化版输出: {simplified_output.shape}")
    assert simplified_output.shape == (64,)
    
    # 测试7: 工厂函数
    print("\n7. 工厂函数测试")
    encoders = {
        'full': create_social_circle('full', feature_dim=64),
        'simplified': create_social_circle('simplified', feature_dim=64)
    }
    
    for name, encoder in encoders.items():
        out = encoder(target_traj, neighbor_trajs, angles)
        print(f"{name}: {out.shape}")
        assert out.shape == (64,)
    
    # 测试8: 参数量统计
    print("\n8. 参数量统计")
    for name, encoder in encoders.items():
        params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"{name}编码器: {params:,} 参数 ({trainable_params:,} 可训练)")
    
    print("\n✓ SocialCircle编码器测试通过！")

