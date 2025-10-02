"""
行人未来轨迹编码器
编码行人的多模态未来预测为固定维度特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PedestrianEncoder(nn.Module):
    """
    行人未来轨迹编码器
    
    处理流程：
    1. 对每个行人的每个预测模态进行时序编码（GRU）
    2. 多模态注意力聚合（20个模态 → 1个特征）
    3. 跨行人注意力聚合（N个行人 → 1个特征）
    
    输入：
        - pedestrian_predictions: (N_peds, pred_horizon, coord_dim, num_modes)
          每个行人的多模态未来轨迹预测
        - pedestrian_mask: (N_peds,) 有效性mask
    
    输出：
        - pedestrian_features: (feature_dim,) 行人未来特征
    """
    
    def __init__(
        self,
        pred_horizon: int = 12,
        coord_dim: int = 2,
        num_modes: int = 20,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            pred_horizon: 预测时间步长
            coord_dim: 坐标维度（2D为2）
            num_modes: 预测模态数量
            feature_dim: 输出特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()
        
        self.pred_horizon = pred_horizon
        self.coord_dim = coord_dim
        self.num_modes = num_modes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 1. 时序编码器（对每个模态的轨迹进行GRU编码）
        self.temporal_encoder = nn.GRU(
            input_size=coord_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 2. 多模态注意力（聚合20个模态）
        self.mode_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 多模态聚合后的处理
        self.mode_aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. 跨行人注意力（聚合N个行人）
        self.pedestrian_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 跨行人聚合后的处理
        self.pedestrian_aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. 最终投影到输出维度
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim)
        )
    
    def _encode_single_pedestrian_modes(
        self,
        pedestrian_prediction: torch.Tensor
    ) -> torch.Tensor:
        """
        编码单个行人的所有预测模态
        
        Args:
            pedestrian_prediction: (pred_horizon, coord_dim, num_modes)
                单个行人的多模态预测
        
        Returns:
            mode_features: (num_modes, hidden_dim)
                每个模态的特征
        """
        pred_horizon, coord_dim, num_modes = pedestrian_prediction.shape
        
        # 重塑为: (num_modes, pred_horizon, coord_dim)
        modes_trajectories = pedestrian_prediction.permute(2, 0, 1)
        
        # 展平: (num_modes * pred_horizon, coord_dim) -> 用于批量处理不可行
        # 改为循环处理每个模态（如果模态数不多，这是可接受的）
        mode_features = []
        for mode_id in range(num_modes):
            mode_traj = modes_trajectories[mode_id]  # (pred_horizon, coord_dim)
            mode_traj = mode_traj.unsqueeze(0)  # (1, pred_horizon, coord_dim)
            
            # GRU编码
            _, h_n = self.temporal_encoder(mode_traj)
            # h_n shape: (num_layers, batch, hidden_dim) = (1, 1, hidden_dim)
            mode_feat = h_n.squeeze()  # 移除所有大小为1的维度 -> (hidden_dim,)
            mode_features.append(mode_feat)
        
        # Stack: (num_modes, hidden_dim)
        mode_features = torch.stack(mode_features)
        
        return mode_features
    
    def _aggregate_modes_with_attention(
        self,
        mode_features: torch.Tensor
    ) -> torch.Tensor:
        """
        使用注意力机制聚合多个模态
        
        Args:
            mode_features: (num_modes, hidden_dim) 或 (batch, num_modes, hidden_dim)
        
        Returns:
            aggregated: (hidden_dim,) 或 (batch, hidden_dim)
        """
        if mode_features.dim() == 2:
            # 添加batch维度
            mode_features = mode_features.unsqueeze(0)  # (1, num_modes, hidden_dim)
            single_input = True
        else:
            single_input = False
        
        # 自注意力
        attended, _ = self.mode_attention(
            query=mode_features,
            key=mode_features,
            value=mode_features
        )  # (batch, num_modes, hidden_dim)
        
        # 平均池化
        aggregated = attended.mean(dim=1)  # (batch, hidden_dim)
        
        # 后处理
        aggregated = self.mode_aggregation(aggregated)
        
        if single_input:
            aggregated = aggregated.squeeze(0)
        
        return aggregated
    
    def forward(
        self,
        pedestrian_predictions: torch.Tensor,
        pedestrian_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pedestrian_predictions: Tensor(N_peds, pred_horizon, coord_dim, num_modes)
                或 Tensor(batch, N_peds, pred_horizon, coord_dim, num_modes)
                所有行人的多模态未来预测
            pedestrian_mask: Tensor(N_peds,) 或 Tensor(batch, N_peds)
                行人有效性mask，1表示有效，0表示padding
        
        Returns:
            pedestrian_features: Tensor(feature_dim,) 或 Tensor(batch, feature_dim)
                聚合后的行人未来特征
        """
        # 处理输入维度
        if pedestrian_predictions.dim() == 4:
            # (N_peds, pred_horizon, coord_dim, num_modes)
            pedestrian_predictions = pedestrian_predictions.unsqueeze(0)
            if pedestrian_mask is not None:
                pedestrian_mask = pedestrian_mask.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size, num_peds, pred_horizon, coord_dim, num_modes = pedestrian_predictions.shape
        
        # 创建默认mask
        if pedestrian_mask is None:
            pedestrian_mask = torch.ones(batch_size, num_peds, device=pedestrian_predictions.device)
        
        # 1. 对每个行人的每个模态进行时序编码
        all_pedestrian_features = []
        
        for batch_idx in range(batch_size):
            batch_pedestrian_features = []
            
            for ped_idx in range(num_peds):
                # 检查该行人是否有效
                if pedestrian_mask[batch_idx, ped_idx] > 0:
                    ped_prediction = pedestrian_predictions[batch_idx, ped_idx]
                    # (pred_horizon, coord_dim, num_modes)
                    
                    # 编码所有模态
                    mode_features = self._encode_single_pedestrian_modes(ped_prediction)
                    # (num_modes, hidden_dim)
                    assert mode_features.dim() == 2, f"mode_features should be 2D, got {mode_features.shape}"
                    
                    # 聚合模态
                    ped_feature = self._aggregate_modes_with_attention(mode_features)
                    # Should be (hidden_dim,)
                    
                    # 强制确保是1D张量
                    while ped_feature.dim() > 1:
                        ped_feature = ped_feature.squeeze(0)
                    
                    assert ped_feature.dim() == 1, f"ped_feature should be 1D, got {ped_feature.shape}"
                    assert ped_feature.size(0) == self.hidden_dim, f"Expected size {self.hidden_dim}, got {ped_feature.size(0)}"
                else:
                    # 无效行人，使用零特征
                    ped_feature = torch.zeros(self.hidden_dim, device=pedestrian_predictions.device)
                
                batch_pedestrian_features.append(ped_feature)
            
            # Stack: (num_peds, hidden_dim)
            batch_pedestrian_features = torch.stack(batch_pedestrian_features)
            all_pedestrian_features.append(batch_pedestrian_features)
        
        # Stack: (batch, num_peds, hidden_dim)
        all_pedestrian_features = torch.stack(all_pedestrian_features)
        
        # 检查是否所有行人都无效
        num_valid_peds = pedestrian_mask.sum(dim=1)  # (batch,)
        
        # 如果某个batch中所有行人都无效，直接返回零特征
        if (num_valid_peds == 0).any():
            output_features = torch.zeros(batch_size, self.feature_dim, device=pedestrian_predictions.device)
            if single_input:
                output_features = output_features.squeeze(0)
            return output_features
        
        # 2. 跨行人注意力聚合
        # 创建key_padding_mask
        key_padding_mask = (pedestrian_mask == 0)  # True表示需要mask的位置
        
        # 自注意力
        attended_peds, _ = self.pedestrian_attention(
            query=all_pedestrian_features,
            key=all_pedestrian_features,
            value=all_pedestrian_features,
            key_padding_mask=key_padding_mask
        )  # (batch, num_peds, hidden_dim)
        
        # 3. 聚合（平均池化，考虑mask）
        mask_expanded = pedestrian_mask.unsqueeze(-1)  # (batch, num_peds, 1)
        masked_features = attended_peds * mask_expanded
        num_valid = pedestrian_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        aggregated_peds = masked_features.sum(dim=1) / num_valid  # (batch, hidden_dim)
        
        # 后处理
        aggregated_peds = self.pedestrian_aggregation(aggregated_peds)
        
        # 4. 投影到输出维度
        output_features = self.output_projection(aggregated_peds)  # (batch, feature_dim)
        
        if single_input:
            output_features = output_features.squeeze(0)
        
        return output_features


class SimplePedestrianEncoder(nn.Module):
    """
    简化版行人编码器
    
    直接平均所有模态和所有行人，不使用注意力机制
    """
    
    def __init__(
        self,
        pred_horizon: int = 12,
        coord_dim: int = 2,
        num_modes: int = 20,
        feature_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.pred_horizon = pred_horizon
        self.coord_dim = coord_dim
        self.num_modes = num_modes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 时序编码
        self.temporal_encoder = nn.GRU(
            input_size=coord_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(
        self,
        pedestrian_predictions: torch.Tensor,
        pedestrian_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """简化版前向传播"""
        if pedestrian_predictions.dim() == 4:
            pedestrian_predictions = pedestrian_predictions.unsqueeze(0)
            if pedestrian_mask is not None:
                pedestrian_mask = pedestrian_mask.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size, num_peds, pred_horizon, coord_dim, num_modes = pedestrian_predictions.shape
        
        if pedestrian_mask is None:
            pedestrian_mask = torch.ones(batch_size, num_peds, device=pedestrian_predictions.device)
        
        # 重塑: (batch * num_peds * num_modes, pred_horizon, coord_dim)
        predictions_flat = pedestrian_predictions.permute(0, 1, 4, 2, 3).reshape(-1, pred_horizon, coord_dim)
        
        # GRU编码
        _, h_n = self.temporal_encoder(predictions_flat)
        features = h_n.squeeze(0)  # (batch * num_peds * num_modes, hidden_dim)
        
        # 重塑回: (batch, num_peds, num_modes, hidden_dim)
        features = features.reshape(batch_size, num_peds, num_modes, self.hidden_dim)
        
        # 平均所有模态
        features = features.mean(dim=2)  # (batch, num_peds, hidden_dim)
        
        # 平均所有行人（考虑mask）
        mask_expanded = pedestrian_mask.unsqueeze(-1)
        masked_features = features * mask_expanded
        num_valid = pedestrian_mask.sum(dim=1, keepdim=True).clamp(min=1)
        aggregated = masked_features.sum(dim=1) / num_valid  # (batch, hidden_dim)
        
        # 投影
        output = self.output_projection(aggregated)
        
        if single_input:
            output = output.squeeze(0)
        
        return output


# 工厂函数
def create_pedestrian_encoder(
    encoder_type: str = 'attention',
    **kwargs
) -> nn.Module:
    """
    创建行人编码器的工厂函数
    
    Args:
        encoder_type: 编码器类型
            - 'attention': 注意力版（推荐）
            - 'simple': 简化版
        **kwargs: 其他参数
    
    Returns:
        encoder: 行人编码器
    """
    if encoder_type == 'attention':
        return PedestrianEncoder(**kwargs)
    elif encoder_type == 'simple':
        return SimplePedestrianEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == '__main__':
    """简单测试"""
    print("测试行人未来轨迹编码器...")
    
    # 测试1: 基础功能
    print("\n1. 基础功能测试")
    encoder = PedestrianEncoder(
        pred_horizon=12,
        coord_dim=2,
        num_modes=20,
        feature_dim=64,
        hidden_dim=128
    )
    
    # 创建测试数据: 5个行人的多模态预测
    ped_predictions = torch.randn(5, 12, 2, 20)
    ped_mask = torch.ones(5)
    ped_mask[3:] = 0  # 后两个行人无效
    
    output = encoder(ped_predictions, ped_mask)
    print(f"输入: {ped_predictions.shape}, mask: {ped_mask}")
    print(f"输出: {output.shape}")
    assert output.shape == (64,), f"Expected (64,), got {output.shape}"
    
    # 测试2: 批量处理
    print("\n2. 批量处理测试")
    batch_predictions = torch.randn(4, 10, 12, 2, 20)
    batch_mask = torch.rand(4, 10) > 0.3
    batch_mask = batch_mask.float()
    
    batch_output = encoder(batch_predictions, batch_mask)
    print(f"批量输入: {batch_predictions.shape}")
    print(f"批量输出: {batch_output.shape}")
    assert batch_output.shape == (4, 64)
    
    # 测试3: 无有效行人
    print("\n3. 无有效行人测试")
    empty_mask = torch.zeros(5)
    empty_output = encoder(ped_predictions, empty_mask)
    print(f"无有效行人输出: {empty_output.shape}")
    assert torch.isfinite(empty_output).all()
    
    # 测试4: 简化版
    print("\n4. 简化版测试")
    simple_encoder = SimplePedestrianEncoder(
        pred_horizon=12,
        feature_dim=64
    )
    
    simple_output = simple_encoder(ped_predictions, ped_mask)
    print(f"简化版输出: {simple_output.shape}")
    assert simple_output.shape == (64,)
    
    # 测试5: 梯度流
    print("\n5. 梯度流测试")
    ped_predictions_grad = torch.randn(5, 12, 2, 20, requires_grad=True)
    output_grad = encoder(ped_predictions_grad, ped_mask)
    loss = output_grad.sum()
    loss.backward()
    
    assert ped_predictions_grad.grad is not None
    assert torch.isfinite(ped_predictions_grad.grad).all()
    print("✓ 梯度流正常")
    
    # 测试6: 工厂函数
    print("\n6. 工厂函数测试")
    encoders = {
        'attention': create_pedestrian_encoder('attention', feature_dim=64),
        'simple': create_pedestrian_encoder('simple', feature_dim=64)
    }
    
    for name, enc in encoders.items():
        out = enc(ped_predictions, ped_mask)
        print(f"{name}: {out.shape}")
        assert out.shape == (64,)
    
    # 测试7: 参数量统计
    print("\n7. 参数量统计")
    for name, enc in encoders.items():
        params = sum(p.numel() for p in enc.parameters())
        trainable_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        print(f"{name}编码器: {params:,} 参数 ({trainable_params:,} 可训练)")
    
    print("\n✓ 行人未来轨迹编码器测试通过！")

