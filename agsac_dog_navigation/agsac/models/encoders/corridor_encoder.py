"""
通路几何编码器
处理通路多边形特征（已通过PointNet编码后），使用注意力机制聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CorridorEncoder(nn.Module):
    """
    通路几何编码器
    
    输入：
        - corridor_features: (max_corridors, feature_dim) 已编码的通路特征
        - corridor_mask: (max_corridors,) 有效性mask (1=有效, 0=padding)
        - robot_position: (2,) 机器人位置（可选，用于位置编码）
    
    输出：
        - corridor_feature: (output_dim,) 聚合后的通路特征
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        max_corridors: int = 10
    ):
        """
        Args:
            input_dim: 输入特征维度（PointNet编码后的维度）
            output_dim: 输出特征维度
            num_heads: 注意力头数
            dropout: Dropout比例
            use_positional_encoding: 是否使用位置编码
            max_corridors: 最大通路数量
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        self.max_corridors = max_corridors
        
        # 位置编码（可学习）
        if use_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, max_corridors, input_dim) * 0.02
            )
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力后的层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # 前馈网络后的层归一化
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 最终的聚合和投影网络
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        
        # 层归一化
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        corridor_features: torch.Tensor,
        corridor_mask: torch.Tensor,
        robot_position: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            corridor_features: Tensor(max_corridors, feature_dim) 
                             或 Tensor(batch, max_corridors, feature_dim)
                             已经通过PointNet编码的通路特征
            corridor_mask: Tensor(max_corridors,) 
                          或 Tensor(batch, max_corridors)
                          有效性mask，1表示有效，0表示padding
            robot_position: Tensor(2,) 或 Tensor(batch, 2)
                          机器人位置（可选，暂未使用）
        
        Returns:
            aggregated_features: Tensor(output_dim,) 
                               或 Tensor(batch, output_dim)
                               聚合后的通路特征
        """
        # 处理输入维度
        if corridor_features.dim() == 2:
            # (max_corridors, feature_dim)
            corridor_features = corridor_features.unsqueeze(0)  # (1, max_corridors, feature_dim)
            corridor_mask = corridor_mask.unsqueeze(0)  # (1, max_corridors)
            single_input = True
        else:
            single_input = False
        
        batch_size, num_corridors, feature_dim = corridor_features.shape
        
        # 1. 添加位置编码
        if self.use_positional_encoding:
            # 截取或扩展位置编码以匹配实际通路数量
            pos_enc = self.positional_encoding[:, :num_corridors, :]
            corridor_features = corridor_features + pos_enc
        
        # 2. 创建注意力mask
        # PyTorch的MultiheadAttention使用key_padding_mask，True表示需要mask的位置
        # 我们的mask中1表示有效，0表示padding，所以需要取反
        key_padding_mask = (corridor_mask == 0)  # (batch, max_corridors)
        
        # 3. 自注意力
        # 如果所有通路都被mask了，跳过注意力
        if key_padding_mask.all():
            # 所有通路都无效，返回零特征
            output = torch.zeros(batch_size, self.output_dim, device=corridor_features.device)
            if single_input:
                output = output.squeeze(0)
            return output
        
        attn_out, attn_weights = self.self_attention(
            query=corridor_features,
            key=corridor_features,
            value=corridor_features,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )  # (batch, max_corridors, feature_dim)
        
        # 残差连接 + 层归一化
        corridor_features = self.norm1(corridor_features + self.dropout(attn_out))
        
        # 4. 前馈网络
        ff_out = self.feed_forward(corridor_features)
        
        # 残差连接 + 层归一化
        corridor_features = self.norm2(corridor_features + self.dropout(ff_out))
        # (batch, max_corridors, feature_dim)
        
        # 5. 聚合有效特征
        # 方法：仅对有效通路进行平均池化
        # 扩展mask维度用于广播
        mask_expanded = corridor_mask.unsqueeze(-1)  # (batch, max_corridors, 1)
        
        # 将padding位置置零
        masked_features = corridor_features * mask_expanded  # (batch, max_corridors, feature_dim)
        
        # 计算有效通路数量
        num_valid = corridor_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        
        # 平均池化（只计算有效通路）
        aggregated = masked_features.sum(dim=1) / num_valid  # (batch, feature_dim)
        
        # 6. 通过MLP投影到输出维度
        output = self.aggregation_mlp(aggregated)  # (batch, output_dim)
        
        # 层归一化
        output = self.output_norm(output)
        
        # 如果输入是单个样本，返回单个特征向量
        if single_input:
            output = output.squeeze(0)
        
        return output
    
    def get_attention_weights(
        self,
        corridor_features: torch.Tensor,
        corridor_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        获取注意力权重（用于可视化和分析）
        
        Args:
            corridor_features: Tensor(max_corridors, feature_dim) 或 Tensor(batch, max_corridors, feature_dim)
            corridor_mask: Tensor(max_corridors,) 或 Tensor(batch, max_corridors)
        
        Returns:
            attn_weights: Tensor(num_heads, max_corridors, max_corridors) 
                         或 Tensor(batch, num_heads, max_corridors, max_corridors)
        """
        if corridor_features.dim() == 2:
            corridor_features = corridor_features.unsqueeze(0)
            corridor_mask = corridor_mask.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # 添加位置编码
        if self.use_positional_encoding:
            num_corridors = corridor_features.size(1)
            pos_enc = self.positional_encoding[:, :num_corridors, :]
            corridor_features = corridor_features + pos_enc
        
        # 创建mask
        key_padding_mask = (corridor_mask == 0)
        
        # 获取注意力权重
        _, attn_weights = self.self_attention(
            query=corridor_features,
            key=corridor_features,
            value=corridor_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # 返回所有头的权重
        )
        
        if single_input:
            attn_weights = attn_weights.squeeze(0)
        
        return attn_weights


class SimpleCorridorEncoder(nn.Module):
    """
    简化版通路编码器
    
    不使用注意力机制，直接平均池化
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 简单的MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        corridor_features: torch.Tensor,
        corridor_mask: torch.Tensor,
        robot_position: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """简化版前向传播"""
        if corridor_features.dim() == 2:
            corridor_features = corridor_features.unsqueeze(0)
            corridor_mask = corridor_mask.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # 平均池化（只计算有效通路）
        mask_expanded = corridor_mask.unsqueeze(-1)
        masked_features = corridor_features * mask_expanded
        num_valid = corridor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        aggregated = masked_features.sum(dim=1) / num_valid
        
        # MLP
        output = self.mlp(aggregated)
        
        if single_input:
            output = output.squeeze(0)
        
        return output


class HierarchicalCorridorEncoder(nn.Module):
    """
    层次化通路编码器
    
    使用多层注意力进行层次化特征提取
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_corridors: int = 10
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_corridors, input_dim) * 0.02
        )
        
        # 多层注意力
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 最终投影
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        corridor_features: torch.Tensor,
        corridor_mask: torch.Tensor,
        robot_position: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """层次化前向传播"""
        if corridor_features.dim() == 2:
            corridor_features = corridor_features.unsqueeze(0)
            corridor_mask = corridor_mask.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # 添加位置编码
        num_corridors = corridor_features.size(1)
        pos_enc = self.positional_encoding[:, :num_corridors, :]
        features = corridor_features + pos_enc
        
        # 创建mask
        key_padding_mask = (corridor_mask == 0)
        
        # 多层注意力
        for attn_layer, norm_layer in zip(self.attention_layers, self.norm_layers):
            attn_out, _ = attn_layer(
                query=features,
                key=features,
                value=features,
                key_padding_mask=key_padding_mask
            )
            features = norm_layer(features + self.dropout(attn_out))
        
        # 聚合
        mask_expanded = corridor_mask.unsqueeze(-1)
        masked_features = features * mask_expanded
        num_valid = corridor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        aggregated = masked_features.sum(dim=1) / num_valid
        
        # 输出投影
        output = self.output_projection(aggregated)
        
        if single_input:
            output = output.squeeze(0)
        
        return output


# 工厂函数
def create_corridor_encoder(
    encoder_type: str = 'attention',
    input_dim: int = 64,
    output_dim: int = 128,
    **kwargs
) -> nn.Module:
    """
    创建通路编码器的工厂函数
    
    Args:
        encoder_type: 编码器类型
            - 'attention': 注意力版（推荐）
            - 'simple': 简化版
            - 'hierarchical': 层次化版
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        **kwargs: 其他参数
    
    Returns:
        encoder: 通路编码器
    """
    if encoder_type == 'attention':
        return CorridorEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    elif encoder_type == 'simple':
        return SimpleCorridorEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    elif encoder_type == 'hierarchical':
        return HierarchicalCorridorEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == '__main__':
    """简单测试"""
    print("测试通路几何编码器...")
    
    # 测试1: 基础注意力版本
    print("\n1. 基础注意力版本")
    encoder = CorridorEncoder(
        input_dim=64,
        output_dim=128,
        num_heads=4
    )
    
    # 单个样本
    corridor_feats = torch.randn(10, 64)  # 10个通路，每个64维
    corridor_mask = torch.zeros(10)
    corridor_mask[:5] = 1.0  # 只有前5个有效
    
    output = encoder(corridor_feats, corridor_mask)
    print(f"输入: {corridor_feats.shape}, mask有效数: {corridor_mask.sum()}")
    print(f"输出: {output.shape}")
    
    # 批量样本
    batch_feats = torch.randn(4, 10, 64)
    batch_mask = torch.zeros(4, 10)
    batch_mask[0, :3] = 1  # 第1个样本3个有效
    batch_mask[1, :7] = 1  # 第2个样本7个有效
    batch_mask[2, :5] = 1  # 第3个样本5个有效
    batch_mask[3, :10] = 1  # 第4个样本10个有效
    
    batch_output = encoder(batch_feats, batch_mask)
    print(f"批量输入: {batch_feats.shape}")
    print(f"批量输出: {batch_output.shape}")
    print(f"每个样本有效通路数: {batch_mask.sum(dim=1)}")
    
    # 测试2: 简化版本
    print("\n2. 简化版本")
    simple_encoder = SimpleCorridorEncoder(input_dim=64, output_dim=128)
    simple_output = simple_encoder(corridor_feats, corridor_mask)
    print(f"简化版输出: {simple_output.shape}")
    
    # 测试3: 层次化版本
    print("\n3. 层次化版本")
    hierarchical_encoder = HierarchicalCorridorEncoder(
        input_dim=64,
        output_dim=128,
        num_layers=2
    )
    hierarchical_output = hierarchical_encoder(corridor_feats, corridor_mask)
    print(f"层次化版输出: {hierarchical_output.shape}")
    
    # 测试4: 注意力权重可视化
    print("\n4. 注意力权重")
    attn_weights = encoder.get_attention_weights(corridor_feats, corridor_mask)
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 测试5: 边界情况
    print("\n5. 边界情况测试")
    
    # 只有1个有效通路
    single_mask = torch.zeros(10)
    single_mask[0] = 1
    single_output = encoder(corridor_feats, single_mask)
    print(f"单个有效通路输出: {single_output.shape}")
    
    # 所有通路都有效
    all_mask = torch.ones(10)
    all_output = encoder(corridor_feats, all_mask)
    print(f"全部有效通路输出: {all_output.shape}")
    
    # 没有有效通路（应该返回零向量）
    no_mask = torch.zeros(10)
    no_output = encoder(corridor_feats, no_mask)
    print(f"无有效通路输出: {no_output.shape}, 是否全零: {torch.allclose(no_output, torch.zeros_like(no_output))}")
    
    # 测试6: 梯度流
    print("\n6. 梯度流测试")
    corridor_feats_grad = torch.randn(10, 64, requires_grad=True)
    output_grad = encoder(corridor_feats_grad, corridor_mask)
    loss = output_grad.sum()
    loss.backward()
    print(f"梯度形状: {corridor_feats_grad.grad.shape}")
    print(f"梯度是否有限: {torch.isfinite(corridor_feats_grad.grad).all()}")
    
    # 测试7: 参数量统计
    print("\n7. 参数量统计")
    encoders = {
        'attention': encoder,
        'simple': simple_encoder,
        'hierarchical': hierarchical_encoder
    }
    
    for name, enc in encoders.items():
        params = sum(p.numel() for p in enc.parameters())
        trainable_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        print(f"{name}编码器: {params:,} 参数 ({trainable_params:,} 可训练)")
    
    # 测试8: 工厂函数
    print("\n8. 工厂函数")
    factory_encoders = {
        'attention': create_corridor_encoder('attention', 64, 128),
        'simple': create_corridor_encoder('simple', 64, 128),
        'hierarchical': create_corridor_encoder('hierarchical', 64, 128, num_layers=2)
    }
    
    for name, enc in factory_encoders.items():
        out = enc(corridor_feats, corridor_mask)
        print(f"{name}: {out.shape}")
    
    print("\n✓ 通路几何编码器测试通过！")

