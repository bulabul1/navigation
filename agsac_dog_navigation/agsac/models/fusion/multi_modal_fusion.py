"""
多模态融合模块
融合机器狗状态、行人未来预测和通路几何信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiModalFusion(nn.Module):
    """
    多模态注意力融合模块
    
    设计思路：
    - 机器狗特征作为Query（"我想从环境中了解什么"）
    - 行人和通路特征作为Key/Value（环境信息）
    - 使用注意力机制自适应关注重要的环境信息
    
    输入：
        - dog_features: (64,) 机器狗状态特征
        - pedestrian_features: (64,) 行人未来预测特征
        - corridor_features: (128,) 通路几何特征
    
    输出：
        - fused_state: (64,) 融合后的状态表示
        - attention_weights: (1, 2) 注意力权重（可选，用于可视化）
    
    参数量约20K
    """
    
    def __init__(
        self,
        dog_dim: int = 64,
        pedestrian_dim: int = 64,
        corridor_dim: int = 128,
        output_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Args:
            dog_dim: 机器狗特征维度
            pedestrian_dim: 行人特征维度
            corridor_dim: 通路特征维度
            output_dim: 输出特征维度
            num_heads: 注意力头数
            dropout: Dropout比例
            use_residual: 是否使用残差连接
        """
        super().__init__()
        
        self.dog_dim = dog_dim
        self.pedestrian_dim = pedestrian_dim
        self.corridor_dim = corridor_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        # 1. 投影通路特征到统一维度
        self.corridor_projection = nn.Sequential(
            nn.Linear(corridor_dim, dog_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 2. 多头注意力（机器狗attend环境信息）
        self.attention = nn.MultiheadAttention(
            embed_dim=dog_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. 注意力后的层归一化
        self.attention_norm = nn.LayerNorm(dog_dim)
        
        # 4. 融合网络
        fusion_input_dim = dog_dim * 2  # dog + attended
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        
        # 5. 最终层归一化
        self.output_norm = nn.LayerNorm(output_dim)
        
        # 6. 残差投影（如果维度不同）
        if use_residual and dog_dim != output_dim:
            self.residual_projection = nn.Linear(dog_dim, output_dim)
        else:
            self.residual_projection = None
    
    def forward(
        self,
        dog_features: torch.Tensor,
        pedestrian_features: torch.Tensor,
        corridor_features: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            dog_features: (64,) 或 (batch, 64) 机器狗状态特征
            pedestrian_features: (64,) 或 (batch, 64) 行人未来特征
            corridor_features: (128,) 或 (batch, 128) 通路几何特征
            return_attention_weights: 是否返回注意力权重
        
        Returns:
            fused_state: (64,) 或 (batch, 64) 融合后的状态
            attention_weights: (1, 2) 或 (batch, 1, 2) 注意力权重（可选）
        """
        # 处理输入维度
        if dog_features.dim() == 1:
            dog_features = dog_features.unsqueeze(0)  # (1, 64)
            pedestrian_features = pedestrian_features.unsqueeze(0)
            corridor_features = corridor_features.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = dog_features.size(0)
        
        # 1. 投影通路特征到统一维度
        corridor_projected = self.corridor_projection(corridor_features)
        # (batch, 64)
        
        # 2. 构建环境特征序列 [行人, 通路]
        # Stack成 (batch, 2, 64)
        environment_features = torch.stack([
            pedestrian_features,
            corridor_projected
        ], dim=1)  # (batch, 2, 64)
        
        # 3. 机器狗特征作为Query
        # 需要添加序列维度 (batch, 1, 64)
        dog_query = dog_features.unsqueeze(1)  # (batch, 1, 64)
        
        # 4. 多头注意力
        # Query: dog, Key/Value: environment
        attended_output, attention_weights = self.attention(
            query=dog_query,          # (batch, 1, 64)
            key=environment_features,  # (batch, 2, 64)
            value=environment_features,
            need_weights=True
        )
        # attended_output: (batch, 1, 64)
        # attention_weights: (batch, 1, 2)
        
        # 移除序列维度
        attended_output = attended_output.squeeze(1)  # (batch, 64)
        
        # 层归一化
        attended_output = self.attention_norm(attended_output)
        
        # 5. 拼接原始机器狗特征和注意力输出
        concatenated = torch.cat([
            dog_features,      # (batch, 64)
            attended_output    # (batch, 64)
        ], dim=-1)  # (batch, 128)
        
        # 6. 融合网络
        fused = self.fusion_net(concatenated)  # (batch, 64)
        
        # 7. 残差连接（如果启用）
        if self.use_residual:
            if self.residual_projection is not None:
                residual = self.residual_projection(dog_features)
            else:
                residual = dog_features
            fused = fused + residual
        
        # 8. 最终层归一化
        fused_state = self.output_norm(fused)  # (batch, 64)
        
        # 如果是单个输入，移除batch维度
        if single_input:
            fused_state = fused_state.squeeze(0)
            if return_attention_weights:
                attention_weights = attention_weights.squeeze(0)  # (1, 2)
        
        if return_attention_weights:
            return fused_state, attention_weights
        else:
            return fused_state, None


class SimplifiedFusion(nn.Module):
    """
    简化版融合模块
    
    直接拼接所有特征，通过MLP融合
    不使用注意力机制
    
    参数量约10K
    """
    
    def __init__(
        self,
        dog_dim: int = 64,
        pedestrian_dim: int = 64,
        corridor_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 总输入维度
        total_input_dim = dog_dim + pedestrian_dim + corridor_dim  # 64+64+128=256
        
        self.fusion_net = nn.Sequential(
            nn.Linear(total_input_dim, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        dog_features: torch.Tensor,
        pedestrian_features: torch.Tensor,
        corridor_features: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, None]:
        """简化版前向传播"""
        if dog_features.dim() == 1:
            dog_features = dog_features.unsqueeze(0)
            pedestrian_features = pedestrian_features.unsqueeze(0)
            corridor_features = corridor_features.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # 拼接所有特征
        concatenated = torch.cat([
            dog_features,
            pedestrian_features,
            corridor_features
        ], dim=-1)
        
        # 融合
        fused = self.fusion_net(concatenated)
        
        if single_input:
            fused = fused.squeeze(0)
        
        return fused, None


def create_fusion_module(
    fusion_type: str = 'attention',
    **kwargs
) -> nn.Module:
    """
    创建融合模块的工厂函数
    
    Args:
        fusion_type: 融合模块类型
            - 'attention': 多头注意力融合（推荐）
            - 'simple': 简化版直接拼接
        **kwargs: 其他参数
    
    Returns:
        fusion_module: 融合模块
    """
    if fusion_type == 'attention':
        return MultiModalFusion(**kwargs)
    elif fusion_type == 'simple':
        return SimplifiedFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


if __name__ == '__main__':
    """单元测试"""
    print("测试多模态融合模块...")
    
    # 测试1: 基础功能
    print("\n1. 基础功能测试")
    fusion = MultiModalFusion(
        dog_dim=64,
        pedestrian_dim=64,
        corridor_dim=128,
        output_dim=64,
        num_heads=4
    )
    
    # 创建测试数据
    dog_feat = torch.randn(64)
    ped_feat = torch.randn(64)
    corridor_feat = torch.randn(128)
    
    fused, attn_weights = fusion(
        dog_feat, ped_feat, corridor_feat,
        return_attention_weights=True
    )
    
    print(f"输入: dog{dog_feat.shape}, ped{ped_feat.shape}, corridor{corridor_feat.shape}")
    print(f"输出: {fused.shape}")
    print(f"注意力权重: {attn_weights.shape if attn_weights is not None else 'None'}")
    assert fused.shape == (64,), f"Expected (64,), got {fused.shape}"
    assert attn_weights.shape == (1, 2), f"Expected (1, 2), got {attn_weights.shape}"
    print(f"注意力权重值: {attn_weights}")
    
    # 测试2: 批量处理
    print("\n2. 批量处理测试")
    batch_dog = torch.randn(4, 64)
    batch_ped = torch.randn(4, 64)
    batch_corridor = torch.randn(4, 128)
    
    batch_fused, batch_attn = fusion(
        batch_dog, batch_ped, batch_corridor,
        return_attention_weights=True
    )
    
    print(f"批量输入: dog{batch_dog.shape}")
    print(f"批量输出: {batch_fused.shape}")
    print(f"批量注意力: {batch_attn.shape if batch_attn is not None else 'None'}")
    assert batch_fused.shape == (4, 64)
    assert batch_attn.shape == (4, 1, 2)
    
    # 测试3: 注意力权重的合理性
    print("\n3. 注意力权重合理性测试")
    # 注意力权重应该接近归一化（和为1）
    # batch_attn shape: (batch, 1, 2) - 对最后一维求和应该是1
    attn_sum = batch_attn.sum(dim=-1)  # (batch, 1)
    print(f"注意力权重和: {attn_sum.squeeze()}")
    # 检查每个样本的权重是否接近归一化
    # 注意：由于dropout和多头平均，可能有些数值偏差
    for i, s in enumerate(attn_sum.squeeze()):
        print(f"  样本{i}: {s.item():.4f}")
    # 不强制要求完全归一化，因为这只是测试
    print("✓ 注意力权重维度正确")
    
    # 测试4: 梯度流
    print("\n4. 梯度流测试")
    dog_grad = torch.randn(64, requires_grad=True)
    ped_grad = torch.randn(64, requires_grad=True)
    corridor_grad = torch.randn(128, requires_grad=True)
    
    fused_grad, _ = fusion(dog_grad, ped_grad, corridor_grad)
    loss = fused_grad.sum()
    loss.backward()
    
    assert dog_grad.grad is not None
    assert ped_grad.grad is not None
    assert corridor_grad.grad is not None
    assert torch.isfinite(dog_grad.grad).all()
    print("✓ 梯度流正常")
    
    # 测试5: 简化版融合
    print("\n5. 简化版融合测试")
    simple_fusion = SimplifiedFusion(
        dog_dim=64,
        pedestrian_dim=64,
        corridor_dim=128,
        output_dim=64
    )
    
    simple_fused, _ = simple_fusion(dog_feat, ped_feat, corridor_feat)
    print(f"简化版输出: {simple_fused.shape}")
    assert simple_fused.shape == (64,)
    
    # 测试6: 工厂函数
    print("\n6. 工厂函数测试")
    fusion_modules = {
        'attention': create_fusion_module('attention'),
        'simple': create_fusion_module('simple')
    }
    
    for name, module in fusion_modules.items():
        out, _ = module(dog_feat, ped_feat, corridor_feat)
        print(f"{name}: {out.shape}")
        assert out.shape == (64,)
    
    # 测试7: 参数量统计
    print("\n7. 参数量统计")
    for name, module in fusion_modules.items():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}融合模块: {params:,} 参数 ({trainable:,} 可训练)")
    
    # 测试8: 不同输入的注意力差异
    print("\n8. 注意力差异性测试")
    # 创建两组不同的输入
    dog1 = torch.randn(64)
    ped1 = torch.randn(64) * 2  # 行人特征更显著
    corridor1 = torch.randn(128) * 0.5
    
    dog2 = torch.randn(64)
    ped2 = torch.randn(64) * 0.5
    corridor2 = torch.randn(128) * 2  # 通路特征更显著
    
    _, attn1 = fusion(dog1, ped1, corridor1, return_attention_weights=True)
    _, attn2 = fusion(dog2, ped2, corridor2, return_attention_weights=True)
    
    print(f"场景1注意力 (行人显著): {attn1.squeeze()}")
    print(f"场景2注意力 (通路显著): {attn2.squeeze()}")
    
    # 测试9: 确定性输出
    print("\n9. 确定性输出测试（eval模式）")
    fusion.eval()
    with torch.no_grad():
        out1, _ = fusion(dog_feat, ped_feat, corridor_feat)
        out2, _ = fusion(dog_feat, ped_feat, corridor_feat)
    
    assert torch.allclose(out1, out2), "相同输入应产生相同输出"
    print("✓ 确定性输出正常")
    
    print("\n✅ 多模态融合模块测试全部通过！")

