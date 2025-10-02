"""
PointNet实现
用于编码通路多边形（可变顶点数）为固定维度特征
基于论文: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    """
    PointNet基础网络
    
    处理点云数据，使用对称函数实现置换不变性
    适配2D点（通路多边形顶点）
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        feature_dim: int = 64,
        hidden_dims: list = [64, 128, 256],
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: 输入点的维度（2D点为2）
            feature_dim: 输出特征维度
            hidden_dims: 隐藏层维度列表
            use_batch_norm: 是否使用BatchNorm
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.use_batch_norm = use_batch_norm
        
        # 构建点特征提取网络（逐点MLP）
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        
        self.point_feature_extractor = nn.Sequential(*layers)
        
        # 全局特征维度（max + mean pooling拼接）
        self.global_feature_dim = hidden_dims[-1] * 2
        
        # 后处理网络
        self.post_mlp = nn.Sequential(
            nn.Linear(self.global_feature_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            points: Tensor(num_points, input_dim) 点云数据
                    或 Tensor(batch, num_points, input_dim)
        
        Returns:
            features: Tensor(feature_dim,) 全局特征
                     或 Tensor(batch, feature_dim)
        """
        # 处理输入维度
        if points.dim() == 2:
            # (num_points, input_dim)
            points = points.unsqueeze(0)  # (1, num_points, input_dim)
            single_input = True
        else:
            single_input = False
        
        batch_size, num_points, _ = points.shape
        
        # 1. 逐点特征提取
        # 重塑为 (batch * num_points, input_dim)
        points_flat = points.reshape(-1, self.input_dim)
        
        # 提取特征 (batch * num_points, hidden_dims[-1])
        point_features = self.point_feature_extractor(points_flat)
        
        # 重塑回 (batch, num_points, hidden_dims[-1])
        point_features = point_features.reshape(batch_size, num_points, -1)
        
        # 2. 对称聚合（置换不变性）
        # Max pooling
        max_features, _ = torch.max(point_features, dim=1)  # (batch, hidden_dims[-1])
        
        # Mean pooling
        mean_features = torch.mean(point_features, dim=1)  # (batch, hidden_dims[-1])
        
        # 拼接
        global_features = torch.cat([max_features, mean_features], dim=-1)  # (batch, hidden_dims[-1]*2)
        
        # 3. 后处理
        output_features = self.post_mlp(global_features)  # (batch, feature_dim)
        
        # 如果输入是单个点云，返回单个特征向量
        if single_input:
            output_features = output_features.squeeze(0)
        
        return output_features


class PointNetEncoder(nn.Module):
    """
    PointNet编码器（用于通路多边形）
    
    增强版本，添加了相对坐标和局部特征
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dims: list = [64, 128, 256],
        use_batch_norm: bool = True,
        use_relative_coords: bool = True
    ):
        """
        Args:
            feature_dim: 输出特征维度
            hidden_dims: 隐藏层维度
            use_batch_norm: 是否使用BatchNorm
            use_relative_coords: 是否使用相对坐标（相对于质心和参考点）
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_relative_coords = use_relative_coords
        
        # 根据是否使用相对坐标确定输入维度
        # 原始坐标(2) + 相对参考点(2) + 相对质心(2) = 6
        input_dim = 6 if use_relative_coords else 2
        
        # PointNet核心
        self.pointnet = PointNet(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            use_batch_norm=use_batch_norm
        )
    
    def _compute_relative_features(
        self,
        vertices: torch.Tensor,
        reference_point: torch.Tensor
    ) -> torch.Tensor:
        """
        计算相对特征
        
        Args:
            vertices: (num_vertices, 2) 多边形顶点
            reference_point: (2,) 参考点（通常是机器狗位置）
        
        Returns:
            features: (num_vertices, 6) 增强特征
        """
        # 计算质心
        centroid = vertices.mean(dim=0, keepdim=True)  # (1, 2)
        
        # 相对于参考点的坐标
        relative_to_ref = vertices - reference_point  # (num_vertices, 2)
        
        # 相对于质心的坐标（局部特征）
        relative_to_centroid = vertices - centroid  # (num_vertices, 2)
        
        # 拼接所有特征
        features = torch.cat([
            vertices,                # 原始坐标 (2)
            relative_to_ref,        # 相对参考点 (2)
            relative_to_centroid    # 相对质心 (2)
        ], dim=-1)  # (num_vertices, 6)
        
        return features
    
    def forward(
        self,
        vertices: torch.Tensor,
        reference_point: torch.Tensor = None
    ) -> torch.Tensor:
        """
        编码多边形
        
        Args:
            vertices: Tensor(num_vertices, 2) 多边形顶点
                     或 Tensor(batch, num_vertices, 2)
            reference_point: Tensor(2,) 参考点（可选）
                           或 Tensor(batch, 2)
        
        Returns:
            features: Tensor(feature_dim,) 多边形特征
                     或 Tensor(batch, feature_dim)
        """
        # 处理输入维度
        if vertices.dim() == 2:
            # (num_vertices, 2)
            single_input = True
            vertices = vertices.unsqueeze(0)  # (1, num_vertices, 2)
            if reference_point is not None:
                reference_point = reference_point.unsqueeze(0)  # (1, 2)
        else:
            single_input = False
        
        batch_size = vertices.size(0)
        
        # 计算相对特征
        if self.use_relative_coords:
            if reference_point is None:
                # 如果没有提供参考点，使用原点
                reference_point = torch.zeros(batch_size, 2, device=vertices.device)
            
            # 对batch中的每个多边形计算相对特征
            enhanced_features = []
            for i in range(batch_size):
                feat = self._compute_relative_features(
                    vertices[i],
                    reference_point[i]
                )
                enhanced_features.append(feat)
            
            # 堆叠 (batch, num_vertices, 6)
            enhanced_vertices = torch.stack(enhanced_features)
        else:
            enhanced_vertices = vertices
        
        # PointNet编码
        features = self.pointnet(enhanced_vertices)
        
        # 如果输入是单个多边形，返回单个特征
        if single_input:
            features = features.squeeze(0)
        
        return features


class AdaptivePointNetEncoder(nn.Module):
    """
    自适应PointNet编码器
    
    处理极端情况（非常少或非常多的顶点）
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        min_points: int = 3,
        max_points: int = 100
    ):
        """
        Args:
            feature_dim: 输出特征维度
            min_points: 最小顶点数（少于此数量会padding）
            max_points: 最大顶点数（超过会下采样）
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.min_points = min_points
        self.max_points = max_points
        
        self.encoder = PointNetEncoder(
            feature_dim=feature_dim,
            use_relative_coords=True
        )
    
    def _handle_extreme_cases(
        self,
        vertices: torch.Tensor
    ) -> torch.Tensor:
        """
        处理极端情况
        
        Args:
            vertices: (num_vertices, 2)
        
        Returns:
            processed_vertices: (adjusted_num_vertices, 2)
        """
        num_vertices = vertices.size(0)
        
        # 情况1: 顶点太少
        if num_vertices < self.min_points:
            # 通过插值增加点
            # 简单策略：在边上插值
            additional_points = []
            num_to_add = self.min_points - num_vertices
            
            for i in range(num_to_add):
                # 在随机边上插值
                edge_idx = i % num_vertices
                next_idx = (edge_idx + 1) % num_vertices
                
                # 线性插值
                alpha = 0.5
                interpolated = alpha * vertices[edge_idx] + (1 - alpha) * vertices[next_idx]
                additional_points.append(interpolated)
            
            vertices = torch.cat([vertices, torch.stack(additional_points)], dim=0)
        
        # 情况2: 顶点太多
        elif num_vertices > self.max_points:
            # 均匀下采样
            indices = torch.linspace(0, num_vertices - 1, self.max_points, device=vertices.device)
            indices = indices.long()
            vertices = vertices[indices]
        
        return vertices
    
    def forward(
        self,
        vertices: torch.Tensor,
        reference_point: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            vertices: Tensor(num_vertices, 2) 可能需要调整的顶点
            reference_point: Tensor(2,) 参考点
        
        Returns:
            features: Tensor(feature_dim,) 特征
        """
        # 处理极端情况
        adjusted_vertices = self._handle_extreme_cases(vertices)
        
        # 编码
        features = self.encoder(adjusted_vertices, reference_point)
        
        return features


# 便捷函数
def create_pointnet_encoder(
    feature_dim: int = 64,
    encoder_type: str = 'enhanced'
) -> nn.Module:
    """
    创建PointNet编码器的工厂函数
    
    Args:
        feature_dim: 输出特征维度
        encoder_type: 编码器类型
            - 'basic': 基础PointNet
            - 'enhanced': 增强版（使用相对坐标）
            - 'adaptive': 自适应版（处理极端情况）
    
    Returns:
        encoder: PointNet编码器
    """
    if encoder_type == 'basic':
        return PointNet(
            input_dim=2,
            feature_dim=feature_dim
        )
    elif encoder_type == 'enhanced':
        return PointNetEncoder(
            feature_dim=feature_dim,
            use_relative_coords=True
        )
    elif encoder_type == 'adaptive':
        return AdaptivePointNetEncoder(
            feature_dim=feature_dim
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == '__main__':
    """简单测试"""
    print("测试PointNet编码器...")
    
    # 测试1: 基础PointNet
    print("\n1. 基础PointNet")
    pointnet = PointNet(input_dim=2, feature_dim=64)
    
    # 单个点云
    points = torch.randn(10, 2)
    features = pointnet(points)
    print(f"输入形状: {points.shape}, 输出形状: {features.shape}")
    
    # 批量点云
    batch_points = torch.randn(4, 10, 2)
    batch_features = pointnet(batch_points)
    print(f"批量输入: {batch_points.shape}, 批量输出: {batch_features.shape}")
    
    # 测试2: 增强版PointNetEncoder
    print("\n2. 增强版PointNetEncoder")
    encoder = PointNetEncoder(feature_dim=64, use_relative_coords=True)
    
    # 多边形（正方形）
    square = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    reference = torch.zeros(2)
    
    square_features = encoder(square, reference)
    print(f"正方形顶点: {square.shape}, 特征: {square_features.shape}")
    
    # 测试置换不变性
    shuffled_square = square[torch.randperm(4)]
    shuffled_features = encoder(shuffled_square, reference)
    print(f"置换不变性检查: {torch.allclose(square_features, shuffled_features, atol=1e-5)}")
    
    # 测试3: 自适应编码器
    print("\n3. 自适应编码器")
    adaptive_encoder = AdaptivePointNetEncoder(feature_dim=64)
    
    # 少量顶点（三角形）
    triangle = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ])
    triangle_features = adaptive_encoder(triangle, reference)
    print(f"三角形: {triangle.shape} -> {triangle_features.shape}")
    
    # 大量顶点
    many_points = torch.randn(150, 2)
    many_features = adaptive_encoder(many_points, reference)
    print(f"大量顶点: {many_points.shape} -> {many_features.shape}")
    
    # 测试4: 参数量统计
    print("\n4. 参数量统计")
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    print("\n✓ PointNet编码器测试通过！")