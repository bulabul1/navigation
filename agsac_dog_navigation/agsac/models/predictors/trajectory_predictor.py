"""
轨迹预测器
支持加载预训练的SocialCircle+E-V2-Net或使用简化实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import os
from pathlib import Path


class TrajectoryPredictorInterface(nn.Module):
    """
    轨迹预测器接口基类
    所有预测器都应该实现这个接口
    """
    
    def forward(
        self,
        target_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        neighbor_angles: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测未来轨迹
        
        Args:
            target_trajectory: (8, 2) 或 (batch, 8, 2) 目标行人历史轨迹
            neighbor_trajectories: (N, 8, 2) 或 (batch, N, 8, 2) 邻居轨迹
            neighbor_angles: (N,) 或 (batch, N) 相对角度
            neighbor_mask: (N,) 或 (batch, N) 有效性mask
        
        Returns:
            predictions: (12, 2, 20) 或 (batch, 12, 2, 20)
                12个未来时间步，2维坐标，20个模态
        """
        raise NotImplementedError


class SimpleE_V2_Net(nn.Module):
    """
    简化版E-V2-Net
    GRU编码器-解码器架构，用于多模态轨迹预测
    """
    
    def __init__(
        self,
        encoder_input_dim: int = 64,
        encoder_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        prediction_horizon: int = 12,
        num_modes: int = 20,
        coordinate_dim: int = 2
    ):
        super().__init__()
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.prediction_horizon = prediction_horizon
        self.num_modes = num_modes
        self.coordinate_dim = coordinate_dim
        
        # 编码器：从社交特征到隐藏状态
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, encoder_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        )
        
        # 解码器：GRU生成未来轨迹
        self.decoder_cell = nn.GRUCell(
            input_size=coordinate_dim,
            hidden_size=decoder_hidden_dim
        )
        
        # 输出层：预测下一个位置
        self.output_layer = nn.Linear(decoder_hidden_dim, coordinate_dim)
        
        # 模态生成：从一个隐藏状态生成多个模态
        self.mode_generator = nn.ModuleList([
            nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
            for _ in range(num_modes)
        ])
    
    def forward(self, social_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            social_features: (feature_dim,) 或 (batch, feature_dim)
                从SocialCircle编码的社交特征
        
        Returns:
            predictions: (prediction_horizon, coordinate_dim, num_modes) 
                       或 (batch, prediction_horizon, coordinate_dim, num_modes)
        """
        if social_features.dim() == 1:
            social_features = social_features.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = social_features.size(0)
        
        # 1. 编码社交特征
        encoded = self.encoder(social_features)  # (batch, encoder_hidden_dim)
        
        # 2. 为每个模态生成初始隐藏状态
        all_mode_predictions = []
        
        for mode_id in range(self.num_modes):
            # 生成该模态的初始隐藏状态
            h = self.mode_generator[mode_id](encoded)  # (batch, decoder_hidden_dim)
            
            # 解码生成轨迹
            mode_trajectory = []
            current_pos = torch.zeros(batch_size, self.coordinate_dim, device=social_features.device)
            
            for t in range(self.prediction_horizon):
                # GRU解码
                h = self.decoder_cell(current_pos, h)
                
                # 预测下一个位置（增量）
                delta = self.output_layer(h)  # (batch, coordinate_dim)
                current_pos = current_pos + delta
                
                mode_trajectory.append(current_pos)
            
            # Stack: (batch, prediction_horizon, coordinate_dim)
            mode_trajectory = torch.stack(mode_trajectory, dim=1)
            all_mode_predictions.append(mode_trajectory)
        
        # Stack所有模态: (batch, prediction_horizon, coordinate_dim, num_modes)
        predictions = torch.stack(all_mode_predictions, dim=-1)
        
        if single_input:
            # (prediction_horizon, coordinate_dim, num_modes)
            predictions = predictions.squeeze(0)
        
        return predictions


class SimpleTrajectoryPredictor(TrajectoryPredictorInterface):
    """
    简化版轨迹预测器
    组合SocialCircle + SimpleE_V2_Net
    """
    
    def __init__(
        self,
        social_circle_dim: int = 64,
        encoder_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        prediction_horizon: int = 12,
        num_modes: int = 20,
        num_sectors: int = 8,
        freeze: bool = False
    ):
        super().__init__()
        
        # 导入SocialCircle（已经实现）
        from agsac.models.encoders.social_circle import SocialCircle
        
        self.social_circle = SocialCircle(
            num_sectors=num_sectors,
            feature_dim=social_circle_dim,
            hidden_dim=128
        )
        
        self.e_v2_net = SimpleE_V2_Net(
            encoder_input_dim=social_circle_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            prediction_horizon=prediction_horizon,
            num_modes=num_modes
        )
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        target_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        neighbor_angles: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """实现接口"""
        # 1. SocialCircle编码
        social_features = self.social_circle(
            target_trajectory,
            neighbor_trajectories,
            neighbor_angles,
            neighbor_mask
        )
        
        # 2. E-V2-Net预测
        predictions = self.e_v2_net(social_features)
        
        return predictions


class PretrainedTrajectoryPredictor(TrajectoryPredictorInterface):
    """
    预训练轨迹预测器
    加载开源的SocialCircle+E-V2-Net预训练权重
    
    使用说明:
        1. 将预训练权重放在 pretrained/social_circle/weights/ 或 pretrained/e_v2_net/weights/
        2. 创建此类实例时指定权重路径
        3. 如果加载失败，会自动回退到SimpleTrajectoryPredictor
    """
    
    def __init__(
        self,
        weights_path: Union[str, Path],
        freeze: bool = True,
        fallback_to_simple: bool = True
    ):
        super().__init__()
        
        self.weights_path = Path(weights_path)
        self.freeze = freeze
        
        # 尝试加载预训练模型
        try:
            self._load_pretrained_model()
            self.using_pretrained = True
            print(f"✓ 成功加载预训练模型: {weights_path}")
        except Exception as e:
            if fallback_to_simple:
                print(f"⚠ 加载预训练模型失败: {e}")
                print(f"  回退到简化实现...")
                self._use_simple_predictor()
                self.using_pretrained = False
            else:
                raise
        
        if freeze and self.using_pretrained:
            for param in self.parameters():
                param.requires_grad = False
    
    def _load_pretrained_model(self):
        """加载预训练模型（需要根据实际开源代码调整）"""
        # TODO: 根据实际的开源代码实现加载逻辑
        # 这里是一个框架示例
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {self.weights_path}")
        
        # 方法1: 如果有完整的模型文件
        try:
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            
            # 假设checkpoint包含 'social_circle' 和 'e_v2_net' 的权重
            # 实际结构需要根据开源代码调整
            if 'social_circle' in checkpoint and 'e_v2_net' in checkpoint:
                from agsac.models.encoders.social_circle import SocialCircle
                
                self.social_circle = SocialCircle()
                self.social_circle.load_state_dict(checkpoint['social_circle'])
                
                self.e_v2_net = SimpleE_V2_Net()  # 或加载原始E-V2-Net
                self.e_v2_net.load_state_dict(checkpoint['e_v2_net'])
            else:
                raise ValueError("权重文件格式不符合预期")
        
        except Exception as e:
            # 方法2: 如果需要从原始代码导入
            # 尝试从 pretrained/social_circle/original_code 导入
            import sys
            original_code_path = self.weights_path.parent.parent / 'original_code'
            if original_code_path.exists():
                sys.path.insert(0, str(original_code_path))
                try:
                    # 假设原始代码有这些模块
                    # from models.social_circle import SocialCircleModel
                    # from models.e_v2_net import EV2NetModel
                    # self.social_circle = SocialCircleModel()
                    # self.e_v2_net = EV2NetModel()
                    # 加载权重...
                    pass
                finally:
                    sys.path.pop(0)
            
            raise e
    
    def _use_simple_predictor(self):
        """使用简化实现作为后备方案"""
        simple_predictor = SimpleTrajectoryPredictor()
        self.social_circle = simple_predictor.social_circle
        self.e_v2_net = simple_predictor.e_v2_net
    
    def forward(
        self,
        target_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        neighbor_angles: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """实现接口"""
        social_features = self.social_circle(
            target_trajectory,
            neighbor_trajectories,
            neighbor_angles,
            neighbor_mask
        )
        
        predictions = self.e_v2_net(social_features)
        
        return predictions


def create_trajectory_predictor(
    predictor_type: str = 'simple',
    **kwargs
) -> TrajectoryPredictorInterface:
    """
    创建轨迹预测器的工厂函数
    
    Args:
        predictor_type: 预测器类型
            - 'simple': 简化实现
            - 'pretrained': 加载预训练模型
        **kwargs: 其他参数
    
    Returns:
        predictor: 轨迹预测器
    """
    if predictor_type == 'simple':
        return SimpleTrajectoryPredictor(**kwargs)
    elif predictor_type == 'pretrained':
        return PretrainedTrajectoryPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


if __name__ == '__main__':
    """简单测试"""
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("测试轨迹预测器...")
    
    # 测试1: 简化版预测器
    print("\n1. 简化版预测器")
    simple_predictor = SimpleTrajectoryPredictor(
        social_circle_dim=64,
        prediction_horizon=12,
        num_modes=20
    )
    
    # 创建测试数据
    import math
    target_traj = torch.randn(8, 2)
    neighbor_trajs = torch.randn(5, 8, 2)
    angles = torch.rand(5) * 2 * math.pi
    
    predictions = simple_predictor(target_traj, neighbor_trajs, angles)
    print(f"输入: 目标{target_traj.shape}, 邻居{neighbor_trajs.shape}")
    print(f"输出: {predictions.shape}")
    assert predictions.shape == (12, 2, 20), f"Expected (12, 2, 20), got {predictions.shape}"
    
    # 测试2: 批量处理
    print("\n2. 批量处理")
    batch_target = torch.randn(4, 8, 2)
    batch_neighbors = torch.randn(4, 10, 8, 2)
    batch_angles = torch.rand(4, 10) * 2 * math.pi
    
    batch_predictions = simple_predictor(batch_target, batch_neighbors, batch_angles)
    print(f"批量输出: {batch_predictions.shape}")
    assert batch_predictions.shape == (4, 12, 2, 20)
    
    # 测试3: 预训练预测器（会回退到简化实现）
    print("\n3. 预训练预测器（回退测试）")
    pretrained_predictor = PretrainedTrajectoryPredictor(
        weights_path='pretrained/social_circle/weights/model.pth',
        fallback_to_simple=True
    )
    
    pretrained_predictions = pretrained_predictor(target_traj, neighbor_trajs, angles)
    print(f"预训练预测输出: {pretrained_predictions.shape}")
    assert pretrained_predictions.shape == (12, 2, 20)
    
    # 测试4: 工厂函数
    print("\n4. 工厂函数")
    predictor = create_trajectory_predictor('simple')
    output = predictor(target_traj, neighbor_trajs, angles)
    print(f"工厂函数输出: {output.shape}")
    
    # 测试5: 参数量统计
    print("\n5. 参数量统计")
    params = sum(p.numel() for p in simple_predictor.parameters())
    trainable_params = sum(p.numel() for p in simple_predictor.parameters() if p.requires_grad)
    print(f"SimpleTrajectoryPredictor: {params:,} 参数 ({trainable_params:,} 可训练)")
    
    print("\n✓ 轨迹预测器测试通过！")

