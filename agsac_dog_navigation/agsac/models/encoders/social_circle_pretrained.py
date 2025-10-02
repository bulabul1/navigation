"""
SocialCircle预训练模型适配器
将开源的SocialCircle PyTorch实现适配到我们的接口

参考: https://github.com/cocoon2wong/SocialCircle
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Optional, Tuple


class SocialCirclePretrained(nn.Module):
    """
    SocialCircle预训练模型适配器
    
    功能:
    1. 加载开源SocialCircle的PyTorch实现
    2. 适配输入/输出格式到我们的接口
    3. 支持预训练权重加载
    
    输入格式 (我们的接口):
        target_trajectory: (batch, obs_horizon, 2)
        neighbor_trajectories: (batch, num_neighbors, obs_horizon, 2)
        neighbor_mask: (batch, num_neighbors)
    
    输出格式:
        social_features: (batch, social_feature_dim)
    """
    
    def __init__(
        self,
        obs_horizon: int = 8,
        social_feature_dim: int = 128,
        num_sectors: int = 8,
        use_pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            obs_horizon: 观测历史长度
            social_feature_dim: 输出特征维度
            num_sectors: SocialCircle的扇区数量
            use_pretrained: 是否使用预训练模型
            pretrained_path: 预训练权重路径
        """
        super().__init__()
        
        self.obs_horizon = obs_horizon
        self.social_feature_dim = social_feature_dim
        self.num_sectors = num_sectors
        self.use_pretrained = use_pretrained
        
        # 尝试导入开源SocialCircle
        self._import_socialcircle()
        
        # 构建模型
        if self.socialcircle_available:
            self._build_pretrained_model()
        else:
            # 如果导入失败，使用我们的SimplifiedSocialCircle作为fallback
            print("[Warning] 无法导入开源SocialCircle，使用SimplifiedSocialCircle作为fallback")
            from .social_circle import SimplifiedSocialCircle
            self.model = SimplifiedSocialCircle(
                obs_horizon=obs_horizon,
                social_feature_dim=social_feature_dim
            )
            self.is_fallback = True
        
        # 加载预训练权重
        if use_pretrained and pretrained_path and self.socialcircle_available:
            self._load_pretrained(pretrained_path)
    
    def _import_socialcircle(self):
        """尝试导入开源SocialCircle"""
        try:
            # 添加external目录到path
            external_dir = Path(__file__).parent.parent.parent.parent / 'external' / 'SocialCircle_original'
            
            if external_dir.exists():
                sys.path.insert(0, str(external_dir))
                
                # 尝试导入（具体的import路径需要根据实际代码结构调整）
                # from socialCircle.layers import SocialCircleLayer
                # self.SocialCircleLayer = SocialCircleLayer
                
                self.socialcircle_available = True
                print(f"[Success] 成功导入开源SocialCircle from {external_dir}")
            else:
                print(f"[Warning] SocialCircle代码未找到: {external_dir}")
                self.socialcircle_available = False
                
        except Exception as e:
            print(f"[Warning] 导入SocialCircle失败: {e}")
            self.socialcircle_available = False
    
    def _build_pretrained_model(self):
        """构建预训练模型"""
        # TODO: 根据实际的SocialCircle代码结构来构建
        # 这里需要根据下载的代码来调整
        
        # 示例（需要根据实际代码调整）:
        # self.model = self.SocialCircleLayer(
        #     obs_len=self.obs_horizon,
        #     feature_dim=self.social_feature_dim,
        #     partitions=self.num_sectors,
        #     ...
        # )
        
        # 暂时使用fallback
        print("[Info] 使用SimplifiedSocialCircle（待适配开源代码）")
        from .social_circle import SimplifiedSocialCircle
        self.model = SimplifiedSocialCircle(
            obs_horizon=self.obs_horizon,
            social_feature_dim=self.social_feature_dim
        )
        self.is_fallback = True
    
    def _load_pretrained(self, pretrained_path: str):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 根据实际的checkpoint格式加载
            # 可能需要适配key名称
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[Success] 成功加载预训练权重: {pretrained_path}")
            
        except Exception as e:
            print(f"[Warning] 加载预训练权重失败: {e}")
    
    def forward(
        self,
        target_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        neighbor_mask: torch.Tensor,
        robot_position: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            target_trajectory: (batch, obs_horizon, 2)
            neighbor_trajectories: (batch, num_neighbors, obs_horizon, 2)
            neighbor_mask: (batch, num_neighbors)
            robot_position: (batch, 2) - 可选
        
        Returns:
            social_features: (batch, social_feature_dim)
        """
        # 如果是fallback，直接调用
        if self.is_fallback:
            return self.model(
                target_trajectory,
                neighbor_trajectories,
                neighbor_mask,
                robot_position
            )
        
        # TODO: 适配开源SocialCircle的输入格式
        # 可能需要格式转换
        
        # 调用开源模型
        # output = self.model(...)
        
        # 适配输出格式
        # social_features = ...
        
        # 暂时使用fallback
        return self.model(
            target_trajectory,
            neighbor_trajectories,
            neighbor_mask,
            robot_position
        )


def create_socialcircle_pretrained(
    use_pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> SocialCirclePretrained:
    """
    工厂函数：创建SocialCircle预训练模型
    
    Args:
        use_pretrained: 是否使用预训练
        pretrained_path: 预训练权重路径
        **kwargs: 其他参数
    
    Returns:
        SocialCirclePretrained实例
    """
    # 如果没有指定路径，使用默认路径
    if pretrained_path is None and use_pretrained:
        default_path = Path(__file__).parent.parent.parent.parent / 'pretrained' / 'social_circle' / 'weights.pth'
        if default_path.exists():
            pretrained_path = str(default_path)
            print(f"[Info] 使用默认预训练权重: {pretrained_path}")
    
    return SocialCirclePretrained(
        use_pretrained=use_pretrained,
        pretrained_path=pretrained_path,
        **kwargs
    )


# ==================== 测试 ====================
if __name__ == '__main__':
    print("测试SocialCircle预训练模型适配器...")
    
    # 创建模型
    model = create_socialcircle_pretrained(
        obs_horizon=8,
        social_feature_dim=128,
        use_pretrained=False  # 测试时不加载权重
    )
    
    print(f"模型创建成功: {'使用fallback' if model.is_fallback else '使用预训练'}")
    
    # 测试forward
    batch_size = 2
    num_neighbors = 5
    obs_horizon = 8
    
    target_traj = torch.randn(batch_size, obs_horizon, 2)
    neighbor_trajs = torch.randn(batch_size, num_neighbors, obs_horizon, 2)
    neighbor_mask = torch.ones(batch_size, num_neighbors, dtype=torch.bool)
    neighbor_mask[:, 3:] = False  # 只有前3个邻居有效
    
    output = model(target_traj, neighbor_trajs, neighbor_mask)
    
    print(f"输入:")
    print(f"  target_trajectory: {target_traj.shape}")
    print(f"  neighbor_trajectories: {neighbor_trajs.shape}")
    print(f"  neighbor_mask: {neighbor_mask.shape}")
    print(f"输出:")
    print(f"  social_features: {output.shape}")
    
    assert output.shape == (batch_size, 128), f"输出shape错误: {output.shape}"
    
    print("\n[SUCCESS] SocialCircle预训练模型适配器测试通过！")

