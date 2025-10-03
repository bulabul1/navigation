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
        
        # 转换为绝对路径
        self.weights_path = Path(weights_path).absolute()
        self.freeze = freeze
        
        # 尝试加载预训练模型
        try:
            self._load_pretrained_model()
            self.using_pretrained = True
            print(f"[OK] 成功加载预训练模型: {weights_path}")
        except Exception as e:
            if fallback_to_simple:
                print(f"[WARN] 加载预训练模型失败: {e}")
                print(f"  回退到简化实现...")
                self._use_simple_predictor()
                self.using_pretrained = False
            else:
                raise
        
        if freeze and self.using_pretrained:
            for param in self.parameters():
                param.requires_grad = False
    
    def _load_pretrained_model(self):
        """从SocialCircle加载预训练的EVSCModel"""
        import os
        import sys
        
        # 设置环境变量避免OpenMP冲突
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # 保存当前工作目录和sys.path
        original_dir = os.getcwd()
        original_path_len = len(sys.path)
        
        # SocialCircle目录
        sc_dir = Path(__file__).parent.parent.parent.parent / 'external' / 'SocialCircle_original'
        
        if not sc_dir.exists():
            raise FileNotFoundError(f"SocialCircle目录不存在: {sc_dir}")
        
        # 切换到SocialCircle目录
        os.chdir(sc_dir)
        
        # 添加到sys.path（记录位置以便清理）
        sc_dir_str = str(sc_dir)
        if sc_dir_str not in sys.path:
            sys.path.insert(0, sc_dir_str)
        
        try:
            # 导入main函数
            from main import main
            
            # 调用main加载模型（不运行测试）
            # 关键：必须指定 --model evsc，默认使用GPU 0
            print(f"[INFO] 正在加载模型...")
            print(f"  - 权重路径: {self.weights_path}")
            
            # 检查GPU是否可用
            gpu_id = '0' if torch.cuda.is_available() else '-1'
            print(f"  - 使用设备: {'GPU' if gpu_id == '0' else 'CPU'}")
            
            structure = main(
                ['--model', 'evsc', '--load', str(self.weights_path), '--gpu', gpu_id],
                run_train_or_test=False
            )
            
            # 检查返回值
            if structure is None:
                raise ValueError("main()返回None，模型加载失败")
            
            print(f"[INFO] Structure创建成功: {type(structure).__name__}")
            
            # 因为run_train_or_test=False，模型未创建，需要手动创建
            if structure.model is None:
                print(f"[INFO] 手动创建模型...")
                structure.model = structure.create_model().to(structure.device)
                
                # 加载预训练权重
                if self.weights_path.exists():
                    print(f"[INFO] 加载预训练权重...")
                    structure.model.load_weights_from_logDir(str(self.weights_path))
            
            # 保存模型对象和设备信息
            self.evsc_model = structure.model
            self.evsc_model.eval()
            self.device = structure.device  # 保存设备信息
            
            print(f"[INFO] 模型设备: {self.device}")
            
            # 保存配置信息（动态获取模态数）
            self.obs_frames = getattr(structure.args, 'obs_frames', 8)
            self.pred_frames = getattr(structure.args, 'pred_frames', 12)
            
            # 模态数 = K(测试重复) × Kc(风格通道)
            K = getattr(structure.args, 'K', 1)
            Kc = getattr(structure.args, 'Kc', 20)
            self.num_modes = K * Kc
            
            print(f"[OK] EVSCModel加载成功")
            print(f"  - obs_frames: {self.obs_frames}")
            print(f"  - pred_frames: {self.pred_frames}")
            print(f"  - num_modes: {self.num_modes} (K={K} × Kc={Kc})")
            
        finally:
            # 清理sys.path（移除添加的路径）
            while len(sys.path) > original_path_len:
                removed = sys.path.pop(0)
                if removed != sc_dir_str:
                    # 如果不是我们添加的，放回去
                    sys.path.insert(0, removed)
                    break
            
            # 恢复原目录
            os.chdir(original_dir)
    
    def _use_simple_predictor(self):
        """使用简化实现作为后备方案"""
        simple_predictor = SimpleTrajectoryPredictor()
        self.social_circle = simple_predictor.social_circle
        self.e_v2_net = simple_predictor.e_v2_net
        # 设置默认设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _interpolate_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        分段线性插值关键点到完整轨迹
        
        EVSCModel输出3个关键点在时间步 t=[4, 8, 11]
        需要插值到完整的12个时间步 t=[0, 1, ..., 11]
        
        Args:
            keypoints: (batch, K, 3, 2) - K个模态，每个3个关键点
        
        Returns:
            full_traj: (batch, K, 12, 2) - K个模态，每个12个点
        """
        batch, K, n_key, dim = keypoints.shape
        device = keypoints.device
        
        # 创建输出tensor
        full_traj = torch.zeros(batch, K, 12, dim, device=device)
        
        # 对每个时间步进行插值
        # 关键点对应时间: t=[4, 8, 11]
        for t in range(12):
            t_val = float(t)
            
            if t <= 4:
                # 区间 [0, 4]: 保持第一个关键点的值（避免从原点引入假位移）
                full_traj[:, :, t, :] = keypoints[:, :, 0, :]
                
            elif t <= 8:
                # 区间 (4, 8]: 从第一个关键点线性插值到第二个关键点
                alpha = (t_val - 4.0) / 4.0  # (t-4)/(8-4)
                full_traj[:, :, t, :] = (
                    (1 - alpha) * keypoints[:, :, 0, :] +
                    alpha * keypoints[:, :, 1, :]
                )
                
            else:  # t > 8, t <= 11
                # 区间 (8, 11]: 从第二个关键点线性插值到第三个关键点
                alpha = (t_val - 8.0) / 3.0  # (t-8)/(11-8)
                full_traj[:, :, t, :] = (
                    (1 - alpha) * keypoints[:, :, 1, :] +
                    alpha * keypoints[:, :, 2, :]
                )
        
        return full_traj
    
    def forward(
        self,
        target_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        neighbor_angles: torch.Tensor = None,
        neighbor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 - 预测未来轨迹
        
        Args:
            target_trajectory: (batch, obs_frames, 2) 目标行人轨迹
            neighbor_trajectories: (batch, N, obs_frames, 2) 邻居轨迹
            neighbor_angles: 未使用（EVSCModel内部计算）
            neighbor_mask: (batch, N) 邻居有效性mask
        
        Returns:
            predictions: (batch, pred_frames, 2, num_modes) 预测轨迹
        """
        if not self.using_pretrained:
            # 使用简化实现
            social_features = self.social_circle(
                target_trajectory,
                neighbor_trajectories,
                neighbor_angles,
                neighbor_mask
            )
            predictions = self.e_v2_net(social_features)
            return predictions
        
        # 使用预训练EVSCModel
        batch_size = target_trajectory.size(0)
        obs_len = target_trajectory.size(1)
        
        # 输入长度校验
        if obs_len != self.obs_frames:
            raise ValueError(
                f"输入轨迹长度 ({obs_len}) 与预训练模型期望 ({self.obs_frames}) 不匹配。"
                f"请检查数据预处理或重新训练模型。"
            )
        
        # 准备输入（EVSCModel格式）
        # 确保数据在正确的设备上
        obs = target_trajectory.to(self.device)      # (batch, 8, 2)
        nei = neighbor_trajectories.to(self.device)  # (batch, N, 8, 2)
        
        # 应用邻居mask（屏蔽无效邻居）
        if neighbor_mask is not None:
            # mask: (batch, N) -> (batch, N, 1, 1) 用于广播
            mask_expanded = neighbor_mask.unsqueeze(-1).unsqueeze(-1).to(self.device)
            nei = nei * mask_expanded  # 无效邻居轨迹置零
        
        # EVSCModel推理
        with torch.no_grad():
            Y, social_circle, _ = self.evsc_model([obs, nei], training=False)
        
        # Y shape: (batch, K*Kc, 3, 2)
        # K*Kc 个模态，每个模态3个关键点在t=[4,8,11]
        actual_num_modes = Y.size(1)
        
        # 插值：3个关键点 → 12个完整点
        full_predictions = self._interpolate_keypoints(Y)
        # 输出: (batch, actual_num_modes, 12, 2)
        
        # 调整维度顺序以匹配接口
        # 从 (batch, num_modes, 12, 2) → (batch, 12, 2, num_modes)
        predictions = full_predictions.permute(0, 2, 3, 1)
        
        return predictions


def create_trajectory_predictor(
    predictor_type: str = 'pretrained',
    **kwargs
) -> TrajectoryPredictorInterface:
    """
    创建轨迹预测器的工厂函数
    
    Args:
        predictor_type: 预测器类型
            - 'pretrained': 加载预训练模型（默认且推荐）
            - 'simple': 简化实现（已弃用，不推荐使用）
        **kwargs: 其他参数
    
    Returns:
        predictor: 轨迹预测器
    """
    if predictor_type == 'pretrained':
        return PretrainedTrajectoryPredictor(**kwargs)
    elif predictor_type == 'simple':
        print("[WARNING] 简化版预测器已弃用，请使用预训练模型！")
        return SimpleTrajectoryPredictor(**kwargs)
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

