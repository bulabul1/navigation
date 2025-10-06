"""
AGSAC主模型类
整合所有子模块：编码器、融合、SAC、GDE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import os

from .encoders.dog_state_encoder import DogStateEncoder, create_dog_state_encoder
from .encoders.corridor_encoder import CorridorEncoder, create_corridor_encoder
from .encoders.pointnet import PointNet
from .predictors.trajectory_predictor import TrajectoryPredictorInterface, create_trajectory_predictor
from .encoders.pedestrian_encoder import PedestrianEncoder, create_pedestrian_encoder
from .fusion.multi_modal_fusion import MultiModalFusion, create_fusion_module
from .sac.sac_agent import SACAgent
from .evaluator.geometric_evaluator import GeometricDifferentialEvaluator


class AGSACModel(nn.Module):
    """
    AGSAC主模型：整合所有子模块
    
    架构：
    1. 感知编码层：
       - DogStateEncoder: 编码机器狗状态
       - PointNet + CorridorEncoder: 编码走廊几何
       - SocialCircle: 编码社交上下文
    
    2. 预测层：
       - TrajectoryPredictor (E-V2-Net): 预测行人未来轨迹
    
    3. 行人编码层：
       - PedestrianEncoder: 编码预测的未来轨迹
    
    4. 融合层：
       - MultiModalFusion: 融合三种特征
    
    5. 决策层：
       - SACAgent (Actor + Critic): 生成动作和Q值
    
    6. 评估层：
       - GeometricDifferentialEvaluator: 评估路径几何质量
    
    输入格式：
        observation = {
            'dog': {
                'trajectory': (batch, 8, 2),
                'velocity': (batch, 2),
                'position': (batch, 2),
                'goal': (batch, 2)
            },
            'pedestrians': {
                'trajectories': (batch, max_peds, obs_horizon, 2),
                'mask': (batch, max_peds)
            },
            'corridors': {
                'polygons': (batch, max_corridors, max_vertices, 2),
                'vertex_counts': (batch, max_corridors),
                'mask': (batch, max_corridors)
            },
            'reference_line': (batch, 2, 2)  # 用于GDE
        }
    """
    
    def __init__(
        self,
        # 维度配置
        dog_feature_dim: int = 64,
        corridor_feature_dim: int = 128,
        social_feature_dim: int = 128,
        pedestrian_feature_dim: int = 128,
        fusion_dim: int = 64,
        action_dim: int = 22,
        
        # 场景配置
        max_pedestrians: int = 10,
        max_corridors: int = 5,
        max_vertices: int = 20,
        obs_horizon: int = 8,
        pred_horizon: int = 12,
        num_modes: int = 20,
        
        # 网络配置
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        
        # SAC配置
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        encoder_lr: Optional[float] = None,  # 编码器学习率，None则使用critic_lr
        alpha_lr: float = 3e-4,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_entropy: bool = True,
        target_entropy: Optional[float] = None,
        max_grad_norm: float = 1.0,
        
        # 预训练模型配置
        use_pretrained_predictor: bool = False,
        pretrained_weights_path: Optional[str] = None,
        
        # 设备
        device: str = 'cpu'
    ):
        """
        Args:
            dog_feature_dim: 机器狗特征维度
            corridor_feature_dim: 走廊特征维度
            social_feature_dim: 社交上下文特征维度
            pedestrian_feature_dim: 行人特征维度
            fusion_dim: 融合后特征维度
            action_dim: 动作维度
            max_pedestrians: 最大行人数
            max_corridors: 最大走廊数
            max_vertices: 走廊最大顶点数
            obs_horizon: 观测历史长度
            pred_horizon: 预测未来长度
            num_modes: 预测模态数
            hidden_dim: LSTM隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout比例
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            alpha_lr: 熵系数学习率
            alpha: 固定熵系数（当auto_entropy=False时生效）
            gamma: 折扣因子
            tau: 软更新系数
            auto_entropy: 是否自动调节熵系数
            target_entropy: 目标熵
            max_grad_norm: 梯度裁剪阈值
            use_pretrained_predictor: 是否使用预训练轨迹预测器
            pretrained_weights_path: 预训练模型权重路径（如 'weights/SocialCircle/evsczara1'）
            device: 设备
        """
        super().__init__()
        
        self.dog_feature_dim = dog_feature_dim
        self.corridor_feature_dim = corridor_feature_dim
        self.social_feature_dim = social_feature_dim
        self.pedestrian_feature_dim = pedestrian_feature_dim
        self.fusion_dim = fusion_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        
        # ==================== 1. 感知编码层 ====================
        
        # 1.1 机器狗状态编码器
        self.dog_encoder = create_dog_state_encoder(
            encoder_type='gru',
            hidden_dim=dog_feature_dim,
            gru_layers=2,
            dropout=dropout
        )
        
        # 1.2 走廊编码器 (PointNet + CorridorEncoder)
        # PointNet用于编码单个走廊多边形
        self.pointnet = PointNet(
            input_dim=2,  # (x, y)
            feature_dim=64,
            hidden_dims=[64, 128, 256],
            use_batch_norm=True
        )
        
        # CorridorEncoder用于聚合多个走廊
        self.corridor_encoder = create_corridor_encoder(
            encoder_type='attention',
            input_dim=64,
            output_dim=corridor_feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_corridors=max_corridors
        )
        
        # ==================== 2. 预测层 ====================
        
        # 强制使用预训练模型
        if not use_pretrained_predictor or not pretrained_weights_path:
            raise ValueError(
                "必须使用预训练轨迹预测器！\n"
                "请在配置文件中设置:\n"
                "  use_pretrained_predictor: true\n"
                "  pretrained_weights_path: 'external/SocialCircle_original/weights/SocialCircle/evsczara1'"
            )
        
        print(f"[INFO] 使用预训练轨迹预测器: {pretrained_weights_path}")
        self.trajectory_predictor = create_trajectory_predictor(
            predictor_type='pretrained',
            weights_path=pretrained_weights_path,
            freeze=True,
            fallback_to_simple=False,  # 不允许回退到简化版
            device=device
        )
        
        # ==================== 3. 行人编码层 ====================
        
        self.pedestrian_encoder = create_pedestrian_encoder(
            encoder_type='attention',
            pred_horizon=pred_horizon,
            coord_dim=2,
            num_modes=num_modes,
            feature_dim=dog_feature_dim,  # 保持与dog一致，都是64
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ==================== 4. 融合层 ====================
        
        self.fusion = create_fusion_module(
            fusion_type='attention',
            dog_dim=dog_feature_dim,
            pedestrian_dim=dog_feature_dim,  # 与dog一致
            corridor_dim=corridor_feature_dim,
            output_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ==================== 5. 决策层（SAC） ====================
        
        self.sac_agent = SACAgent(
            state_dim=fusion_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            gamma=gamma,
            tau=tau,
            auto_entropy=auto_entropy,
            target_entropy=target_entropy,
            fixed_alpha=alpha,
            max_grad_norm=max_grad_norm,
            device=device
        )
        
        # ==================== 编码器优化器 ====================
        # 收集所有编码器和fusion的参数
        encoder_params = []
        encoder_params.extend(self.dog_encoder.parameters())
        encoder_params.extend(self.pointnet.parameters())
        encoder_params.extend(self.corridor_encoder.parameters())
        encoder_params.extend(self.pedestrian_encoder.parameters())
        encoder_params.extend(self.fusion.parameters())
        # 注意：trajectory_predictor已冻结，不包含
        
        # 创建编码器优化器（独立学习率或与critic同步）
        actual_encoder_lr = encoder_lr if encoder_lr is not None else critic_lr
        self.encoder_optimizer = optim.Adam(encoder_params, lr=actual_encoder_lr)
        print(f"[Model] 编码器优化器创建：{sum(p.numel() for p in encoder_params):,} 参数, lr={actual_encoder_lr}")
        
        # ==================== 6. 评估层（GDE） ====================
        
        self.geo_evaluator = GeometricDifferentialEvaluator(
            eta=0.5,
            M=10
        )
        
        # 移动到设备
        self.to(self.device)
        
        # 打印参数统计
        self._print_parameter_stats()
    
    def _print_parameter_stats(self):
        """打印各模块参数统计"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        stats = {
            'DogEncoder': count_params(self.dog_encoder),
            'PointNet': count_params(self.pointnet),
            'CorridorEncoder': count_params(self.corridor_encoder),
            'TrajectoryPredictor': count_params(self.trajectory_predictor),
            'PedestrianEncoder': count_params(self.pedestrian_encoder),
            'Fusion': count_params(self.fusion),
            'SAC_Actor': count_params(self.sac_agent.actor),
            'SAC_Critic': count_params(self.sac_agent.critic),
            'GDE': count_params(self.geo_evaluator)
        }
        
        total = sum(stats.values())
        
        print("\n" + "="*60)
        print("AGSAC模型参数统计")
        print("="*60)
        for name, params in stats.items():
            print(f"{name:.<30} {params:>10,} ({params/total*100:>5.1f}%)")
        print("-"*60)
        print(f"{'总计':.<30} {total:>10,} (100.0%)")
        print(f"{'预算':.<30} {2000000:>10,}")
        print(f"{'剩余':.<30} {2000000-total:>10,} ({(2000000-total)/2000000*100:>5.1f}%)")
        print("="*60 + "\n")
    
    def init_hidden_states(self, batch_size: int = 1) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        初始化所有隐藏状态
        
        Returns:
            {
                'actor': (h, c),
                'critic1': (h, c),
                'critic2': (h, c)
            }
        """
        hidden_states = {
            'actor': (
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device),
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
            ),
            'critic1': (
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device),
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
            ),
            'critic2': (
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device),
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
            )
        }
        return hidden_states
    
    def encode_corridors(
        self,
        corridor_polygons: torch.Tensor,
        corridor_vertex_counts: torch.Tensor,
        corridor_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        编码走廊几何
        
        Args:
            corridor_polygons: (batch, max_corridors, max_vertices, 2)
            corridor_vertex_counts: (batch, max_corridors)
            corridor_mask: (batch, max_corridors)
        
        Returns:
            corridor_features: (batch, corridor_feature_dim)
        """
        batch_size, max_corridors, max_vertices, coord_dim = corridor_polygons.shape
        
        # 1. 使用PointNet编码每个走廊
        # Reshape: (batch*max_corridors, max_vertices, 2)
        corridors_flat = corridor_polygons.view(batch_size * max_corridors, max_vertices, coord_dim)
        
        # PointNet编码: (batch*max_corridors, 64)
        corridor_features_flat = self.pointnet(corridors_flat)
        
        # Reshape回: (batch, max_corridors, 64)
        corridor_features = corridor_features_flat.view(batch_size, max_corridors, -1)
        
        # 2. 使用CorridorEncoder聚合多个走廊
        # corridor_features: (batch, corridor_feature_dim)
        corridor_features = self.corridor_encoder(
            corridor_features,
            corridor_mask
        )
        
        return corridor_features
    
    def forward(
        self,
        observation: Dict,
        hidden_states: Optional[Dict] = None,
        deterministic: bool = False,
        return_attention: bool = False
    ) -> Dict:
        """
        完整前向传播
        
        Args:
            observation: 观测字典
            hidden_states: 隐藏状态字典
            deterministic: 是否确定性动作
            return_attention: 是否返回注意力权重
        
        Returns:
            {
                'action': (batch, action_dim),
                'log_prob': (batch,),
                'q1': (batch,),
                'q2': (batch,),
                'fused_state': (batch, fusion_dim),
                'hidden_states': 更新后的隐藏状态,
                'debug_info': {
                    'dog_features': (batch, dog_feature_dim),
                    'corridor_features': (batch, corridor_feature_dim),
                    'social_features': (batch, social_feature_dim),
                    'pedestrian_predictions': (batch, max_peds, pred_horizon, 2, num_modes),
                    'pedestrian_features': (batch, pedestrian_feature_dim),
                    'attention_weights': ... (如果return_attention=True)
                }
            }
        """
        # 提取batch_size
        batch_size = observation['dog']['trajectory'].shape[0]
        
        # 初始化隐藏状态
        if hidden_states is None:
            hidden_states = self.init_hidden_states(batch_size)
        
        debug_info = {}
        
        # ==================== 1. 感知编码 ====================
        
        # 1.1 编码机器狗状态
        dog_features = self.dog_encoder(
            past_trajectory=observation['dog']['trajectory'],
            current_velocity=observation['dog']['velocity'],
            current_position=observation['dog']['position'],
            goal_position=observation['dog']['goal']
        )
        debug_info['dog_features'] = dog_features
        
        # 1.2 编码走廊几何
        corridor_features = self.encode_corridors(
            corridor_polygons=observation['corridors']['polygons'],
            corridor_vertex_counts=observation['corridors']['vertex_counts'],
            corridor_mask=observation['corridors']['mask']
        )
        debug_info['corridor_features'] = corridor_features
        
        # ==================== 2. 轨迹预测 ====================
        
        # 预测每个行人的未来轨迹
        # pedestrian_trajectories: (batch, max_peds, obs_horizon, 2)
        max_peds = observation['pedestrians']['trajectories'].shape[1]
        
        # pedestrian_predictions: (batch, max_peds, pred_horizon, 2, num_modes)
        pedestrian_predictions_list = []
        
        for i in range(max_peds):
            # 目标行人历史轨迹
            target_traj = observation['pedestrians']['trajectories'][:, i, :, :]  # (batch, obs_horizon, 2)
            
            # 邻居行人（排除自己）
            neighbor_indices = [j for j in range(max_peds) if j != i]
            if len(neighbor_indices) > 0:
                neighbor_trajs = observation['pedestrians']['trajectories'][:, neighbor_indices, :, :]  # (batch, N-1, obs_horizon, 2)
                neighbor_mask = observation['pedestrians']['mask'][:, neighbor_indices]  # (batch, N-1)
                
                # 计算相对角度（使用最后一个观测位置）
                target_pos = target_traj[:, -1, :]  # (batch, 2)
                neighbor_pos = neighbor_trajs[:, :, -1, :]  # (batch, N-1, 2)
                relative_vec = neighbor_pos - target_pos.unsqueeze(1)  # (batch, N-1, 2)
                neighbor_angles = torch.atan2(relative_vec[:, :, 1], relative_vec[:, :, 0])  # (batch, N-1)
            else:
                # 如果只有一个行人，创建空的邻居
                neighbor_trajs = torch.zeros(batch_size, 0, observation['pedestrians']['trajectories'].shape[2], 2, device=self.device)
                neighbor_angles = torch.zeros(batch_size, 0, device=self.device)
                neighbor_mask = torch.zeros(batch_size, 0, device=self.device)
            
            # 预测第i个行人的未来轨迹
            pred = self.trajectory_predictor(
                target_trajectory=target_traj,
                neighbor_trajectories=neighbor_trajs,
                neighbor_angles=neighbor_angles,
                neighbor_mask=neighbor_mask
            )
            pedestrian_predictions_list.append(pred)
        
        # Stack: (batch, max_peds, pred_horizon, 2, num_modes)
        pedestrian_predictions = torch.stack(pedestrian_predictions_list, dim=1)
        
        # 可选：将EVSC输出的模态数下采样到配置的期望数量（例如3），以加速后续编码
        actual_num_modes = pedestrian_predictions.shape[-1]
        desired_num_modes = getattr(self.pedestrian_encoder, 'num_modes', actual_num_modes)
        if actual_num_modes > desired_num_modes:
            # 均匀选取desired_num_modes个索引（保持覆盖不同风格）
            indices = torch.linspace(0, actual_num_modes - 1, desired_num_modes, device=pedestrian_predictions.device)
            indices = torch.round(indices).long().unique()
            # 如果由于unique导致数量减少，补齐到期望数量
            if indices.numel() < desired_num_modes:
                last_idx = indices[-1].item() if indices.numel() > 0 else 0
                pad = torch.arange(last_idx, last_idx + (desired_num_modes - indices.numel()), device=indices.device)
                pad = torch.clamp(pad, max=actual_num_modes - 1)
                indices = torch.cat([indices, pad.long()])[:desired_num_modes]
            pedestrian_predictions = torch.index_select(pedestrian_predictions, dim=-1, index=indices)
        
        debug_info['pedestrian_predictions'] = pedestrian_predictions
        
        # ==================== 3. 行人编码 ====================
        
        pedestrian_features = self.pedestrian_encoder(
            pedestrian_predictions=pedestrian_predictions,
            pedestrian_mask=observation['pedestrians']['mask']
        )
        debug_info['pedestrian_features'] = pedestrian_features
        
        # ==================== 4. 多模态融合 ====================
        
        if return_attention:
            fused_state, attention_weights = self.fusion(
                dog_features=dog_features,
                pedestrian_features=pedestrian_features,
                corridor_features=corridor_features,
                return_attention_weights=True
            )
            debug_info['attention_weights'] = attention_weights
        else:
            fused_state, _ = self.fusion(
                dog_features=dog_features,
                pedestrian_features=pedestrian_features,
                corridor_features=corridor_features,
                return_attention_weights=False
            )
        
        # ==================== 5. SAC决策 ====================
        
        # 5.1 生成动作
        action, log_prob, hidden_states['actor'] = self.sac_agent.actor(
            state=fused_state,
            hidden_state=hidden_states['actor']
        )
        
        if deterministic:
            # 使用均值作为确定性动作
            action, hidden_states['actor'] = self.sac_agent.select_action(
                state=fused_state,
                hidden_actor=hidden_states['actor'],
                deterministic=True
            )
        
        # 5.2 评估Q值
        q1, hidden_states['critic1'] = self.sac_agent.critic.critic1(
            state=fused_state,
            action=action,
            hidden_state=hidden_states['critic1']
        )
        q2, hidden_states['critic2'] = self.sac_agent.critic.critic2(
            state=fused_state,
            action=action,
            hidden_state=hidden_states['critic2']
        )
        
        return {
            'action': action,
            'log_prob': log_prob,
            'q1': q1,
            'q2': q2,
            'fused_state': fused_state,
            'hidden_states': hidden_states,
            'debug_info': debug_info
        }
    
    def encode_from_obs(self, observation: Dict) -> torch.Tensor:
        """
        从观测编码到fused_state（仅编码，不做决策）
        用于训练时重新编码，使编码器参与梯度传播
        
        Args:
            observation: 观测字典（格式同forward）
        
        Returns:
            fused_state: (batch, fusion_dim) 融合后的状态特征
        """
        # ==================== 1. 感知编码 ====================
        
        # 1.1 编码机器狗状态
        dog_features = self.dog_encoder(
            past_trajectory=observation['dog']['trajectory'],
            current_velocity=observation['dog']['velocity'],
            current_position=observation['dog']['position'],
            goal_position=observation['dog']['goal']
        )
        
        # 1.2 编码走廊几何
        corridor_features = self.encode_corridors(
            corridor_polygons=observation['corridors']['polygons'],
            corridor_vertex_counts=observation['corridors']['vertex_counts'],
            corridor_mask=observation['corridors']['mask']
        )
        
        # ==================== 2. 轨迹预测 ====================
        
        batch_size = observation['dog']['trajectory'].shape[0]
        max_peds = observation['pedestrians']['trajectories'].shape[1]
        pedestrian_predictions_list = []
        
        for i in range(max_peds):
            target_traj = observation['pedestrians']['trajectories'][:, i, :, :]
            neighbor_indices = [j for j in range(max_peds) if j != i]
            
            if len(neighbor_indices) > 0:
                neighbor_trajs = observation['pedestrians']['trajectories'][:, neighbor_indices, :, :]
                neighbor_mask = observation['pedestrians']['mask'][:, neighbor_indices]
                target_pos = target_traj[:, -1, :]
                neighbor_pos = neighbor_trajs[:, :, -1, :]
                relative_vec = neighbor_pos - target_pos.unsqueeze(1)
                neighbor_angles = torch.atan2(relative_vec[:, :, 1], relative_vec[:, :, 0])
            else:
                neighbor_trajs = torch.zeros(batch_size, 0, observation['pedestrians']['trajectories'].shape[2], 2, device=self.device)
                neighbor_angles = torch.zeros(batch_size, 0, device=self.device)
                neighbor_mask = torch.zeros(batch_size, 0, device=self.device)
            
            # 预测（预训练模型，无梯度）
            with torch.no_grad():
                pred = self.trajectory_predictor(
                    target_trajectory=target_traj,
                    neighbor_trajectories=neighbor_trajs,
                    neighbor_angles=neighbor_angles,
                    neighbor_mask=neighbor_mask
                )
            pedestrian_predictions_list.append(pred)
        
        pedestrian_predictions = torch.stack(pedestrian_predictions_list, dim=1)
        
        # 下采样模态数（如果需要）
        actual_num_modes = pedestrian_predictions.shape[-1]
        desired_num_modes = getattr(self.pedestrian_encoder, 'num_modes', actual_num_modes)
        if actual_num_modes > desired_num_modes:
            indices = torch.linspace(0, actual_num_modes - 1, desired_num_modes, device=pedestrian_predictions.device)
            indices = torch.round(indices).long().unique()
            if indices.numel() < desired_num_modes:
                last_idx = indices[-1].item() if indices.numel() > 0 else 0
                pad = torch.arange(last_idx, last_idx + (desired_num_modes - indices.numel()), device=indices.device)
                pad = torch.clamp(pad, max=actual_num_modes - 1)
                indices = torch.cat([indices, pad.long()])[:desired_num_modes]
            pedestrian_predictions = torch.index_select(pedestrian_predictions, dim=-1, index=indices)
        
        # ==================== 3. 行人编码 ====================
        
        pedestrian_features = self.pedestrian_encoder(
            pedestrian_predictions=pedestrian_predictions,
            pedestrian_mask=observation['pedestrians']['mask']
        )
        
        # ==================== 4. 多模态融合 ====================
        
        fused_state, _ = self.fusion(
            dog_features=dog_features,
            pedestrian_features=pedestrian_features,
            corridor_features=corridor_features,
            return_attention_weights=False
        )
        
        return fused_state
    
    def encode_batch(self, observations: List[Dict]) -> torch.Tensor:
        """
        批量编码observations（用于训练）
        
        Args:
            observations: List of observation dicts (length = batch_size)
        
        Returns:
            fused_states: (batch_size, fusion_dim)
        """
        # Stack所有observations为单个batch
        batch_obs = self._stack_observations(observations)
        
        # 调用encode_from_obs
        fused_states = self.encode_from_obs(batch_obs)
        
        return fused_states
    
    def _stack_observations(self, observations: List[Dict]) -> Dict:
        """
        将observation列表stack成batch
        
        Args:
            observations: List of obs dicts (每个可能有batch维度(1, ...)或无)
        
        Returns:
            batched_obs: 单个batch的observation dict
        """
        batch_size = len(observations)
        
        # 辅助函数：处理可能的batch维度
        def squeeze_if_batched(tensor):
            """如果tensor有batch维度(1, ...)，squeeze掉"""
            if tensor.dim() > 0 and tensor.size(0) == 1:
                return tensor.squeeze(0)
            return tensor
        
        # Stack dog observations
        dog_trajectories = torch.stack([squeeze_if_batched(obs['dog']['trajectory']) for obs in observations], dim=0)
        dog_velocities = torch.stack([squeeze_if_batched(obs['dog']['velocity']) for obs in observations], dim=0)
        dog_positions = torch.stack([squeeze_if_batched(obs['dog']['position']) for obs in observations], dim=0)
        dog_goals = torch.stack([squeeze_if_batched(obs['dog']['goal']) for obs in observations], dim=0)
        
        # Stack pedestrian observations
        ped_trajectories = torch.stack([squeeze_if_batched(obs['pedestrians']['trajectories']) for obs in observations], dim=0)
        ped_masks = torch.stack([squeeze_if_batched(obs['pedestrians']['mask']) for obs in observations], dim=0)
        
        # Stack corridor observations
        corridor_polygons = torch.stack([squeeze_if_batched(obs['corridors']['polygons']) for obs in observations], dim=0)
        corridor_vertex_counts = torch.stack([squeeze_if_batched(obs['corridors']['vertex_counts']) for obs in observations], dim=0)
        corridor_masks = torch.stack([squeeze_if_batched(obs['corridors']['mask']) for obs in observations], dim=0)
        
        return {
            'dog': {
                'trajectory': dog_trajectories,
                'velocity': dog_velocities,
                'position': dog_positions,
                'goal': dog_goals
            },
            'pedestrians': {
                'trajectories': ped_trajectories,
                'mask': ped_masks
            },
            'corridors': {
                'polygons': corridor_polygons,
                'vertex_counts': corridor_vertex_counts,
                'mask': corridor_masks
            }
        }
    
    def select_action(
        self,
        observation: Dict,
        hidden_states: Optional[Dict] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        选择动作（推理/数据收集）
        
        Returns:
            action: (batch, action_dim)
            log_prob: (batch,)
            hidden_states: 更新后的隐藏状态
        """
        with torch.no_grad():
            result = self.forward(
                observation=observation,
                hidden_states=hidden_states,
                deterministic=deterministic,
                return_attention=False
            )
        
        return result['action'], result['log_prob'], result['hidden_states']
    
    def update(self, segment_batch: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        训练更新（调用SAC Agent的update）
        
        Args:
            segment_batch: 序列段batch
        
        Returns:
            losses: {
                'critic_loss': float,
                'actor_loss': float,
                'alpha_loss': float (如果auto_entropy=True),
                'alpha': float
            }
        """
        return self.sac_agent.update(segment_batch)
    
    def soft_update_target(self):
        """软更新目标网络"""
        self.sac_agent.soft_update_target()
    
    def save_checkpoint(self, filepath: str):
        """
        保存检查点
        
        Args:
            filepath: 保存路径
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'sac_actor_optimizer': self.sac_agent.actor_optimizer.state_dict(),
            'sac_critic_optimizer': self.sac_agent.critic_optimizer.state_dict(),
        }
        
        # 保存编码器优化器
        if hasattr(self, 'encoder_optimizer'):
            checkpoint['encoder_optimizer'] = self.encoder_optimizer.state_dict()
        
        if self.sac_agent.auto_entropy:
            checkpoint['sac_alpha_optimizer'] = self.sac_agent.alpha_optimizer.state_dict()
            checkpoint['log_alpha'] = self.sac_agent.log_alpha.item()
        
        torch.save(checkpoint, filepath)
        print(f"[SAVE] 检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizers: bool = True):
        """
        加载检查点
        
        Args:
            filepath: 检查点路径
            load_optimizers: 是否加载优化器状态
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizers:
            self.sac_agent.actor_optimizer.load_state_dict(checkpoint['sac_actor_optimizer'])
            self.sac_agent.critic_optimizer.load_state_dict(checkpoint['sac_critic_optimizer'])
            
            # 加载编码器优化器
            if hasattr(self, 'encoder_optimizer') and 'encoder_optimizer' in checkpoint:
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            
            if self.sac_agent.auto_entropy and 'sac_alpha_optimizer' in checkpoint:
                self.sac_agent.alpha_optimizer.load_state_dict(checkpoint['sac_alpha_optimizer'])
                self.sac_agent.log_alpha.data.fill_(checkpoint['log_alpha'])
        
        print(f"[LOAD] 检查点已加载: {filepath}")


# ==================== 内置测试 ====================
if __name__ == '__main__':
    print("测试AGSACModel...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建模型
    model = AGSACModel(
        dog_feature_dim=64,
        corridor_feature_dim=128,
        social_feature_dim=128,
        pedestrian_feature_dim=128,
        fusion_dim=64,
        action_dim=22,
        max_pedestrians=5,
        max_corridors=3,
        max_vertices=10,
        obs_horizon=8,
        pred_horizon=12,
        num_modes=20,
        hidden_dim=128,
        device=device
    )
    
    # 创建模拟观测
    batch_size = 2
    observation = {
        'dog': {
            'trajectory': torch.randn(batch_size, 8, 2, device=device),
            'velocity': torch.randn(batch_size, 2, device=device),
            'position': torch.randn(batch_size, 2, device=device),
            'goal': torch.randn(batch_size, 2, device=device)
        },
        'pedestrians': {
            'trajectories': torch.randn(batch_size, 5, 8, 2, device=device),
            'mask': torch.ones(batch_size, 5, device=device)
        },
        'corridors': {
            'polygons': torch.randn(batch_size, 3, 10, 2, device=device),
            'vertex_counts': torch.tensor([[10, 8, 6], [10, 10, 4]], device=device),
            'mask': torch.ones(batch_size, 3, device=device)
        },
        'reference_line': torch.randn(batch_size, 2, 2, device=device)
    }
    
    print("\n1. 测试前向传播...")
    result = model.forward(observation, deterministic=False, return_attention=True)
    
    print(f"[OK] action shape: {result['action'].shape}")
    print(f"[OK] log_prob shape: {result['log_prob'].shape}")
    print(f"[OK] q1 shape: {result['q1'].shape}")
    print(f"[OK] q2 shape: {result['q2'].shape}")
    print(f"[OK] fused_state shape: {result['fused_state'].shape}")
    
    print("\n2. 测试select_action...")
    action, log_prob, hidden_states = model.select_action(observation, deterministic=True)
    print(f"[OK] action shape: {action.shape}")
    print(f"[OK] log_prob shape: {log_prob.shape}")
    
    print("\n3. 测试隐藏状态传递...")
    action2, log_prob2, hidden_states2 = model.select_action(
        observation, 
        hidden_states=hidden_states, 
        deterministic=True
    )
    print(f"[OK] 隐藏状态传递成功")
    
    print("\n4. 测试保存/加载检查点...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name
    
    model.save_checkpoint(temp_path)
    model.load_checkpoint(temp_path)
    os.remove(temp_path)
    print(f"[OK] 检查点保存/加载成功")
    
    print("\n" + "="*60)
    print("[SUCCESS] AGSACModel所有测试通过！")
    print("="*60)

