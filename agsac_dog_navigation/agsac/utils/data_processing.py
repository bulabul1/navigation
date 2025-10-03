"""
数据处理模块
负责处理可变长度输入，包括padding、mask生成、截断等
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional


class DataProcessor:
    """
    AGSAC数据预处理器
    处理可变长度的行人轨迹和通路多边形输入
    """
    
    def __init__(
        self,
        max_pedestrians: int = 10,
        max_corridors: int = 10,
        device: str = 'cpu'
    ):
        """
        Args:
            max_pedestrians: 最大行人数量
            max_corridors: 最大通路数量
            device: 计算设备
        """
        self.max_pedestrians = max_pedestrians
        self.max_corridors = max_corridors
        self.device = device
    
    def process_pedestrian_trajectories(
        self,
        trajectories: List[torch.Tensor],
        robot_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理行人历史轨迹
        
        Args:
            trajectories: List[Tensor(seq_len, 2)] 可变数量的行人轨迹
            robot_position: Tensor(2,) 机器狗当前位置，用于选择最近的行人
            
        Returns:
            padded_trajs: Tensor(max_pedestrians, seq_len, 2) padding后的轨迹
            mask: Tensor(max_pedestrians,) 有效性mask (1=有效, 0=padding)
        """
        if len(trajectories) == 0:
            # 没有行人的情况
            seq_len = 8  # 默认序列长度
            padded_trajs = torch.zeros(
                self.max_pedestrians, seq_len, 2,
                device=self.device
            )
            mask = torch.zeros(self.max_pedestrians, device=self.device)
            return padded_trajs, mask
        
        # 截断超限输入
        if len(trajectories) > self.max_pedestrians:
            trajectories = self._select_closest_trajectories(
                trajectories, 
                robot_position, 
                self.max_pedestrians
            )
        
        num_peds = len(trajectories)
        seq_len = trajectories[0].size(0)
        
        # 创建padding后的张量
        padded_trajs = torch.zeros(
            self.max_pedestrians, seq_len, 2,
            device=self.device
        )
        
        # 填充真实轨迹
        for i, traj in enumerate(trajectories):
            padded_trajs[i] = traj.to(self.device)
        
        # 生成mask
        mask = torch.zeros(self.max_pedestrians, device=self.device)
        mask[:num_peds] = 1.0
        
        return padded_trajs, mask
    
    def process_corridor_features(
        self,
        corridor_features: List[torch.Tensor],
        robot_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理通路特征（已通过PointNet编码后）
        
        Args:
            corridor_features: List[Tensor(feature_dim,)] 每个通路的特征向量
            robot_position: Tensor(2,) 机器狗当前位置
            
        Returns:
            padded_features: Tensor(max_corridors, feature_dim) padding后的特征
            mask: Tensor(max_corridors,) 有效性mask
        """
        if len(corridor_features) == 0:
            # 没有通路的情况（理论上不应该发生）
            feature_dim = 64  # 默认特征维度
            padded_features = torch.zeros(
                self.max_corridors, feature_dim,
                device=self.device
            )
            mask = torch.zeros(self.max_corridors, device=self.device)
            return padded_features, mask
        
        # 截断超限输入（注意：这里需要原始多边形信息来选择最近的）
        # 实际使用时，应该在编码前截断，这里假设已经截断
        if len(corridor_features) > self.max_corridors:
            corridor_features = corridor_features[:self.max_corridors]
        
        num_corridors = len(corridor_features)
        feature_dim = corridor_features[0].size(0)
        
        # 创建padding后的张量
        padded_features = torch.zeros(
            self.max_corridors, feature_dim,
            device=self.device
        )
        
        # 填充真实特征
        for i, feat in enumerate(corridor_features):
            padded_features[i] = feat.to(self.device)
        
        # 生成mask
        mask = torch.zeros(self.max_corridors, device=self.device)
        mask[:num_corridors] = 1.0
        
        return padded_features, mask
    
    def process_corridors(
        self,
        corridors: List[torch.Tensor],
        robot_position: torch.Tensor
    ) -> Tuple[List[torch.Tensor], int]:
        """
        处理原始通路多边形（在PointNet编码前）
        主要用于截断和选择
        
        Args:
            corridors: List[Tensor(N_i, 2)] 可变数量和顶点数的多边形
            robot_position: Tensor(2,) 机器狗当前位置
            
        Returns:
            selected_corridors: List[Tensor(N_i, 2)] 截断后的通路
            num_valid: int 有效通路数量
        """
        if len(corridors) == 0:
            return [], 0
        
        # 截断超限输入
        if len(corridors) > self.max_corridors:
            corridors = self._select_closest_corridors(
                corridors,
                robot_position,
                self.max_corridors
            )
        
        return corridors, len(corridors)
    
    def _select_closest_trajectories(
        self,
        trajectories: List[torch.Tensor],
        robot_position: torch.Tensor,
        max_num: int
    ) -> List[torch.Tensor]:
        """
        选择距离机器狗最近的N个行人轨迹
        
        Args:
            trajectories: List[Tensor(seq_len, 2)] 所有行人轨迹
            robot_position: Tensor(2,) 机器狗位置
            max_num: 最大保留数量
            
        Returns:
            List[Tensor] 选择后的轨迹
        """
        # 计算每个行人当前位置（最后一帧）到机器狗的距离
        distances = []
        for traj in trajectories:
            last_pos = traj[-1]  # (2,)
            dist = torch.norm(last_pos - robot_position)
            distances.append(dist.item())
        
        # 获取距离最近的max_num个索引
        sorted_indices = np.argsort(distances)[:max_num]
        
        # 选择对应的轨迹
        selected_trajectories = [trajectories[i] for i in sorted_indices]
        
        return selected_trajectories
    
    def _select_closest_corridors(
        self,
        corridors: List[torch.Tensor],
        robot_position: torch.Tensor,
        max_num: int
    ) -> List[torch.Tensor]:
        """
        选择距离机器狗最近的N个通路
        
        Args:
            corridors: List[Tensor(N_i, 2)] 所有通路多边形
            robot_position: Tensor(2,) 机器狗位置
            max_num: 最大保留数量
            
        Returns:
            List[Tensor] 选择后的通路
        """
        # 计算每个通路质心到机器狗的距离
        distances = []
        for corridor in corridors:
            centroid = corridor.mean(dim=0)  # (2,)
            dist = torch.norm(centroid - robot_position)
            distances.append(dist.item())
        
        # 获取距离最近的max_num个索引
        sorted_indices = np.argsort(distances)[:max_num]
        
        # 选择对应的通路
        selected_corridors = [corridors[i] for i in sorted_indices]
        
        return selected_corridors
    
    def create_batch(
        self,
        samples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        将多个样本组合成batch
        
        Args:
            samples: List of dicts, 每个dict包含:
                - 'pedestrian_trajs': Tensor(max_peds, seq_len, 2)
                - 'pedestrian_mask': Tensor(max_peds,)
                - 'corridor_features': Tensor(max_corridors, feat_dim)
                - 'corridor_mask': Tensor(max_corridors,)
                - 'dog_past_traj': Tensor(seq_len, 2)
                - 'dog_vel': Tensor(2,)
                - 'dog_pos': Tensor(2,)
                - 'goal_pos': Tensor(2,)
                
        Returns:
            Dict[str, Tensor] batch后的数据，增加batch维度
        """
        batch_size = len(samples)
        
        # 初始化batch字典
        batch = {}
        
        # 获取第一个样本的key来初始化
        first_sample = samples[0]
        
        for key in first_sample.keys():
            # Stack所有样本的该字段
            batch[key] = torch.stack([sample[key] for sample in samples])
        
        return batch
    
    def normalize_coordinates(
        self,
        coords: torch.Tensor,
        reference_point: torch.Tensor,
        scale: float = 10.0
    ) -> torch.Tensor:
        """
        归一化坐标到[-1, 1]范围
        
        Args:
            coords: Tensor(..., 2) 原始坐标
            reference_point: Tensor(2,) 参考点（通常是机器狗位置）
            scale: float 归一化尺度（米）
            
        Returns:
            Tensor(..., 2) 归一化后的坐标
        """
        # 转换为相对坐标
        relative_coords = coords - reference_point
        
        # 归一化到[-1, 1]
        normalized_coords = relative_coords / scale
        
        # 裁剪到[-1, 1]范围
        normalized_coords = torch.clamp(normalized_coords, -1.0, 1.0)
        
        return normalized_coords
    
    def denormalize_coordinates(
        self,
        normalized_coords: torch.Tensor,
        reference_point: torch.Tensor,
        scale: float = 10.0
    ) -> torch.Tensor:
        """
        反归一化坐标
        
        Args:
            normalized_coords: Tensor(..., 2) 归一化后的坐标
            reference_point: Tensor(2,) 参考点
            scale: float 归一化尺度
            
        Returns:
            Tensor(..., 2) 原始坐标
        """
        # 反归一化
        relative_coords = normalized_coords * scale
        
        # 转换回全局坐标
        global_coords = relative_coords + reference_point
        
        return global_coords


def pad_sequence_list(
    sequences: List[torch.Tensor],
    max_length: int,
    padding_value: float = 0.0
) -> torch.Tensor:
    """
    将可变长度的序列列表padding到固定长度
    
    Args:
        sequences: List[Tensor(seq_len, ...)] 可变长度的序列列表
        max_length: int 目标长度
        padding_value: float padding值
        
    Returns:
        Tensor(max_length, num_sequences, ...) padding后的序列
    """
    if len(sequences) == 0:
        raise ValueError("Empty sequence list")
    
    num_sequences = len(sequences)
    # 获取序列的特征维度
    feature_shape = sequences[0].shape[1:]  # 除第一维(seq_len)外的形状
    
    # 创建padding后的张量 (max_length, num_sequences, *feature_shape)
    padded = torch.full(
        (max_length, num_sequences) + feature_shape,
        padding_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device
    )
    
    # 填充实际序列
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length)
        padded[:seq_len, i] = seq[:seq_len]
    
    return padded


def create_attention_mask(
    valid_lengths: torch.Tensor,
    max_length: int,
    inverted: bool = False
) -> torch.Tensor:
    """
    创建注意力mask
    
    Args:
        valid_lengths: Tensor(batch,) 每个样本的有效长度
        max_length: int 序列最大长度
        inverted: bool 是否反转（True表示padding位置）
        
    Returns:
        Tensor(batch, max_length) mask矩阵
    """
    batch_size = valid_lengths.size(0)
    
    # 创建位置索引 (1, max_length)
    positions = torch.arange(max_length, device=valid_lengths.device).unsqueeze(0)
    
    # 扩展valid_lengths (batch, 1)
    valid_lengths = valid_lengths.unsqueeze(1)
    
    # 创建mask (batch, max_length)
    if inverted:
        # True表示padding位置（用于PyTorch的key_padding_mask）
        mask = positions >= valid_lengths
    else:
        # True表示有效位置
        mask = positions < valid_lengths
    
    return mask


# 便捷函数
def process_single_input(
    pedestrian_trajs: List[torch.Tensor],
    corridors: List[torch.Tensor],
    dog_state: Dict[str, torch.Tensor],
    max_pedestrians: int = 10,
    max_corridors: int = 10,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    处理单个时间步的完整输入
    
    Args:
        pedestrian_trajs: List[Tensor(8,2)] 行人轨迹
        corridors: List[Tensor(N_i,2)] 通路多边形
        dog_state: Dict包含 'past_traj', 'vel', 'pos', 'goal'
        max_pedestrians: 最大行人数
        max_corridors: 最大通路数
        device: 计算设备
        
    Returns:
        Dict[str, Tensor] 处理后的输入
    """
    processor = DataProcessor(max_pedestrians, max_corridors, device)
    
    robot_pos = dog_state['pos']
    
    # 处理行人轨迹
    ped_trajs_padded, ped_mask = processor.process_pedestrian_trajectories(
        pedestrian_trajs,
        robot_pos
    )
    
    # 处理通路（返回截断后的列表）
    corridors_selected, num_valid_corridors = processor.process_corridors(
        corridors,
        robot_pos
    )
    
    # 组装输出
    processed = {
        'pedestrian_trajs': ped_trajs_padded,      # (max_peds, 8, 2)
        'pedestrian_mask': ped_mask,                # (max_peds,)
        'corridors': corridors_selected,            # List[Tensor(N_i,2)]
        'num_corridors': num_valid_corridors,       # int
        'dog_past_traj': dog_state['past_traj'].to(device),  # (8, 2)
        'dog_vel': dog_state['vel'].to(device),               # (2,)
        'dog_pos': dog_state['pos'].to(device),               # (2,)
        'goal_pos': dog_state['goal'].to(device),             # (2,)
    }
    
    return processed


if __name__ == '__main__':
    """简单测试"""
    print("测试数据处理模块...")
    
    # 创建测试数据
    ped_trajs = [
        torch.randn(8, 2) for _ in range(3)  # 3个行人
    ]
    corridors = [
        torch.randn(5, 2),  # 5边形
        torch.randn(4, 2),  # 4边形
        torch.randn(6, 2),  # 6边形
    ]
    dog_state = {
        'past_traj': torch.randn(8, 2),
        'vel': torch.randn(2),
        'pos': torch.zeros(2),
        'goal': torch.tensor([5.0, 5.0])
    }
    
    # 处理
    processed = process_single_input(
        ped_trajs,
        corridors,
        dog_state,
        max_pedestrians=10,
        max_corridors=10
    )
    
    print(f"行人轨迹: {processed['pedestrian_trajs'].shape}")
    print(f"行人mask: {processed['pedestrian_mask'].shape}")
    print(f"mask值: {processed['pedestrian_mask']}")
    print(f"通路数量: {processed['num_corridors']}")
    print(f"机器狗轨迹: {processed['dog_past_traj'].shape}")
    
    print("\n✓ 数据处理模块测试通过！")