"""
几何计算工具
包括角度、距离、坐标转换等
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


def compute_relative_angles(
    target_position: torch.Tensor,
    neighbor_positions: torch.Tensor
) -> torch.Tensor:
    """
    计算目标点到多个邻居点的相对角度
    用于SocialCircle的角度分区
    
    Args:
        target_position: Tensor(2,) 目标点位置 [x, y]
        neighbor_positions: Tensor(N, 2) N个邻居的位置
        
    Returns:
        angles: Tensor(N,) 相对角度，范围[0, 2π]
    """
    # 计算相对向量
    relative_vectors = neighbor_positions - target_position  # (N, 2)
    
    # 使用atan2计算角度（范围[-π, π]）
    angles = torch.atan2(relative_vectors[:, 1], relative_vectors[:, 0])
    
    # 转换到[0, 2π]范围
    angles = torch.where(angles < 0, angles + 2 * torch.pi, angles)
    
    return angles


def compute_distance(
    point1: torch.Tensor,
    point2: torch.Tensor,
    keepdim: bool = False
) -> torch.Tensor:
    """
    计算两点之间的欧氏距离
    
    Args:
        point1: Tensor(..., 2) 点1
        point2: Tensor(..., 2) 点2
        keepdim: 是否保持维度
        
    Returns:
        distance: Tensor(...) 或 Tensor(..., 1) 距离
    """
    diff = point2 - point1
    distance = torch.norm(diff, dim=-1, keepdim=keepdim)
    return distance


def compute_heading_angle(velocity: torch.Tensor) -> torch.Tensor:
    """
    从速度向量计算航向角
    
    Args:
        velocity: Tensor(2,) 或 Tensor(batch, 2) 速度向量 [vx, vy]
        
    Returns:
        heading: Tensor() 或 Tensor(batch,) 航向角（弧度）
    """
    if velocity.dim() == 1:
        heading = torch.atan2(velocity[1], velocity[0])
    else:
        heading = torch.atan2(velocity[:, 1], velocity[:, 0])
    return heading


def rotate_2d(
    points: torch.Tensor,
    angle: torch.Tensor
) -> torch.Tensor:
    """
    2D旋转变换
    
    Args:
        points: Tensor(..., 2) 待旋转的点
        angle: Tensor() 旋转角度（弧度），逆时针为正
        
    Returns:
        rotated_points: Tensor(..., 2) 旋转后的点
    """
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # 构造旋转矩阵
    rotation_matrix = torch.tensor([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ], device=points.device, dtype=points.dtype)
    
    # 应用旋转
    # points: (..., 2) -> (..., 2, 1)
    # rotation_matrix: (2, 2)
    # result: (..., 2, 1) -> (..., 2)
    
    original_shape = points.shape
    points_flat = points.reshape(-1, 2)  # (N, 2)
    rotated_flat = points_flat @ rotation_matrix.T  # (N, 2)
    rotated_points = rotated_flat.reshape(original_shape)
    
    return rotated_points


def transform_to_local_frame(
    global_points: torch.Tensor,
    robot_position: torch.Tensor,
    robot_heading: torch.Tensor
) -> torch.Tensor:
    """
    将全局坐标转换到机器人局部坐标系
    
    Args:
        global_points: Tensor(..., 2) 全局坐标点
        robot_position: Tensor(2,) 机器人位置
        robot_heading: Tensor() 机器人航向角
        
    Returns:
        local_points: Tensor(..., 2) 局部坐标点
    """
    # 平移到原点
    translated = global_points - robot_position
    
    # 旋转到局部坐标系（反向旋转）
    local_points = rotate_2d(translated, -robot_heading)
    
    return local_points


def transform_to_global_frame(
    local_points: torch.Tensor,
    robot_position: torch.Tensor,
    robot_heading: torch.Tensor
) -> torch.Tensor:
    """
    将局部坐标转换到全局坐标系
    
    Args:
        local_points: Tensor(..., 2) 局部坐标点
        robot_position: Tensor(2,) 机器人位置
        robot_heading: Tensor() 机器人航向角
        
    Returns:
        global_points: Tensor(..., 2) 全局坐标点
    """
    # 旋转到全局坐标系
    rotated = rotate_2d(local_points, robot_heading)
    
    # 平移到机器人位置
    global_points = rotated + robot_position
    
    return global_points


def incremental_path_to_global(
    incremental_path: torch.Tensor,
    robot_position: torch.Tensor,
    robot_heading: torch.Tensor,
    max_range: float = 5.0
) -> torch.Tensor:
    """
    将增量式路径（归一化）转换为全局坐标路径
    
    Args:
        incremental_path: Tensor(num_points, 2) 归一化增量路径 in [-1, 1]
        robot_position: Tensor(2,) 机器人当前位置
        robot_heading: Tensor() 机器人当前航向
        max_range: float 路径最大范围（米）
        
    Returns:
        global_path: Tensor(num_points, 2) 全局坐标路径
    """
    # 反归一化
    increments = incremental_path * (max_range / incremental_path.size(0))
    
    # 累积求和得到相对起点的坐标（局部坐标系）
    local_path = torch.cumsum(increments, dim=0)
    
    # 转换到全局坐标系
    global_path = transform_to_global_frame(
        local_path,
        robot_position,
        robot_heading
    )
    
    return global_path


def compute_path_curvature(
    path: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    计算路径的曲率（近似）
    
    Args:
        path: Tensor(num_points, 2) 路径点
        epsilon: float 数值稳定项
        
    Returns:
        curvature: Tensor(num_points-2,) 每段的曲率
    """
    if path.size(0) < 3:
        return torch.tensor([], device=path.device)
    
    # 计算连续三点的向量
    v1 = path[1:-1] - path[:-2]  # (N-2, 2)
    v2 = path[2:] - path[1:-1]   # (N-2, 2)
    
    # 计算叉积（2D中返回标量）
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # (N-2,)
    
    # 计算点积
    dot = (v1 * v2).sum(dim=-1)  # (N-2,)
    
    # 计算转角
    angles = torch.atan2(cross, dot)  # (N-2,)
    
    # 曲率近似为转角除以段长
    segment_lengths = torch.norm(v1, dim=-1) + epsilon
    curvature = torch.abs(angles) / segment_lengths
    
    return curvature


def compute_path_length(path: torch.Tensor) -> torch.Tensor:
    """
    计算路径总长度
    
    Args:
        path: Tensor(num_points, 2) 路径点
        
    Returns:
        length: Tensor() 总长度
    """
    if path.size(0) < 2:
        return torch.tensor(0.0, device=path.device)
    
    # 计算每段长度
    segments = path[1:] - path[:-1]  # (N-1, 2)
    segment_lengths = torch.norm(segments, dim=-1)  # (N-1,)
    
    # 求和
    total_length = segment_lengths.sum()
    
    return total_length


def compute_closest_point_on_path(
    path: torch.Tensor,
    query_point: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """
    找到路径上距离查询点最近的点
    
    Args:
        path: Tensor(num_points, 2) 路径点
        query_point: Tensor(2,) 查询点
        
    Returns:
        closest_idx: int 最近点的索引
        closest_point: Tensor(2,) 最近点坐标
    """
    # 计算所有点到查询点的距离
    distances = torch.norm(path - query_point, dim=-1)  # (num_points,)
    
    # 找到最小距离的索引
    closest_idx = torch.argmin(distances).item()
    closest_point = path[closest_idx]
    
    return closest_idx, closest_point


def compute_lookahead_point(
    path: torch.Tensor,
    current_position: torch.Tensor,
    lookahead_distance: float
) -> Tuple[int, torch.Tensor]:
    """
    计算前视点（Pure Pursuit算法用）
    
    Args:
        path: Tensor(num_points, 2) 路径点
        current_position: Tensor(2,) 当前位置
        lookahead_distance: float 前视距离
        
    Returns:
        lookahead_idx: int 前视点索引
        lookahead_point: Tensor(2,) 前视点坐标
    """
    # 计算所有点到当前位置的距离
    distances = torch.norm(path - current_position, dim=-1)  # (num_points,)
    
    # 找到距离最接近lookahead_distance的点
    distance_diff = torch.abs(distances - lookahead_distance)
    lookahead_idx = torch.argmin(distance_diff).item()
    
    # 确保不会选择当前位置后面的点
    # 如果所有点都太近，选择最远的
    if distances[lookahead_idx] < lookahead_distance * 0.5:
        lookahead_idx = torch.argmax(distances).item()
    
    lookahead_point = path[lookahead_idx]
    
    return lookahead_idx, lookahead_point


def check_collision(
    point: torch.Tensor,
    obstacles: torch.Tensor,
    collision_radius: float
) -> torch.Tensor:
    """
    检查点是否与障碍物碰撞
    
    Args:
        point: Tensor(2,) 或 Tensor(batch, 2) 待检查的点
        obstacles: Tensor(num_obstacles, 2) 障碍物位置
        collision_radius: float 碰撞半径
        
    Returns:
        collision: Tensor() 或 Tensor(batch,) bool，True表示碰撞
    """
    if point.dim() == 1:
        point = point.unsqueeze(0)  # (1, 2)
        single_point = True
    else:
        single_point = False
    
    # 计算到所有障碍物的距离
    # point: (batch, 2), obstacles: (num_obstacles, 2)
    # distances: (batch, num_obstacles)
    distances = torch.cdist(point, obstacles)  # (batch, num_obstacles)
    
    # 检查是否有任何障碍物距离小于碰撞半径
    collision = (distances < collision_radius).any(dim=-1)  # (batch,)
    
    if single_point:
        collision = collision.squeeze(0)  # scalar
    
    return collision


def compute_polygon_centroid(vertices: torch.Tensor) -> torch.Tensor:
    """
    计算多边形质心
    
    Args:
        vertices: Tensor(num_vertices, 2) 多边形顶点
        
    Returns:
        centroid: Tensor(2,) 质心坐标
    """
    centroid = vertices.mean(dim=0)
    return centroid


def point_to_line_distance(
    point: torch.Tensor,
    line_start: torch.Tensor,
    line_end: torch.Tensor
) -> torch.Tensor:
    """
    计算点到线段的距离
    
    Args:
        point: Tensor(..., 2) 点坐标
        line_start: Tensor(2,) 线段起点
        line_end: Tensor(2,) 线段终点
        
    Returns:
        distance: Tensor(...) 距离
    """
    # 线段向量
    line_vec = line_end - line_start  # (2,)
    line_length_sq = (line_vec ** 2).sum()
    
    if line_length_sq < 1e-8:
        # 退化为点
        return torch.norm(point - line_start, dim=-1)
    
    # 点到起点的向量
    point_vec = point - line_start  # (..., 2)
    
    # 投影参数
    t = (point_vec * line_vec).sum(dim=-1) / line_length_sq  # (...)
    t = torch.clamp(t, 0.0, 1.0)  # 限制在线段上
    
    # 投影点
    projection = line_start + t.unsqueeze(-1) * line_vec  # (..., 2)
    
    # 距离
    distance = torch.norm(point - projection, dim=-1)
    
    return distance


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """
    将角度归一化到[-π, π]范围
    
    Args:
        angle: Tensor(...) 角度（弧度）
        
    Returns:
        normalized_angle: Tensor(...) 归一化后的角度
    """
    # 归一化到[-π, π]
    normalized = torch.atan2(torch.sin(angle), torch.cos(angle))
    return normalized


def angle_difference(angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
    """
    计算两个角度之间的最小差值
    
    Args:
        angle1: Tensor(...) 角度1
        angle2: Tensor(...) 角度2
        
    Returns:
        diff: Tensor(...) 角度差，范围[-π, π]
    """
    diff = angle2 - angle1
    diff = normalize_angle(diff)
    return diff


def interpolate_path(
    path: torch.Tensor,
    num_points: int
) -> torch.Tensor:
    """
    对路径进行插值，生成更密集的点
    
    Args:
        path: Tensor(N, 2) 原始路径
        num_points: int 目标点数
        
    Returns:
        interpolated_path: Tensor(num_points, 2) 插值后的路径
    """
    if path.size(0) >= num_points:
        # 如果已经够密集，进行下采样
        indices = torch.linspace(0, path.size(0) - 1, num_points, device=path.device)
        indices = indices.long()
        return path[indices]
    
    # 计算累积路径长度
    segments = path[1:] - path[:-1]  # (N-1, 2)
    segment_lengths = torch.norm(segments, dim=-1)  # (N-1,)
    cumulative_lengths = torch.cat([
        torch.zeros(1, device=path.device),
        torch.cumsum(segment_lengths, dim=0)
    ])  # (N,)
    total_length = cumulative_lengths[-1]
    
    # 生成均匀分布的目标长度
    target_lengths = torch.linspace(0, total_length, num_points, device=path.device)
    
    # 对每个目标长度进行插值
    interpolated_points = []
    for target_length in target_lengths:
        # 找到对应的线段
        idx = torch.searchsorted(cumulative_lengths, target_length)
        if idx == 0:
            interpolated_points.append(path[0])
        elif idx >= path.size(0):
            interpolated_points.append(path[-1])
        else:
            # 线性插值
            segment_start_length = cumulative_lengths[idx - 1]
            segment_end_length = cumulative_lengths[idx]
            segment_length = segment_end_length - segment_start_length
            
            if segment_length < 1e-8:
                interpolated_points.append(path[idx])
            else:
                alpha = (target_length - segment_start_length) / segment_length
                point = path[idx - 1] + alpha * (path[idx] - path[idx - 1])
                interpolated_points.append(point)
    
    interpolated_path = torch.stack(interpolated_points)
    return interpolated_path


# 批量处理的便捷函数
def batch_compute_distances(
    points1: torch.Tensor,
    points2: torch.Tensor
) -> torch.Tensor:
    """
    批量计算点对之间的距离
    
    Args:
        points1: Tensor(N, 2) N个点
        points2: Tensor(M, 2) M个点
        
    Returns:
        distances: Tensor(N, M) 距离矩阵
    """
    return torch.cdist(points1, points2)


if __name__ == '__main__':
    """简单测试"""
    print("测试几何工具模块...")
    
    # 测试角度计算
    target_pos = torch.tensor([0.0, 0.0])
    neighbor_pos = torch.tensor([
        [1.0, 0.0],   # 0度
        [0.0, 1.0],   # 90度
        [-1.0, 0.0],  # 180度
        [0.0, -1.0]   # 270度
    ])
    angles = compute_relative_angles(target_pos, neighbor_pos)
    print(f"相对角度（度）: {angles * 180 / torch.pi}")
    
    # 测试坐标转换
    global_point = torch.tensor([5.0, 3.0])
    robot_pos = torch.tensor([2.0, 1.0])
    robot_heading = torch.tensor(torch.pi / 4)  # 45度
    
    local_point = transform_to_local_frame(global_point, robot_pos, robot_heading)
    print(f"\n全局坐标 {global_point} -> 局部坐标 {local_point}")
    
    back_to_global = transform_to_global_frame(local_point, robot_pos, robot_heading)
    print(f"局部坐标 {local_point} -> 全局坐标 {back_to_global}")
    
    # 测试增量式路径转换
    incremental_path = torch.randn(11, 2) * 0.5  # 归一化路径
    global_path = incremental_path_to_global(
        incremental_path,
        robot_pos,
        robot_heading,
        max_range=5.0
    )
    print(f"\n增量式路径形状: {incremental_path.shape}")
    print(f"全局路径形状: {global_path.shape}")
    print(f"路径长度: {compute_path_length(global_path):.2f}m")
    
    # 测试曲率计算
    if global_path.size(0) >= 3:
        curvature = compute_path_curvature(global_path)
        print(f"平均曲率: {curvature.mean():.4f}")
    
    # 测试碰撞检测
    test_point = torch.tensor([3.0, 2.0])
    obstacles = torch.tensor([
        [3.1, 2.1],
        [5.0, 5.0]
    ])
    collision = check_collision(test_point, obstacles, collision_radius=0.5)
    print(f"\n碰撞检测: {collision}")
    
    # 测试前视点
    lookahead_idx, lookahead_point = compute_lookahead_point(
        global_path,
        robot_pos,
        lookahead_distance=2.0
    )
    print(f"前视点索引: {lookahead_idx}, 位置: {lookahead_point}")
    
    print("\n✓ 几何工具模块测试通过！")