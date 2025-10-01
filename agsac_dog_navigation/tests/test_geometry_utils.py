"""
几何工具模块的单元测试
"""

import pytest
import torch
import numpy as np
from agsac.utils.geometry_utils import (
    compute_relative_angles,
    compute_distance,
    compute_heading_angle,
    rotate_2d,
    transform_to_local_frame,
    transform_to_global_frame,
    incremental_path_to_global,
    compute_path_curvature,
    compute_path_length,
    compute_closest_point_on_path,
    compute_lookahead_point,
    check_collision,
    compute_polygon_centroid,
    point_to_line_distance,
    normalize_angle,
    angle_difference,
    interpolate_path,
    batch_compute_distances
)


class TestAngleComputation:
    """测试角度计算相关函数"""
    
    def test_compute_relative_angles_basic(self):
        """测试基本的相对角度计算"""
        target_pos = torch.tensor([0.0, 0.0])
        neighbor_pos = torch.tensor([
            [1.0, 0.0],   # 0度
            [0.0, 1.0],   # 90度
            [-1.0, 0.0],  # 180度
            [0.0, -1.0]   # 270度
        ])
        
        angles = compute_relative_angles(target_pos, neighbor_pos)
        
        # 转换为度数便于检查
        angles_deg = angles * 180 / torch.pi
        
        torch.testing.assert_close(angles_deg[0], torch.tensor(0.0), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(angles_deg[1], torch.tensor(90.0), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(angles_deg[2], torch.tensor(180.0), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(angles_deg[3], torch.tensor(270.0), atol=1e-4, rtol=1e-4)
    
    def test_compute_relative_angles_range(self):
        """测试角度范围在[0, 2π]"""
        target_pos = torch.zeros(2)
        neighbor_pos = torch.randn(10, 2)
        
        angles = compute_relative_angles(target_pos, neighbor_pos)
        
        assert (angles >= 0).all()
        assert (angles <= 2 * torch.pi).all()
    
    def test_compute_heading_angle_single(self):
        """测试单个速度向量的航向角"""
        velocity = torch.tensor([1.0, 0.0])  # 向右
        heading = compute_heading_angle(velocity)
        torch.testing.assert_close(heading, torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        
        velocity = torch.tensor([0.0, 1.0])  # 向上
        heading = compute_heading_angle(velocity)
        torch.testing.assert_close(heading, torch.tensor(torch.pi / 2), atol=1e-5, rtol=1e-5)
    
    def test_compute_heading_angle_batch(self):
        """测试批量速度向量的航向角"""
        velocities = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0]
        ])
        
        headings = compute_heading_angle(velocities)
        
        assert headings.shape == (3,)
        torch.testing.assert_close(headings[0], torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(headings[1], torch.tensor(torch.pi / 2), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(headings[2], torch.tensor(torch.pi), atol=1e-5, rtol=1e-5)
    
    def test_normalize_angle(self):
        """测试角度归一化"""
        angles = torch.tensor([0.0, torch.pi, 2 * torch.pi, 3 * torch.pi, -torch.pi, -2 * torch.pi])
        normalized = normalize_angle(angles)
        
        # 所有角度应该在[-π, π]范围
        assert (normalized >= -torch.pi).all()
        assert (normalized <= torch.pi).all()
        
        # 检查具体值
        torch.testing.assert_close(normalized[0], torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(normalized[1], torch.tensor(torch.pi), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(normalized[2], torch.tensor(0.0), atol=1e-5, rtol=1e-5)
    
    def test_angle_difference(self):
        """测试角度差计算"""
        angle1 = torch.tensor(0.0)
        angle2 = torch.tensor(torch.pi / 2)
        
        diff = angle_difference(angle1, angle2)
        torch.testing.assert_close(diff, torch.tensor(torch.pi / 2), atol=1e-5, rtol=1e-5)
        
        # 测试跨越±π边界的情况
        angle1 = torch.tensor(torch.pi - 0.1)
        angle2 = torch.tensor(-torch.pi + 0.1)
        diff = angle_difference(angle1, angle2)
        
        # 差值应该是0.2，而不是接近2π
        assert abs(diff) < 1.0


class TestDistanceComputation:
    """测试距离计算相关函数"""
    
    def test_compute_distance_single(self):
        """测试单点距离"""
        point1 = torch.tensor([0.0, 0.0])
        point2 = torch.tensor([3.0, 4.0])
        
        distance = compute_distance(point1, point2)
        
        torch.testing.assert_close(distance, torch.tensor(5.0), atol=1e-5, rtol=1e-5)
    
    def test_compute_distance_batch(self):
        """测试批量距离计算"""
        points1 = torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0]
        ])
        points2 = torch.tensor([
            [3.0, 4.0],
            [4.0, 5.0]
        ])
        
        distances = compute_distance(points1, points2)
        
        assert distances.shape == (2,)
        torch.testing.assert_close(distances[0], torch.tensor(5.0), atol=1e-5, rtol=1e-5)
    
    def test_batch_compute_distances(self):
        """测试距离矩阵计算"""
        points1 = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        points2 = torch.tensor([
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0]
        ])
        
        distances = batch_compute_distances(points1, points2)
        
        assert distances.shape == (2, 3)
        
        # 检查几个具体值
        torch.testing.assert_close(distances[0, 0], torch.tensor(1.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(distances[1, 2], torch.tensor(1.0), atol=1e-5, rtol=1e-5)


class TestCoordinateTransform:
    """测试坐标变换相关函数"""
    
    def test_rotate_2d_90deg(self):
        """测试90度旋转"""
        point = torch.tensor([1.0, 0.0])
        angle = torch.tensor(torch.pi / 2)  # 90度
        
        rotated = rotate_2d(point, angle)
        
        expected = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(rotated, expected, atol=1e-5, rtol=1e-5)
    
    def test_rotate_2d_batch(self):
        """测试批量旋转"""
        points = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        angle = torch.tensor(torch.pi / 2)
        
        rotated = rotate_2d(points, angle)
        
        assert rotated.shape == (2, 2)
    
    def test_transform_roundtrip(self):
        """测试局部-全局坐标转换的往返一致性"""
        global_point = torch.tensor([5.0, 3.0])
        robot_pos = torch.tensor([2.0, 1.0])
        robot_heading = torch.tensor(torch.pi / 4)
        
        # 全局 -> 局部
        local_point = transform_to_local_frame(global_point, robot_pos, robot_heading)
        
        # 局部 -> 全局
        recovered_global = transform_to_global_frame(local_point, robot_pos, robot_heading)
        
        torch.testing.assert_close(global_point, recovered_global, atol=1e-4, rtol=1e-4)
    
    def test_transform_batch(self):
        """测试批量坐标转换"""
        global_points = torch.randn(5, 2)
        robot_pos = torch.zeros(2)
        robot_heading = torch.tensor(torch.pi / 3)
        
        local_points = transform_to_local_frame(global_points, robot_pos, robot_heading)
        
        assert local_points.shape == (5, 2)
        
        # 验证往返一致性
        recovered = transform_to_global_frame(local_points, robot_pos, robot_heading)
        torch.testing.assert_close(global_points, recovered, atol=1e-4, rtol=1e-4)
    
    def test_incremental_path_to_global(self):
        """测试增量式路径转换"""
        incremental_path = torch.randn(11, 2) * 0.5  # 归一化到[-0.5, 0.5]
        robot_pos = torch.tensor([1.0, 2.0])
        robot_heading = torch.tensor(0.0)  # 朝向0度
        
        global_path = incremental_path_to_global(
            incremental_path,
            robot_pos,
            robot_heading,
            max_range=5.0
        )
        
        assert global_path.shape == (11, 2)
        
        # 第一个点应该接近机器人位置（因为是累积的）
        # 注意：第一个点是第一个增量累积的结果，不是机器人位置
        assert torch.isfinite(global_path).all()


class TestPathAnalysis:
    """测试路径分析相关函数"""
    
    def test_compute_path_length_straight(self):
        """测试直线路径长度"""
        path = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0]
        ])
        
        length = compute_path_length(path)
        
        torch.testing.assert_close(length, torch.tensor(3.0), atol=1e-5, rtol=1e-5)
    
    def test_compute_path_length_empty(self):
        """测试空路径"""
        path = torch.tensor([]).reshape(0, 2)
        length = compute_path_length(path)
        torch.testing.assert_close(length, torch.tensor(0.0), atol=1e-5, rtol=1e-5)
    
    def test_compute_path_curvature(self):
        """测试路径曲率计算"""
        # 创建一个弯曲的路径
        path = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        
        curvature = compute_path_curvature(path)
        
        assert curvature.shape == (2,)  # N-2 = 4-2 = 2
        assert (curvature >= 0).all()  # 曲率非负
    
    def test_compute_path_curvature_straight(self):
        """测试直线路径的曲率（应该接近0）"""
        path = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0]
        ])
        
        curvature = compute_path_curvature(path)
        
        # 直线曲率应该非常小
        assert curvature.abs().max() < 1e-3
    
    def test_compute_closest_point_on_path(self):
        """测试找最近路径点"""
        path = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0]
        ])
        query_point = torch.tensor([1.5, 0.5])
        
        closest_idx, closest_point = compute_closest_point_on_path(path, query_point)
        
        assert closest_idx in [1, 2]  # 应该是中间的某个点
        assert closest_point.shape == (2,)
    
    def test_compute_lookahead_point(self):
        """测试前视点计算"""
        path = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0]
        ])
        current_pos = torch.tensor([0.5, 0.0])
        lookahead_distance = 2.0
        
        lookahead_idx, lookahead_point = compute_lookahead_point(
            path,
            current_pos,
            lookahead_distance
        )
        
        assert 0 <= lookahead_idx < path.size(0)
        assert lookahead_point.shape == (2,)
    
    def test_interpolate_path_upsample(self):
        """测试路径上采样插值"""
        path = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0]
        ])
        
        interpolated = interpolate_path(path, num_points=5)
        
        assert interpolated.shape == (5, 2)
        
        # 起点和终点应该不变
        torch.testing.assert_close(interpolated[0], path[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(interpolated[-1], path[-1], atol=1e-5, rtol=1e-5)
    
    def test_interpolate_path_downsample(self):
        """测试路径下采样"""
        path = torch.randn(20, 2)
        
        interpolated = interpolate_path(path, num_points=10)
        
        assert interpolated.shape == (10, 2)


class TestCollisionDetection:
    """测试碰撞检测相关函数"""
    
    def test_check_collision_single_point(self):
        """测试单点碰撞检测"""
        point = torch.tensor([1.0, 1.0])
        obstacles = torch.tensor([
            [1.1, 1.1],  # 很近
            [5.0, 5.0]   # 很远
        ])
        
        collision = check_collision(point, obstacles, collision_radius=0.5)
        
        assert collision == True  # 应该碰撞
        
        # 测试无碰撞情况
        collision = check_collision(point, obstacles, collision_radius=0.1)
        assert collision == False
    
    def test_check_collision_batch(self):
        """测试批量碰撞检测"""
        points = torch.tensor([
            [1.0, 1.0],
            [10.0, 10.0]
        ])
        obstacles = torch.tensor([
            [1.1, 1.1],
            [5.0, 5.0]
        ])
        
        collisions = check_collision(points, obstacles, collision_radius=0.5)
        
        assert collisions.shape == (2,)
        assert collisions[0] == True   # 第一个点碰撞
        assert collisions[1] == False  # 第二个点不碰撞
    
    def test_point_to_line_distance(self):
        """测试点到线段距离"""
        point = torch.tensor([1.0, 1.0])
        line_start = torch.tensor([0.0, 0.0])
        line_end = torch.tensor([2.0, 0.0])
        
        distance = point_to_line_distance(point, line_start, line_end)
        
        # 垂直距离应该是1.0
        torch.testing.assert_close(distance, torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    
    def test_point_to_line_distance_endpoint(self):
        """测试点到线段端点的距离"""
        point = torch.tensor([3.0, 0.0])
        line_start = torch.tensor([0.0, 0.0])
        line_end = torch.tensor([2.0, 0.0])
        
        distance = point_to_line_distance(point, line_start, line_end)
        
        # 距离应该是到终点的距离 = 1.0
        torch.testing.assert_close(distance, torch.tensor(1.0), atol=1e-5, rtol=1e-5)


class TestPolygonUtils:
    """测试多边形相关函数"""
    
    def test_compute_polygon_centroid(self):
        """测试多边形质心计算"""
        # 正方形
        vertices = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        
        centroid = compute_polygon_centroid(vertices)
        
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(centroid, expected, atol=1e-5, rtol=1e-5)
    
    def test_compute_polygon_centroid_triangle(self):
        """测试三角形质心"""
        vertices = torch.tensor([
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0]
        ])
        
        centroid = compute_polygon_centroid(vertices)
        
        expected = torch.tensor([1.0, 1.0])
        torch.testing.assert_close(centroid, expected, atol=1e-5, rtol=1e-5)


class TestEdgeCases:
    """测试边界情况"""
    
    def test_zero_velocity_heading(self):
        """测试零速度的航向角"""
        velocity = torch.tensor([0.0, 0.0])
        heading = compute_heading_angle(velocity)
        
        # 应该返回一个有效的角度（虽然没有物理意义）
        assert torch.isfinite(heading)
    
    def test_very_short_path(self):
        """测试非常短的路径"""
        path = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        
        length = compute_path_length(path)
        torch.testing.assert_close(length, torch.tensor(0.0), atol=1e-5, rtol=1e-5)
    
    def test_single_point_path(self):
        """测试单点路径"""
        path = torch.tensor([[1.0, 2.0]])
        
        length = compute_path_length(path)
        torch.testing.assert_close(length, torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        
        # 曲率计算应该返回空
        curvature = compute_path_curvature(path)
        assert curvature.numel() == 0
    
    def test_collinear_points_curvature(self):
        """测试共线点的曲率"""
        path = torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        
        curvature = compute_path_curvature(path)
        
        # 共线点曲率应该接近0
        assert curvature.abs().max() < 1e-3
    
    def test_nan_handling(self):
        """测试NaN处理"""
        # 大多数函数应该对有效输入不产生NaN
        path = torch.randn(10, 2)
        
        length = compute_path_length(path)
        assert torch.isfinite(length)
        
        curvature = compute_path_curvature(path)
        assert torch.isfinite(curvature).all()


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])