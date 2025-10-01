"""
工具函数模块
"""

from .data_processing import (
    DataProcessor,
    pad_sequence_list,
    create_attention_mask,
    process_single_input
)

from .geometry_utils import (
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

__all__ = [
    # 数据处理
    'DataProcessor',
    'pad_sequence_list',
    'create_attention_mask',
    'process_single_input',
    
    # 几何工具
    'compute_relative_angles',
    'compute_distance',
    'compute_heading_angle',
    'rotate_2d',
    'transform_to_local_frame',
    'transform_to_global_frame',
    'incremental_path_to_global',
    'compute_path_curvature',
    'compute_path_length',
    'compute_closest_point_on_path',
    'compute_lookahead_point',
    'check_collision',
    'compute_polygon_centroid',
    'point_to_line_distance',
    'normalize_angle',
    'angle_difference',
    'interpolate_path',
    'batch_compute_distances'
]