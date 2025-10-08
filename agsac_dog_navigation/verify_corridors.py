"""
验证Corridor生成的正确性
检查起点和终点是否真的在每个corridor内部
"""

import numpy as np
from agsac.envs.corridor_generator import CorridorGenerator


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    判断点是否在多边形内（射线法）
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        # 射线与边相交判断
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        
        j = i
    
    return inside


def check_point_distance_to_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
    """计算点到多边形边界的最短距离"""
    min_dist = float('inf')
    
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        
        # 点到线段的距离
        dist = point_to_segment_distance(point, p1, p2)
        min_dist = min(min_dist, dist)
    
    return min_dist


def point_to_segment_distance(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """计算点到线段的最短距离"""
    # 向量
    v = p2 - p1
    w = point - p1
    
    # 线段长度的平方
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(point - p1)
    
    c2 = np.dot(v, v)
    if c1 >= c2:
        return np.linalg.norm(point - p2)
    
    # 投影点
    b = c1 / c2
    pb = p1 + b * v
    return np.linalg.norm(point - pb)


def verify_scenario(scenario, scenario_name="Scenario"):
    """验证一个场景中所有corridor"""
    print(f"\n{'='*70}")
    print(f"验证 {scenario_name}")
    print(f"{'='*70}")
    
    start = scenario['start']
    goal = scenario['goal']
    
    print(f"起点: ({start[0]:.3f}, {start[1]:.3f})")
    print(f"终点: ({goal[0]:.3f}, {goal[1]:.3f})")
    print(f"直线距离: {np.linalg.norm(goal - start):.3f}m")
    print(f"\nCorridor总数: {len(scenario['corridors'])}")
    
    all_valid = True
    
    for i, corridor in enumerate(scenario['corridors']):
        print(f"\n--- Corridor {i+1} ({len(corridor)} 个顶点) ---")
        
        # 检查起点
        start_inside = point_in_polygon(start, corridor)
        start_dist = check_point_distance_to_polygon(start, corridor)
        
        print(f"  起点检查:")
        print(f"    在内部: {'[YES]' if start_inside else '[NO]'}")
        print(f"    到边界距离: {start_dist:.4f}m")
        
        if not start_inside:
            print(f"    [WARNING] 起点不在corridor内部！")
            all_valid = False
        elif start_dist < 0.1:
            print(f"    [WARNING] 起点离边界太近（<10cm）！")
        
        # 检查终点
        goal_inside = point_in_polygon(goal, corridor)
        goal_dist = check_point_distance_to_polygon(goal, corridor)
        
        print(f"  终点检查:")
        print(f"    在内部: {'[YES]' if goal_inside else '[NO]'}")
        print(f"    到边界距离: {goal_dist:.4f}m")
        
        if not goal_inside:
            print(f"    [WARNING] 终点不在corridor内部！")
            all_valid = False
        elif goal_dist < 0.1:
            print(f"    [WARNING] 终点离边界太近（<10cm）！")
        
        # 打印corridor边界框
        min_x, min_y = corridor[:, 0].min(), corridor[:, 1].min()
        max_x, max_y = corridor[:, 0].max(), corridor[:, 1].max()
        print(f"  Corridor边界框:")
        print(f"    X: [{min_x:.3f}, {max_x:.3f}] (宽度: {max_x - min_x:.3f}m)")
        print(f"    Y: [{min_y:.3f}, {max_y:.3f}] (高度: {max_y - min_y:.3f}m)")
        
        # 打印前4个顶点
        print(f"  前4个顶点:")
        for j in range(min(4, len(corridor))):
            print(f"    顶点{j+1}: ({corridor[j][0]:.3f}, {corridor[j][1]:.3f})")
    
    print(f"\n{'='*70}")
    if all_valid:
        print(f"[PASS] {scenario_name} 验证通过：所有起点和终点都在corridor内部")
    else:
        print(f"[FAIL] {scenario_name} 验证失败：存在起点或终点不在corridor内部的情况")
    print(f"{'='*70}")
    
    return all_valid


def main():
    """主验证流程"""
    print("=" * 70)
    print("Corridor起点终点包含性验证")
    print("=" * 70)
    
    all_scenarios_valid = True
    
    # 场景1: 固定水平路径
    print("\n\n[场景1] 固定水平路径 (2, 6) → (10, 6)")
    generator1 = CorridorGenerator(seed=123)
    scenario1 = generator1.generate_scenario(
        'medium',
        fixed_start=np.array([2.0, 6.0]),
        fixed_goal=np.array([10.0, 6.0])
    )
    valid1 = verify_scenario(scenario1, "场景1: 水平路径")
    all_scenarios_valid = all_scenarios_valid and valid1
    
    # 场景2: 固定对角线路径
    print("\n\n[场景2] 固定对角线路径 (2, 2) → (10, 10)")
    generator2 = CorridorGenerator(seed=456)
    scenario2 = generator2.generate_scenario(
        'medium',
        fixed_start=np.array([2.0, 2.0]),
        fixed_goal=np.array([10.0, 10.0])
    )
    valid2 = verify_scenario(scenario2, "场景2: 对角线路径")
    all_scenarios_valid = all_scenarios_valid and valid2
    
    # 场景3: 随机场景（Easy）
    print("\n\n[场景3] 随机Easy场景")
    generator3 = CorridorGenerator(seed=42)
    scenario3 = generator3.generate_scenario('easy')
    valid3 = verify_scenario(scenario3, "场景3: 随机Easy")
    all_scenarios_valid = all_scenarios_valid and valid3
    
    # 场景4: 随机场景（Medium）
    print("\n\n[场景4] 随机Medium场景")
    generator4 = CorridorGenerator(seed=789)
    scenario4 = generator4.generate_scenario('medium')
    valid4 = verify_scenario(scenario4, "场景4: 随机Medium")
    all_scenarios_valid = all_scenarios_valid and valid4
    
    # 场景5: 随机场景（Hard）
    print("\n\n[场景5] 随机Hard场景")
    generator5 = CorridorGenerator(seed=999)
    scenario5 = generator5.generate_scenario('hard')
    valid5 = verify_scenario(scenario5, "场景5: 随机Hard")
    all_scenarios_valid = all_scenarios_valid and valid5
    
    # 最终总结
    print("\n\n" + "=" * 70)
    print("最终验证结果")
    print("=" * 70)
    if all_scenarios_valid:
        print("[SUCCESS] 所有场景验证通过！所有起点终点都在corridor内部。")
    else:
        print("[FAILURE] 存在场景验证失败！需要修复corridor生成逻辑。")
    print("=" * 70)


if __name__ == '__main__':
    main()

