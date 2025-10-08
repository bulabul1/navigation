"""
Corridor场景生成器
自动生成从起点到终点的多条可通行路线
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class CorridorGenerator:
    """
    Corridor场景生成器
    
    功能：
    - 随机生成起点和终点
    - 随机生成障碍物
    - 生成多条绕过障碍物的可行路线
    - 支持难度分级（easy/medium/hard）
    """
    
    def __init__(
        self,
        map_size: Tuple[float, float] = (12.0, 12.0),
        seed: Optional[int] = None
    ):
        """
        Args:
            map_size: 地图大小 (width, height)
            seed: 随机种子
        """
        self.map_size = map_size
        self.rng = np.random.RandomState(seed)
    
    def generate_scenario(
        self,
        difficulty: str = 'medium',
        fixed_start: Optional[np.ndarray] = None,
        fixed_goal: Optional[np.ndarray] = None
    ) -> Dict:
        """
        生成一个完整场景
        
        Args:
            difficulty: 难度级别 ('easy', 'medium', 'hard')
            fixed_start: 固定起点（可选）
            fixed_goal: 固定终点（可选）
        
        Returns:
            scenario: {
                'start': (x, y),
                'goal': (x, y),
                'corridors': [polygon1, polygon2, ...],
                'obstacles': [polygon1, polygon2, ...],
                'num_pedestrians': int
            }
        """
        # 1. 确定难度参数
        if difficulty == 'easy':
            num_corridors = self.rng.randint(2, 4)  # 2-3个走廊（提供多路径选择）
            num_obstacles = 0  # 取消障碍物
            num_pedestrians = self.rng.randint(2, 4)  # 2-3个行人
        elif difficulty == 'medium':
            num_corridors = self.rng.randint(3, 5)  # 3-4个走廊
            num_obstacles = 0  # 取消障碍物
            num_pedestrians = self.rng.randint(3, 5)  # 3-4个行人
        else:  # hard
            num_corridors = self.rng.randint(3, 5)  # 3-4个走廊
            num_obstacles = 0  # 取消障碍物
            num_pedestrians = self.rng.randint(3, 5)  # 3-4个行人
        
        # 2. 生成起点和终点
        if fixed_start is not None:
            start = fixed_start
        else:
            start = self._generate_start()
        
        if fixed_goal is not None:
            goal = fixed_goal
        else:
            goal = self._generate_goal(start)
        
        # 3. 生成障碍物
        obstacles = self._generate_obstacles(start, goal, num_obstacles)
        
        # 4. 生成corridors（绕过障碍物的路线）
        corridors = self._generate_corridors(
            start, goal, obstacles, num_corridors
        )
        
        return {
            'start': start,
            'goal': goal,
            'corridors': corridors,
            'obstacles': obstacles,
            'num_pedestrians': num_pedestrians
        }
    
    def _generate_start(self) -> np.ndarray:
        """生成随机起点（地图任意位置，避免太靠近边界）"""
        w, h = self.map_size
        # 全地图范围随机生成起点
        x = self.rng.uniform(w * 0.1, w * 0.9)
        y = self.rng.uniform(h * 0.1, h * 0.9)
        return np.array([x, y])
    
    def _generate_goal(self, start: np.ndarray) -> np.ndarray:
        """生成随机终点（全方向，距离起点足够远）"""
        w, h = self.map_size
        
        # 确保终点距离起点足够远，但方向完全随机
        min_distance = max(w, h) * 0.5  # 降低最小距离要求
        
        for _ in range(100):  # 最多尝试100次
            # 全地图范围随机生成终点
            x = self.rng.uniform(w * 0.1, w * 0.9)
            y = self.rng.uniform(h * 0.1, h * 0.9)
            goal = np.array([x, y])
            
            if np.linalg.norm(goal - start) >= min_distance:
                return goal
        
        # 如果找不到，返回起点对角位置
        return np.array([w - start[0], h - start[1]])
    
    def _generate_obstacles(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_obstacles: int
    ) -> List[np.ndarray]:
        """
        生成障碍物（在起点和终点之间）
        """
        obstacles = []
        w, h = self.map_size
        
        # 中间区域
        mid_x = (start[0] + goal[0]) / 2
        mid_y = (start[1] + goal[1]) / 2
        
        for i in range(num_obstacles):
            # 障碍物中心（在中间区域，避开起点和终点）
            center_x = self.rng.uniform(
                max(start[0] + 1.5, mid_x - 2),
                min(goal[0] - 1.5, mid_x + 2)
            )
            center_y = self.rng.uniform(
                max(0, mid_y - 2),
                min(h, mid_y + 2)
            )
            
            # 障碍物大小
            width = self.rng.uniform(1.0, 2.5)
            height = self.rng.uniform(1.0, 2.5)
            
            # 矩形障碍物
            obstacle = np.array([
                [center_x - width/2, center_y - height/2],
                [center_x + width/2, center_y - height/2],
                [center_x + width/2, center_y + height/2],
                [center_x - width/2, center_y + height/2]
            ])
            
            obstacles.append(obstacle)
        
        return obstacles
    
    def _generate_corridors(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[np.ndarray],
        num_corridors: int
    ) -> List[np.ndarray]:
        """
        生成多条corridor（绕过障碍物的路线）
        
        策略：
        - 上方绕行
        - 下方绕行
        - 宽敞变体
        - 直接路线（如果没有障碍物阻挡）
        """
        corridors = []
        
        # 预定义策略
        strategies = ['upper', 'lower', 'wide_upper', 'wide_lower', 'direct']
        
        # 随机选择策略
        selected = self.rng.choice(
            strategies,
            size=min(num_corridors, len(strategies)),
            replace=False
        )
        
        for strategy in selected:
            corridor = self._generate_single_corridor(
                start, goal, obstacles, strategy
            )
            if corridor is not None:
                # 验证并确保起点终点在corridor内
                corridor = self._ensure_endpoints_inside(corridor, start, goal)
                corridors.append(corridor)
        
        # 确保至少有一条corridor
        if len(corridors) == 0:
            corridor = self._generate_direct_corridor(start, goal)
            corridor = self._ensure_endpoints_inside(corridor, start, goal)
            corridors.append(corridor)
        
        return corridors
    
    def _generate_single_corridor(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[np.ndarray],
        strategy: str
    ) -> Optional[np.ndarray]:
        """根据策略生成单条corridor"""
        
        if strategy == 'upper':
            return self._upper_detour_corridor(start, goal, obstacles, width=2.5)
        
        elif strategy == 'lower':
            return self._lower_detour_corridor(start, goal, obstacles, width=2.5)
        
        elif strategy == 'wide_upper':
            return self._upper_detour_corridor(start, goal, obstacles, width=3.0)
        
        elif strategy == 'wide_lower':
            return self._lower_detour_corridor(start, goal, obstacles, width=3.0)
        
        elif strategy == 'direct':
            return self._generate_direct_corridor(start, goal, width=2.5)
        
        return None
    
    def _upper_detour_corridor(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[np.ndarray],
        width: float = 2.5
    ) -> np.ndarray:
        """
        上方绕行corridor（修复版：确保无自相交）
        
        路径：起点 → 向上走 → 转弯向goal方向 → 向下到达终点
        
        生成一个"凹"字形或"┐"形的通道，确保：
        1. 起点和终点在通道内部
        2. 多边形不自相交
        3. 形成连续的可行走路径
        """
        
        # 找到障碍物的最高点（如果有障碍物）
        if obstacles:
            max_y = max(obs[:, 1].max() for obs in obstacles)
            detour_y = max_y + width + self.rng.uniform(1.0, 2.0)
        else:
            # 无障碍物时，向上绕行一段距离
            max_y = max(start[1], goal[1])
            detour_y = max_y + self.rng.uniform(2.0, 3.0)
        
        half_width = width / 2
        
        # 确定起点和终点的相对位置
        # 统一处理：从左到右（如果start在右侧，交换）
        if start[0] > goal[0]:
            left_point = goal
            right_point = start
            left_y_offset = -1.0  # goal下方
            right_y_offset = -1.0  # start下方
        else:
            left_point = start
            right_point = goal
            left_y_offset = -1.0
            right_y_offset = -1.0
        
        # 构建L型凹多边形（逆时针绘制，确保无自相交）
        # 形状类似"┐"或"┌"
        corridor = np.array([
            # 外边界（逆时针）
            [left_point[0] - half_width, left_point[1] + left_y_offset],      # 1. 左侧起点
            [left_point[0] - half_width, detour_y + half_width],               # 2. 向上
            [right_point[0] + half_width, detour_y + half_width],              # 3. 向右
            [right_point[0] + half_width, right_point[1] + right_y_offset],    # 4. 向下
            # 内边界（从右向左，形成内凹）
            [right_point[0] - half_width, right_point[1] + right_y_offset],    # 5. 右侧内边
            [right_point[0] - half_width, detour_y - half_width],              # 6. 向上（内侧）
            [left_point[0] + half_width, detour_y - half_width],               # 7. 向左（内侧）
            [left_point[0] + half_width, left_point[1] + left_y_offset],       # 8. 向下回左侧
        ])
        
        return corridor
    
    def _lower_detour_corridor(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[np.ndarray],
        width: float = 2.5
    ) -> np.ndarray:
        """
        下方绕行corridor（修复版：确保无自相交）
        
        路径：起点 → 向下走 → 转弯向goal方向 → 向上到达终点
        
        生成一个"凹"字形或"┘"形的通道，确保：
        1. 起点和终点在通道内部
        2. 多边形不自相交
        3. 形成连续的可行走路径
        """
        
        # 找到障碍物的最低点
        if obstacles:
            min_y = min(obs[:, 1].min() for obs in obstacles)
            detour_y = min_y - width - self.rng.uniform(1.0, 2.0)
        else:
            # 无障碍物时，向下绕行一段距离
            min_y = min(start[1], goal[1])
            detour_y = min_y - self.rng.uniform(2.0, 3.0)
        
        detour_y = max(0.5, detour_y)  # 不超出地图边界（至少留0.5米）
        
        half_width = width / 2
        
        # 确定起点和终点的相对位置
        # 统一处理：从左到右（如果start在右侧，交换）
        if start[0] > goal[0]:
            left_point = goal
            right_point = start
            left_y_offset = 1.0  # goal上方
            right_y_offset = 1.0  # start上方
        else:
            left_point = start
            right_point = goal
            left_y_offset = 1.0
            right_y_offset = 1.0
        
        # 构建L型凹多边形（逆时针绘制，确保无自相交）
        # 形状类似"┘"或"└"
        corridor = np.array([
            # 外边界（逆时针）
            [left_point[0] - half_width, left_point[1] + left_y_offset],      # 1. 左侧起点
            [left_point[0] - half_width, detour_y - half_width],               # 2. 向下
            [right_point[0] + half_width, detour_y - half_width],              # 3. 向右
            [right_point[0] + half_width, right_point[1] + right_y_offset],    # 4. 向上
            # 内边界（从右向左，形成内凹）
            [right_point[0] - half_width, right_point[1] + right_y_offset],    # 5. 右侧内边
            [right_point[0] - half_width, detour_y + half_width],              # 6. 向下（内侧）
            [left_point[0] + half_width, detour_y + half_width],               # 7. 向左（内侧）
            [left_point[0] + half_width, left_point[1] + left_y_offset],       # 8. 向上回左侧
        ])
        
        return corridor
    
    def _generate_direct_corridor(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        width: float = 2.5
    ) -> np.ndarray:
        """
        直线corridor（矩形）
        
        Args:
            start: 起点
            goal: 终点
            width: 通道宽度（默认2.5米，足够机器狗+行人通行）
        """
        
        direction = goal - start
        distance = np.linalg.norm(direction)
        direction_normalized = direction / (distance + 1e-6)
        
        # 垂直于方向的向量（通道宽度）
        perpendicular = np.array([-direction[1], direction[0]])
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)
        perpendicular *= width / 2
        
        # 向前后各延伸1米，确保起点和终点在corridor内部（增加到1米）
        extension = 1.0
        start_extended = start - direction_normalized * extension
        goal_extended = goal + direction_normalized * extension
        
        corridor = np.array([
            start_extended + perpendicular,
            goal_extended + perpendicular,
            goal_extended - perpendicular,
            start_extended - perpendicular
        ])
        
        return corridor
    
    def _ensure_endpoints_inside(
        self,
        corridor: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray
    ) -> np.ndarray:
        """
        确保起点和终点在corridor内部
        
        注意：新版本的corridor生成逻辑已经在构建时确保了起点终点在内部，
        这个函数只做验证，不再进行扩展（避免将L型拍扁成矩形）。
        
        Args:
            corridor: (N, 2) 多边形顶点
            start: (2,) 起点
            goal: (2,) 终点
        
        Returns:
            原始corridor（不做修改）
        """
        # 检查起点和终点是否在内部（只做验证）
        start_inside = self._point_in_polygon(start, corridor)
        goal_inside = self._point_in_polygon(goal, corridor)
        
        # 如果不在内部，打印警告但不修改（避免破坏L型结构）
        if not start_inside:
            print(f"  [警告] 起点 ({start[0]:.2f}, {start[1]:.2f}) 可能在corridor边界上")
        if not goal_inside:
            print(f"  [警告] 终点 ({goal[0]:.2f}, {goal[1]:.2f}) 可能在corridor边界上")
        
        # 直接返回原始corridor，保留其形状（包括L型）
        return corridor
    
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        判断点是否在多边形内（射线法）
        
        Args:
            point: (2,) 点坐标 [x, y]
            polygon: (n, 2) 多边形顶点
        
        Returns:
            inside: True if point is inside polygon
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


# ==================== 测试 ====================
if __name__ == '__main__':
    print("测试Corridor生成器...")
    
    generator = CorridorGenerator(seed=42)
    
    # 测试1: Easy难度
    print("\n[测试1] Easy难度")
    scenario = generator.generate_scenario('easy')
    print(f"  起点: {scenario['start']}")
    print(f"  终点: {scenario['goal']}")
    print(f"  Corridor数量: {len(scenario['corridors'])}")
    print(f"  障碍物数量: {len(scenario['obstacles'])}")
    print(f"  行人数量: {scenario['num_pedestrians']}")
    
    # 测试2: Medium难度
    print("\n[测试2] Medium难度")
    scenario = generator.generate_scenario('medium')
    print(f"  起点: {scenario['start']}")
    print(f"  终点: {scenario['goal']}")
    print(f"  Corridor数量: {len(scenario['corridors'])}")
    print(f"  障碍物数量: {len(scenario['obstacles'])}")
    print(f"  行人数量: {scenario['num_pedestrians']}")
    
    # 测试3: Hard难度
    print("\n[测试3] Hard难度")
    scenario = generator.generate_scenario('hard')
    print(f"  起点: {scenario['start']}")
    print(f"  终点: {scenario['goal']}")
    print(f"  Corridor数量: {len(scenario['corridors'])}")
    print(f"  障碍物数量: {len(scenario['obstacles'])}")
    print(f"  行人数量: {scenario['num_pedestrians']}")
    
    # 测试4: 固定起点终点
    print("\n[测试4] 固定起点终点")
    fixed_start = np.array([0.0, 5.0])
    fixed_goal = np.array([10.0, 5.0])
    scenario = generator.generate_scenario(
        'medium',
        fixed_start=fixed_start,
        fixed_goal=fixed_goal
    )
    print(f"  起点: {scenario['start']} (固定)")
    print(f"  终点: {scenario['goal']} (固定)")
    print(f"  Corridor数量: {len(scenario['corridors'])}")
    
    # 测试5: 多次生成验证随机性
    print("\n[测试5] 随机性验证")
    for i in range(3):
        scenario = generator.generate_scenario('medium')
        print(f"  场景{i+1}: "
              f"起点({scenario['start'][0]:.1f},{scenario['start'][1]:.1f}), "
              f"终点({scenario['goal'][0]:.1f},{scenario['goal'][1]:.1f}), "
              f"{len(scenario['corridors'])}条corridor")
    
    print("\n[OK] Corridor生成器测试完成！")

