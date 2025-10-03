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
            num_corridors = 2
            num_obstacles = 1
            num_pedestrians = self.rng.randint(1, 3)
        elif difficulty == 'medium':
            num_corridors = self.rng.randint(2, 4)
            num_obstacles = self.rng.randint(1, 3)
            num_pedestrians = self.rng.randint(3, 6)
        else:  # hard
            num_corridors = self.rng.randint(3, 5)
            num_obstacles = self.rng.randint(2, 4)
            num_pedestrians = self.rng.randint(5, 10)
        
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
        """生成随机起点（地图左侧）"""
        w, h = self.map_size
        x = self.rng.uniform(0, w * 0.2)
        y = self.rng.uniform(h * 0.3, h * 0.7)
        return np.array([x, y])
    
    def _generate_goal(self, start: np.ndarray) -> np.ndarray:
        """生成随机终点（地图右侧，距离起点足够远）"""
        w, h = self.map_size
        
        # 确保终点在右侧，且距离起点至少一定距离
        min_distance = max(w, h) * 0.6
        
        for _ in range(100):  # 最多尝试100次
            x = self.rng.uniform(w * 0.7, w * 0.95)
            y = self.rng.uniform(h * 0.3, h * 0.7)
            goal = np.array([x, y])
            
            if np.linalg.norm(goal - start) >= min_distance:
                return goal
        
        # 如果找不到，返回固定位置
        return np.array([w * 0.85, h * 0.5])
    
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
                corridors.append(corridor)
        
        # 确保至少有一条corridor
        if len(corridors) == 0:
            corridors.append(self._generate_direct_corridor(start, goal))
        
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
            return self._upper_detour_corridor(start, goal, obstacles, width=1.5)
        
        elif strategy == 'lower':
            return self._lower_detour_corridor(start, goal, obstacles, width=1.5)
        
        elif strategy == 'wide_upper':
            return self._upper_detour_corridor(start, goal, obstacles, width=2.5)
        
        elif strategy == 'wide_lower':
            return self._lower_detour_corridor(start, goal, obstacles, width=2.5)
        
        elif strategy == 'direct':
            return self._generate_direct_corridor(start, goal)
        
        return None
    
    def _upper_detour_corridor(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[np.ndarray],
        width: float = 1.5
    ) -> np.ndarray:
        """上方绕行corridor"""
        
        # 找到障碍物的最高点
        if obstacles:
            max_y = max(obs[:, 1].max() for obs in obstacles)
        else:
            max_y = max(start[1], goal[1])
        
        detour_y = max_y + width + self.rng.uniform(0.5, 1.0)
        
        # 构建L型多边形
        # 从起点出发，向上绕，再到终点
        mid_x = (start[0] + goal[0]) / 2
        
        corridor = np.array([
            [start[0] - 0.5, start[1] - 0.5],  # 起点区域
            [start[0] - 0.5, detour_y + width],  # 向上
            [mid_x - 1, detour_y + width],       # 向右
            [mid_x + 1, detour_y + width],
            [goal[0] + 0.5, detour_y + width],   # 到达终点上方
            [goal[0] + 0.5, goal[1] + 0.5],      # 向下到终点
            [goal[0] + 0.5, start[1] - 0.5],     # 回到起点高度
            [mid_x + 1, start[1] - 0.5],
            [mid_x + 1, detour_y],               # 内边界
            [mid_x - 1, detour_y],
            [mid_x - 1, start[1] - 0.5],
            [start[0] - 0.5, start[1] - 0.5]
        ])
        
        return corridor
    
    def _lower_detour_corridor(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[np.ndarray],
        width: float = 1.5
    ) -> np.ndarray:
        """下方绕行corridor"""
        
        # 找到障碍物的最低点
        if obstacles:
            min_y = min(obs[:, 1].min() for obs in obstacles)
        else:
            min_y = min(start[1], goal[1])
        
        detour_y = min_y - width - self.rng.uniform(0.5, 1.0)
        detour_y = max(0, detour_y)  # 不超出地图
        
        # 构建L型多边形
        mid_x = (start[0] + goal[0]) / 2
        
        corridor = np.array([
            [start[0] - 0.5, start[1] + 0.5],  # 起点区域
            [start[0] - 0.5, detour_y - width],  # 向下
            [mid_x - 1, detour_y - width],
            [mid_x + 1, detour_y - width],
            [goal[0] + 0.5, detour_y - width],   # 到达终点下方
            [goal[0] + 0.5, goal[1] - 0.5],      # 向上到终点
            [goal[0] + 0.5, start[1] + 0.5],
            [mid_x + 1, start[1] + 0.5],
            [mid_x + 1, detour_y],               # 内边界
            [mid_x - 1, detour_y],
            [mid_x - 1, start[1] + 0.5],
            [start[0] - 0.5, start[1] + 0.5]
        ])
        
        return corridor
    
    def _generate_direct_corridor(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        width: float = 2.0
    ) -> np.ndarray:
        """直线corridor（矩形）"""
        
        direction = goal - start
        perpendicular = np.array([-direction[1], direction[0]])
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)
        perpendicular *= width / 2
        
        corridor = np.array([
            start + perpendicular,
            goal + perpendicular,
            goal - perpendicular,
            start - perpendicular
        ])
        
        return corridor


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

