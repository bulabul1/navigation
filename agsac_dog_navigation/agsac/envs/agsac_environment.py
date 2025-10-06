"""
AGSAC Environment Interface
AGSAC环境接口 - 标准化的环境交互层

功能:
1. 环境接口抽象（类似OpenAI Gym）
2. 观测标准化
3. 动作执行
4. 奖励计算（基础 + GDE）
5. 碰撞检测
6. Episode管理
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from abc import ABC, abstractmethod
from agsac.models.evaluator.geometric_evaluator import GeometricDifferentialEvaluator
from agsac.envs.corridor_generator import CorridorGenerator


class AGSACEnvironment(ABC):
    """
    AGSAC环境抽象基类
    
    设计原则:
    - 类似OpenAI Gym的接口
    - 标准化的观测格式
    - 灵活的奖励设计
    - 支持批量环境（可选）
    
    子类需要实现的抽象方法:
    - _reset_env()
    - _step_env(action)
    - _get_raw_observation()
    - _check_collision()
    - _compute_base_reward()
    """
    
    def __init__(
        self,
        max_pedestrians: int = 10,
        max_corridors: int = 10,
        max_vertices: int = 20,
        obs_horizon: int = 8,
        pred_horizon: int = 12,
        max_episode_steps: int = 500,
        use_geometric_reward: bool = True,
        reward_weights: Optional[Dict] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            max_pedestrians: 最大行人数量
            max_corridors: 最大走廊数量
            max_vertices: 每个走廊的最大顶点数
            obs_horizon: 观测历史长度
            pred_horizon: 预测未来长度
            max_episode_steps: 最大episode长度
            use_geometric_reward: 是否使用几何奖励
            reward_weights: 奖励权重配置
            device: 计算设备
        """
        self.max_pedestrians = max_pedestrians
        self.max_corridors = max_corridors
        self.max_vertices = max_vertices
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.max_episode_steps = max_episode_steps
        self.use_geometric_reward = use_geometric_reward
        self.device = torch.device(device)
        
        # 奖励权重（默认值）
        self.reward_weights = {
            'goal_progress': 1.0,
            'collision': -10.0,
            'geometric': 0.5,
            'timeout': -1.0,
            'goal_reached': 10.0,
            'step_penalty': -0.01
        }
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)
        
        # Episode状态
        self.current_step = 0
        self.episode_return = 0.0
        self.done = False
        
        # 历史观测（用于构建obs_horizon）
        self.observation_history = []
        
        # 路径历史（用于GDE）
        self.path_history = []
        
    @abstractmethod
    def _reset_env(self) -> Dict:
        """
        重置环境（由子类实现）
        
        Returns:
            raw_state: 原始环境状态
        """
        pass
    
    @abstractmethod
    def _step_env(self, action: np.ndarray) -> Dict:
        """
        执行动作（由子类实现）
        
        Args:
            action: 动作 (action_dim,)
        
        Returns:
            raw_state: 执行后的原始环境状态
        """
        pass
    
    @abstractmethod
    def _get_raw_observation(self) -> Dict:
        """
        获取原始观测（由子类实现）
        
        Returns:
            {
                'robot_state': {...},
                'pedestrians': [...],
                'corridors': [...],
                'goal': [x, y],
                'reference_line': [[x1, y1], [x2, y2]]
            }
        """
        pass
    
    @abstractmethod
    def _check_collision(self) -> bool:
        """
        碰撞检测（由子类实现）
        
        Returns:
            collision: True if collision occurred
        """
        pass
    
    @abstractmethod
    def _compute_base_reward(self, action: np.ndarray, collision: bool) -> float:
        """
        计算基础奖励（由子类实现）
        
        Args:
            action: 执行的动作
            collision: 是否碰撞
        
        Returns:
            base_reward: 基础奖励值
        """
        pass
    
    def reset(self) -> Dict:
        """
        重置环境
        
        Returns:
            observation: 标准化的观测
        """
        # 重置状态
        self.current_step = 0
        self.episode_return = 0.0
        self.done = False
        self.observation_history = []
        self.path_history = []
        
        # 调用子类实现
        raw_state = self._reset_env()
        
        # 获取初始观测
        observation = self._process_observation()
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: 动作 (action_dim,) 范围[-1, 1]
        
        Returns:
            observation: 观测
            reward: 奖励
            done: 是否结束
            info: 附加信息
        """
        if self.done:
            raise RuntimeError("Episode已经结束，请先调用reset()")
        
        # 1. 执行动作
        raw_state = self._step_env(action)
        
        # 2. 碰撞检测
        collision = self._check_collision()
        
        # 3. 计算奖励
        reward, reward_info = self._compute_reward(action, collision)
        
        # 4. 检查终止条件
        done, done_reason = self._check_done(collision)
        
        # 5. 获取观测
        observation = self._process_observation()
        
        # 6. 更新状态
        self.current_step += 1
        self.episode_return += reward
        self.done = done
        
        # 7. 构建info
        info = {
            'collision': collision,
            'done_reason': done_reason,
            'episode_step': self.current_step,
            'episode_return': self.episode_return,
            **reward_info
        }
        
        return observation, reward, done, info
    
    def _process_observation(self) -> Dict:
        """
        处理原始观测，转换为标准格式
        
        Returns:
            observation: 标准化的观测
        """
        raw_obs = self._get_raw_observation()
        
        # 标准化观测格式
        observation = {
            # 机器狗状态
            'robot_state': self._process_robot_state(raw_obs['robot_state']),
            
            # 行人观测历史 (num_peds, obs_horizon, 2)
            'pedestrian_observations': self._process_pedestrian_obs(
                raw_obs['pedestrians']
            ),
            
            # 行人mask (num_peds,)
            'pedestrian_mask': self._create_pedestrian_mask(
                raw_obs['pedestrians']
            ),
            
            # 走廊几何 (num_corridors, num_vertices, 2)
            'corridor_vertices': self._process_corridors(
                raw_obs['corridors']
            ),
            
            # 走廊mask (num_corridors,)
            'corridor_mask': self._create_corridor_mask(
                raw_obs['corridors']
            ),
            
            # 目标点
            'goal': torch.tensor(
                raw_obs['goal'], dtype=torch.float32, device=self.device
            ),
            
            # 参考线（用于GDE）
            'reference_line': raw_obs.get('reference_line', None)
        }
        
        return observation
    
    def _process_robot_state(self, robot_state: Dict) -> Dict:
        """
        处理机器狗状态
        
        Args:
            robot_state: {
                'position': [x, y],
                'velocity': [vx, vy],
                'orientation': theta,
                'goal_vector': [dx, dy]
            }
        
        Returns:
            processed_state: torch.Tensor化的状态
        """
        return {
            'position': torch.tensor(
                robot_state['position'], dtype=torch.float32, device=self.device
            ),
            'velocity': torch.tensor(
                robot_state['velocity'], dtype=torch.float32, device=self.device
            ),
            'orientation': torch.tensor(
                robot_state['orientation'], dtype=torch.float32, device=self.device
            ),
            'goal_vector': torch.tensor(
                robot_state['goal_vector'], dtype=torch.float32, device=self.device
            )
        }
    
    def _process_pedestrian_obs(self, pedestrians: List[Dict]) -> torch.Tensor:
        """
        处理行人观测历史
        
        Args:
            pedestrians: List of {
                'trajectory': [[x, y], ...],  # 观测历史
                'id': int
            }
        
        Returns:
            pedestrian_obs: (max_pedestrians, obs_horizon, 2)
        """
        num_peds = min(len(pedestrians), self.max_pedestrians)
        
        # 初始化（全0）
        ped_obs = torch.zeros(
            self.max_pedestrians, self.obs_horizon, 2,
            dtype=torch.float32, device=self.device
        )
        
        # 填充实际数据
        for i in range(num_peds):
            traj = pedestrians[i]['trajectory']
            # 取最近的obs_horizon个点
            traj = traj[-self.obs_horizon:]
            traj_len = len(traj)
            
            # 填充（如果不足obs_horizon，前面补0）
            if traj_len > 0:
                traj_tensor = torch.tensor(
                    traj, dtype=torch.float32, device=self.device
                )
                ped_obs[i, -traj_len:, :] = traj_tensor
        
        return ped_obs
    
    def _create_pedestrian_mask(self, pedestrians: List[Dict]) -> torch.Tensor:
        """
        创建行人mask
        
        Args:
            pedestrians: 行人列表
        
        Returns:
            mask: (max_pedestrians,) 1=有效, 0=padding
        """
        num_peds = min(len(pedestrians), self.max_pedestrians)
        mask = torch.zeros(self.max_pedestrians, dtype=torch.bool, device=self.device)
        mask[:num_peds] = True
        return mask
    
    def _process_corridors(self, corridors: List[np.ndarray]) -> torch.Tensor:
        """
        处理走廊几何
        
        Args:
            corridors: List of np.ndarray (num_vertices, 2)
        
        Returns:
            corridor_vertices: (max_corridors, max_vertices, 2)
        """
        num_corrs = min(len(corridors), self.max_corridors)
        
        # 初始化
        corr_vertices = torch.zeros(
            self.max_corridors, self.max_vertices, 2,
            dtype=torch.float32, device=self.device
        )
        
        # 填充
        for i in range(num_corrs):
            vertices = corridors[i]
            num_verts = min(len(vertices), self.max_vertices)
            
            if num_verts > 0:
                vert_tensor = torch.tensor(
                    vertices[:num_verts], dtype=torch.float32, device=self.device
                )
                corr_vertices[i, :num_verts, :] = vert_tensor
        
        return corr_vertices
    
    def _create_corridor_mask(self, corridors: List[np.ndarray]) -> torch.Tensor:
        """
        创建走廊mask
        
        Args:
            corridors: 走廊列表
        
        Returns:
            mask: (max_corridors,) 1=有效, 0=padding
        """
        num_corrs = min(len(corridors), self.max_corridors)
        mask = torch.zeros(self.max_corridors, dtype=torch.bool, device=self.device)
        mask[:num_corrs] = True
        return mask
    
    def _compute_reward(
        self, action: np.ndarray, collision: bool
    ) -> Tuple[float, Dict]:
        """
        计算总奖励
        
        注意：DummyAGSACEnvironment已在_compute_base_reward中完整实现了
        所有奖励组件（progress, GDE, collision, step等），这里直接返回
        避免双重计分。
        
        Args:
            action: 执行的动作
            collision: 是否碰撞
        
        Returns:
            total_reward: 总奖励
            reward_info: 奖励详情（包含所有分量）
        """
        # 子类已完整实现所有奖励，直接使用
        total_reward, reward_components = self._compute_base_reward(action, collision)
        
        # 详情（包含所有奖励分量，用于日志和调试）
        reward_info = {
            'total_reward': total_reward,
            **reward_components  # 展开所有奖励分量
        }
        
        return total_reward, reward_info
    
    def _compute_geometric_reward(self) -> float:
        """
        计算几何奖励（基于GDE）
        
        Returns:
            geometric_reward: 几何对齐奖励
        """
        # 需要至少2个点来计算方向
        if len(self.path_history) < 2:
            return 0.0
        
        # 使用最近的N个点计算路径段
        N = min(10, len(self.path_history))
        recent_path = self.path_history[-N:]
        
        # 简化的几何奖励：路径平滑度
        # 计算方向变化的平滑性
        angles = []
        for i in range(1, len(recent_path)):
            dx = recent_path[i][0] - recent_path[i-1][0]
            dy = recent_path[i][1] - recent_path[i-1][1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        if len(angles) > 1:
            # 角度变化的标准差（越小越平滑）
            angle_std = np.std(angles)
            # 转换为奖励（0-1）
            smoothness = np.exp(-angle_std)
            return smoothness
        
        return 0.0
    
    def _check_done(self, collision: bool) -> Tuple[bool, str]:
        """
        检查episode是否结束
        
        Args:
            collision: 是否碰撞
        
        Returns:
            done: 是否结束
            reason: 结束原因
        """
        # 碰撞
        if collision:
            return True, 'collision'
        
        # 超时
        if self.current_step >= self.max_episode_steps:
            return True, 'timeout'
        
        # 到达目标（由子类通过_compute_base_reward中的标志传递）
        # 这里简化处理，子类可以override
        raw_obs = self._get_raw_observation()
        robot_pos = np.array(raw_obs['robot_state']['position'])
        goal_pos = np.array(raw_obs['goal'])
        distance_to_goal = np.linalg.norm(robot_pos - goal_pos)
        
        if distance_to_goal < 1.0:  # 阈值与奖励函数一致
            return True, 'goal_reached'
        
        # 提前终止：连续严重违规corridor约束
        # 如果连续20步都在corridor外且距离>1.0，视为无效探索，提前终止
        if (hasattr(self, 'consecutive_violations') and 
            self.consecutive_violations >= self.consecutive_violation_threshold):
            return True, 'corridor_violation'
        
        return False, 'running'
    
    def get_info(self) -> Dict:
        """
        获取环境信息
        
        Returns:
            info: 环境配置和状态信息
        """
        return {
            'max_pedestrians': self.max_pedestrians,
            'max_corridors': self.max_corridors,
            'max_vertices': self.max_vertices,
            'obs_horizon': self.obs_horizon,
            'pred_horizon': self.pred_horizon,
            'max_episode_steps': self.max_episode_steps,
            'current_step': self.current_step,
            'episode_return': self.episode_return,
            'done': self.done
        }


class DummyAGSACEnvironment(AGSACEnvironment):
    """
    Dummy环境实现（用于测试）
    
    模拟一个简单的2D导航任务:
    - 机器狗从start到goal
    - 固定的行人和走廊
    - 简单的碰撞检测
    """
    
    def __init__(
        self, 
        use_corridor_generator: bool = False,
        curriculum_learning: bool = False,
        scenario_seed: Optional[int] = None,
        corridor_constraint_mode: str = 'soft',  # 新增：'soft', 'medium', 'hard'
        corridor_penalty_weight: float = 10.0,   # 新增：软约束惩罚权重
        corridor_penalty_cap: float = 30.0,      # 新增：corridor惩罚上限（每步）
        progress_reward_weight: float = 20.0,    # 新增：可配置进展奖励权重
        step_penalty_weight: float = 0.01,       # 新增：可配置步数惩罚权重
        enable_step_limit: bool = True,          # 新增：是否启用步长限幅
        **kwargs
    ):
        """
        Args:
            use_corridor_generator: 是否使用自动场景生成器（默认False，使用固定场景）
            curriculum_learning: 是否启用课程学习（从easy到hard）
            scenario_seed: 场景生成的随机种子
            corridor_constraint_mode: Corridor约束模式
                - 'soft': 软约束，偏离惩罚（适合训练初期）
                - 'medium': 中等约束，更大惩罚（适合训练中期）
                - 'hard': 硬约束，离开即碰撞（适合训练后期）
            corridor_penalty_weight: 软约束惩罚权重（分/米）
            corridor_penalty_cap: corridor惩罚上限（每步最多扣多少分）
            progress_reward_weight: 进展奖励权重（默认20.0）
            step_penalty_weight: 步数惩罚权重（默认0.01）
            enable_step_limit: 是否启用步长限幅（防止超冲）
        """
        super().__init__(**kwargs)
        
        # 场景生成器配置
        self.use_corridor_generator = use_corridor_generator
        self.curriculum_learning = curriculum_learning
        self.scenario_seed = scenario_seed
        
        # Corridor约束配置
        self.corridor_constraint_mode = corridor_constraint_mode
        self.corridor_penalty_weight = corridor_penalty_weight
        self.corridor_penalty_cap = corridor_penalty_cap
        
        # 奖励权重配置（可调节）
        self.progress_reward_weight = progress_reward_weight
        self.step_penalty_weight = step_penalty_weight
        
        # 运动控制配置
        self.enable_step_limit = enable_step_limit
        
        # Corridor violation统计
        self.corridor_violations = 0
        self.corridor_violation_distances = []
        self.consecutive_violations = 0  # 连续违规计数
        self.consecutive_violation_threshold = 20  # 连续违规20步提前终止
        
        # Corridor边界框缓存（性能优化）
        self.corridor_bboxes = []  # 存储每个corridor的边界框
        
        if self.use_corridor_generator:
            self.corridor_generator = CorridorGenerator(
                map_size=(12.0, 12.0),
                seed=scenario_seed
            )
            # 课程学习：初始难度（如果禁用课程学习，固定使用easy）
            self.current_difficulty = 'easy' if curriculum_learning else 'easy'
            self.episode_count = 0
        
        # 环境参数（默认固定场景）
        self.start_pos = np.array([0.0, 0.0])
        self.goal_pos = np.array([10.0, 10.0])
        
        # 当前状态
        self.robot_position = self.start_pos.copy()
        self.robot_velocity = np.array([0.0, 0.0])
        self.robot_orientation = 0.0
        
        # 固定的行人（模拟）- 使用min来避免超过max_pedestrians
        self.num_pedestrians = min(3, self.max_pedestrians)
        self.pedestrian_trajectories = []
        
        # 固定的走廊（模拟）- 使用min来避免超过max_corridors
        self.num_corridors = min(2, self.max_corridors)
        self.corridor_data = []
        
        # 静态障碍物（仅用于固定场景）
        self.static_obstacle = None
        
        # 几何微分评估器（方向一致性GDE）
        self.gde = GeometricDifferentialEvaluator(eta=0.3, M=10)
    
    def _reset_env(self) -> Dict:
        """重置环境"""
        
        # ===== 1. 场景生成（动态 vs 固定） =====
        if self.use_corridor_generator:
            # 动态生成场景
            self._generate_dynamic_scenario()
        else:
            # 使用固定场景（向后兼容）
            self._setup_fixed_scenario()
        
        # ===== 2. 重置机器狗状态 =====
        self.robot_position = self.start_pos.copy()
        self.robot_velocity = np.array([0.0, 0.0])
        self.robot_orientation = 0.0
        
        # 重置距离记录（用于计算进展奖励）
        self.last_distance = np.linalg.norm(self.goal_pos - self.start_pos)
        
        # 重置corridor violation统计
        self.corridor_violations = 0
        self.corridor_violation_distances = []
        self.consecutive_violations = 0  # 重置连续违规计数
        
        return {}
    
    def _generate_dynamic_scenario(self):
        """使用生成器动态生成场景（带课程学习）"""
        
        # 课程学习：根据episode数量调整难度和约束模式
        if self.curriculum_learning:
            # 难度渐进
            if self.episode_count < 50:
                self.current_difficulty = 'easy'
            elif self.episode_count < 150:
                self.current_difficulty = 'medium'
            else:
                self.current_difficulty = 'hard'
            
            # Corridor约束渐进：soft → medium → hard
            if self.episode_count < 100:
                self.corridor_constraint_mode = 'soft'
            elif self.episode_count < 300:
                self.corridor_constraint_mode = 'medium'
            else:
                self.corridor_constraint_mode = 'hard'
            
            # Corridor惩罚权重渐进递增（每100 episodes +2，最多到15）
            base_weight = 8.0  # 初始权重
            increment_per_100 = 2.0
            max_weight = 15.0
            increments = min(self.episode_count // 100, 3)  # 最多3次递增
            self.corridor_penalty_weight = min(
                base_weight + increments * increment_per_100,
                max_weight
            )
        
        # 生成场景
        scenario = self.corridor_generator.generate_scenario(
            difficulty=self.current_difficulty
        )
        
        # 更新环境状态
        self.start_pos = scenario['start']
        self.goal_pos = scenario['goal']
        self.corridor_data = scenario['corridors']
        self.static_obstacle = scenario['obstacles'] if scenario['obstacles'] else None
        
        # 预计算corridor边界框（性能优化）
        self._compute_corridor_bboxes()
        
        # 验证起点和终点是否在corridor内（诊断）
        if self.corridor_data:
            start_in = self._is_in_any_corridor(self.start_pos)
            goal_in = self._is_in_any_corridor(self.goal_pos)
            if not start_in:
                print(f"  ⚠️ [警告] 起点不在corridor内! 起点: ({self.start_pos[0]:.2f}, {self.start_pos[1]:.2f})")
            if not goal_in:
                print(f"  ⚠️ [警告] 终点不在corridor内! 终点: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})")
        
        # 更新行人数量和位置（在corridor内随机生成）
        self.num_pedestrians = min(
            scenario['num_pedestrians'],
            self.max_pedestrians
        )
        self.pedestrian_trajectories = self._generate_pedestrian_positions(
            self.num_pedestrians
        )
        
        # 增加episode计数
        self.episode_count += 1
    
    def _setup_fixed_scenario(self):
        """设置固定场景（向后兼容）"""
        
        # 固定起点和终点
        self.start_pos = np.array([0.0, 0.0])
        self.goal_pos = np.array([10.0, 10.0])
        
        # 固定行人位置（修复：让行人远离起点，避免初始碰撞）
        self.pedestrian_trajectories = [
            {'id': 0, 'trajectory': [[3.0, 1.0]]},  # 行人1在侧面
            {'id': 1, 'trajectory': [[5.0, 3.0]]},  # 行人2在中间
            {'id': 2, 'trajectory': [[7.0, 7.0]]},  # 行人3靠近目标
        ][:self.num_pedestrians]  # 只取需要的数量
        
        # 固定走廊：两条从起点到终点的独立可选路线
        # 场景：中间有障碍物(4,3)-(6,7)，提供上下两条绕行路径
        self.corridor_data = [
            # 通路1：上方绕行路线（L型区域）
            np.array([
                [-0.5, -0.5],   # 起点区域外扩（包含0,0）
                [-0.5, 11.0],   # 左上
                [3.5, 11.0],    # 向右（障碍物左侧）
                [3.5, 7.5],     # 向下到障碍物上方
                [6.5, 7.5],     # 越过障碍物
                [6.5, 11.0],    # 向上
                [11.0, 11.0],   # 右上（包含10,10）
                [11.0, -0.5],   # 右下
                [6.5, -0.5],    # 回来
                [6.5, 6.0],     # 内边界（避开障碍物）
                [3.5, 6.0],     # 障碍物左侧
                [3.5, -0.5]     # 回到起点
            ]),
            
            # 通路2：下方绕行路线（L型区域）
            np.array([
                [-0.5, 11.0],   # 起点区域左上
                [-0.5, -0.5],   # 左下（包含0,0）
                [3.5, -0.5],    # 向右
                [3.5, 2.5],     # 向上到障碍物下方
                [6.5, 2.5],     # 越过障碍物
                [6.5, -0.5],    # 向下
                [11.0, -0.5],   # 右下
                [11.0, 11.0],   # 右上（包含10,10）
                [6.5, 11.0],    # 回来
                [6.5, 4.0],     # 内边界（避开障碍物）
                [3.5, 4.0],     # 障碍物左侧
                [3.5, 11.0]     # 回到起点
            ])
        ]
        
        # 静态障碍物
        self.static_obstacle = np.array([
            [4.0, 3.0], [6.0, 3.0], [6.0, 7.0], [4.0, 7.0]
        ])
        
        # 预计算corridor边界框（性能优化）
        self._compute_corridor_bboxes()
    
    def _generate_pedestrian_positions(self, num_pedestrians: int) -> List[Dict]:
        """在corridor内随机生成行人初始位置（确保与起点有安全距离）"""
        pedestrians = []
        
        # 在第一条corridor内随机选择位置
        if self.corridor_data:
            corridor = self.corridor_data[0]
            min_x, max_x = corridor[:, 0].min(), corridor[:, 0].max()
            min_y, max_y = corridor[:, 1].min(), corridor[:, 1].max()
            
            for i in range(num_pedestrians):
                # 随机位置（重试机制确保与起点、终点和其他行人保持安全距离）
                min_safe_distance = 2.5  # 与起点/终点的最小安全距离（米）← 增加到2.5米
                max_attempts = 50
                
                for attempt in range(max_attempts):
                    x = np.random.uniform(min_x + 1, max_x - 1)
                    y = np.random.uniform(min_y + 1, max_y - 1)
                    pos = np.array([x, y])
                    
                    # 检查与起点的距离
                    dist_to_start = np.linalg.norm(pos - self.start_pos)
                    # 检查与终点的距离
                    dist_to_goal = np.linalg.norm(pos - self.goal_pos)
                    # 检查与已有行人的距离
                    too_close_to_others = False
                    for ped in pedestrians:
                        other_pos = np.array(ped['trajectory'][-1])
                        if np.linalg.norm(pos - other_pos) < 1.0:  # 行人之间至少1米
                            too_close_to_others = True
                            break
                    
                    # 如果位置合法，接受
                    if dist_to_start >= min_safe_distance and dist_to_goal >= min_safe_distance and not too_close_to_others:
                        pedestrians.append({
                            'id': i,
                            'trajectory': [[x, y]]
                        })
                        break
                else:
                    # 重试失败，使用备选位置（corridor中心偏移）
                    center_x = (min_x + max_x) / 2 + np.random.uniform(-2, 2)
                    center_y = (min_y + max_y) / 2 + np.random.uniform(-2, 2)
                    pedestrians.append({
                        'id': i,
                        'trajectory': [[center_x, center_y]]
                    })
        
        return pedestrians
    
    def _step_env(self, action: np.ndarray) -> Dict:
        """
        执行动作
        
        Args:
            action: (22,) = 11个路径点(x,y)，或兼容旧版(2,)
        """
        # 路径跟踪简化版：朝向第一个路径点移动
        if len(action) == 22:
            # 标准格式：11个路径点
            path_normalized = action.reshape(11, 2)  # [-1, 1] 范围（假设Actor输出使用tanh）
            
            # 坐标转换：归一化 → 相对 → 全局
            scale = 2.0  # 每个点的最大偏移范围 ±2米
            path_relative = path_normalized * scale  # [-2m, +2m]
            path_global = self.robot_position + path_relative  # 全局坐标
            
            # 保存完整路径用于GDE评估
            self.current_planned_path = path_global.copy()
            
            # 取第一个路径点作为短期目标
            target_point = path_global[0]
            
            # 计算朝向目标的位移
            direction = target_point - self.robot_position
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-6:
                direction = direction / direction_norm
            
            # 以固定速度朝目标移动（带步长限幅）
            speed = 1.5  # m/s （提高3倍速度：1.5 * 0.1 = 0.15米/步）
            dt = 0.1     # s
            max_displacement = speed * dt  # 最大移动距离
            
            # 步长限幅：防止超冲和抖动
            if self.enable_step_limit:
                # 到第一个路径点的剩余距离（修正：基于首点而非最终目标）
                remaining_distance = np.linalg.norm(target_point - self.robot_position)
                # 限制位移不超过到首点的距离
                actual_displacement = min(max_displacement, remaining_distance)
            else:
                actual_displacement = max_displacement
            
            displacement = direction * actual_displacement
        elif len(action) == 2:
            # 兼容旧版：直接位移
            dx = action[0] * 0.1
            dy = action[1] * 0.1
            displacement = np.array([dx, dy])
        else:
            raise ValueError(f"Action维度错误: {len(action)}，应为22或2")
        
        self.robot_position += displacement
        self.robot_velocity = displacement / 0.1  # dt=0.1
        
        # 更新朝向
        if np.linalg.norm(displacement) > 1e-6:
            self.robot_orientation = np.arctan2(displacement[1], displacement[0])
        
        # 更新行人轨迹（模拟移动）
        for ped in self.pedestrian_trajectories:
            # 简单的线性移动
            last_pos = ped['trajectory'][-1]
            new_pos = [last_pos[0] + 0.05, last_pos[1] + 0.05]
            ped['trajectory'].append(new_pos)
            
            # 限制历史长度
            if len(ped['trajectory']) > self.obs_horizon * 2:
                ped['trajectory'] = ped['trajectory'][-self.obs_horizon:]
        
        # 记录路径
        self.path_history.append(self.robot_position.copy())
        
        return {}
    
    def _get_raw_observation(self) -> Dict:
        """获取原始观测"""
        goal_vector = self.goal_pos - self.robot_position
        
        return {
            'robot_state': {
                'position': self.robot_position.tolist(),
                'velocity': self.robot_velocity.tolist(),
                'orientation': float(self.robot_orientation),
                'goal_vector': goal_vector.tolist()
            },
            'pedestrians': self.pedestrian_trajectories,
            'corridors': self.corridor_data,
            'goal': self.goal_pos.tolist(),
            'reference_line': [
                self.start_pos.tolist(),
                self.goal_pos.tolist()
            ]
        }
    
    def _compute_corridor_bboxes(self):
        """
        预计算所有corridor的边界框（性能优化）
        边界框用于快速筛选，避免对每个点都做复杂的polygon检查
        """
        self.corridor_bboxes = []
        for corridor in self.corridor_data:
            if len(corridor) > 0:
                min_x = np.min(corridor[:, 0])
                max_x = np.max(corridor[:, 0])
                min_y = np.min(corridor[:, 1])
                max_y = np.max(corridor[:, 1])
                self.corridor_bboxes.append((min_x, max_x, min_y, max_y))
            else:
                self.corridor_bboxes.append(None)
    
    def _point_in_bbox(self, point: np.ndarray, bbox: tuple) -> bool:
        """
        快速检查点是否在边界框内
        
        Args:
            point: (2,) 点坐标
            bbox: (min_x, max_x, min_y, max_y)
        
        Returns:
            inside: True if point is inside bbox
        """
        if bbox is None:
            return False
        min_x, max_x, min_y, max_y = bbox
        x, y = point
        return min_x <= x <= max_x and min_y <= y <= max_y
    
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
               (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def _is_in_any_corridor(self, point: np.ndarray) -> bool:
        """
        检测点是否在任意corridor内（性能优化：先用bbox筛选）
        
        Args:
            point: (2,) 点坐标
        
        Returns:
            in_corridor: True if in any corridor
        """
        for i, corridor in enumerate(self.corridor_data):
            # 性能优化：先用边界框快速筛选
            if i < len(self.corridor_bboxes) and self.corridor_bboxes[i] is not None:
                if not self._point_in_bbox(point, self.corridor_bboxes[i]):
                    continue  # 不在bbox内，直接跳过
            
            # 在bbox内，才做精确的polygon检查
            if self._point_in_polygon(point, corridor):
                return True
        return False
    
    def _distance_to_nearest_corridor(self, point: np.ndarray) -> float:
        """
        计算点到最近corridor边界的距离
        
        Args:
            point: (2,) 点坐标
        
        Returns:
            min_distance: 到最近corridor边界的距离（米）
        """
        if not self.corridor_data:
            return 0.0
        
        min_dist = float('inf')
        
        for corridor in self.corridor_data:
            # 如果在corridor内，距离为0
            if self._point_in_polygon(point, corridor):
                return 0.0
            
            # 否则计算到每条边的距离
            for i in range(len(corridor)):
                j = (i + 1) % len(corridor)
                edge_start = corridor[i]
                edge_end = corridor[j]
                
                # 点到线段的距离
                dist = self._point_to_segment_distance(point, edge_start, edge_end)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _point_to_segment_distance(
        self, 
        point: np.ndarray, 
        seg_start: np.ndarray, 
        seg_end: np.ndarray
    ) -> float:
        """
        计算点到线段的最短距离
        
        Args:
            point: 点坐标
            seg_start: 线段起点
            seg_end: 线段终点
        
        Returns:
            distance: 最短距离
        """
        # 线段向量
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq < 1e-8:
            # 退化为点
            return np.linalg.norm(point - seg_start)
        
        # 投影参数 t
        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        
        # 线段上最近点
        nearest_point = seg_start + t * seg_vec
        
        return np.linalg.norm(point - nearest_point)
    
    def _check_collision(self) -> bool:
        """碰撞检测（含corridor约束）"""
        # 边界检测（更宽松的边界）
        if np.any(self.robot_position < -5.0) or np.any(self.robot_position > 15.0):
            # 诊断日志：边界碰撞
            if self.current_step <= 1:  # 仅第一步输出
                print(f"  [碰撞] 边界碰撞! 位置: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})")
            return True
        
        # 行人碰撞检测（简化：距离阈值）
        for i, ped in enumerate(self.pedestrian_trajectories):
            if len(ped['trajectory']) > 0:
                ped_pos = np.array(ped['trajectory'][-1])
                dist = np.linalg.norm(self.robot_position - ped_pos)
                if dist < 0.2:  # 碰撞阈值（从0.3降到0.25，更宽容）
                    # 诊断日志：行人碰撞
                    if self.current_step <= 1:  # 仅第一步输出
                        print(f"  [碰撞] 行人碰撞! 行人{i}距离: {dist:.3f}m < 0.2m")
                        print(f"       机器人: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})")
                        print(f"       行人{i}: ({ped_pos[0]:.2f}, {ped_pos[1]:.2f})")
                    return True
        
        # Corridor约束检测（课程学习）
        if hasattr(self, 'corridor_constraint_mode') and \
           self.corridor_constraint_mode == 'hard':
            # 硬约束模式：不在corridor内 = 碰撞
            if self.corridor_data and not self._is_in_any_corridor(self.robot_position):
                if self.current_step <= 1:
                    print(f"  [碰撞] Corridor约束! 位置: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})")
                return True
        
        return False
    
    def _evaluate_path_curvature(self, path: np.ndarray) -> float:
        """
        基于夹角积分的路径平滑度评估
        
        数学原理：
        1. 计算相邻向量：vᵢ = pᵢ₊₁ - pᵢ
        2. 计算转角：θᵢ = arccos(vᵢ·vᵢ₊₁ / (‖vᵢ‖·‖vᵢ₊₁‖))
        3. 总转角（夹角积分）：Θ = Σθᵢ
        4. 指数评分：score = exp(-α·Θ)
        
        Args:
            path: (11, 2) 全局坐标的路径点
        
        Returns:
            score: float ∈ (0, 1]
                1.0 = 完美直线（无转角）
                0.0 = 极度曲折（大量转角）
        """
        if len(path) < 3:
            # 路径太短，无法评估
            return 0.0
        
        # 步骤1：计算相邻向量
        vectors = np.diff(path, axis=0)  # (10, 2)
        
        # 步骤2：计算向量长度
        lengths = np.linalg.norm(vectors, axis=1)  # (10,)
        
        # 过滤零向量（避免除零）
        eps = 1e-6
        valid_mask = lengths > eps
        
        if valid_mask.sum() < 2:
            # 有效向量太少，无法计算转角
            return 0.0
        
        # 步骤3：计算相邻向量之间的转角
        angles = []
        
        for i in range(len(vectors) - 1):
            if valid_mask[i] and valid_mask[i + 1]:
                # 归一化向量
                v1 = vectors[i] / lengths[i]
                v2 = vectors[i + 1] / lengths[i + 1]
                
                # 点积（余弦值）
                cos_theta = np.dot(v1, v2)
                
                # 限制到[-1, 1]（数值稳定性）
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                
                # 转角
                theta = np.arccos(cos_theta)  # [0, π]
                
                angles.append(theta)
        
        if len(angles) == 0:
            return 0.0
        
        # 步骤4：总转角（夹角积分）
        total_angle = np.sum(angles)
        
        # 步骤5：指数评分
        alpha = 1.0  # 衰减系数
        score = np.exp(-alpha * total_angle)  # (0, 1]
        
        return score
    
    def _compute_base_reward(self, action: np.ndarray, collision: bool) -> Tuple[float, Dict]:
        """
        计算基础奖励（改进版本）
        
        Returns:
            total_reward: 总奖励值
            reward_components: 奖励分量字典（用于日志和调试）
        """
        # 当前到目标的距离
        current_distance = np.linalg.norm(self.goal_pos - self.robot_position)
        
        # 计算进展（如果有上一步的距离记录）
        if not hasattr(self, 'last_distance'):
            self.last_distance = np.linalg.norm(self.goal_pos - self.start_pos)
        
        # 进展奖励：向目标靠近得正奖励，远离得负奖励（可配置权重）
        progress = self.last_distance - current_distance
        progress_reward = progress * self.progress_reward_weight  # 默认20.0
        
        # 更新距离记录
        self.last_distance = current_distance
        
        # 到达目标的大奖励
        goal_reached_reward = 0.0
        if current_distance < 1.0:  # 放宽到达阈值（速度更快了）
            goal_reached_reward = 100.0  # 大幅增加到达奖励
        
        # 碰撞惩罚（增加到-100以强调安全重要性）
        collision_penalty = -100.0 if collision else 0.0
        
        # 步数惩罚（可配置权重）
        step_penalty = -self.step_penalty_weight  # 默认-0.01
        
        # ========== GDE评估（路径质量）==========
        direction_reward = 0.0
        direction_score_raw = 0.0
        curvature_reward = 0.0
        curvature_score_raw = 0.0
        
        if hasattr(self, 'current_planned_path'):
            # 1. 方向一致性GDE（朝向目标，修改为对称奖励）
            try:
                path_tensor = torch.from_numpy(self.current_planned_path).float()
                reference_line = torch.from_numpy(self.goal_pos - self.robot_position).float()
                
                direction_score_raw = self.gde(path_tensor, reference_line).item()
                # direction_score ∈ [0, 1]，归一化到[-1, 1]
                direction_normalized = 2.0 * direction_score_raw - 1.0
                # 权重0.3，范围[-0.3, 0.3]
                direction_reward = direction_normalized * 0.3
            except Exception as e:
                # 如果评估失败，不加分也不扣分
                direction_reward = 0.0
            
            # 2. 路径平滑度GDE（曲率评估）
            try:
                curvature_score_raw = self._evaluate_path_curvature(self.current_planned_path)
                # curvature_score ∈ (0, 1]，转换到[-1, 1]再加权
                # score=1.0 → +0.5, score=0.5 → 0, score=0 → -0.5
                normalized_curvature = 2.0 * curvature_score_raw - 1.0
                curvature_reward = normalized_curvature * 0.5
            except Exception as e:
                curvature_reward = 0.0
        
        # ========== Corridor约束惩罚 ==========
        corridor_penalty = 0.0
        corridor_violation_distance = 0.0
        in_corridor = True
        
        # 性能优化：权重为0时跳过检查
        if self.corridor_data and self.corridor_penalty_weight > 0:  
            in_corridor = self._is_in_any_corridor(self.robot_position)
            
            if not in_corridor:
                # 计算到最近corridor的距离
                corridor_violation_distance = self._distance_to_nearest_corridor(self.robot_position)
                
                # 更新连续违规计数（用于提前终止）
                # 只有当距离>1.0且真正严重违规时才计数
                if corridor_violation_distance > 1.0:
                    self.consecutive_violations += 1
                else:
                    self.consecutive_violations = 0  # 轻微违规不算
                
                # 根据约束模式计算惩罚
                if self.corridor_constraint_mode == 'soft':
                    # 软约束：轻微惩罚
                    raw_penalty = -corridor_violation_distance * self.corridor_penalty_weight
                elif self.corridor_constraint_mode == 'medium':
                    # 中等约束：更大惩罚
                    raw_penalty = -corridor_violation_distance * (self.corridor_penalty_weight * 2.0)
                elif self.corridor_constraint_mode == 'hard':
                    # 硬约束：在_check_collision中处理，这里不加惩罚
                    raw_penalty = 0.0
                else:
                    raw_penalty = -corridor_violation_distance * self.corridor_penalty_weight
                
                # 裁剪惩罚：防止单步惩罚过大，保持主导但不失控
                corridor_penalty = max(raw_penalty, -self.corridor_penalty_cap)
            else:
                # 在corridor内，重置连续违规计数
                self.consecutive_violations = 0
        
        # ========== 总奖励 ==========
        total_reward = (
            progress_reward +       # 主导：~10.0 per meter
            direction_reward +      # 方向一致性：-0.3~0.3 (对称)
            curvature_reward +      # 路径平滑度：-0.5~0.5
            corridor_penalty +      # 新增：Corridor约束：0 或 -10~-50
            goal_reached_reward +   # 稀疏：100.0
            collision_penalty +     # 稀疏：-100.0 (增加)
            step_penalty            # -0.01 (增加)
        )
        # 注：删除了distance_penalty（与progress_reward冗余）
        
        # 构建详细的奖励分量信息
        reward_components = {
            'progress_reward': progress_reward,
            'progress_meters': progress,
            'direction_reward': direction_reward,
            'direction_score': direction_score_raw,
            'curvature_reward': curvature_reward,
            'curvature_score': curvature_score_raw,
            'corridor_penalty': corridor_penalty,
            'corridor_violation_distance': corridor_violation_distance,
            'in_corridor': in_corridor,
            'goal_reached_reward': goal_reached_reward,
            'collision_penalty': collision_penalty,
            'step_penalty': step_penalty,
            'current_distance': current_distance
        }
        
        return total_reward, reward_components


# ==================== 内置测试 ====================
if __name__ == '__main__':
    print("测试AGSACEnvironment...")
    
    # 创建环境
    env = DummyAGSACEnvironment(
        max_pedestrians=10,
        max_corridors=10,
        max_vertices=20,
        obs_horizon=8,
        pred_horizon=12,
        max_episode_steps=100,
        use_geometric_reward=True,
        device='cpu'
    )
    
    print("[OK] 环境创建成功")
    print(f"配置: max_peds={env.max_pedestrians}, max_corrs={env.max_corridors}")
    
    # ========== 测试1: Reset ==========
    print("\n1. 测试reset...")
    obs = env.reset()
    
    print(f"[OK] Reset成功")
    print(f"观测keys: {obs.keys()}")
    print(f"robot_state keys: {obs['robot_state'].keys()}")
    print(f"pedestrian_obs shape: {obs['pedestrian_observations'].shape}")
    print(f"pedestrian_mask shape: {obs['pedestrian_mask'].shape}")
    print(f"corridor_vertices shape: {obs['corridor_vertices'].shape}")
    print(f"corridor_mask shape: {obs['corridor_mask'].shape}")
    print(f"goal: {obs['goal']}")
    
    # ========== 测试2: Step ==========
    print("\n2. 测试step...")
    
    # 随机动作
    action = np.random.randn(22)  # action_dim=22
    obs, reward, done, info = env.step(action)
    
    print(f"[OK] Step成功")
    print(f"reward: {reward:.4f}")
    print(f"done: {done}")
    print(f"info keys: {info.keys()}")
    print(f"collision: {info['collision']}")
    print(f"episode_step: {info['episode_step']}")
    print(f"reward详情:")
    for k, v in info.items():
        if 'reward' in k or 'penalty' in k:
            print(f"  {k}: {v:.4f}")
    
    # ========== 测试3: Episode运行 ==========
    print("\n3. 测试完整episode...")
    
    env.reset()
    total_reward = 0.0
    steps = 0
    
    for i in range(50):
        action = np.random.randn(22) * 0.1  # 小动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            print(f"[OK] Episode结束: {info['done_reason']}")
            break
    
    print(f"[OK] Episode完成:")
    print(f"  总步数: {steps}")
    print(f"  总奖励: {total_reward:.4f}")
    print(f"  平均奖励: {total_reward/steps:.4f}")
    
    # ========== 测试4: 观测格式 ==========
    print("\n4. 测试观测格式...")
    
    env.reset()
    obs = env.step(np.random.randn(22))[0]
    
    # 验证类型
    assert isinstance(obs['robot_state']['position'], torch.Tensor)
    assert isinstance(obs['pedestrian_observations'], torch.Tensor)
    assert isinstance(obs['pedestrian_mask'], torch.Tensor)
    assert isinstance(obs['corridor_vertices'], torch.Tensor)
    assert isinstance(obs['corridor_mask'], torch.Tensor)
    
    # 验证shape
    assert obs['pedestrian_observations'].shape == (10, 8, 2)
    assert obs['pedestrian_mask'].shape == (10,)
    assert obs['corridor_vertices'].shape == (10, 20, 2)
    assert obs['corridor_mask'].shape == (10,)
    
    print("[OK] 所有观测格式正确")
    
    # ========== 测试5: 碰撞检测 ==========
    print("\n5. 测试碰撞检测...")
    
    env.reset()
    
    # 移动到边界外（触发碰撞）
    for _ in range(5):
        action = np.ones(22) * 10.0  # 大动作，快速移出边界
        obs, reward, done, info = env.step(action)
        
        if info['collision']:
            print(f"[OK] 检测到碰撞，episode终止")
            assert done == True
            assert info['done_reason'] == 'collision'
            break
    
    # ========== 测试6: 超时检测 ==========
    print("\n6. 测试超时...")
    
    # 重新创建环境（避免状态污染）
    env = DummyAGSACEnvironment(max_episode_steps=10, device='cpu')
    env.reset()
    
    for i in range(15):
        action = np.random.randn(22) * 0.01  # 很小的动作
        obs, reward, done, info = env.step(action)
        
        if done:
            if info['done_reason'] == 'timeout':
                print(f"[OK] 超时终止，步数: {info['episode_step']}")
            else:
                print(f"[INFO] Episode提前结束: {info['done_reason']}，步数: {info['episode_step']}")
            break
    
    # ========== 测试7: 几何奖励 ==========
    print("\n7. 测试几何奖励...")
    
    # 重新创建环境
    env = DummyAGSACEnvironment(
        max_episode_steps=100,
        use_geometric_reward=True,
        device='cpu'
    )
    env.reset()
    
    geometric_rewards = []
    for i in range(20):
        action = np.random.randn(22) * 0.1
        obs, reward, done, info = env.step(action)
        
        if 'geometric_reward' in info:
            geometric_rewards.append(info['geometric_reward'])
        
        if done:
            break
    
    if geometric_rewards:
        avg_geo = np.mean(geometric_rewards)
        print(f"[OK] 平均几何奖励: {avg_geo:.4f}")
        print(f"[OK] 几何奖励范围: [{min(geometric_rewards):.4f}, {max(geometric_rewards):.4f}]")
    
    # ========== 测试8: Info方法 ==========
    print("\n8. 测试get_info...")
    
    info = env.get_info()
    print(f"[OK] 环境信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("[SUCCESS] AGSACEnvironment所有测试通过！")
    print("="*60)

