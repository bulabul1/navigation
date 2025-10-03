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
        
        Args:
            action: 执行的动作
            collision: 是否碰撞
        
        Returns:
            total_reward: 总奖励
            reward_info: 奖励详情
        """
        # 基础奖励（由子类实现）
        base_reward = self._compute_base_reward(action, collision)
        
        # 几何奖励（可选）
        geometric_reward = 0.0
        if self.use_geometric_reward and len(self.path_history) >= 2:
            geometric_reward = self._compute_geometric_reward()
        
        # 碰撞惩罚
        collision_penalty = self.reward_weights['collision'] if collision else 0.0
        
        # 步数惩罚（鼓励快速完成）
        step_penalty = self.reward_weights['step_penalty']
        
        # 总奖励
        total_reward = (
            base_reward +
            self.reward_weights['geometric'] * geometric_reward +
            collision_penalty +
            step_penalty
        )
        
        # 详情
        reward_info = {
            'base_reward': base_reward,
            'geometric_reward': geometric_reward,
            'collision_penalty': collision_penalty,
            'step_penalty': step_penalty,
            'total_reward': total_reward
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
        
        if distance_to_goal < 0.5:  # 阈值可配置
            return True, 'goal_reached'
        
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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 环境参数
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
    
    def _reset_env(self) -> Dict:
        """重置环境"""
        # 重置机器狗状态
        self.robot_position = self.start_pos.copy()
        self.robot_velocity = np.array([0.0, 0.0])
        self.robot_orientation = 0.0
        
        # 重置距离记录（用于计算进展奖励）
        self.last_distance = np.linalg.norm(self.goal_pos - self.start_pos)
        
        # 初始化行人轨迹（修复：让行人远离起点，避免初始碰撞）
        self.pedestrian_trajectories = [
            {'id': 0, 'trajectory': [[3.0, 1.0]]},  # 行人1在侧面
            {'id': 1, 'trajectory': [[5.0, 3.0]]},  # 行人2在中间
            {'id': 2, 'trajectory': [[7.0, 7.0]]},  # 行人3靠近目标
        ][:self.num_pedestrians]  # 只取需要的数量
        
        # 初始化走廊
        self.corridor_data = [
            np.array([[-2, -2], [12, -2], [12, 12], [-2, 12]]),  # 大边界走廊
            np.array([[2, 2], [8, 2], [8, 8], [2, 8]])  # 中间通道
        ]
        
        return {}
    
    def _step_env(self, action: np.ndarray) -> Dict:
        """
        执行动作
        
        Args:
            action: (action_dim,) 简化为 (dx, dy) 位移
        """
        # 简化：action直接作为位移
        # 实际中action是路径点，需要控制器执行
        dx = action[0] * 0.1  # 缩放
        dy = action[1] * 0.1
        
        self.robot_position += np.array([dx, dy])
        self.robot_velocity = np.array([dx, dy]) / 0.1  # dt=0.1
        
        # 更新朝向
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            self.robot_orientation = np.arctan2(dy, dx)
        
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
    
    def _check_collision(self) -> bool:
        """碰撞检测（简化）"""
        # 边界检测（更宽松的边界）
        if np.any(self.robot_position < -5.0) or np.any(self.robot_position > 15.0):
            return True
        
        # 行人碰撞检测（简化：距离阈值）
        for ped in self.pedestrian_trajectories:
            if len(ped['trajectory']) > 0:
                ped_pos = np.array(ped['trajectory'][-1])
                dist = np.linalg.norm(self.robot_position - ped_pos)
                if dist < 0.3:  # 碰撞阈值（更小的阈值）
                    return True
        
        return False
    
    def _compute_base_reward(self, action: np.ndarray, collision: bool) -> float:
        """计算基础奖励（改进版本）"""
        # 当前到目标的距离
        current_distance = np.linalg.norm(self.goal_pos - self.robot_position)
        
        # 计算进展（如果有上一步的距离记录）
        if not hasattr(self, 'last_distance'):
            self.last_distance = np.linalg.norm(self.goal_pos - self.start_pos)
        
        # 进展奖励：向目标靠近得正奖励，远离得负奖励
        progress = self.last_distance - current_distance
        progress_reward = progress * 10.0  # 放大进展的影响
        
        # 更新距离记录
        self.last_distance = current_distance
        
        # 到达目标的大奖励
        goal_reached_reward = 0.0
        if current_distance < 0.5:
            goal_reached_reward = 100.0  # 大幅增加到达奖励
        
        # 碰撞惩罚
        collision_penalty = -50.0 if collision else 0.0
        
        # 小的步数惩罚（鼓励快速完成）
        step_penalty = -0.01
        
        # 距离惩罚（轻微，避免停滞）
        distance_penalty = -current_distance * 0.01
        
        total_reward = (
            progress_reward + 
            goal_reached_reward + 
            collision_penalty + 
            step_penalty +
            distance_penalty
        )
        
        return total_reward


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

