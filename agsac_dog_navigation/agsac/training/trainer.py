"""
AGSAC Trainer
AGSAC训练器 - 完整的训练循环

功能:
1. 训练循环管理
2. 数据收集（环境交互）
3. 模型更新（SAC训练）
4. 评估与监控
5. Checkpointing
6. 日志记录
"""

import os
import time
import json
import torch
import numpy as np
import random
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from collections import deque

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[Warning] TensorBoard not available. Install with: pip install tensorboard")

from ..models import AGSACModel
from ..envs import AGSACEnvironment
from .replay_buffer import SequenceReplayBuffer


class AGSACTrainer:
    """
    AGSAC训练器
    
    完整的训练流程:
    1. 环境交互收集episode
    2. Episode存入ReplayBuffer
    3. 采样segment batch训练SAC
    4. 定期评估
    5. 保存checkpoint
    """
    
    def __init__(
        self,
        model: AGSACModel,
        env: AGSACEnvironment,
        eval_env: Optional[AGSACEnvironment] = None,
        buffer_capacity: int = 10000,
        seq_len: int = 16,
        burn_in: int = 0,
        batch_size: int = 32,
        warmup_episodes: int = 10,
        updates_per_episode: int = 100,
        eval_interval: int = 10,
        eval_episodes: int = 5,
        save_interval: int = 50,
        log_interval: int = 1,
        max_episodes: int = 1000,
        device: str = 'cpu',
        save_dir: str = './outputs',
        experiment_name: str = 'agsac_experiment',
        use_tensorboard: bool = True,
        eval_seed: Optional[int] = None
    ):
        """
        Args:
            model: AGSACModel实例
            env: 训练环境
            eval_env: 评估环境（可选，默认用训练环境）
            buffer_capacity: ReplayBuffer容量
            seq_len: 序列段长度
            burn_in: Burn-in长度
            batch_size: 批次大小
            warmup_episodes: 预热episodes（开始训练前）
            updates_per_episode: 每episode的更新次数
            eval_interval: 评估间隔（episodes）
            eval_episodes: 每次评估的episodes数
            save_interval: 保存间隔（episodes）
            log_interval: 日志间隔（episodes）
            max_episodes: 最大训练episodes
            device: 设备
            save_dir: 保存目录
            experiment_name: 实验名称
        """
        self.model = model
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        
        # 训练参数
        self.batch_size = batch_size
        self.warmup_episodes = warmup_episodes
        self.updates_per_episode = updates_per_episode
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.max_episodes = max_episodes
        self.device = torch.device(device)
        
        # ReplayBuffer
        self.buffer = SequenceReplayBuffer(
            capacity=buffer_capacity,
            seq_len=seq_len,
            burn_in=burn_in,
            device=device
        )
        
        # 保存路径
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
        self.total_updates = 0
        
        # 训练历史
        self.train_history = {
            'episode_returns': [],
            'episode_lengths': [],
            'eval_returns': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'alpha_values': [],
            # 新增：Episode详细信息
            'done_reasons': [],           # 终止原因: 'collision', 'goal_reached', 'max_steps'
            'collision_types': [],        # 碰撞类型: 'pedestrian', 'corridor', 'boundary', 'none'
            'corridor_violations': [],    # Corridor违规次数
            'avg_progress_reward': [],    # 平均progress奖励
            'avg_corridor_penalty': []    # 平均corridor惩罚
        }
        
        # 最佳模型追踪
        self.best_eval_return = -float('inf')
        
        # 固定评估随机种子（可选）
        self.eval_seed = eval_seed
        
        # 性能分析标志（确保只profiling一次）
        self._profiled_collect = False
        self._profiled_train = False
        self._profiled_episode = False
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        if self.use_tensorboard:
            tensorboard_dir = self.save_dir / 'tensorboard'
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
            print(f"[Trainer] TensorBoard日志: {tensorboard_dir}")
        
        print(f"[Trainer] 初始化完成")
        print(f"  - 实验名称: {experiment_name}")
        print(f"  - 保存目录: {self.save_dir}")
        print(f"  - Buffer容量: {buffer_capacity}")
        print(f"  - 序列长度: {seq_len}")
        print(f"  - Batch大小: {batch_size}")
        print(f"  - TensorBoard: {'启用' if self.use_tensorboard else '禁用'}")
    
    def collect_episode(self, deterministic: bool = False) -> Dict:
        """
        收集一个完整episode
        
        Args:
            deterministic: 是否使用确定性策略
        
        Returns:
            episode_data: {
                'fused_states': List[Tensor],
                'actions': List[Tensor],
                'rewards': List[float],
                'dones': List[bool],
                'hidden_states': List[Dict],
                'geo_scores': List[float],
                'episode_return': float,
                'episode_length': int,
                'start_pos': np.ndarray,
                'goal_pos': np.ndarray,
                'actual_path': List[np.ndarray],
                'done_reason': str
            }
        """
        # 同步episode_count到环境（确保课程学习正确）
        if hasattr(self.env, 'episode_count'):
            self.env.episode_count = self.episode_count
        
        # 重置环境
        obs = self.env.reset()
        
        # 记录起点和终点
        start_pos = self.env.robot_position.copy()
        goal_pos = self.env.goal_pos.copy()
        actual_path = [start_pos.copy()]  # 记录实际走过的路径
        
        # 初始化hidden states
        hidden_states = self.model.init_hidden_states(batch_size=1)
        
        # Episode数据
        observations = []  # 新增：原始观测（用于训练时重编码）
        fused_states = []
        actions = []
        rewards = []
        dones = []
        hidden_states_list = []
        geo_scores = []
        reward_infos = []  # 新增：收集每步的奖励详情
        
        episode_return = 0.0
        episode_length = 0
        done_reason = 'unknown'
        
        done = False
        
        # 性能分析（只在第一次collect时，不管是否resume）
        enable_profile = not self._profiled_collect
        if enable_profile:
            self._profiled_collect = True  # 标记已profiling
            profile_data = {
                'add_batch': [],
                'model_forward': [],
                'data_transfer': [],
                'env_step': []
            }
        
        while not done:
            step_start = time.time() if enable_profile else None
            
            # 添加batch维度
            t0 = time.time() if enable_profile else None
            obs_batch = self._add_batch_dim(obs)
            if enable_profile:
                profile_data['add_batch'].append(time.time() - t0)
            
            # Forward pass（获取fused_state）
            t0 = time.time() if enable_profile else None
            with torch.no_grad():
                model_output = self.model(
                    obs_batch,
                    hidden_states=hidden_states,
                    deterministic=deterministic,
                    return_attention=False
                )
            if enable_profile:
                profile_data['model_forward'].append(time.time() - t0)
            
            # 提取数据
            action = model_output['action'].squeeze(0)  # 移除batch维度
            fused_state = model_output['fused_state'].squeeze(0)
            new_hidden_states = model_output['hidden_states']
            
            # 执行动作
            t0 = time.time() if enable_profile else None
            action_np = action.cpu().numpy()
            if enable_profile:
                profile_data['data_transfer'].append(time.time() - t0)
            
            t0 = time.time() if enable_profile else None
            next_obs, reward, done, info = self.env.step(action_np)
            if enable_profile:
                profile_data['env_step'].append(time.time() - t0)
            
            # 记录当前位置到路径
            actual_path.append(self.env.robot_position.copy())
            
            # 记录结束原因
            if 'done_reason' in info:
                done_reason = info['done_reason']
            
            # 记录数据
            # 新增：保存原始观测（深拷贝并移到CPU，节省GPU显存）
            obs_cpu = self._to_device_observation(obs_batch, 'cpu', deep_copy=True)
            observations.append(obs_cpu)
            fused_states.append(fused_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            hidden_states_list.append(self._copy_hidden_states(hidden_states))
            
            # 收集奖励详情（新增collision_type）
            reward_infos.append({
                'progress_reward': info.get('progress_reward', 0.0),
                'direction_reward': info.get('direction_reward', 0.0),
                'curvature_reward': info.get('curvature_reward', 0.0),
                'corridor_penalty': info.get('corridor_penalty', 0.0),
                'corridor_violation_distance': info.get('corridor_violation_distance', 0.0),
                'in_corridor': info.get('in_corridor', True),
                'goal_reached_reward': info.get('goal_reached_reward', 0.0),
                'collision_penalty': info.get('collision_penalty', 0.0),
                'collision_type': info.get('collision_type', 'none'),  # 新增：碰撞类型
                'step_penalty': info.get('step_penalty', 0.0),
            })
            
            if 'geometric_reward' in info:
                geo_scores.append(info['geometric_reward'])
            else:
                geo_scores.append(0.0)
            
            # 更新状态
            episode_return += reward
            episode_length += 1
            obs = next_obs
            hidden_states = new_hidden_states
            
            # 防止无限循环
            if episode_length >= self.env.max_episode_steps:
                break
        
        # 输出性能分析（只在第一次collect时）
        if enable_profile and profile_data['model_forward']:
            print("\n" + "="*70)
            print(f"🔍 性能分析: 环境交互 (Episode {self.episode_count})")
            print("="*70)
            
            import numpy as np
            
            # 只分析前10步
            max_analyze = min(10, len(profile_data['model_forward']))
            
            print(f"\n前{max_analyze}步详细耗时:")
            print("-"*70)
            for i in range(max_analyze):
                batch_time = profile_data['add_batch'][i] * 1000
                model_time = profile_data['model_forward'][i] * 1000
                transfer_time = profile_data['data_transfer'][i] * 1000
                env_time = profile_data['env_step'][i] * 1000
                total_time = batch_time + model_time + transfer_time + env_time
                
                print(f"Step {i+1:2d}: Total={total_time:6.1f}ms "
                      f"(Batch={batch_time:4.1f}ms + Model={model_time:5.1f}ms + "
                      f"Transfer={transfer_time:4.1f}ms + Env={env_time:6.1f}ms)")
            
            # 统计平均值
            print("\n" + "-"*70)
            print(f"平均值 (基于{max_analyze}步):")
            print("-"*70)
            avg_batch = np.mean(profile_data['add_batch'][:max_analyze]) * 1000
            avg_model = np.mean(profile_data['model_forward'][:max_analyze]) * 1000
            avg_transfer = np.mean(profile_data['data_transfer'][:max_analyze]) * 1000
            avg_env = np.mean(profile_data['env_step'][:max_analyze]) * 1000
            avg_total = avg_batch + avg_model + avg_transfer + avg_env
            
            print(f"  添加Batch:   {avg_batch:6.2f}ms ({avg_batch/avg_total*100:4.1f}%)")
            print(f"  模型推理:    {avg_model:6.2f}ms ({avg_model/avg_total*100:4.1f}%)")
            print(f"  数据传输:    {avg_transfer:6.2f}ms ({avg_transfer/avg_total*100:4.1f}%)")
            print(f"  环境执行:    {avg_env:6.2f}ms ({avg_env/avg_total*100:4.1f}%)")
            print(f"  总计:        {avg_total:6.2f}ms")
            
            # 预估完整episode时间
            print("\n" + "-"*70)
            print("完整Episode预估:")
            print("-"*70)
            print(f"  每步平均: {avg_total:.2f}ms")
            print(f"  50步:  {avg_total * 50 / 1000:.1f}s")
            print(f"  100步: {avg_total * 100 / 1000:.1f}s")
            print(f"  200步: {avg_total * 200 / 1000:.1f}s")
            print("="*70 + "\n")
        
        # 构建episode数据
        episode_data = {
            'observations': observations,  # 新增：原始观测（用于训练）
            'fused_states': fused_states,  # 保留：向后兼容
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'hidden_states': hidden_states_list,
            'geo_scores': geo_scores,
            'reward_infos': reward_infos,  # 新增：奖励详情
            'episode_return': episode_return,
            'episode_length': episode_length,
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'actual_path': actual_path,
            'done_reason': done_reason
        }
        
        return episode_data
    
    def train_step(self) -> Dict[str, float]:
        """
        执行一次训练更新（多个segment batch）
        
        Returns:
            avg_losses: 平均损失
        """
        if len(self.buffer) < self.warmup_episodes:
            return {}
        
        # 额外检查：确保buffer真的有数据（防止空buffer采样）
        if len(self.buffer.episodes) == 0:
            print("[Warning] Buffer为空，跳过训练")
            return {}
        
        actor_losses = []
        critic_losses = []
        alpha_losses = []
        alpha_values = []
        
        # 性能分析（只在第一次训练时，不管是否resume）
        enable_profile = not self._profiled_train
        if enable_profile:
            self._profiled_train = True  # 标记已profiling
            profile_data = {
                'sample': [],
                'update': [],
                'total': []
            }
            print("\n" + "="*70)
            print("性能分析: 训练更新 (前5次)")
            print("="*70)
        
        for i in range(self.updates_per_episode):
            update_start = time.time() if enable_profile else None
            
            # 采样segment batch
            t0 = time.time() if enable_profile else None
            segment_batch = self.buffer.sample(self.batch_size)
            sample_time = (time.time() - t0) if enable_profile else 0
            
            # 重新编码observations（使编码器参与梯度传播）
            t_encode = time.time() if enable_profile else None
            if segment_batch and len(segment_batch[0].get('observations', [])) > 0:
                # 批量重编码observations和next_observations
                for segment in segment_batch:
                    obs_list = segment['observations']
                    next_obs_list = segment['next_observations']
                    
                    # 重新编码（with grad）
                    if obs_list and len(obs_list) > 0:
                        # 移动到模型设备（从CPU搬回GPU）
                        obs_on_device = [self._to_device_observation(obs, str(self.model.device)) for obs in obs_list]
                        next_obs_on_device = [self._to_device_observation(obs, str(self.model.device)) for obs in next_obs_list]
                        
                        # 批量编码（减少Python循环）
                        fused_states_batch = self.model.encode_batch(obs_on_device)  # (seq_len, 64)
                        next_fused_states_batch = self.model.encode_batch(next_obs_on_device)  # (seq_len, 64)
                        
                        # 更新segment的states和next_states
                        segment['states'] = fused_states_batch
                        segment['next_states'] = next_fused_states_batch
            
            encode_time = (time.time() - t_encode) if enable_profile else 0
            
            # SAC更新（同时更新编码器）
            t0 = time.time() if enable_profile else None
            
            # 清空编码器梯度
            if hasattr(self.model, 'encoder_optimizer'):
                self.model.encoder_optimizer.zero_grad()
            
            # SAC更新（内部会组合Critic+Actor loss，一次backward）
            # 编码器梯度来自combined_loss = critic_loss + actor_loss
            # - Critic loss → 学习评估价值的特征
            # - Actor loss → 学习选择动作的特征
            losses = self.model.sac_agent.update(segment_batch)
            
            # 更新编码器参数（使用combined loss的梯度）
            if hasattr(self.model, 'encoder_optimizer'):
                # 梯度裁剪（防止梯度爆炸）
                encoder_params = []
                encoder_params.extend(self.model.dog_encoder.parameters())
                encoder_params.extend(self.model.pointnet.parameters())
                encoder_params.extend(self.model.corridor_encoder.parameters())
                encoder_params.extend(self.model.pedestrian_encoder.parameters())
                encoder_params.extend(self.model.fusion.parameters())
                
                encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                    encoder_params, max_norm=1.0
                )
                
                self.model.encoder_optimizer.step()
            
            update_time = (time.time() - t0) if enable_profile else 0
            
            # 记录编码时间（如果profiling）
            if enable_profile and i < 5:
                if 'encode' not in profile_data:
                    profile_data['encode'] = []
                profile_data['encode'].append(encode_time)
            
            # 记录
            actor_losses.append(losses['actor_loss'])
            critic_losses.append(losses['critic_loss'])
            if 'alpha_loss' in losses:
                alpha_losses.append(losses['alpha_loss'])
            alpha_values.append(losses['alpha'])
            
            self.total_updates += 1
            
            # 记录性能数据（只记录前5次）
            if enable_profile and i < 5:
                total_time = time.time() - update_start
                profile_data['sample'].append(sample_time)
                profile_data['update'].append(update_time)
                profile_data['total'].append(total_time)
        
        # 输出性能分析
        if enable_profile and profile_data['total']:
            print("\n每次更新的时间分布:")
            print("-"*70)
            for i in range(len(profile_data['total'])):
                sample_ms = profile_data['sample'][i] * 1000
                encode_ms = profile_data['encode'][i] * 1000 if 'encode' in profile_data and i < len(profile_data['encode']) else 0
                update_ms = profile_data['update'][i] * 1000
                total_ms = profile_data['total'][i] * 1000
                if encode_ms > 0:
                    print(f"  更新{i+1}: Sample={sample_ms:6.2f}ms  Encode={encode_ms:7.2f}ms  Update={update_ms:8.2f}ms  Total={total_ms:8.2f}ms")
                else:
                    print(f"  更新{i+1}: Sample={sample_ms:7.2f}ms  Update={update_ms:8.2f}ms  Total={total_ms:8.2f}ms")
            
            # 平均值
            avg_sample = np.mean(profile_data['sample']) * 1000
            avg_encode = np.mean(profile_data['encode']) * 1000 if 'encode' in profile_data else 0
            avg_update = np.mean(profile_data['update']) * 1000
            avg_total = np.mean(profile_data['total']) * 1000
            
            print("-"*70)
            print("平均耗时:")
            print(f"  Buffer采样:  {avg_sample:7.2f}ms ({avg_sample/avg_total*100:4.1f}%)")
            if avg_encode > 0:
                print(f"  重新编码:    {avg_encode:7.2f}ms ({avg_encode/avg_total*100:4.1f}%)")
            print(f"  SAC更新:     {avg_update:7.2f}ms ({avg_update/avg_total*100:4.1f}%)")
            print(f"  单次总计:    {avg_total:7.2f}ms")
            
            # 预估完整train_step时间
            estimated_total = avg_total * self.updates_per_episode / 1000
            print(f"\n完整train_step预估 ({self.updates_per_episode}次更新):")
            print(f"  预估总时间: {estimated_total:.1f}s")
            print("="*70 + "\n")
        
        # 计算平均
        avg_losses = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'alpha': np.mean(alpha_values)
        }
        
        if alpha_losses:
            avg_losses['alpha_loss'] = np.mean(alpha_losses)
        
        return avg_losses
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估当前策略
        
        Returns:
            eval_stats: 评估统计
        """
        # 保存当前RNG状态与cudnn配置
        rng_python = random.getstate()
        rng_numpy = np.random.get_state()
        rng_torch = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        cudnn_benchmark = torch.backends.cudnn.benchmark if torch.backends.cudnn.is_available() else None
        cudnn_deterministic = torch.backends.cudnn.deterministic if torch.backends.cudnn.is_available() else None
        
        # 设定评估种子以保证一致性
        if self.eval_seed is not None:
            random.seed(self.eval_seed)
            np.random.seed(self.eval_seed)
            torch.manual_seed(self.eval_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.eval_seed)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        
        self.model.eval()
        
        eval_returns = []
        eval_lengths = []
        
        for _ in range(self.eval_episodes):
            episode_data = self.collect_episode(deterministic=True)
            eval_returns.append(episode_data['episode_return'])
            eval_lengths.append(episode_data['episode_length'])
        
        self.model.train()
        
        # 恢复RNG状态与cudnn配置
        random.setstate(rng_python)
        np.random.set_state(rng_numpy)
        torch.set_rng_state(rng_torch)
        if torch.cuda.is_available() and rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        if torch.backends.cudnn.is_available():
            if cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = cudnn_benchmark
            if cudnn_deterministic is not None:
                torch.backends.cudnn.deterministic = cudnn_deterministic
        
        eval_stats = {
            'eval_return_mean': np.mean(eval_returns),
            'eval_return_std': np.std(eval_returns),
            'eval_length_mean': np.mean(eval_lengths)
        }
        
        return eval_stats
    
    def train(self) -> Dict:
        """
        完整训练循环
        
        Returns:
            train_history: 训练历史
        """
        # 保存起始episode（支持resume）
        start_episode = self.episode_count
        
        print(f"\n{'='*60}")
        print(f"开始训练: {self.max_episodes} episodes")
        if start_episode > 0:
            print(f"从 Episode {start_episode} 继续")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for episode in range(self.max_episodes):
            self.episode_count = start_episode + episode + 1
            
            # 性能分析标记（只在第一次有训练的episode）
            will_train = len(self.buffer) >= self.warmup_episodes
            enable_episode_profile = (not self._profiled_episode and will_train)
            if enable_episode_profile:
                self._profiled_episode = True  # 标记已profiling
            
            # 1. 收集episode
            episode_start = time.time()
            collect_start = time.time()
            episode_data = self.collect_episode(deterministic=False)
            collect_time = time.time() - collect_start
            
            buffer_start = time.time()
            self.buffer.add_episode(episode_data)
            buffer_time = time.time() - buffer_start
            
            episode_return = episode_data['episode_return']
            episode_length = episode_data['episode_length']
            self.total_steps += episode_length
            
            # 记录训练历史
            self.train_history['episode_returns'].append(episode_return)
            self.train_history['episode_lengths'].append(episode_length)
            
            # 2. 训练（如果buffer足够）
            train_losses = {}
            train_time = 0
            if len(self.buffer) >= self.warmup_episodes:
                train_start = time.time()
                train_losses = self.train_step()
                train_time = time.time() - train_start
                
                # 记录losses
                if train_losses:
                    self.train_history['actor_losses'].append(train_losses['actor_loss'])
                    self.train_history['critic_losses'].append(train_losses['critic_loss'])
                    if 'alpha_loss' in train_losses:
                        self.train_history['alpha_losses'].append(train_losses['alpha_loss'])
                    self.train_history['alpha_values'].append(train_losses['alpha'])
            
            episode_time = time.time() - episode_start
            
            # 输出episode级别的性能分析（第一次有训练时）
            if enable_episode_profile:
                other_time = episode_time - collect_time - buffer_time - train_time
                print("\n" + "="*70)
                print("性能分析: Episode级别时间分布")
                print("="*70)
                print(f"Episode长度: {episode_length}步")
                print(f"Buffer大小: {len(self.buffer)}个episodes")
                print("-"*70)
                print(f"  1. 收集Episode:    {collect_time:8.2f}s ({collect_time/episode_time*100:5.1f}%)")
                print(f"  2. 添加到Buffer:   {buffer_time:8.2f}s ({buffer_time/episode_time*100:5.1f}%)")
                print(f"  3. 训练更新:       {train_time:8.2f}s ({train_time/episode_time*100:5.1f}%)")
                print(f"  4. 其他(日志等):   {other_time:8.2f}s ({other_time/episode_time*100:5.1f}%)")
                print("-"*70)
                print(f"  总计:              {episode_time:8.2f}s")
                print("="*70 + "\n")
            
            # 3. 日志（每个episode都输出）
            self._log_episode(episode, episode_return, episode_length, 
                             train_losses, episode_time, episode_data)
            
            # 4. 评估
            if episode % self.eval_interval == 0 and episode > 0:
                eval_stats = self.evaluate()
                self.train_history['eval_returns'].append(eval_stats['eval_return_mean'])
                
                self._log_evaluation(episode, eval_stats)
                
                # 保存最佳模型
                if eval_stats['eval_return_mean'] > self.best_eval_return:
                    self.best_eval_return = eval_stats['eval_return_mean']
                    self.save_checkpoint(is_best=True)
                    print(f"  [Best] 新的最佳模型! Return={self.best_eval_return:.2f}")
            
            # 5. 定期保存
            if episode % self.save_interval == 0 and episode > 0:
                self.save_checkpoint(is_best=False)
        
        total_time = time.time() - start_time
        
        # 最终保存
        self.save_checkpoint(is_best=False, suffix='final')
        self.save_training_history()
        
        # 关闭TensorBoard
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()
            print("[TensorBoard] 日志已保存并关闭")
        
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"  总时间: {total_time/60:.2f} 分钟")
        print(f"  总episodes: {self.episode_count}")
        print(f"  总steps: {self.total_steps}")
        print(f"  总updates: {self.total_updates}")
        print(f"  最佳eval return: {self.best_eval_return:.2f}")
        print(f"{'='*60}\n")
        
        return self.train_history
    
    def _add_batch_dim(self, obs: Dict) -> Dict:
        """
        为观测添加batch维度并转换为AGSACModel期望的格式
        
        环境格式 → 模型格式:
        {
            'robot_state': {...},
            'pedestrian_observations': (10, 8, 2),
            'corridor_vertices': (10, 20, 2),
            ...
        }
        →
        {
            'dog': {
                'trajectory': (1, 8, 2),
                'velocity': (1, 2),
                'position': (1, 2),
                'goal': (1, 2)
            },
            'pedestrians': {
                'trajectories': (1, 10, 8, 2),
                'mask': (1, 10)
            },
            'corridors': {
                'polygons': (1, 10, 20, 2),
                'vertex_counts': (1, 10),
                'mask': (1, 10)
            }
        }
        """
        # 提取机器狗信息
        robot_state = obs['robot_state']
        position = robot_state['position'].unsqueeze(0)  # (1, 2)
        velocity = robot_state['velocity'].unsqueeze(0)  # (1, 2)
        goal = obs['goal'].unsqueeze(0)  # (1, 2)
        
        # 构造trajectory（使用真实path_history而不是重复当前位置）
        obs_horizon = self.env.obs_horizon
        if hasattr(self.env, 'path_history') and len(self.env.path_history) > 0:
            # 从path_history取最近obs_horizon个点
            path_hist = self.env.path_history[-obs_horizon:]
            # 不足时用起点填充
            while len(path_hist) < obs_horizon:
                if hasattr(self.env, 'start_pos'):
                    path_hist.insert(0, self.env.start_pos.copy())
                else:
                    path_hist.insert(0, path_hist[0].copy() if path_hist else position.squeeze(0).cpu().numpy())
            # 优化：先转numpy array再转tensor，避免性能警告
            path_hist_array = np.array(path_hist, dtype=np.float32)  # (obs_horizon, 2)
            trajectory = torch.from_numpy(path_hist_array).to(position.device).unsqueeze(0)  # (1, obs_horizon, 2)
        else:
            # 回退：使用position重复（向后兼容）
            trajectory = position.unsqueeze(1).repeat(1, obs_horizon, 1)  # (1, obs_horizon, 2)
        
        # 构造模型期望的格式
        model_obs = {
            'dog': {
                'trajectory': trajectory,  # (1, obs_horizon, 2)
                'velocity': velocity,      # (1, 2)
                'position': position,      # (1, 2)
                'goal': goal               # (1, 2)
            },
            'pedestrians': {
                'trajectories': obs['pedestrian_observations'].unsqueeze(0),  # (1, max_peds, obs_horizon, 2)
                'mask': obs['pedestrian_mask'].unsqueeze(0)                   # (1, max_peds)
            },
            'corridors': {
                'polygons': obs['corridor_vertices'].unsqueeze(0),  # (1, max_corridors, max_vertices, 2)
                'vertex_counts': self._compute_vertex_counts(obs['corridor_vertices'], obs['corridor_mask']).unsqueeze(0),  # (1, max_corridors)
                'mask': obs['corridor_mask'].unsqueeze(0)          # (1, max_corridors)
            }
        }
        
        return model_obs
    
    def _compute_vertex_counts(self, corridor_vertices: torch.Tensor, 
                               corridor_mask: torch.Tensor) -> torch.Tensor:
        """
        计算每个走廊的实际顶点数
        
        Args:
            corridor_vertices: (max_corridors, max_vertices, 2)
            corridor_mask: (max_corridors,)
        
        Returns:
            vertex_counts: (max_corridors,)
        """
        max_corridors, max_vertices, _ = corridor_vertices.shape
        vertex_counts = torch.zeros(max_corridors, dtype=torch.long, device=corridor_vertices.device)
        
        for i in range(max_corridors):
            if corridor_mask[i]:
                # 找到第一个全零的顶点（表示padding开始）
                vertices = corridor_vertices[i]  # (max_vertices, 2)
                norms = torch.norm(vertices, dim=1)  # (max_vertices,)
                non_zero = (norms > 1e-6).sum()
                vertex_counts[i] = max(non_zero.item(), 4)  # 至少4个顶点（矩形）
            else:
                vertex_counts[i] = max_vertices  # padding的走廊也设置为max_vertices
        
        return vertex_counts
    
    def _copy_hidden_states(self, hidden_states: Dict) -> Dict:
        """深拷贝hidden states"""
        copied = {}
        for key, (h, c) in hidden_states.items():
            copied[key] = (h.clone(), c.clone())
        return copied
    
    def _to_device_observation(self, obs: Dict, device: str, deep_copy: bool = False) -> Dict:
        """
        将observation移动到指定设备
        
        Args:
            obs: observation dict
            device: 目标设备 ('cpu' or 'cuda')
            deep_copy: 是否深拷贝（避免修改原始数据）
        
        Returns:
            obs_on_device: 移动后的observation
        """
        target_device = torch.device(device)
        
        def move_tensor(tensor):
            if deep_copy:
                return tensor.clone().detach().to(target_device)
            else:
                return tensor.to(target_device)
        
        def move_dict(d):
            result = {}
            for key, value in d.items():
                if isinstance(value, torch.Tensor):
                    result[key] = move_tensor(value)
                elif isinstance(value, dict):
                    result[key] = move_dict(value)
                else:
                    result[key] = value
            return result
        
        return move_dict(obs)
    
    def _compute_average_reward_components(self, episode_data: Dict) -> Optional[Dict]:
        """
        计算episode中奖励分量的平均值
        
        Args:
            episode_data: episode数据（包含reward_infos）
        
        Returns:
            avg_rewards: 平均奖励分量字典
        """
        if 'reward_infos' not in episode_data or not episode_data['reward_infos']:
            return None
        
        reward_infos = episode_data['reward_infos']
        
        # 收集所有步骤的奖励分量
        progress_rewards = []
        direction_rewards = []
        curvature_rewards = []
        corridor_penalties = []
        corridor_violations = []
        goal_rewards = []
        collision_penalties = []
        step_penalties = []
        
        for info in reward_infos:
            if 'progress_reward' in info:
                progress_rewards.append(info['progress_reward'])
            if 'direction_reward' in info:
                direction_rewards.append(info['direction_reward'])
            if 'curvature_reward' in info:
                curvature_rewards.append(info['curvature_reward'])
            if 'corridor_penalty' in info:
                corridor_penalties.append(info['corridor_penalty'])
            if 'in_corridor' in info:
                corridor_violations.append(0 if info['in_corridor'] else 1)
            if 'goal_reached_reward' in info:
                goal_rewards.append(info['goal_reached_reward'])
            if 'collision_penalty' in info:
                collision_penalties.append(info['collision_penalty'])
            if 'step_penalty' in info:
                step_penalties.append(info['step_penalty'])
        
        # 计算平均值和统计
        avg_rewards = {
            'progress': np.mean(progress_rewards) if progress_rewards else 0.0,
            'direction': np.mean(direction_rewards) if direction_rewards else 0.0,
            'curvature': np.mean(curvature_rewards) if curvature_rewards else 0.0,
            'corridor': np.mean(corridor_penalties) if corridor_penalties else 0.0,
            'goal': np.sum(goal_rewards) if goal_rewards else 0.0,  # goal是稀疏的，用sum
            'collision': np.sum(collision_penalties) if collision_penalties else 0.0,  # collision是稀疏的，用sum
            'step': np.mean(step_penalties) if step_penalties else 0.0,
            'corridor_violations': np.sum(corridor_violations) if corridor_violations else 0,  # 违规次数
            'corridor_violation_rate': np.mean(corridor_violations) if corridor_violations else 0.0  # 违规率
        }
        
        return avg_rewards
    
    def _log_episode(self, episode: int, episode_return: float, 
                     episode_length: int, train_losses: Dict, 
                     episode_time: float, episode_data: Dict = None):
        """记录episode日志"""
        # 提取路径信息
        if episode_data is not None:
            start_pos = episode_data['start_pos']
            goal_pos = episode_data['goal_pos']
            actual_path = episode_data['actual_path']
            done_reason = episode_data['done_reason']
            
            # 计算实际距离
            path_distance = 0.0
            for i in range(1, len(actual_path)):
                path_distance += np.linalg.norm(actual_path[i] - actual_path[i-1])
            
            # 计算直线距离和最终距离
            straight_distance = np.linalg.norm(goal_pos - start_pos)
            final_distance = np.linalg.norm(goal_pos - actual_path[-1])
            
            # 第一行：基础信息
            log_str = f"[Episode {episode:4d}]"
            log_str += f" Return={episode_return:7.2f}"
            log_str += f" Length={episode_length:3d}"
            log_str += f" Buffer={len(self.buffer):4d}"
            
            # 训练损失
            if train_losses:
                log_str += f" | Actor={train_losses['actor_loss']:.4f}"
                log_str += f" Critic={train_losses['critic_loss']:.4f}"
                log_str += f" Alpha={train_losses['alpha']:.4f}"
            
            # 时间
            log_str += f" | Time={episode_time:.2f}s"
            print(log_str)
            

            # 第二行：路径详情（新增碰撞类型统计）
            path_str = f"  ├─ Start: ({start_pos[0]:5.2f},{start_pos[1]:5.2f})"
            path_str += f" → Goal: ({goal_pos[0]:5.2f},{goal_pos[1]:5.2f})"
            path_str += f" | Dist: {path_distance:5.2f}m (直线:{straight_distance:.2f}m)"
            path_str += f" | 剩余: {final_distance:4.2f}m"
            path_str += f" | {done_reason}"
            
            # 如果是碰撞结束，显示碰撞类型
            if done_reason == 'collision' and 'reward_infos' in episode_data:
                collision_types = [info.get('collision_type', 'none') for info in episode_data['reward_infos']]
                # 找到最后一个非'none'的碰撞类型
                final_collision_type = 'none'
                for ct in reversed(collision_types):
                    if ct != 'none':
                        final_collision_type = ct
                        break
                
                # 映射碰撞类型为中文
                collision_type_map = {
                    'pedestrian': '行人碰撞',
                    'corridor': 'corridor碰撞',
                    'boundary': '边界碰撞',
                    'none': '未知'
                }
                path_str += f" [{collision_type_map.get(final_collision_type, final_collision_type)}]"
            
            print(path_str)
            
            # 第三行：奖励分量详情（新增）
            if 'reward_infos' in episode_data:
                # 计算平均奖励分量（如果有多个step的话）
                avg_rewards = self._compute_average_reward_components(episode_data)
                
                if avg_rewards:
                    reward_str = f"  ├─ Rewards: "
                    reward_str += f"Prog={avg_rewards['progress']:.3f} "
                    reward_str += f"Dir={avg_rewards['direction']:.3f} "
                    reward_str += f"Curv={avg_rewards['curvature']:.3f} "
                    reward_str += f"Corr={avg_rewards['corridor']:.3f} "
                    reward_str += f"Goal={avg_rewards['goal']:.1f} "
                    reward_str += f"Coll={avg_rewards['collision']:.1f} "
                    reward_str += f"Step={avg_rewards['step']:.3f}"
                    print(reward_str)
                    
                    # 第四行：Corridor violation统计
                    if avg_rewards['corridor_violations'] > 0:
                        corridor_str = f"  ├─ Corridor: "
                        corridor_str += f"Violations={int(avg_rewards['corridor_violations'])}/{episode_length} "
                        corridor_str += f"({avg_rewards['corridor_violation_rate']:.1%})"
                        print(corridor_str)
            
            # 第五行：路径点（每5步显示一个，避免太长）
            if len(actual_path) > 2:
                step_interval = max(1, len(actual_path) // 10)  # 最多显示10个点
                sample_indices = list(range(0, len(actual_path), step_interval))
                if sample_indices[-1] != len(actual_path) - 1:
                    sample_indices.append(len(actual_path) - 1)  # 确保包含最后一个点
                
                path_points_str = f"  └─ Path: "
                for idx in sample_indices[:10]:  # 最多显示10个点
                    pos = actual_path[idx]
                    path_points_str += f"({pos[0]:5.2f},{pos[1]:5.2f})"
                    if idx != sample_indices[-1] or idx != len(actual_path) - 1:
                        path_points_str += " → "
                
                if len(sample_indices) > 10:
                    path_points_str += "..."
                print(path_points_str)
            
            # 收集详细数据到train_history（新增）
            self.train_history['done_reasons'].append(done_reason)
            
            # 提取碰撞类型
            final_collision_type = 'none'
            if done_reason == 'collision' and 'reward_infos' in episode_data:
                collision_types = [info.get('collision_type', 'none') for info in episode_data['reward_infos']]
                for ct in reversed(collision_types):
                    if ct != 'none':
                        final_collision_type = ct
                        break
            self.train_history['collision_types'].append(final_collision_type)
            
            # 提取corridor违规和奖励分量
            if 'reward_infos' in episode_data:
                avg_rewards = self._compute_average_reward_components(episode_data)
                if avg_rewards:
                    self.train_history['corridor_violations'].append(int(avg_rewards['corridor_violations']))
                    self.train_history['avg_progress_reward'].append(avg_rewards['progress'])
                    self.train_history['avg_corridor_penalty'].append(avg_rewards['corridor'])
                else:
                    # 没有reward_infos时填充默认值
                    self.train_history['corridor_violations'].append(0)
                    self.train_history['avg_progress_reward'].append(0.0)
                    self.train_history['avg_corridor_penalty'].append(0.0)
            else:
                # 没有reward_infos时填充默认值
                self.train_history['corridor_violations'].append(0)
                self.train_history['avg_progress_reward'].append(0.0)
                self.train_history['avg_corridor_penalty'].append(0.0)
                
        else:
            # 旧版本兼容：只显示基础信息
            log_str = f"[Episode {episode:4d}]"
            log_str += f" Return={episode_return:7.2f}"
            log_str += f" Length={episode_length:3d}"
            log_str += f" Buffer={len(self.buffer):4d}"
            
            if train_losses:
                log_str += f" | Actor={train_losses['actor_loss']:.4f}"
                log_str += f" Critic={train_losses['critic_loss']:.4f}"
                log_str += f" Alpha={train_losses['alpha']:.4f}"
            
            log_str += f" | Time={episode_time:.2f}s"
            print(log_str)
            
            # 旧版本兼容：填充默认值
            self.train_history['done_reasons'].append('unknown')
            self.train_history['collision_types'].append('none')
            self.train_history['corridor_violations'].append(0)
            self.train_history['avg_progress_reward'].append(0.0)
            self.train_history['avg_corridor_penalty'].append(0.0)
        
        # TensorBoard记录
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('train/episode_return', episode_return, episode)
            self.writer.add_scalar('train/episode_length', episode_length, episode)
            self.writer.add_scalar('train/buffer_size', len(self.buffer), episode)
            
            if train_losses:
                self.writer.add_scalar('train/actor_loss', train_losses['actor_loss'], episode)
                self.writer.add_scalar('train/critic_loss', train_losses['critic_loss'], episode)
                self.writer.add_scalar('train/alpha', train_losses['alpha'], episode)
            
            self.writer.add_scalar('train/episode_time', episode_time, episode)
            
            # 奖励分量记录（新增）
            if episode_data is not None and 'reward_infos' in episode_data:
                avg_rewards = self._compute_average_reward_components(episode_data)
                if avg_rewards:
                    self.writer.add_scalar('reward/progress', avg_rewards['progress'], episode)
                    self.writer.add_scalar('reward/direction', avg_rewards['direction'], episode)
                    self.writer.add_scalar('reward/curvature', avg_rewards['curvature'], episode)
                    self.writer.add_scalar('reward/corridor', avg_rewards['corridor'], episode)
                    self.writer.add_scalar('reward/goal', avg_rewards['goal'], episode)
                    self.writer.add_scalar('reward/collision', avg_rewards['collision'], episode)
                    self.writer.add_scalar('reward/step', avg_rewards['step'], episode)
                    
                    # Corridor violation统计
                    self.writer.add_scalar('corridor/violations', avg_rewards['corridor_violations'], episode)
                    self.writer.add_scalar('corridor/violation_rate', avg_rewards['corridor_violation_rate'], episode)
    
    def _log_evaluation(self, episode: int, eval_stats: Dict):
        """记录评估日志"""
        print(f"\n{'='*60}")
        print(f"[Evaluation @ Episode {episode}]")
        print(f"  Mean Return: {eval_stats['eval_return_mean']:.2f} ± {eval_stats['eval_return_std']:.2f}")
        print(f"  Mean Length: {eval_stats['eval_length_mean']:.1f}")
        print(f"{'='*60}\n")
        
        # TensorBoard记录
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('eval/mean_return', eval_stats['eval_return_mean'], episode)
            self.writer.add_scalar('eval/std_return', eval_stats['eval_return_std'], episode)
            self.writer.add_scalar('eval/mean_length', eval_stats['eval_length_mean'], episode)
    
    def save_checkpoint(self, is_best: bool = False, suffix: str = ''):
        """
        保存checkpoint
        
        Args:
            is_best: 是否是最佳模型
            suffix: 文件名后缀
        """
        # 确定文件名
        if is_best:
            filename = 'best_model.pt'
        elif suffix:
            filename = f'checkpoint_{suffix}.pt'
        else:
            filename = f'checkpoint_ep{self.episode_count}.pt'
        
        filepath = self.save_dir / filename
        
        # 保存内容
        checkpoint = {
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'model_state_dict': self.model.state_dict(),
            'best_eval_return': self.best_eval_return,
            'train_history': self.train_history
        }
        
        torch.save(checkpoint, filepath)
        
        if not is_best:
            print(f"  [Save] Checkpoint保存至: {filename}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载checkpoint
        
        Args:
            filepath: checkpoint文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # 恢复状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.episode_count = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        self.total_updates = checkpoint['total_updates']
        self.best_eval_return = checkpoint['best_eval_return']
        self.train_history = checkpoint['train_history']
        
        # 兼容性修复：为旧checkpoint添加新字段（如果缺失）
        if 'done_reasons' not in self.train_history:
            print("[Load] 检测到旧版checkpoint，添加新字段...")
            self.train_history['done_reasons'] = []
            self.train_history['collision_types'] = []
            self.train_history['corridor_violations'] = []
            self.train_history['avg_progress_reward'] = []
            self.train_history['avg_corridor_penalty'] = []
        
        # 同步episode_count到环境（确保resume训练时课程学习正确）
        if hasattr(self.env, 'episode_count'):
            self.env.episode_count = self.episode_count
            print(f"[Load] 同步episode_count到环境: {self.episode_count}")
        
        print(f"[Load] Checkpoint加载成功: {filepath}")
        print(f"  - Episode: {self.episode_count}")
        print(f"  - Total steps: {self.total_steps}")
        print(f"  - Best eval return: {self.best_eval_return:.2f}")
    
    def save_training_history(self):
        """保存训练历史为JSON"""
        filepath = self.save_dir / 'training_history.json'
        
        # 转换为可序列化格式
        history_serializable = {}
        for key, values in self.train_history.items():
            # 区分数值类型和字符串类型
            if key in ['done_reasons', 'collision_types']:
                # 字符串类型，直接保存
                history_serializable[key] = values
            elif key == 'corridor_violations':
                # 整数类型
                history_serializable[key] = [int(v) for v in values]
            else:
                # 浮点数类型
                history_serializable[key] = [float(v) for v in values]
        
        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"[Save] 训练历史保存至: training_history.json")
    
    def get_stats(self) -> Dict:
        """
        获取训练统计信息
        
        Returns:
            stats: 统计字典
        """
        stats = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'buffer_size': len(self.buffer),
            'best_eval_return': self.best_eval_return
        }
        
        # 最近100个episode的统计
        if self.train_history['episode_returns']:
            recent_returns = self.train_history['episode_returns'][-100:]
            stats['recent_return_mean'] = np.mean(recent_returns)
            stats['recent_return_std'] = np.std(recent_returns)
        
        return stats


# ==================== 内置测试 ====================
if __name__ == '__main__':
    print("测试AGSACTrainer...")
    
    # 创建dummy环境和模型
    from ..envs import DummyAGSACEnvironment
    from ..models import AGSACModel
    
    print("\n1. 创建环境和模型...")
    env = DummyAGSACEnvironment(
        max_pedestrians=5,   # 减少数量
        max_corridors=5,     # 减少数量
        max_vertices=10,     # 减少数量
        obs_horizon=8,
        pred_horizon=12,
        max_episode_steps=50,
        device='cpu'
    )
    
    model = AGSACModel(
        dog_feature_dim=64,
        corridor_feature_dim=128,
        social_feature_dim=128,
        pedestrian_feature_dim=64,
        fusion_dim=64,
        action_dim=22,
        max_pedestrians=5,   # 匹配环境
        max_corridors=5,     # 匹配环境
        max_vertices=10,     # 匹配环境
        device='cpu'
    )
    
    print("[OK] 环境和模型创建成功")
    
    # 2. 创建Trainer
    print("\n2. 创建Trainer...")
    trainer = AGSACTrainer(
        model=model,
        env=env,
        buffer_capacity=100,
        seq_len=16,
        batch_size=4,
        warmup_episodes=2,
        updates_per_episode=5,
        eval_interval=5,
        eval_episodes=2,
        save_interval=10,
        log_interval=1,
        max_episodes=10,  # 少量episodes用于测试
        device='cpu',
        save_dir='./test_outputs',
        experiment_name='test_agsac'
    )
    
    print("[OK] Trainer创建成功")
    
    # 3. 测试collect_episode
    print("\n3. 测试collect_episode...")
    episode_data = trainer.collect_episode(deterministic=False)
    
    print(f"[OK] Episode收集成功")
    print(f"  - Episode length: {episode_data['episode_length']}")
    print(f"  - Episode return: {episode_data['episode_return']:.2f}")
    print(f"  - Fused states: {len(episode_data['fused_states'])}")
    print(f"  - Actions: {len(episode_data['actions'])}")
    
    # 4. 测试添加到buffer
    print("\n4. 测试ReplayBuffer...")
    trainer.buffer.add_episode(episode_data)
    print(f"[OK] Episode添加到buffer, 大小: {len(trainer.buffer)}")
    
    # 5. 测试train_step（需要足够的数据）
    print("\n5. 测试train_step...")
    # 收集更多episodes
    for i in range(3):
        ep = trainer.collect_episode(deterministic=False)
        trainer.buffer.add_episode(ep)
    
    print(f"[OK] Buffer大小: {len(trainer.buffer)}")
    
    if len(trainer.buffer) >= trainer.warmup_episodes:
        losses = trainer.train_step()
        print(f"[OK] 训练更新成功")
        print(f"  - Actor loss: {losses['actor_loss']:.4f}")
        print(f"  - Critic loss: {losses['critic_loss']:.4f}")
        print(f"  - Alpha: {losses['alpha']:.4f}")
    
    # 6. 测试evaluate
    print("\n6. 测试evaluate...")
    eval_stats = trainer.evaluate()
    print(f"[OK] 评估完成")
    print(f"  - Mean return: {eval_stats['eval_return_mean']:.2f}")
    print(f"  - Std return: {eval_stats['eval_return_std']:.2f}")
    
    # 7. 测试完整训练循环（短版）
    print("\n7. 测试完整训练循环...")
    history = trainer.train()
    
    print(f"[OK] 训练完成")
    print(f"  - Episodes: {trainer.episode_count}")
    print(f"  - Total steps: {trainer.total_steps}")
    
    # 8. 测试checkpoint
    print("\n8. 测试checkpoint...")
    trainer.save_checkpoint(is_best=False, suffix='test')
    
    # 创建新trainer并加载
    new_trainer = AGSACTrainer(
        model=AGSACModel(device='cpu'),
        env=env,
        device='cpu',
        save_dir='./test_outputs',
        experiment_name='test_agsac'
    )
    
    checkpoint_path = trainer.save_dir / 'checkpoint_test.pt'
    new_trainer.load_checkpoint(str(checkpoint_path))
    
    print(f"[OK] Checkpoint加载成功")
    
    # 9. 测试get_stats
    print("\n9. 测试get_stats...")
    stats = trainer.get_stats()
    print(f"[OK] 统计信息:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.2f}")
        else:
            print(f"  - {k}: {v}")
    
    print("\n" + "="*60)
    print("[SUCCESS] AGSACTrainer所有测试通过！")
    print("="*60)
    
    # 清理测试文件
    import shutil
    test_dir = Path('./test_outputs')
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("\n[Clean] 测试文件已清理")

