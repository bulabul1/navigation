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
        use_tensorboard: bool = True
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
            'alpha_values': []
        }
        
        # 最佳模型追踪
        self.best_eval_return = -float('inf')
        
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
                'episode_length': int
            }
        """
        # 重置环境
        obs = self.env.reset()
        
        # 初始化hidden states
        hidden_states = self.model.init_hidden_states(batch_size=1)
        
        # Episode数据
        fused_states = []
        actions = []
        rewards = []
        dones = []
        hidden_states_list = []
        geo_scores = []
        
        episode_return = 0.0
        episode_length = 0
        
        done = False
        
        while not done:
            # 添加batch维度
            obs_batch = self._add_batch_dim(obs)
            
            # Forward pass（获取fused_state）
            with torch.no_grad():
                model_output = self.model(
                    obs_batch,
                    hidden_states=hidden_states,
                    deterministic=deterministic,
                    return_attention=False
                )
            
            # 提取数据
            action = model_output['action'].squeeze(0)  # 移除batch维度
            fused_state = model_output['fused_state'].squeeze(0)
            new_hidden_states = model_output['hidden_states']
            
            # 执行动作
            action_np = action.cpu().numpy()
            next_obs, reward, done, info = self.env.step(action_np)
            
            # 记录数据
            fused_states.append(fused_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            hidden_states_list.append(self._copy_hidden_states(hidden_states))
            
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
        
        # 构建episode数据
        episode_data = {
            'fused_states': fused_states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'hidden_states': hidden_states_list,
            'geo_scores': geo_scores,
            'episode_return': episode_return,
            'episode_length': episode_length
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
        
        actor_losses = []
        critic_losses = []
        alpha_losses = []
        alpha_values = []
        
        for _ in range(self.updates_per_episode):
            # 采样segment batch
            segment_batch = self.buffer.sample(self.batch_size)
            
            # SAC更新
            losses = self.model.sac_agent.update(segment_batch)
            
            # 记录
            actor_losses.append(losses['actor_loss'])
            critic_losses.append(losses['critic_loss'])
            if 'alpha_loss' in losses:
                alpha_losses.append(losses['alpha_loss'])
            alpha_values.append(losses['alpha'])
            
            self.total_updates += 1
        
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
        self.model.eval()
        
        eval_returns = []
        eval_lengths = []
        
        for _ in range(self.eval_episodes):
            episode_data = self.collect_episode(deterministic=True)
            eval_returns.append(episode_data['episode_return'])
            eval_lengths.append(episode_data['episode_length'])
        
        self.model.train()
        
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
        print(f"\n{'='*60}")
        print(f"开始训练: {self.max_episodes} episodes")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for episode in range(self.max_episodes):
            self.episode_count = episode + 1
            
            # 1. 收集episode
            episode_start = time.time()
            episode_data = self.collect_episode(deterministic=False)
            self.buffer.add_episode(episode_data)
            
            episode_return = episode_data['episode_return']
            episode_length = episode_data['episode_length']
            self.total_steps += episode_length
            
            # 记录训练历史
            self.train_history['episode_returns'].append(episode_return)
            self.train_history['episode_lengths'].append(episode_length)
            
            # 2. 训练（如果buffer足够）
            train_losses = {}
            if len(self.buffer) >= self.warmup_episodes:
                train_losses = self.train_step()
                
                # 记录losses
                if train_losses:
                    self.train_history['actor_losses'].append(train_losses['actor_loss'])
                    self.train_history['critic_losses'].append(train_losses['critic_loss'])
                    if 'alpha_loss' in train_losses:
                        self.train_history['alpha_losses'].append(train_losses['alpha_loss'])
                    self.train_history['alpha_values'].append(train_losses['alpha'])
            
            episode_time = time.time() - episode_start
            
            # 3. 日志
            if episode % self.log_interval == 0:
                self._log_episode(episode, episode_return, episode_length, 
                                 train_losses, episode_time)
            
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
        
        # 构造trajectory（使用position复制obs_horizon次）
        # 注：实际应用中应维护真实的历史轨迹，这里简化处理
        obs_horizon = self.env.obs_horizon
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
    
    def _log_episode(self, episode: int, episode_return: float, 
                     episode_length: int, train_losses: Dict, 
                     episode_time: float):
        """记录episode日志"""
        # 基础信息
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

