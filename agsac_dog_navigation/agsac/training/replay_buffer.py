"""
Sequence Replay Buffer
序列段回放缓冲区 - 用于LSTM-based SAC训练

核心功能:
1. 存储完整episode轨迹
2. 采样固定长度的sequence segment
3. 管理hidden state
4. 支持burn-in预热（可选）
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import deque


class SequenceReplayBuffer:
    """
    序列段回放缓冲区
    
    与传统ReplayBuffer的区别:
    - 传统: 存储单个transition，采样独立样本
    - 序列段: 存储完整episode，采样连续的sequence segment
    
    优势:
    - 保留时序信息，充分利用LSTM记忆能力
    - Hidden state在segment内传递
    - 更好的样本效率
    """
    
    def __init__(
        self,
        capacity: int = 10000,      # episode容量
        seq_len: int = 16,          # segment长度
        burn_in: int = 0,           # burn-in长度（可选）
        device: str = 'cpu'
    ):
        """
        Args:
            capacity: 最大存储episode数量
            seq_len: 每个segment的序列长度
            burn_in: burn-in长度（用于预热hidden state）
            device: 设备（cpu/cuda）
        """
        self.capacity = capacity
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.device = torch.device(device)
        
        # 存储完整episode
        self.episodes = deque(maxlen=capacity)
        
        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
    
    def add_episode(self, episode_data: Dict):
        """
        添加完整episode到buffer
        
        Args:
            episode_data: {
                'fused_states': List[Tensor],   # T个状态 (64,)
                'actions': List[Tensor],         # T个动作 (22,)
                'rewards': List[float],          # T个奖励
                'dones': List[bool],             # T个终止标志
                'hidden_states': List[Dict],     # T个隐藏状态
                'geo_scores': List[float],       # T个几何评分（可选）
                'episode_return': float,
                'episode_length': int
            }
        """
        # 验证episode长度
        episode_length = len(episode_data['fused_states'])
        
        if episode_length < self.seq_len:
            print(f"[Warning] Episode长度({episode_length}) < seq_len({self.seq_len}), 跳过")
            return
        
        # 转换为tensor（如果还不是）
        processed_episode = {
            'observations': episode_data.get('observations', []),  # 新增：原始观测（存CPU，节省GPU内存）
            'fused_states': [
                s.to(self.device) if isinstance(s, torch.Tensor) else torch.tensor(s, device=self.device)
                for s in episode_data['fused_states']
            ],
            'actions': [
                a.to(self.device) if isinstance(a, torch.Tensor) else torch.tensor(a, device=self.device)
                for a in episode_data['actions']
            ],
            'rewards': [
                float(r) for r in episode_data['rewards']
            ],
            'dones': [
                bool(d) for d in episode_data['dones']
            ],
            'hidden_states': episode_data['hidden_states'],
            'episode_return': float(episode_data['episode_return']),
            'episode_length': int(episode_length)
        }
        
        # 可选字段
        if 'geo_scores' in episode_data:
            processed_episode['geo_scores'] = [float(g) for g in episode_data['geo_scores']]
        
        # 添加到buffer
        self.episodes.append(processed_episode)
        
        # 更新统计
        self.episode_count += 1
        self.total_steps += episode_length
    
    def sample(self, batch_size: int) -> List[Dict]:
        """
        采样segment batch
        
        Args:
            batch_size: segment数量
        
        Returns:
            segment_batch: List of segments
            [
                {
                    'states': (seq_len, 64),
                    'actions': (seq_len, 22),
                    'rewards': (seq_len,),
                    'next_states': (seq_len, 64),
                    'dones': (seq_len,),
                    'init_hidden': {
                        'actor': (h, c),
                        'critic1': (h, c),
                        'critic2': (h, c)
                    }
                },
                ...
            ]
        """
        if len(self.episodes) == 0:
            raise ValueError("Buffer is empty! Cannot sample.")
        
        segments = []
        
        for _ in range(batch_size):
            # 1. 随机选择一个episode
            episode = random.choice(self.episodes)
            
            # 2. 计算可用的起始位置范围
            # 需要留出seq_len+1的空间（因为要有next_state）
            max_start_idx = episode['episode_length'] - self.seq_len - 1
            
            if max_start_idx < self.burn_in:
                # Episode太短，重新采样
                continue
            
            # 3. 随机选择起始位置（考虑burn-in）
            start_idx = random.randint(self.burn_in, max_start_idx)
            end_idx = start_idx + self.seq_len
            
            # 4. 提取segment
            segment = {
                'observations': episode['observations'][start_idx:end_idx] if 'observations' in episode else [],  # 新增
                'next_observations': episode['observations'][start_idx+1:end_idx+1] if 'observations' in episode else [],  # 新增
                'states': torch.stack(
                    episode['fused_states'][start_idx:end_idx]
                ),  # (seq_len, 64) - 保留兼容
                
                'actions': torch.stack(
                    episode['actions'][start_idx:end_idx]
                ),  # (seq_len, 22)
                
                'rewards': torch.tensor(
                    episode['rewards'][start_idx:end_idx],
                    dtype=torch.float32,
                    device=self.device
                ),  # (seq_len,)
                
                'next_states': torch.stack(
                    episode['fused_states'][start_idx+1:end_idx+1]
                ),  # (seq_len, 64) - 保留兼容
                
                'dones': torch.tensor(
                    episode['dones'][start_idx:end_idx],
                    dtype=torch.float32,
                    device=self.device
                ),  # (seq_len,)
                
                'init_hidden': episode['hidden_states'][start_idx]
            }
            
            # 5. Burn-in处理（如果启用）
            if self.burn_in > 0 and start_idx >= self.burn_in:
                # 提取burn-in段用于预热hidden state
                burn_in_start = start_idx - self.burn_in
                segment['burn_in'] = {
                    'states': torch.stack(
                        episode['fused_states'][burn_in_start:start_idx]
                    ),  # (burn_in, 64)
                    'actions': torch.stack(
                        episode['actions'][burn_in_start:start_idx]
                    ),  # (burn_in, 22)
                    'init_hidden': episode['hidden_states'][burn_in_start]
                }
            
            segments.append(segment)
        
        # 确保采样到了足够的segment
        if len(segments) < batch_size:
            # 递归补充
            remaining = batch_size - len(segments)
            segments.extend(self.sample(remaining))
        
        return segments[:batch_size]
    
    def sample_sequence(self, episode_idx: Optional[int] = None) -> Dict:
        """
        采样单个完整episode序列（用于评估或可视化）
        
        Args:
            episode_idx: episode索引，None则随机选择
        
        Returns:
            完整episode数据
        """
        if len(self.episodes) == 0:
            raise ValueError("Buffer is empty!")
        
        if episode_idx is None:
            episode = random.choice(self.episodes)
        else:
            episode = self.episodes[episode_idx]
        
        return episode
    
    def __len__(self) -> int:
        """返回当前buffer中的episode数量"""
        return len(self.episodes)
    
    def size(self) -> int:
        """返回当前buffer中的episode数量"""
        return len(self.episodes)
    
    def get_stats(self) -> Dict:
        """
        获取buffer统计信息
        
        Returns:
            {
                'num_episodes': int,
                'total_steps': int,
                'avg_episode_length': float,
                'avg_episode_return': float,
                'capacity_usage': float
            }
        """
        if len(self.episodes) == 0:
            return {
                'num_episodes': 0,
                'total_steps': 0,
                'avg_episode_length': 0.0,
                'avg_episode_return': 0.0,
                'capacity_usage': 0.0
            }
        
        total_length = sum(ep['episode_length'] for ep in self.episodes)
        total_return = sum(ep['episode_return'] for ep in self.episodes)
        
        return {
            'num_episodes': len(self.episodes),
            'total_steps': total_length,
            'avg_episode_length': total_length / len(self.episodes),
            'avg_episode_return': total_return / len(self.episodes),
            'capacity_usage': len(self.episodes) / self.capacity
        }
    
    def clear(self):
        """清空buffer"""
        self.episodes.clear()
        self.episode_count = 0
        self.total_steps = 0


# ==================== 内置测试 ====================
if __name__ == '__main__':
    print("测试SequenceReplayBuffer...")
    
    # 创建buffer
    buffer = SequenceReplayBuffer(
        capacity=100,
        seq_len=16,
        burn_in=4,
        device='cpu'
    )
    
    print(f"Buffer配置: capacity={buffer.capacity}, seq_len={buffer.seq_len}, burn_in={buffer.burn_in}")
    
    # ========== 测试1: 添加episode ==========
    print("\n1. 测试添加episode...")
    
    # 模拟一个episode
    episode_length = 50
    hidden_dim = 128
    
    episode_data = {
        'fused_states': [torch.randn(64) for _ in range(episode_length)],
        'actions': [torch.randn(22) for _ in range(episode_length)],
        'rewards': [random.random() for _ in range(episode_length)],
        'dones': [False] * (episode_length - 1) + [True],
        'hidden_states': [
            {
                'actor': (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim)),
                'critic1': (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim)),
                'critic2': (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
            }
            for _ in range(episode_length)
        ],
        'geo_scores': [random.random() for _ in range(episode_length)],
        'episode_return': 100.5,
        'episode_length': episode_length
    }
    
    buffer.add_episode(episode_data)
    print(f"[OK] 添加episode成功, buffer大小: {len(buffer)}")
    
    # 添加更多episode
    for i in range(5):
        ep_len = random.randint(30, 60)
        ep_data = {
            'fused_states': [torch.randn(64) for _ in range(ep_len)],
            'actions': [torch.randn(22) for _ in range(ep_len)],
            'rewards': [random.random() for _ in range(ep_len)],
            'dones': [False] * (ep_len - 1) + [True],
            'hidden_states': [
                {
                    'actor': (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim)),
                    'critic1': (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim)),
                    'critic2': (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
                }
                for _ in range(ep_len)
            ],
            'episode_return': random.uniform(50, 150),
            'episode_length': ep_len
        }
        buffer.add_episode(ep_data)
    
    print(f"[OK] 总共添加了 {len(buffer)} 个episodes")
    
    # ========== 测试2: 采样segment ==========
    print("\n2. 测试采样segment...")
    
    batch_size = 4
    segments = buffer.sample(batch_size)
    
    print(f"[OK] 采样了 {len(segments)} 个segments")
    
    # 验证segment结构
    seg = segments[0]
    print(f"[OK] Segment keys: {seg.keys()}")
    print(f"[OK] states shape: {seg['states'].shape}")
    print(f"[OK] actions shape: {seg['actions'].shape}")
    print(f"[OK] rewards shape: {seg['rewards'].shape}")
    print(f"[OK] next_states shape: {seg['next_states'].shape}")
    print(f"[OK] dones shape: {seg['dones'].shape}")
    print(f"[OK] init_hidden keys: {seg['init_hidden'].keys()}")
    
    assert seg['states'].shape == (16, 64), f"states shape错误: {seg['states'].shape}"
    assert seg['actions'].shape == (16, 22), f"actions shape错误: {seg['actions'].shape}"
    assert seg['rewards'].shape == (16,), f"rewards shape错误: {seg['rewards'].shape}"
    assert seg['next_states'].shape == (16, 64), f"next_states shape错误: {seg['next_states'].shape}"
    assert seg['dones'].shape == (16,), f"dones shape错误: {seg['dones'].shape}"
    
    # ========== 测试3: Burn-in ==========
    print("\n3. 测试burn-in...")
    
    if 'burn_in' in seg:
        print(f"[OK] Burn-in存在")
        print(f"[OK] burn_in states shape: {seg['burn_in']['states'].shape}")
        print(f"[OK] burn_in actions shape: {seg['burn_in']['actions'].shape}")
        assert seg['burn_in']['states'].shape == (4, 64), "burn-in shape错误"
    else:
        print("[INFO] 当前segment没有burn-in（可能起始位置太早）")
    
    # ========== 测试4: 统计信息 ==========
    print("\n4. 测试统计信息...")
    
    stats = buffer.get_stats()
    print(f"[OK] Buffer统计:")
    print(f"  - Episodes数量: {stats['num_episodes']}")
    print(f"  - 总步数: {stats['total_steps']}")
    print(f"  - 平均episode长度: {stats['avg_episode_length']:.1f}")
    print(f"  - 平均episode回报: {stats['avg_episode_return']:.2f}")
    print(f"  - 容量使用率: {stats['capacity_usage']*100:.1f}%")
    
    # ========== 测试5: 边界情况 ==========
    print("\n5. 测试边界情况...")
    
    # 太短的episode（应该被跳过）
    short_episode = {
        'fused_states': [torch.randn(64) for _ in range(10)],
        'actions': [torch.randn(22) for _ in range(10)],
        'rewards': [1.0] * 10,
        'dones': [False] * 9 + [True],
        'hidden_states': [
            {'actor': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128)),
             'critic1': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128)),
             'critic2': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))}
            for _ in range(10)
        ],
        'episode_return': 10.0,
        'episode_length': 10
    }
    
    before_size = len(buffer)
    buffer.add_episode(short_episode)
    after_size = len(buffer)
    
    if before_size == after_size:
        print("[OK] 正确跳过了太短的episode")
    else:
        print(f"[WARNING] 预期跳过短episode，但buffer大小变化了")
    
    # ========== 测试6: 清空buffer ==========
    print("\n6. 测试清空buffer...")
    
    buffer.clear()
    print(f"[OK] Buffer已清空, 大小: {len(buffer)}")
    assert len(buffer) == 0, "清空失败"
    
    # ========== 测试7: Batch一致性 ==========
    print("\n7. 测试batch一致性...")
    
    # 重新添加episodes
    for _ in range(10):
        ep_len = 40
        buffer.add_episode({
            'fused_states': [torch.randn(64) for _ in range(ep_len)],
            'actions': [torch.randn(22) for _ in range(ep_len)],
            'rewards': [1.0] * ep_len,
            'dones': [False] * (ep_len-1) + [True],
            'hidden_states': [
                {'actor': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128)),
                 'critic1': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128)),
                 'critic2': (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))}
                for _ in range(ep_len)
            ],
            'episode_return': float(ep_len),
            'episode_length': ep_len
        })
    
    # 采样多个batch
    for _ in range(3):
        segments = buffer.sample(8)
        assert len(segments) == 8, f"Batch size不一致: {len(segments)}"
        for seg in segments:
            assert seg['states'].shape[0] == 16, "Segment长度不一致"
    
    print("[OK] Batch一致性测试通过")
    
    print("\n" + "="*60)
    print("[SUCCESS] SequenceReplayBuffer所有测试通过！")
    print("="*60)

