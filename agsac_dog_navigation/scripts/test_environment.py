#!/usr/bin/env python3
"""快速测试环境，验证episode长度"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from agsac.envs import DummyAGSACEnvironment

def main():
    print("="*60)
    print("测试DummyEnvironment Episode长度")
    print("="*60)
    
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
    
    print("\n运行5个episodes测试...")
    for ep in range(5):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < 100:
            # 随机动作
            action = np.random.randn(2) * 0.1
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        print(f"Episode {ep+1}: Length={steps:3d}, Reward={total_reward:8.2f}, Done={done}, Reason={info.get('done_reason', 'N/A')}")
    
    print("\n"+"="*60)
    print("测试完成！")
    print("="*60)

if __name__ == "__main__":
    main()

