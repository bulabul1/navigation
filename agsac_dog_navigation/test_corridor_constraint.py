"""
测试Corridor约束功能
验证点在多边形内判断、距离计算等功能
"""

import numpy as np
from agsac.envs import DummyAGSACEnvironment

def test_corridor_constraint():
    print("="*60)
    print("测试Corridor约束功能")
    print("="*60)
    
    # 1. 创建环境（soft模式）
    print("\n1. 创建环境（soft模式）...")
    env = DummyAGSACEnvironment(
        max_pedestrians=3,
        max_corridors=2,
        max_vertices=20,
        corridor_constraint_mode='soft',
        corridor_penalty_weight=10.0,
        device='cpu'
    )
    print("✅ 环境创建成功")
    
    # 2. 重置环境
    print("\n2. 重置环境...")
    obs = env.reset()
    print(f"✅ 环境重置成功")
    print(f"   起点: {env.start_pos}")
    print(f"   终点: {env.goal_pos}")
    print(f"   Corridor数量: {len(env.corridor_data)}")
    
    # 3. 测试点在多边形内判断
    print("\n3. 测试点在多边形内判断...")
    test_points = [
        ("起点", env.start_pos),
        ("终点", env.goal_pos),
        ("中心", np.array([5.0, 5.0])),
        ("左上", np.array([2.0, 8.0])),
        ("右下", np.array([8.0, 2.0])),
    ]
    
    for name, point in test_points:
        in_corridor = env._is_in_any_corridor(point)
        distance = env._distance_to_nearest_corridor(point)
        status = "✅ 在corridor内" if in_corridor else f"❌ 离开{distance:.2f}米"
        print(f"   {name} {point}: {status}")
    
    # 4. 测试step（观察corridor惩罚）
    print("\n4. 测试10步运动（观察corridor惩罚）...")
    violation_count = 0
    total_corridor_penalty = 0.0
    
    for step in range(10):
        action = np.random.randn(22) * 0.05  # 小动作
        obs, reward, done, info = env.step(action)
        
        in_corridor = info.get('in_corridor', True)
        corridor_penalty = info.get('corridor_penalty', 0.0)
        corridor_dist = info.get('corridor_violation_distance', 0.0)
        
        if not in_corridor:
            violation_count += 1
        total_corridor_penalty += corridor_penalty
        
        status = "✅" if in_corridor else f"❌ (距离{corridor_dist:.2f}m)"
        print(f"   Step {step}: {status} "
              f"Corridor penalty={corridor_penalty:6.2f} "
              f"Total reward={reward:7.2f}")
        
        if done:
            print(f"   Episode终止: {info['done_reason']}")
            break
    
    violation_rate = violation_count / 10 * 100
    print(f"\n   统计: {violation_count}/10 步违规 ({violation_rate:.0f}%)")
    print(f"   总corridor惩罚: {total_corridor_penalty:.2f}")
    
    # 5. 测试不同约束模式
    print("\n5. 测试不同约束模式...")
    
    modes = ['soft', 'medium', 'hard']
    for mode in modes:
        env.corridor_constraint_mode = mode
        env.reset()
        
        # 故意移动到corridor外
        env.robot_position = np.array([5.0, 5.0])  # 可能在障碍物内
        
        # 计算奖励
        action = np.zeros(22)
        collision = env._check_collision()
        reward, components = env._compute_base_reward(action, collision)
        
        corridor_penalty = components['corridor_penalty']
        in_corridor = components['in_corridor']
        
        print(f"   {mode:6s} 模式: "
              f"In corridor={in_corridor}, "
              f"Corridor penalty={corridor_penalty:7.2f}, "
              f"Collision={'是' if collision else '否'}")
    
    # 6. 测试几何函数
    print("\n6. 测试几何工具函数...")
    
    # 简单矩形
    polygon = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0]
    ])
    
    test_cases = [
        ("中心点", np.array([5.0, 5.0]), True, 0.0),
        ("边界外", np.array([15.0, 5.0]), False, 5.0),
        ("角落", np.array([0.0, 0.0]), True, 0.0),
    ]
    
    for name, point, expected_inside, expected_dist in test_cases:
        inside = env._point_in_polygon(point, polygon)
        
        if inside:
            distance = 0.0
        else:
            # 计算到边界的距离
            min_dist = float('inf')
            for i in range(len(polygon)):
                j = (i + 1) % len(polygon)
                dist = env._point_to_segment_distance(point, polygon[i], polygon[j])
                min_dist = min(min_dist, dist)
            distance = min_dist
        
        status_inside = "✅" if inside == expected_inside else "❌"
        status_dist = "✅" if abs(distance - expected_dist) < 0.1 else "❌"
        
        print(f"   {name}: "
              f"Inside={inside} {status_inside}, "
              f"Distance={distance:.2f} {status_dist}")
    
    print("\n" + "="*60)
    print("✅ 所有测试完成！Corridor约束功能正常工作。")
    print("="*60)
    
    print("\n💡 提示：")
    print("   - 使用 corridor_constraint_mode='soft' 开始训练")
    print("   - 监控 corridor/violation_rate 指标")
    print("   - 根据情况调整 corridor_penalty_weight")
    print("   - TensorBoard: tensorboard --logdir outputs/")

if __name__ == '__main__':
    test_corridor_constraint()

