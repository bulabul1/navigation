"""
简单的Corridor可视化脚本
快速生成corridor场景图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from agsac.envs.corridor_generator import CorridorGenerator


def visualize_scenario(scenario, title="Corridor Scenario", save_path=None):
    """可视化一个完整场景"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 设置地图范围
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 1. 绘制corridors
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']
    for i, corridor in enumerate(scenario['corridors']):
        polygon = Polygon(
            corridor,
            facecolor=colors[i % len(colors)],
            edgecolor='blue',
            linewidth=2,
            alpha=0.5,
            label=f'Corridor {i+1}'
        )
        ax.add_patch(polygon)
        
        # 标注corridor顶点数
        centroid = corridor.mean(axis=0)
        ax.text(centroid[0], centroid[1], f'C{i+1}\n({len(corridor)} vertices)', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 2. 绘制起点和终点
    start = scenario['start']
    goal = scenario['goal']
    
    ax.plot(start[0], start[1], 'go', markersize=20, label='Start', zorder=10)
    ax.plot(goal[0], goal[1], 'r*', markersize=25, label='Goal', zorder=10)
    
    # 标注坐标
    ax.text(start[0], start[1] - 0.5, f'S ({start[0]:.1f}, {start[1]:.1f})', 
            ha='center', fontsize=10, fontweight='bold')
    ax.text(goal[0], goal[1] + 0.5, f'G ({goal[0]:.1f}, {goal[1]:.1f})', 
            ha='center', fontsize=10, fontweight='bold')
    
    # 3. 绘制直线连接（参考）
    ax.plot([start[0], goal[0]], [start[1], goal[1]], 
            'k--', linewidth=1, alpha=0.5, label='Direct Line')
    
    # 4. 计算距离
    distance = np.linalg.norm(goal - start)
    ax.text(6, 11.5, f'Distance: {distance:.2f}m | Corridors: {len(scenario["corridors"])}',
            ha='center', fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax.legend(loc='upper left', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()


def main():
    """生成并可视化场景"""
    print("=" * 60)
    print("Corridor Visualization")
    print("=" * 60)
    
    # 场景1: Easy难度
    print("\n[Scenario 1] Easy Difficulty")
    gen1 = CorridorGenerator(seed=42)
    scene1 = gen1.generate_scenario('easy')
    print(f"  Start: ({scene1['start'][0]:.2f}, {scene1['start'][1]:.2f})")
    print(f"  Goal: ({scene1['goal'][0]:.2f}, {scene1['goal'][1]:.2f})")
    print(f"  Corridors: {len(scene1['corridors'])}")
    visualize_scenario(scene1, "Easy Difficulty", "corridor_easy.png")
    
    # 场景2: Medium难度
    print("\n[Scenario 2] Medium Difficulty")
    gen2 = CorridorGenerator(seed=123)
    scene2 = gen2.generate_scenario('medium')
    print(f"  Start: ({scene2['start'][0]:.2f}, {scene2['start'][1]:.2f})")
    print(f"  Goal: ({scene2['goal'][0]:.2f}, {scene2['goal'][1]:.2f})")
    print(f"  Corridors: {len(scene2['corridors'])}")
    visualize_scenario(scene2, "Medium Difficulty", "corridor_medium.png")
    
    # 场景3: Hard难度
    print("\n[Scenario 3] Hard Difficulty")
    gen3 = CorridorGenerator(seed=456)
    scene3 = gen3.generate_scenario('hard')
    print(f"  Start: ({scene3['start'][0]:.2f}, {scene3['start'][1]:.2f})")
    print(f"  Goal: ({scene3['goal'][0]:.2f}, {scene3['goal'][1]:.2f})")
    print(f"  Corridors: {len(scene3['corridors'])}")
    visualize_scenario(scene3, "Hard Difficulty", "corridor_hard.png")
    
    # 场景4: 固定起点终点（水平）
    print("\n[Scenario 4] Fixed Horizontal Path")
    gen4 = CorridorGenerator(seed=789)
    scene4 = gen4.generate_scenario(
        'medium',
        fixed_start=np.array([2.0, 6.0]),
        fixed_goal=np.array([10.0, 6.0])
    )
    print(f"  Start: ({scene4['start'][0]:.2f}, {scene4['start'][1]:.2f})")
    print(f"  Goal: ({scene4['goal'][0]:.2f}, {scene4['goal'][1]:.2f})")
    print(f"  Corridors: {len(scene4['corridors'])}")
    visualize_scenario(scene4, "Horizontal Path: L-shape Demo", "corridor_horizontal.png")
    
    print("\n" + "=" * 60)
    print("[Complete] Images saved:")
    print("  - corridor_easy.png")
    print("  - corridor_medium.png")
    print("  - corridor_hard.png")
    print("  - corridor_horizontal.png")
    print("=" * 60)


if __name__ == '__main__':
    main()

