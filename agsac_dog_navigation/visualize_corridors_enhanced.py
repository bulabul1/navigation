"""
增强的Corridor可视化脚本
显示L型通道的绕行路径
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from agsac.envs.corridor_generator import CorridorGenerator


def visualize_corridor_with_paths(scenario, title="Corridor with Path", save_path=None):
    """可视化corridor并标注可行路径"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 设置地图范围
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Y (meters)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    start = scenario['start']
    goal = scenario['goal']
    
    # 1. 绘制corridors
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']
    for i, corridor in enumerate(scenario['corridors']):
        polygon = Polygon(
            corridor,
            facecolor=colors[i % len(colors)],
            edgecolor='darkblue',
            linewidth=3,
            alpha=0.4,
            label=f'Corridor {i+1} ({len(corridor)} vertices)'
        )
        ax.add_patch(polygon)
        
        # 绘制corridor边界顶点（小圆点）
        ax.plot(corridor[:, 0], corridor[:, 1], 'b.', markersize=8, alpha=0.6)
    
    # 2. 绘制起点和终点
    ax.plot(start[0], start[1], 'go', markersize=25, label='Start', zorder=10, 
            markeredgecolor='darkgreen', markeredgewidth=3)
    ax.plot(goal[0], goal[1], 'r*', markersize=35, label='Goal', zorder=10,
            markeredgecolor='darkred', markeredgewidth=2)
    
    # 标注坐标（更大字体）
    ax.text(start[0], start[1] - 0.7, f'S\n({start[0]:.1f}, {start[1]:.1f})', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', linewidth=2))
    ax.text(goal[0], goal[1] + 0.7, f'G\n({goal[0]:.1f}, {goal[1]:.1f})', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))
    
    # 3. 绘制直线连接（参考，虚线）
    ax.plot([start[0], goal[0]], [start[1], goal[1]], 
            'k--', linewidth=2, alpha=0.4, label='Direct Line (reference)')
    
    # 4. 分析并标注corridor类型
    for i, corridor in enumerate(scenario['corridors']):
        centroid = corridor.mean(axis=0)
        
        # 判断corridor类型
        if len(corridor) == 4:
            corridor_type = "Direct\n(Straight)"
        elif len(corridor) == 8:
            # 判断是上绕还是下绕
            max_y = corridor[:, 1].max()
            min_y = corridor[:, 1].min()
            mid_y = (start[1] + goal[1]) / 2
            
            if max_y > max(start[1], goal[1]) + 1:
                corridor_type = "Upper Detour\n(L-shape)"
            elif min_y < min(start[1], goal[1]) - 1:
                corridor_type = "Lower Detour\n(L-shape)"
            else:
                corridor_type = "L-shape"
        else:
            corridor_type = f"{len(corridor)} vertices"
        
        ax.text(centroid[0], centroid[1], f'C{i+1}\n{corridor_type}', 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=colors[i % len(colors)], linewidth=2, alpha=0.9))
    
    # 5. 添加场景信息
    distance = np.linalg.norm(goal - start)
    info_text = (f'Direct Distance: {distance:.2f}m\n'
                 f'Corridors: {len(scenario["corridors"])} paths\n'
                 f'Pedestrians: {scenario["num_pedestrians"]}')
    ax.text(6, 11.2, info_text,
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', 
                     edgecolor='orange', linewidth=2, alpha=0.9))
    
    # 6. 添加图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='black', fancybox=True)
    
    # 7. 添加说明文字
    explanation = (
        "Corridor = 允许通行的路径区域\n"
        "机器狗必须在彩色区域内移动\n"
        "L-shape = 绕行路径（避开障碍）\n"
        "Direct = 直线路径（最短距离）"
    )
    ax.text(0.5, 0.5, explanation,
            ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                     edgecolor='black', linewidth=1, alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[保存] {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()


def main():
    """生成增强可视化"""
    print("=" * 60)
    print("增强Corridor可视化 - 展示L型绕行路径")
    print("=" * 60)
    
    # 场景1: 固定起点终点（水平）
    print("\n[场景1] 水平路径 - 展示L型绕行")
    generator1 = CorridorGenerator(seed=123)
    fixed_start = np.array([2.0, 6.0])
    fixed_goal = np.array([10.0, 6.0])
    scenario1 = generator1.generate_scenario(
        'medium',
        fixed_start=fixed_start,
        fixed_goal=fixed_goal
    )
    print(f"  生成了 {len(scenario1['corridors'])} 条corridor")
    visualize_corridor_with_paths(scenario1, 
                                   "Horizontal Path: L-shape Detour Corridors",
                                   "corridor_enhanced_horizontal.png")
    
    # 场景2: 固定起点终点（对角线）
    print("\n[场景2] 对角线路径")
    generator2 = CorridorGenerator(seed=456)
    fixed_start2 = np.array([2.0, 2.0])
    fixed_goal2 = np.array([10.0, 10.0])
    scenario2 = generator2.generate_scenario(
        'medium',
        fixed_start=fixed_start2,
        fixed_goal=fixed_goal2
    )
    print(f"  生成了 {len(scenario2['corridors'])} 条corridor")
    visualize_corridor_with_paths(scenario2,
                                   "Diagonal Path: Multiple Route Options",
                                   "corridor_enhanced_diagonal.png")
    
    # 场景3: 随机场景（Easy难度）
    print("\n[场景3] 随机Easy场景")
    generator3 = CorridorGenerator(seed=789)
    scenario3 = generator3.generate_scenario('easy')
    print(f"  起点: ({scenario3['start'][0]:.1f}, {scenario3['start'][1]:.1f})")
    print(f"  终点: ({scenario3['goal'][0]:.1f}, {scenario3['goal'][1]:.1f})")
    print(f"  生成了 {len(scenario3['corridors'])} 条corridor")
    visualize_corridor_with_paths(scenario3,
                                   "Easy Difficulty: Random Scenario",
                                   "corridor_enhanced_easy.png")
    
    print("\n" + "=" * 60)
    print("[完成] 增强可视化已保存")
    print("  - corridor_enhanced_horizontal.png")
    print("  - corridor_enhanced_diagonal.png")
    print("  - corridor_enhanced_easy.png")
    print("=" * 60)


if __name__ == '__main__':
    main()

