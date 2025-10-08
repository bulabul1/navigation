"""
可视化训练时实际生成的Corridor场景
模拟训练过程中的corridor生成
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from agsac.envs.corridor_generator import CorridorGenerator


def visualize_training_scenario(scenario, episode_num, save_path=None):
    """可视化单个训练场景"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 设置地图范围
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (meters)', fontsize=13)
    ax.set_ylabel('Y (meters)', fontsize=13)
    ax.set_title(f'Training Episode {episode_num} - Easy Difficulty', 
                 fontsize=15, fontweight='bold')
    
    start = scenario['start']
    goal = scenario['goal']
    
    # 1. 绘制corridors
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for i, corridor in enumerate(scenario['corridors']):
        polygon = Polygon(
            corridor,
            facecolor=colors[i % len(colors)],
            edgecolor='darkblue',
            linewidth=2.5,
            alpha=0.5,
            label=f'Corridor {i+1}' if i == 0 else None
        )
        ax.add_patch(polygon)
        
        # 标注corridor类型
        centroid = corridor.mean(axis=0)
        if len(corridor) == 4:
            corridor_type = "Direct"
        elif len(corridor) == 8:
            max_y = corridor[:, 1].max()
            min_y = corridor[:, 1].min()
            mid_y = (start[1] + goal[1]) / 2
            
            if max_y > max(start[1], goal[1]) + 1:
                corridor_type = "Upper\nDetour"
            elif min_y < min(start[1], goal[1]) - 1:
                corridor_type = "Lower\nDetour"
            else:
                corridor_type = "L-shape"
        else:
            corridor_type = f"{len(corridor)}v"
        
        ax.text(centroid[0], centroid[1], f'C{i+1}\n{corridor_type}', 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='blue', linewidth=2, alpha=0.9))
    
    # 2. 绘制起点和终点
    ax.plot(start[0], start[1], 'go', markersize=25, label='Start', zorder=10,
            markeredgecolor='darkgreen', markeredgewidth=3)
    ax.plot(goal[0], goal[1], 'r*', markersize=35, label='Goal', zorder=10,
            markeredgecolor='darkred', markeredgewidth=2)
    
    # 标注坐标
    ax.text(start[0], start[1] - 0.6, f'START\n({start[0]:.1f}, {start[1]:.1f})', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', 
                     edgecolor='green', linewidth=2))
    ax.text(goal[0], goal[1] + 0.6, f'GOAL\n({goal[0]:.1f}, {goal[1]:.1f})', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', 
                     edgecolor='red', linewidth=2))
    
    # 3. 绘制直线连接
    ax.plot([start[0], goal[0]], [start[1], goal[1]], 
            'k--', linewidth=2, alpha=0.4, label='Direct Line')
    
    # 4. 添加场景信息
    distance = np.linalg.norm(goal - start)
    info_text = (
        f"Distance: {distance:.2f}m\n"
        f"Corridors: {len(scenario['corridors'])}\n"
        f"Pedestrians: {scenario['num_pedestrians']}\n"
        f"Difficulty: Easy"
    )
    ax.text(6, 11.3, info_text, ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', 
                     edgecolor='orange', linewidth=2, alpha=0.9))
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, 
             edgecolor='black', fancybox=True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [保存] {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()


def main():
    """生成多个训练场景示例"""
    print("=" * 70)
    print("训练时实际生成的Corridor场景可视化")
    print("=" * 70)
    print("\n配置信息:")
    print("  - 难度: Easy (固定)")
    print("  - Corridor数量: 1-2条")
    print("  - 行人数量: 2-3个")
    print("  - 起点终点: 随机生成")
    print("  - 最小距离: 6米")
    
    # 生成10个训练场景示例
    print(f"\n{'='*70}")
    print("生成训练场景示例 (模拟Episode 1-10)")
    print(f"{'='*70}")
    
    # 创建一个大图，显示多个场景
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Corridors - Easy Difficulty (Episodes 1-6)', 
                 fontsize=16, fontweight='bold')
    
    colors_list = ['lightblue', 'lightgreen', 'lightyellow']
    
    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        
        # 使用不同的随机种子生成不同场景
        generator = CorridorGenerator(seed=100 + idx)
        scenario = generator.generate_scenario('easy')
        
        start = scenario['start']
        goal = scenario['goal']
        
        # 设置子图
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_title(f'Episode {idx+1}', fontsize=12, fontweight='bold')
        
        # 绘制corridors
        for i, corridor in enumerate(scenario['corridors']):
            polygon = Polygon(
                corridor,
                facecolor=colors_list[i % len(colors_list)],
                edgecolor='darkblue',
                linewidth=2,
                alpha=0.4
            )
            ax.add_patch(polygon)
        
        # 绘制起点终点
        ax.plot(start[0], start[1], 'go', markersize=15, zorder=10)
        ax.plot(goal[0], goal[1], 'r*', markersize=20, zorder=10)
        ax.plot([start[0], goal[0]], [start[1], goal[1]], 
                'k--', linewidth=1, alpha=0.3)
        
        # 添加信息
        distance = np.linalg.norm(goal - start)
        ax.text(6, 11, f'{len(scenario["corridors"])}C | {distance:.1f}m', 
               ha='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # 打印场景信息
        print(f"\nEpisode {idx+1}:")
        print(f"  起点: ({start[0]:.2f}, {start[1]:.2f})")
        print(f"  终点: ({goal[0]:.2f}, {goal[1]:.2f})")
        print(f"  距离: {distance:.2f}m")
        print(f"  Corridor数量: {len(scenario['corridors'])}")
        print(f"  行人数量: {scenario['num_pedestrians']}")
    
    plt.tight_layout()
    plt.savefig('training_corridors_overview.png', dpi=150, bbox_inches='tight')
    print(f"\n[保存] training_corridors_overview.png")
    plt.close()
    
    # 生成3个详细的单场景示例
    print(f"\n{'='*70}")
    print("生成详细单场景示例")
    print(f"{'='*70}")
    
    for i in range(3):
        print(f"\n详细场景 {i+1}:")
        generator = CorridorGenerator(seed=200 + i)
        scenario = generator.generate_scenario('easy')
        
        start = scenario['start']
        goal = scenario['goal']
        distance = np.linalg.norm(goal - start)
        
        print(f"  起点: ({start[0]:.2f}, {start[1]:.2f})")
        print(f"  终点: ({goal[0]:.2f}, {goal[1]:.2f})")
        print(f"  距离: {distance:.2f}m")
        print(f"  Corridor数量: {len(scenario['corridors'])}")
        print(f"  行人数量: {scenario['num_pedestrians']}")
        
        visualize_training_scenario(scenario, i+1, 
                                    f'training_corridor_detail_{i+1}.png')
    
    print("\n" + "=" * 70)
    print("[完成] 训练场景可视化")
    print("\n生成的文件:")
    print("  - training_corridors_overview.png  (6个场景总览)")
    print("  - training_corridor_detail_1.png   (详细场景1)")
    print("  - training_corridor_detail_2.png   (详细场景2)")
    print("  - training_corridor_detail_3.png   (详细场景3)")
    print("=" * 70)


if __name__ == '__main__':
    main()

