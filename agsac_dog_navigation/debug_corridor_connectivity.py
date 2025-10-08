"""
调试Corridor连通性
详细显示顶点编号和连接顺序
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from agsac.envs.corridor_generator import CorridorGenerator


def visualize_corridor_debug(scenario, title="Corridor Debug", save_path=None):
    """详细可视化corridor的连通性"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 设置地图范围
    ax.set_xlim(0, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Y (meters)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    start = scenario['start']
    goal = scenario['goal']
    
    # 绘制每个corridor
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for i, corridor in enumerate(scenario['corridors']):
        # 1. 绘制填充的多边形
        polygon = Polygon(
            corridor,
            facecolor=colors[i % len(colors)],
            edgecolor='darkblue',
            linewidth=3,
            alpha=0.3,
            label=f'Corridor {i+1}'
        )
        ax.add_patch(polygon)
        
        # 2. 绘制顶点和编号
        for j, vertex in enumerate(corridor):
            # 顶点标记
            ax.plot(vertex[0], vertex[1], 'bo', markersize=12, zorder=5)
            # 顶点编号
            ax.text(vertex[0] + 0.2, vertex[1] + 0.2, f'{j+1}', 
                   fontsize=12, fontweight='bold', color='blue',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='blue'))
        
        # 3. 绘制边的连接顺序（箭头）
        for j in range(len(corridor)):
            start_v = corridor[j]
            end_v = corridor[(j + 1) % len(corridor)]
            
            # 画箭头（从顶点j到顶点j+1）
            dx = end_v[0] - start_v[0]
            dy = end_v[1] - start_v[1]
            
            # 箭头位置在边的中点附近
            mid_x = start_v[0] + dx * 0.4
            mid_y = start_v[1] + dy * 0.4
            
            ax.annotate('', xy=(end_v[0], end_v[1]), xytext=(start_v[0], start_v[1]),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.6),
                       zorder=4)
        
        # 4. 打印顶点坐标信息
        print(f"\n=== Corridor {i+1} ({len(corridor)} vertices) ===")
        for j, v in enumerate(corridor):
            print(f"  Vertex {j+1}: ({v[0]:.3f}, {v[1]:.3f})")
        
        # 5. 检查多边形是否自相交
        is_simple = check_simple_polygon(corridor)
        if not is_simple:
            print(f"  ⚠️ 警告：Corridor {i+1} 可能自相交！")
            ax.text(6, -1.5 + i*0.3, f'⚠️ Corridor {i+1}: May have self-intersection!',
                   fontsize=11, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9))
    
    # 绘制起点和终点
    ax.plot(start[0], start[1], 'go', markersize=30, label='Start', zorder=10,
            markeredgecolor='darkgreen', markeredgewidth=3)
    ax.plot(goal[0], goal[1], 'r*', markersize=40, label='Goal', zorder=10,
            markeredgecolor='darkred', markeredgewidth=2)
    
    # 标注起点终点
    ax.text(start[0], start[1] - 0.8, f'START\n({start[0]:.2f}, {start[1]:.2f})', 
            ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(goal[0], goal[1] + 0.8, f'GOAL\n({goal[0]:.2f}, {goal[1]:.2f})', 
            ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', edgecolor='red', linewidth=2))
    
    # 直线参考
    ax.plot([start[0], goal[0]], [start[1], goal[1]], 
            'k--', linewidth=2, alpha=0.4, label='Direct Line')
    
    # 图例和说明
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # 添加说明
    info_text = (
        f"Distance: {np.linalg.norm(goal - start):.2f}m\n"
        f"Corridors: {len(scenario['corridors'])}\n"
        f"Blue numbers = Vertex order\n"
        f"Red arrows = Connection sequence"
    )
    ax.text(11.5, 11, info_text, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow', 
                     edgecolor='orange', linewidth=2, alpha=0.9))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[Saved] {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()


def check_simple_polygon(vertices):
    """
    检查多边形是否简单（无自相交）
    简化版检查：检测是否有边相交
    """
    n = len(vertices)
    
    for i in range(n):
        seg1_p1 = vertices[i]
        seg1_p2 = vertices[(i + 1) % n]
        
        for j in range(i + 2, n):
            if j == (i + n - 1) % n:  # 跳过相邻边
                continue
            
            seg2_p1 = vertices[j]
            seg2_p2 = vertices[(j + 1) % n]
            
            if segments_intersect(seg1_p1, seg1_p2, seg2_p1, seg2_p2):
                return False
    
    return True


def segments_intersect(p1, p2, p3, p4):
    """检查两条线段是否相交"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def main():
    """调试Corridor连通性"""
    print("=" * 70)
    print("Corridor连通性调试 - 详细检查")
    print("=" * 70)
    
    # 场景1: Easy难度（用户提到的问题场景）
    print("\n\n[场景1] Easy难度 - 检查连通性")
    gen1 = CorridorGenerator(seed=42)
    scene1 = gen1.generate_scenario('easy')
    print(f"起点: ({scene1['start'][0]:.3f}, {scene1['start'][1]:.3f})")
    print(f"终点: ({scene1['goal'][0]:.3f}, {scene1['goal'][1]:.3f})")
    print(f"直线距离: {np.linalg.norm(scene1['goal'] - scene1['start']):.3f}m")
    visualize_corridor_debug(scene1, 
                             "Easy Difficulty - Connectivity Debug",
                             "corridor_debug_easy.png")
    
    # 场景2: 固定水平路径
    print("\n\n[场景2] 固定水平路径 - 检查L型结构")
    gen2 = CorridorGenerator(seed=789)
    scene2 = gen2.generate_scenario(
        'medium',
        fixed_start=np.array([2.0, 6.0]),
        fixed_goal=np.array([10.0, 6.0])
    )
    print(f"起点: ({scene2['start'][0]:.3f}, {scene2['start'][1]:.3f})")
    print(f"终点: ({scene2['goal'][0]:.3f}, {scene2['goal'][1]:.3f})")
    visualize_corridor_debug(scene2,
                             "Horizontal Path - L-shape Debug",
                             "corridor_debug_horizontal.png")
    
    # 场景3: 对角线路径
    print("\n\n[场景3] 对角线路径 - 多corridor检查")
    gen3 = CorridorGenerator(seed=456)
    scene3 = gen3.generate_scenario(
        'medium',
        fixed_start=np.array([2.0, 2.0]),
        fixed_goal=np.array([10.0, 10.0])
    )
    print(f"起点: ({scene3['start'][0]:.3f}, {scene3['start'][1]:.3f})")
    print(f"终点: ({scene3['goal'][0]:.3f}, {scene3['goal'][1]:.3f})")
    visualize_corridor_debug(scene3,
                             "Diagonal Path - Multiple Corridors Debug",
                             "corridor_debug_diagonal.png")
    
    print("\n" + "=" * 70)
    print("[完成] 调试图片已保存")
    print("  - corridor_debug_easy.png")
    print("  - corridor_debug_horizontal.png")
    print("  - corridor_debug_diagonal.png")
    print("\n请查看图片中的顶点编号和红色箭头连接顺序")
    print("如果箭头交叉或顶点顺序混乱，说明多边形自相交！")
    print("=" * 70)


if __name__ == '__main__':
    main()

