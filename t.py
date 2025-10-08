import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import datetime

# --- 複製您提供的資料 ---
# 场景：公园从西门到东门
start_point = (0, 5)
end_point = (20, 5)

corridor_data = [
    # 通路1：北侧小径（风景好，窄）
    np.array([
        [0, 4], [0, 8], [5, 8], [5, 10], [15, 10], [15, 8], [20, 8], 
        [20, 4], [15, 4], [15, 6], [5, 6], [5, 4]
    ]),
    # 通路2：中央大道（快速，直）
    np.array([[0, 3], [20, 3], [20, 7], [0, 7]]),
    # 通路3：南侧小径（少人，稍远）
    np.array([
        [0, 6], [0, 0], [5, 0], [5, -2], [15, -2], [15, 0], [20, 0], 
        [20, 6], [15, 6], [15, 4], [5, 4], [5, 6]
    ])
]

# --- 開始繪圖 ---
# 1. 建立畫布與座標軸
fig, ax = plt.subplots(figsize=(18, 12))

# 2. 繪製三條通路
# 通路1 (北側)
poly1 = patches.Polygon(corridor_data[0], closed=True, facecolor='skyblue', alpha=0.7, label='Path 1: North Trail (Scenic)')
ax.add_patch(poly1)

# 通路2 (中央)
poly2 = patches.Polygon(corridor_data[1], closed=True, facecolor='lightgray', alpha=0.8, label='Path 2: Central Avenue (Fastest)')
ax.add_patch(poly2)

# 通路3 (南側)
poly3 = patches.Polygon(corridor_data[2], closed=True, facecolor='lightgreen', alpha=0.7, label='Path 3: South Trail (Quiet)')
ax.add_patch(poly3)

# 3. 標示起點和終點
ax.plot(start_point[0], start_point[1], 'go', markersize=20, label='Start (West Gate)', zorder=10)
ax.text(start_point[0] + 0.3, start_point[1], 'West Gate', va='center', ha='left', fontsize=12)
ax.plot(end_point[0], end_point[1], 'ro', markersize=20, label='End (East Gate)', zorder=10)
ax.text(end_point[0] - 0.3, end_point[1], 'East Gate', va='center', ha='right', fontsize=12)

# 4. 視覺化智能決策邏輯
# 獲取當前時間來決定哪個是active decision
# (這裡我們硬編碼為傍晚，以匹配您的請求時間)
current_time_period = 'Evening'

# --- 決策面板 ---
# 早晨的決策
morning_box = dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=1)
ax.text(5, 12.5, "IF: Morning ☀️", ha='center', va='center', fontsize=14, bbox=morning_box)
ax.text(8.5, 9, "Busy 🏃‍♂️", fontsize=20, ha='center', va='center', color='red')
ax.annotate('AVOID North Path', xy=(8, 9), xytext=(5, 11),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8, connectionstyle="arc3,rad=0.1"))
ax.annotate('CHOOSE South Path', xy=(10, -1), xytext=(5, 11.5),
            arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=10, connectionstyle="angle3,angleA=0,angleB=-90"),
            fontsize=12, color='green', weight='bold')

# 中午的決策
noon_box = dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=1)
ax.text(10, 5, "IF: Noon 🕛\nCHOOSE Central Avenue\n(Fastest & Clear)", ha='center', va='center', 
        fontsize=14, bbox=noon_box, color='darkblue', weight='bold')

# 傍晚的決策 (高亮顯示)
evening_box_active = dict(boxstyle='round,pad=0.5', fc='gold', ec='darkorange', lw=3)
ax.text(15, -4.5, "IF: Evening 🌙 (Current Time)", ha='center', va='center', fontsize=14, bbox=evening_box_active)
ax.text(11.5, -1, "Busy 🐕", fontsize=20, ha='center', va='center', color='red')
ax.annotate('AVOID South Path', xy=(11, -1), xytext=(15, -3),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8, connectionstyle="arc3,rad=-0.1"))
ax.annotate('CHOOSE North Path', xy=(12, 9), xytext=(15, -2.5),
            arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=10, connectionstyle="angle3,angleA=0,angleB=90"),
            fontsize=14, color='green', weight='bold')


# --- 美化圖表 ---
ax.set_xlim(-1, 21)
ax.set_ylim(-6, 14)
ax.set_xlabel("East-West Position (meters)")
ax.set_ylabel("North-South Position (meters)")
ax.set_title("Intelligent Path Selection Based on Time of Day", fontsize=18, weight='bold')
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(loc='lower left')

plt.show()