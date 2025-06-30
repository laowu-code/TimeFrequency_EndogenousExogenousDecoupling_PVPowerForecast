import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 1. 读取数据（假设文件名为data.csv）
df = pd.read_csv("../data/site_24/site_24_2021_3_2021_5_5min.csv")
df=df.iloc[:,1:]
# 2. 计算相关系数矩阵
corr_matrix = df.corr()
corr_matrix.to_csv('corr_matrix.csv')
# 3. 创建自定义渐变色
colors = ["#2E86C1", "#F4D03F", "#E74C3C"]
cmap = LinearSegmentedColormap.from_list("custom", colors)

# 创建极坐标系
plt.figure(figsize=(12, 12), facecolor='#212121')
ax = plt.subplot(111, polar=True)

# 环形参数设置
n_vars = len(corr_matrix.columns)
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合环形
radius = 0.8  # 热力图半径

# 绘制环形热力图
for i in range(n_vars):
    for j in range(i + 1, n_vars):
        # 计算区块角度范围
        start_angle = angles[j] - np.pi / (2 * n_vars)
        end_angle = angles[j] + np.pi / (2 * n_vars)

        # 创建扇形区块
        ax.fill_between(
            np.linspace(start_angle, end_angle, 50),
            i * 0.1 + 0.3,
            i * 0.1 + 0.3 + corr_matrix.iloc[i, j] * 0.5,
            color=cmap((corr_matrix.iloc[i, j] + 1) / 2),
            edgecolor='#34495E',
            lw=0.5
        )

# 装饰参数设置
ax.set_theta_offset(np.pi / 2)
ax.set_ylim(0, 5)
ax.set_axis_off()

# 添加标签
for idx, (label, angle) in enumerate(zip(corr_matrix.columns, angles)):
    # 旋转文字方向
    rotation_angle = np.degrees(angle)
    if angle > np.pi:
        rotation_angle += 180

    ax.text(
        angle, 5.2, label,
        color='#F4D03F',
        ha='center', va='center',
        rotation=rotation_angle,
        fontsize=12,
        fontweight='bold'
    )

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.08)
cbar.ax.tick_params(labelsize=12, colors='white')
cbar.outline.set_edgecolor('#7F8C8D')

# 添加光效装饰
ax.add_artist(plt.Circle((0, 0), 4.5, fc='none', ec='#F4D03F', lw=2, alpha=0.3))
ax.add_artist(plt.Circle((0, 0), 4.8, fc='none', ec='#E74C3C', lw=2, alpha=0.3))

# 保存和显示
plt.tight_layout()
# plt.savefig('circular_heatmap.png', dpi=300)
plt.show()