import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'SimHei'  # 指定中文字体为黑体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 英文使用 Arial
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 你的数据
data = {
    "A": {
        "MAE": [0.1428, 0.178, 0.1952, 0.2196, 0.2442, 0.249],
        "RMSE": [0.3755, 0.4625, 0.5025, 0.5290, 0.5555, 0.5627],
        r'$\mathrm{R}^2$': [0.9763, 0.9641, 0.9576, 0.9530, 0.9482, 0.9469],
        "MBE": [-0.0388, 0.0168, 0.0203, 0.0375, 0.0585, 0.0444],
    },
    "B": {
        "MAE": [0.0959, 0.1220, 0.1376, 0.1466, 0.1503, 0.1612],
        "RMSE": [0.2667, 0.3225, 0.3566, 0.3782, 0.3884, 0.3972],
        r'$\mathrm{R}^2$': [0.9741, 0.9621, 0.9536, 0.9479, 0.9450, 0.9425],
        "MBE": [0.0205, 0.0380, 0.0417, 0.0447, 0.0485, 0.0547],
    },
    "C": {
        "MAE": [0.4963, 0.5749, 0.6646, 0.7293],
        "RMSE": [1.1292, 1.2975, 1.4500, 1.5778],
        r'$\mathrm{R}^2$': [0.9640, 0.9524, 0.9406, 0.9297],
        "MBE": [0.0273, 0.0486, 0.0106, 0.0269],
    },
    "D": {
        "MAE": [0.4660, 0.5614, 0.6358, 0.6918],
        "RMSE": [1.0247, 1.1810, 1.3060, 1.3784],
        r'$\mathrm{R}^2$': [0.9514, 0.9355, 0.9212, 0.9122],
        "MBE": [0.1245, 0.0695, 0.1275, 0.0552],
    },
}

# 预测步长
timesteps = {
    "A": ["5min", "10min", "15min", "20min", "25min", "30min"],
    "B": ["5min", "10min", "15min", "20min", "25min", "30min"],
    "C": ["15min", "30min", "45min", "60min"],
    "D": ["15min", "30min", "45min", "60min"],
}

# 设置子图布局
fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharex=False, sharey=True, gridspec_kw={"wspace": 0.2, "hspace": 0.3})
cmap = sns.light_palette("green", as_cmap=True)  # 颜色从浅到深
datasets = ["A", "B", "C", "D"]

# 设置单独的颜色条位置
# cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # 颜色条
# cbar_ax=None
# 遍历数据集绘制子图
for i, dataset in enumerate(datasets):
    ax = axes[i]  # 计算子图位置
    df = pd.DataFrame(data[dataset], index=timesteps[dataset]).T  # 生成 DataFrame

    # **归一化处理（按行）**
    df_normalized = df.copy()
    for row in df.index:
        min_val, max_val = df.loc[row].min(), df.loc[row].max()
        df_normalized.loc[row] = (df.loc[row] - min_val) / (max_val - min_val + 1e-6)  # 避免除零
    ksize = 16
    # 绘制热力图（每个指标的颜色独立归一化）
    sns.heatmap(df_normalized, annot=df, fmt=".4g", cmap=cmap, linewidths=0.5, ax=ax, cbar=False, annot_kws={"size": 12}
                )#i == 3,cbar_ax=None if i != 3 else cbar_ax

    # ax.set_title(f"Dataset {dataset}",fontsize=ksize)  # 设置子图标题
    # ax.set_xlabel("Horizon", fontsize=ksize)  # x 轴
    ax.set_title(f"数据集 {dataset}", fontsize=ksize)  # 设置子图标题
    ax.set_xlabel("步长", fontsize=ksize)  # x 轴

    if i==0:
        # ax.set_ylabel("Metric",fontsize=ksize)  # y 轴
        ax.set_ylabel("指标", fontsize=ksize)
    ax.tick_params(axis='y', labelsize=ksize)
    ax.tick_params(axis='x', labelsize=ksize)
# 调整布局
# plt.tight_layout(rect=[0, 0, 0.9, 1])  # 预留颜色条空间
plt.subplots_adjust(top=0.9,bottom=0.15,)
plt.savefig('../pic/heatmap_horizons.svg', format='SVG', dpi=800)
plt.show()
