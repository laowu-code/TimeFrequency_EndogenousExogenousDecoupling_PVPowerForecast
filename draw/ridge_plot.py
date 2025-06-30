import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.font_manager import FontProperties
rc('font', family='Arial')
# 读取数据（根据实际分隔符调整，例如 sep='\t' 表示制表符分隔）
file_path = '../data/data.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 取前 1500 行数据
df_sub = df.iloc[0:1500]

# 创建 5 行 1 列的子图，figsize 可根据需要调整
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 12), sharex=True)

# 待绘制的变量名称（与 CSV 表头对应）
variables = ['AP', 'TP', 'RH', 'GHI', 'DHI']
name_label = ['AP(KW)', 'TP(\u2103)', 'RH(%)', 'GHI(Wh/m$^{2}$)', 'DHI(Wh/m$^{2}$)']
# 设置字体大小参数
  # 坐标轴刻度字体大小
ksize=20
colors=['#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E']
# 分别绘制每个变量的折线图
i=0
simhei_font=FontProperties(family='SimHei', size=ksize)
for ax, var in zip(axes, variables):
    ax.plot(df_sub[var], color=colors[i], linewidth=4,alpha=0.7)  # 可修改颜色和线宽
    ax.set_ylabel(name_label[i], fontsize=ksize, fontproperties=simhei_font)    # 设置 y 轴标签，并调整字体大小
    ax.tick_params(axis='y', labelsize=ksize)  # 设置 x 和 y 坐标刻度的字体大小
    i+=1
# 设置 x 轴标签
axes[-1].set_xlabel("时间步（5min）", fontsize=ksize, fontproperties=simhei_font)
axes[-1].tick_params(axis='x', labelsize=ksize)

plt.tight_layout()  # 调整子图间距
plt.savefig('../pic/var_plot.svg', format='SVG', dpi=800)

plt.show()