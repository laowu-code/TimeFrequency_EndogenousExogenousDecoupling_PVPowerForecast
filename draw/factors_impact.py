import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = 'Arial'
# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体字体，SimHei是黑体的字体名

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
site = 'site_24'
dataset = '2021_3_2021_5_5min'
file_path = f'../data/{site}/{site}_{dataset}.csv'
df_all = pd.read_csv(file_path, header=0)
x_start, x_end = 0, 1520

pv = df_all.iloc[:, 1][x_start:x_end].values
t = np.arange(len(pv))
idx = 5
temp = df_all.iloc[:, idx][x_start:x_end].values
name_list = ['功率', '温度', '相对湿度', '全局辐照强度', '水平漫射辐照强度']
name_list_en = ['AP', 'T', 'RH', 'GHI', 'DHI']
name_label = ['功率(KW)', '温度(\u2103)', '相对湿度(%)', '全局辐照强度(Wh/m$^{2}$)', '水平漫射辐照强度(Wh/m$^{2}$)']
colors_ = ['#0079BF', '#DF573F', '#E6671A', '#FFC432']
ksize = 30
ls = 3
fig, ax = plt.subplots(figsize=(20, 10))
ax2 = ax.twinx()
ax.set_ylabel('功率(KW)', color=colors_[0], fontsize=ksize)
ax2.set_ylabel(name_label[idx - 1], color=colors_[1], fontsize=ksize)
ax.tick_params(axis='y', colors=colors_[0], labelsize=ksize)
ax2.tick_params(axis='y', colors=colors_[1], labelsize=ksize)
# y_ticks = np.linspace(np.min(temp), np.max(temp), 8)
# # 设置 y 轴的刻度位置
# ax2.set_yticks(y_ticks)
# ax2.set_ylim(top=1500)
ax.plot(t, pv, color=colors_[0], linewidth=ls, label='功率')
ax2.plot(t, temp, color=colors_[1], linewidth=ls, label=name_list[idx - 1])
# ax2.set_yticks(np.arange(0.0, 0.045, 0.01))
# ax.set_yticks(np.arange(0.15, 0.53, 0.1))
# ax.set_ylim(0.15, 0.53)
t_s = np.arange(0, len(t), 300)
l_s = [t[i] for i in t_s]
ax.set_xticks(t_s)
ax.set_xticklabels(l_s, fontsize=ksize, )
ax.set_xlabel('时间步（5min）', fontsize=ksize)
legend = fig.legend(loc='upper right', ncol=4, fontsize=ksize, frameon=True, edgecolor='black',
                    bbox_to_anchor=(0.9, 0.88))
# fig.legend(all_handles, all_labels, ncol=4, loc='upper center',fontsize=ksize,bbox_to_anchor=(0.5, 0.98))
# plt.subplots_adjust(hspace=0.2, wspace=0.3,top=0.91,bottom=0.15,left=0.05,right=0.95)# 调整图例的位置
# plt.suptitle('MAE and PL values (Node & Level)', fontsize=ksize, y=1.02)
plt.savefig(f'../pic/{name_list[idx - 1]}.svg', format='SVG', dpi=800)
# plt.savefig('../pic_/metrics_all_node.svg', format='SVG', dpi=800)
plt.show()
