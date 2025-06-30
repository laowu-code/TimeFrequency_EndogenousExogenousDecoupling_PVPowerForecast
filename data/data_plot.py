import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# 解决中文显示问题，使用系统自带的中文字体 SimHei
# plt.rcParams['font.sans-serif']=['SimHei']
# # 解决负号显示问题
# plt.rcParams['axes.unicode_minus']=False
# 读取四个数据集
# data1 = pd.read_csv('../data/site_1B/site_1B_2021_3_2021_5_5min.csv')
# data2 = pd.read_csv('../data/site_24/site_24_2021_3_2021_5_5min.csv')
# data3 = pd.read_csv('../data/site_PVOD/site_PVOD_01_15min.csv')
# data4 = pd.read_csv('../data/site_PVOD/site_PVOD_03_15min.csv')
data1 = pd.read_csv('../data/1B/Spring_1B_2017_2020_h.csv')
data2 = pd.read_csv('../data/1B/Summer_1B_2017_2020_h.csv')
data3 = pd.read_csv('../data/1B/Autumn_1B_2017_2020_h.csv')
data4 = pd.read_csv('../data/1B/Winter_1B_2017_2020_h.csv')
# 提取第二列数据
col1 = data1.iloc[:, 1]
col2 = data2.iloc[:, 1]
col3 = data3.iloc[:, 1]
col4 = data4.iloc[:, 1]
col_list=[col1,col2,col3,col4]
# 计算统计量
def calculate_statistics(col):
    return {
        'length': len(col),
        'max': col.max(),
        'min': col.min(),
        'mean': col.mean(),
        'std': col.std(),
        'median': col.median()
    }

stats1 = calculate_statistics(col1)
stats2 = calculate_statistics(col2)
stats3 = calculate_statistics(col3)
stats4 = calculate_statistics(col4)

# 打印统计信息
print("Data 1 Statistics:", stats1)
print("Data 2 Statistics:", stats2)
print("Data 3 Statistics:", stats3)
print("Data 4 Statistics:", stats4)

# 划分数据集
# 生成颜色映射
# train_colors = '#0079BF'
# val_colors ='#DF573F'
# test_colors = '#C492EE'
#
# # 设置子图布局
# fig, axs = plt.subplots(4, 1, figsize=(10, 10))
#
# train_size = int(0.8 * len(col1))
# val_size = int(0.1 * len(col1))
# test_size = len(col1) - train_size - val_size
# dataset=['A','B','C','D']
# time_resolutions = ['5min', '5min', '15min', '15min']
# # 绘制数据1的曲线
# for i in range(4):
#     data=col_list[i]
#     train_size = int(0.8 * len(data))
#     val_size = int(0.1 * len(data))
#     test_size = len(data) - train_size - val_size
#     axs[i].plot(range(train_size), data.iloc[:train_size], color=train_colors, label='Train')
#     axs[i].plot(range(train_size, train_size + val_size), data.iloc[train_size:train_size + val_size], color=val_colors, label='Validation')
#     axs[i].plot(range(train_size + val_size, len(data)), data.iloc[train_size + val_size:], color=test_colors, label='Test')
#     axs[i].set_title(f'Dataset{dataset[i]}')
#     axs[i].set_xlabel(f'Time ({time_resolutions[i]})')
#     axs[i].set_ylabel('PV Power(kW)')
#     axs[i].legend(ncol=5, loc='upper center')
# # 调整布局
# plt.tight_layout()
# plt.savefig('../pic/data_plot.svg',format='SVG',dpi=600)
# plt.show()
