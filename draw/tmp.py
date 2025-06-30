import numpy as np
import pandas as pd
from chord import Chord

# 构造相关系数矩阵
matrix = np.array([
    [1,           0.52250049,  -0.500852325, 0.994140576, 0.533870442],
    [0.52250049,  1,           -0.600619111, 0.541646067, 0.392343401],
    [-0.500852325,-0.600619111, 1,           -0.496897859, -0.235173076],
    [0.994140576, 0.541646067, -0.496897859, 1,           0.542579797],
    [0.533870442, 0.392343401, -0.235173076, 0.542579797, 1]
])
names = ["AP", "TP", "RH", "GHI", "DHI"]

# 将矩阵整理成适合 Chord 的输入
data_list = []
for i in range(len(names)):
    for j in range(len(names)):
        data_list.append([names[i], names[j], matrix[i, j]])

# 生成并展示 Chord 图
Chord(data_list, names).show()
