# -*- coding: utf-8 -*-
"""
@Time: 2024/12/6 15:28
@Author: cong
@Description: 描述文件的功能
"""
import torch
from matplotlib import pyplot as plt

from models import *
from sklearn.metrics import accuracy_score
from readxllsx import read_xlsx
import numpy as np


test_xlsx_path = "./data/val_data/*.xlsx"
max_min_values = [4195.6, 2997.6, 2500.0, 62.340000000000146]
time_step = 3
discharge_datas = read_xlsx(test_xlsx_path)
vol_datas = discharge_datas[0]
cap_datas = discharge_datas[1]
vol_max, vol_min,cap_max, cap_min = tuple(max_min_values)

# 测试数据进行Max-Min 归一化
vol_normalized_datas = [(data - vol_min) / (vol_max - vol_min) for data in vol_datas]
test_datasets = []
for k in range(len(vol_normalized_datas)):
    x = []
    for i in range(len(vol_normalized_datas[k]) - time_step + 1):
        x.append(vol_normalized_datas[k][i:(i + time_step)])
    test_datasets.append(np.expand_dims(np.vstack(x), axis=-1))

test_datasets = [torch.tensor(test_datasets[i], dtype=torch.float32) for i in range(len(test_datasets))]

# 参数设置
input_size = 1  # 输入的特征数量
hidden_size = 128  # LSTM隐藏层的大小
output_size = 1  # 输出的类别数
num_layers = 2  # LSTM的层数

model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('best_model.pth'))
# 假设你已经有了训练好的模型
model.eval()  # 切换到评估模式

test_outputs = []
with torch.no_grad():  # 禁用梯度计算

    for i in range(len(test_datasets)):
        outputs = []
        for j in range(len(test_datasets[i])):
            output = model(test_datasets[i][j].unsqueeze(0))
            outputs.append(output)
        test_outputs.append(outputs)

test_outputs_values = [[tensor.item() for tensor in sublist] for sublist in test_outputs]
# 预测数据进行还原
cap_inverse_datas = [( np.array(data) * (cap_max - cap_min) + cap_min) for data in test_outputs_values]

# 创建一个包含 1 行 2 列的子图
fig, axes = plt.subplots(1, 2, figsize=(20, 12))  # 1 行 2 列

# 在第一个子图上绘制两条线
x = list(range(1, len(cap_datas[0][2:]) + 1))
axes[0].plot(x, cap_datas[0][2:], marker='o', linestyle='-', color='b', label='true')
axes[0].plot(x, cap_inverse_datas[0], marker='o', linestyle='--', color='r', label='predict')
axes[0].set_title('Plot 1')
axes[0].set_xlabel('X-axis')
axes[0].set_ylabel('Y-axis')
axes[0].legend()  # 显示图例
x = list(range(1, len(cap_datas[1][2:]) + 1))

# 在第二个子图上绘制两条线
axes[1].plot(x,cap_datas[1][2:], marker='o', linestyle='-', color='g', label='true')
axes[1].plot(x, cap_inverse_datas[1], marker='o', linestyle='--', color='purple', label='predict')
axes[1].set_title('Plot 2')
axes[1].set_xlabel('X-axis')
axes[1].set_ylabel('Y-axis')
axes[1].legend()  # 显示图例

# 自动调整布局，使子图不会重叠
plt.tight_layout()
# 显示图形
plt.show()