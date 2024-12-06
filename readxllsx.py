# -*- coding: utf-8 -*-
"""
@Time: 2024/12/5 19:17
@Author: cong
@Description: 读取excel表格
"""
import glob
import pandas as pd
import numpy as np


def read_xlsx(path):
    # 读取所有文件
    voltage_datas = []
    capacity_datas = []
    count = 0
    # 使用 glob 获取所有 .xlsx 文件的路径

    files = glob.glob(path)
    for file in files:
        count += 1
        print("文件个数：", count)
        df = pd.read_excel(file, sheet_name='详细数据1')
        data = df["工步类型"]
        # 查找第一个为"CC-CV充电"的索引
        index = data[data.str.contains('CC-CV充电')].index.min()
        # 删除"CC-CV充电"之前的"恒流放电"
        df_ = df.iloc[index:]
        # 从剩余的数据中，找出"恒流放电"的数据
        df_discharge = df_[df_["工步类型"]=="恒流放电"]
        # 取出电压和容量两列数据
        df_use = df_discharge[["电压(mV)", "容量(mAh)"]]
        df_use.iloc[:, 1] += 2500
        discharge_data = df_use.values
        voltage_data = discharge_data[:,0]
        capacity_data = discharge_data[:,1]
        voltage_datas.append(voltage_data)
        capacity_datas.append(capacity_data)
    return voltage_datas, capacity_datas

def find_max_min(data_list):
    # 将列表中的所有数组堆叠成一个大数组
    stacked_data = np.hstack(data_list)
    # 计算最大值和最小值
    max_value = np.max(stacked_data)
    min_value = np.min(stacked_data)
    return max_value, min_value


# 创建时序数据集，使用前n个电压值来预测剩余电量
def create_dataset(voltage, soc, time_step=1):
    x, y = [], []
    for i in range(len(voltage) - time_step):
        x.append(voltage[i:(i + time_step)])
        y.append(soc[i + time_step])  # 使用n个时间步的电压来预测当前时刻的剩余电量
    return np.array(x), np.array(y)

def make_datasets(xl_path, time_step):
    max_min_values = []
    discharge_datas = read_xlsx(xl_path)
    vol_datas = discharge_datas[0]
    vol_max, vol_min = find_max_min(vol_datas)
    cap_datas = discharge_datas[1]
    cap_max, cap_min = find_max_min(cap_datas)
    max_min_values.append([vol_max, vol_min, cap_max, cap_min])
    # 数据进行Max-Min 归一化
    vol_normalized_datas = [(data - vol_min) / (vol_max - vol_min) for data in vol_datas]
    cal_normalized_datas = [(data - cap_min) / (cap_max - cap_min) for data in cap_datas]

    datasets = []
    labels = []
    for k in range(len(vol_normalized_datas)):
        xs, ys = create_dataset(vol_normalized_datas[k], cal_normalized_datas[k], time_step=time_step)
        datasets.append(xs)
        labels.append(ys)

    x_ = np.expand_dims(np.vstack(datasets), axis=-1)
    y_ = np.hstack(labels)

    return x_, y_, max_min_values

