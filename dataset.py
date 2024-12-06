# -*- coding: utf-8 -*-
"""
@Time: 2024/12/6 14:00
@Author: cong
@Description: 创建数据集
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        # 返回数据集的大小
        return len(self.features)

    def __getitem__(self, idx):
        # 返回单个样本（特征和标签）
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

# # 示例数据
# X = np.random.randn(100, 5)  # 100个样本，每个样本5个特征
# y = np.random.randint(0, 2, size=(100,))  # 100个标签
#
# # 创建数据集
# dataset = MyDataset(X, y)
#
# # 使用DataLoader加载数据
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#
# # 迭代加载数据
# for batch_idx, (features, labels) in enumerate(dataloader):
#     print(f"Batch {batch_idx+1}")
#     print(f"Features: {features}")
#     print(f"Labels: {labels}")
