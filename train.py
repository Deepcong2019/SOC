# -*- coding: utf-8 -*-
"""
@Time: 2024/12/5 19:16
@Author: cong
@Description: 描述文件的功能
"""
from readxllsx import make_datasets
import matplotlib.pyplot as plt
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader


xlsx_path = "./data/*.xlsx"
X, Y = make_datasets(xlsx_path)
dataset = MyDataset(X, Y)
# 使用DataLoader加载数据
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 迭代加载数据
# for batch_idx, (features, labels) in enumerate(dataloader):
#     print(f"Batch {batch_idx+1}")
#     print(f"Features: {features}")
#     print(f"Labels: {labels}")



