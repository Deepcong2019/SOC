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
from torch.utils.data import random_split
from models import *
import torch.optim as optim
import torch

xlsx_path = "./data/*.xlsx"
X, Y, max_min_values = make_datasets(xlsx_path, time_step=3)
dataset = MyDataset(X, Y)

# 数据集划分比例
train_size = int(0.8 * len(dataset))  # 80% 用于训练
val_size = len(dataset) - train_size  # 20% 用于验证

# 使用 random_split 划分数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# 使用DataLoader加载数据
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# 迭代加载数据
for batch_idx, (features, labels) in enumerate(train_dataloader):
    print(f"Batch {batch_idx+1}")
    print(f"Features: {features}")
    print(f"Labels: {labels}")
# 参数设置
input_size = 1  # 输入的特征数量
hidden_size = 64  # LSTM隐藏层的大小
output_size = 1  # 输出的类别数
num_layers = 1  # LSTM的层数

# 创建模型
model = LSTMModel(input_size, hidden_size, output_size)

# 打印模型结构
print(model)


# 使用二分类交叉熵损失函数
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (seqs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # 将标签转换为正确的形状 (batch_size, 1)
        labels = labels.view(-1, 1)

        # 将数据传入LSTM模型
        outputs = model(seqs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播并优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    # 验证过程
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算
        for val_seqs, val_labels in val_dataloader:
            val_labels = val_labels.view(-1, 1)
            val_outputs = model(val_seqs)
            loss = criterion(val_outputs, val_labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")



