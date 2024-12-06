# -*- coding: utf-8 -*-
"""
@Time: 2024/12/6 14:40
@Author: cong
@Description: 描述文件的功能
"""
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 层的输出，包含 hidden state 和 cell state
        lstm_out, (hn, cn) = self.lstm(x)
        # 取序列最后一个时间步的输出
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output


