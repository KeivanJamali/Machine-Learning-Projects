import numpy as np
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def create_sequence(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        x.append(seq)
        y.append(target)
    return np.array(x), np.array(y)
