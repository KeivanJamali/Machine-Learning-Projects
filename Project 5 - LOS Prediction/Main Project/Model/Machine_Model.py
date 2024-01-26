import torch
from torch import nn


class LOS_Classification_V0(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


class LOS_Classification_V1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        # self.bn_1 = nn.BatchNorm1d(hidden_size)
        self.dropout_1 = nn.Dropout(0.1)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        # self.bn_2 = nn.BatchNorm1d(hidden_size)
        self.dropout_2 = nn.Dropout(0.1)
        self.layer_3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout_1(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout_2(x)
        x = self.layer_3(x)
        return x


class LOS_Classification_V2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, hidden_size)
        self.layer_4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))
