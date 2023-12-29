import torch
from torch import nn


class LOS_Classification_V0(nn.Module):
    def __init__(self, in_put, hidden_units, out_put):
        super().__init__()
        self.layer_1 = nn.Linear(in_put, hidden_units)
        self.layer_2 = nn.Linear(hidden_units, hidden_units)
        self.layer_3 = nn.Linear(hidden_units, out_put)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

