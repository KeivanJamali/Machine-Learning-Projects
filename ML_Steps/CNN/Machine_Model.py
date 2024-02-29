import torch

from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class CNN_V0(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.l1(x)))
        x = self.dropout(self.relu(self.l2(x)))
        x = self.l3(x)
        return x
