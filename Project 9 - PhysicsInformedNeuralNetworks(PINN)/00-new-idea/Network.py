import torch

from torch import nn


class MLP_02(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(input_size, hidden_size)
        self.layer3 = nn.Linear(input_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.layer6 = nn.Linear(hidden_size, hidden_size)
        self.layer7 = nn.Linear(hidden_size, hidden_size)
        self.layer8 = nn.Linear(hidden_size, hidden_size)
        self.layer9 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.layer1(x))
        out2 = self.relu(self.layer2(x))
        out3 = self.relu(self.layer3(x))

        out1 = self.relu(self.layer4(out1))
        out2 = self.relu(self.layer5(out2))
        out3 = self.relu(self.layer6(torch.sin(out3)))

        out1 = self.relu(self.layer7(torch.sin(out1)))
        out2 = self.relu(self.layer8(out2))

        out = self.layer9(out1 + out2 - out3)
        return out
