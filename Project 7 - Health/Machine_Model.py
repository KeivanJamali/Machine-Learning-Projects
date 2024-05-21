import torch

from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class RNN_V0(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth_number):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth_number = depth_number
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=depth_number, batch_first=True,
                          nonlinearity="relu")
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.depth_number, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class GRU_V0(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth_number):
        super().__init__()
        self.hidden_units = hidden_size
        self.depth_number = depth_number
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=depth_number, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.depth_number, x.size(0), self.hidden_units).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class LSTM_V0(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth_number):
        super().__init__()
        self.hidden_units = hidden_size
        self.depth_number = depth_number
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=depth_number, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.depth_number, x.size(0), self.hidden_units).to(device)
        c0 = torch.zeros(self.depth_number, x.size(0), self.hidden_units).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class NN_V0(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.fc3 = nn.Linear(in_features=input_size, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out_2 = self.fc3(x)
        # print(out_2)
        out = torch.sin(out_2*torch.pi) + out
        out = self.relu(out)
        out = self.fc2(out)
        return out
