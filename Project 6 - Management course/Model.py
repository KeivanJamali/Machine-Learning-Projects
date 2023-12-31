import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class RNN_V0(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_units, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(device)
        out, _ = self.rnn(x, h0)
        # out: batch_size, seq_length, hidden_size
        # out: (32, 10, 64)
        out = out[:, -1, :]
        # out: (32, 64)
        out = self.fc(out)
        return out


class GRU_V0(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_units, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class LSTM_V0(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_units, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
