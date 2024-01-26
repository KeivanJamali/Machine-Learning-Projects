import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class Health_Time_Series(nn.Module):
    def __init__(self, input_size, layer_number, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_number
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=layer_number, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.layer_num, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class GRU_V0(nn.Module):
    def __init__(self, input_size, hidden_size, layer_number, output_size):
        super().__init__()
        self.num_layers = layer_number
        self.hidden_units = hidden_size
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=layer_number, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class LSTM_V0(nn.Module):
    def __init__(self, input_size, hidden_size, layer_number, output_size):
        super().__init__()
        self.num_layers = layer_number
        self.hidden_units = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layer_number, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
