import torch
import Information

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
        self.layer_in = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.layer1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer_in(x))
        out = self.relu(self.layer1(out))
        return self.layer2(out)

class NN_V1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(input_size, hidden_size)
        self.layer3 = nn.Linear(input_size, hidden_size)
        self.layer4 = nn.Linear(input_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.layer6 = nn.Linear(hidden_size, hidden_size)
        self.layer7 = nn.Linear(hidden_size, hidden_size)
        self.layer8 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.layer1(x))
        out2 = self.relu(self.layer2(x))
        out3 = self.relu(self.layer3(x))
        out4 = self.relu(self.layer4(x))

        out1 = self.relu(self.layer5(torch.sin(out1)))
        out2 = self.relu(self.layer6(torch.sin(out2)))
        out3 = self.relu(self.layer7(out3))

        out = out1 - out2 + out3
        out = self.layer8(out)
        return out


class LSTM_V1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth_number):
        super().__init__()
        self.hidden_units = hidden_size
        self.depth_number = depth_number
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=depth_number, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=depth_number, batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc4 = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.layer1 = nn.Linear(in_features=len(Information.features)-1, out_features=hidden_size)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.new = nn.Linear(in_features=1, out_features=hidden_size)
        self.new2 = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x1 = x[:, [0,1,-2], 0].unsqueeze(dim=2)
        x2 = x[:, -1, 1:]

        # h0 = torch.zeros(self.depth_number, x1.size(0), self.hidden_units).to(device)
        # c0 = torch.zeros(self.depth_number, x1.size(0), self.hidden_units).to(device)
        # out2, _ = self.lstm1(x1, (h0, c0))
        # out2 = self.fc1(out2[:, -1, :])
        # outs1 = torch.sin(self.relu(out2))
        # outs1 = self.fc2(outs1)

        h0 = torch.zeros(self.depth_number, x1.size(0), self.hidden_units).to(device)
        c0 = torch.zeros(self.depth_number, x1.size(0), self.hidden_units).to(device)
        out2, _ = self.lstm2(x1, (h0, c0))
        out2 = self.fc3(out2[:, -1, :])
        outs2 = self.relu(out2)
        outs2 = self.fc4(outs2)

        lout = self.relu(self.layer1(x2))
        lout = self.layer2(lout)

        # nout = self.relu(self.new(x3))
        # nout = self.relu(self.new2(nout))

        # out = outs2 + self.relu(self.fcs3(outs1 + lout)) + torch.exp(-nout)
        out = outs2
        return out
