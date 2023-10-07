import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class DataLoader_Me:
    def __init__(self, data, seq, batch_size, dr=False):
        self.data = data
        self.x_data = self.data.iloc[:, :]
        self.y_data = self.data.iloc[:, 0]
        # self._scale()
        self.x_data, self.y_data = self._create_sequence([self.x_data, self.y_data], len_seq=seq)
        self.x_train, self.y_train, self.x_test, self.y_test = self._scale_split()
        self._make_tensor()
        self._dataloader(batch_size=batch_size, dr=dr)

    def _scale(self):
        self.scaler = StandardScaler()
        self.x_data = self.scaler.fit_transform(self.x_data)

    def _scale_split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x_data, self.y_data, test_size=0.2, shuffle=False)
        return x_train, y_train, x_test, y_test

    def _make_tensor(self):
        self.x_train = torch.tensor(self.x_train, dtype=torch.float)
        self.y_train = torch.tensor(self.y_train.reshape(-1, 1), dtype=torch.float)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float)
        self.y_test = torch.tensor(self.y_test.reshape(-1, 1), dtype=torch.float)

    def _create_sequence(self, data, len_seq):
        x = []
        y = []
        for i in range(len(data[0]) - len_seq):
            seq = data[0][i:i + len_seq]
            target = data[1][i + len_seq + 1]
            x.append(seq)
            y.append(target)
        return np.array(x), np.array(y)

    def _dataloader(self, batch_size, dr):
        self.train = DataLoader(list(zip(self.x_train, self.y_train)), batch_size=batch_size, drop_last=dr,
                                shuffle=False)
        self.test = DataLoader(list(zip(self.x_test, self.y_test)), batch_size=batch_size, drop_last=dr, shuffle=True)


class Health(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_model(model, train, loss_fn, optimizer, epochs):
    epoch_count = []
    loss_values = []
    best_loss = float("inf")
    patient = 5
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for batch, (x, y) in enumerate(train):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            train_loss /= len(train)
            if train_loss < best_loss:
                early_stop_count = 0
            else:
                early_stop_count += 1
            epoch_count.append(epoch)
            loss_values.append(train_loss)
            if early_stop_count >= patient:
                print(f"Early Stop At Epoch {epoch}")
                break

    return [epoch_count, loss_values], model


def test_model(model, test, loss_fn):
    test_loss = 0
    r2 = 0
    with torch.inference_mode():
        for x, y in test:
            y_pred = model(x)
            y_pred = torch.round(y_pred)
            test_loss += loss_fn(y_pred, y)
            r2 += r2_score(y, y_pred)

        test_loss /= len(test)
        r2 /= len(test)

    return [test_loss, r2]
