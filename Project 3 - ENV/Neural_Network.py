import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class AOV_prediction(nn.Module):
    def __init__(self, input_layer, hidden_layer_1, hidden_layer_2, out_layer):
        super().__init__()
        self.l1 = nn.Linear(in_features=input_layer, out_features=hidden_layer_1)
        self.l2 = nn.Linear(in_features=hidden_layer_1, out_features=hidden_layer_2)
        self.out = nn.Linear(in_features=hidden_layer_2, out_features=out_layer)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.out(x)


class DataLoader:
    def __init__(self, file_path, random_state):
        self.data = pd.read_csv(file_path)
        self.random_state = random_state

    def _split_data(self) -> list:
        data = self.data.iloc[:, 2:]
        y = self.data.iloc[:, 1]
        train_data, test_data, train_y, test_y = train_test_split(data, y, test_size=0.2,
                                                                  random_state=self.random_state)
        return [train_data.values, train_y.values.reshape(-1, 1), test_data.values, test_y.values.reshape(-1, 1)]

    def _scale_data(self, train_data, test_data) -> list:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        return [train_data, test_data]

    def _tensor_maker(self, train_data, train_y, test_data, test_y) -> list:
        train_data = torch.tensor(train_data, requires_grad=True, dtype=torch.float)
        train_y = torch.tensor(train_y, requires_grad=True, dtype=torch.float)
        test_data = torch.tensor(test_data, requires_grad=True, dtype=torch.float)
        test_y = torch.tensor(test_y, requires_grad=True, dtype=torch.float)
        return [train_data, train_y, test_data, test_y]

    def data_prepared(self):
        train_data, train_y, test_data, test_y = self._split_data()
        train_data, test_data = self._scale_data(train_data, test_data)
        train_data, train_y, test_data, test_y = self._tensor_maker(train_data, train_y, test_data, test_y)
        return [train_data, train_y, test_data, test_y]


def plot_predictions(train: list = None, test: list = None, predict=None) -> None:
    plt.figure(figsize=(6, 4))
    if train:
        plt.scatter(range(len(train[0])), train[1], c="b", s=5, label="Train data")
    if test:
        plt.scatter(range(len(test[0])), test[1].detach(), c="g", s=5, label="Test data")
    if predict is not None:
        plt.scatter(range(len(test[0])), predict, c="r", s=5, label="Predicted data")
    plt.legend(prop={"size": 15})


def flow(model: AOV_prediction, train: list, test: list, epochs: int = 100, batch_size: int = 32, lr: float = 0.1):
    loss_values = []
    test_loss_values = []
    epoch_count = []
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i in range(0, len(train[0]), batch_size):
            model.train()
            X_train = train[0][i:i + batch_size]
            y_train = train[1][i:i + batch_size]
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        y_pred2 = model(train[0])
        loss1 = loss_fn(y_pred2, train[1])
        with torch.inference_mode():
            test_pred = model(test[0])
            test_loss = loss_fn(test_pred, test[1])

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss1.item())
            test_loss_values.append(test_loss.item())
            print(f"Epoch: {epoch} | Loss: {loss1} | Test_Loss: {test_loss}")

    return model, [epoch_count, loss_values, test_loss_values]


def test_model(model: AOV_prediction, test: list) -> None:
    model.eval()
    with torch.inference_mode():
        pred = model(test[0])
    print(f"r2 is : {r2_score(pred, test[1].detach())}")
    plot_predictions(test=test, predict=pred)
