import numpy as np
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from typing import List


class FlowPredict(nn.Module):
    """MLP with two layer and ReLU activation."""

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        """
        Make your model.
        """
        super(FlowPredict, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DataLoader_Me:
    """Prepare data and return as DataLoder"""

    def __init__(self, folder_name: str, miss_rate: float, batch_size: int, device: str, dr: bool = True) -> None:
        """
        :param folder_name: Folder for getting data from.
        :param miss_rate: Missing rate we are working with.[0.2]
        :param batch_size: Size data batch in dataloader.
        :param device: cpu or cuda
        :param dr: Drop the last layer or not.
        """
        self._get_data(folder_name, miss_rate)
        self._device = device
        self.batch_size = batch_size
        self.dr = dr

        index_len = len(self.travel_time.index) * len(self.travel_time.columns)
        attraction = pd.concat([self.attraction] * len(self.travel_time.index), axis=0)
        attraction = attraction.reset_index(drop=True)

        production = pd.DataFrame(np.repeat(self.production.values, len(self.travel_time.columns), axis=0),
                                  columns=self.production.columns)
        production = production.reset_index(drop=True)
        self.data = pd.DataFrame({"travel_time": self.travel_time.values.reshape(index_len),
                                  "production": production.iloc[:, 0],
                                  "attraction": attraction.iloc[:, 0]}, index=range(index_len))

        y_train = pd.DataFrame(self.train_data.values.reshape(index_len), index=range(index_len))
        mask_train = ~y_train.isin(["False", "No_connection"])
        self.test = mask_train
        self.y_train = y_train[mask_train]
        self.y_train.dropna(inplace=True)

        y_val = pd.DataFrame(self.val_data.values.reshape(index_len), index=range(index_len))
        mask_val = ~y_val.isin(["False", "No_connection"])
        self.y_val = y_val[mask_val]
        self.y_val.dropna(inplace=True)

        y_test = pd.DataFrame(self.test_data.values.reshape(index_len), index=range(index_len))
        mask_test = ~y_test.isin(["False", "No_connection"])
        self.y_test = y_test[mask_test]
        self.y_test.dropna(inplace=True)

        mask_train = pd.DataFrame({"travel_time": mask_train.iloc[:, 0], "production": mask_train.iloc[:, 0],
                                   "attraction": mask_train.iloc[:, 0]})
        self.x_train = self.data[mask_train]
        self.x_train.dropna(inplace=True)

        mask_val = pd.DataFrame(
            {"travel_time": mask_val.iloc[:, 0], "production": mask_val.iloc[:, 0], "attraction": mask_val.iloc[:, 0]})
        self.x_val = self.data[mask_val]
        self.x_val.dropna(inplace=True)

        mask_test = pd.DataFrame({"travel_time": mask_test.iloc[:, 0], "production": mask_test.iloc[:, 0],
                                  "attraction": mask_test.iloc[:, 0]})
        self.x_test = self.data[mask_test]
        self.x_test.dropna(inplace=True)

        self._make_tensor()

    def _get_data(self, folder_name: str, miss_rate: float) -> None:
        """
        Collect the data prom the path.
        :param folder_name: Name of folder we get data from.
        :param miss_rate: The missing rate we are working with[0.2].
        """
        self.production = pd.read_csv(f"{folder_name}/production.csv")
        self.production.index = self.production.iloc[:, 0]
        self.production = self.production.iloc[:, 1:]
        self.attraction = pd.read_csv(f"{folder_name}/attraction.csv")
        self.attraction.index = self.attraction.iloc[:, 0]
        self.attraction = self.attraction.iloc[:, 1:]
        self.travel_time = pd.read_csv(f"{folder_name}/travel_time_matrix.csv")
        self.travel_time.index = self.travel_time.iloc[:, 0]
        self.travel_time = self.travel_time.iloc[:, 1:]
        self.train_data = pd.read_csv(f"{folder_name}/at_miss{miss_rate}_train_od_matrix.csv")
        self.train_data.index = self.train_data.iloc[:, 0]
        self.train_data = self.train_data.iloc[:, 1:]
        self.val_data = pd.read_csv(f"{folder_name}/at_miss{miss_rate}_val_od_matrix.csv", low_memory=False)
        self.val_data.index = self.val_data.iloc[:, 0]
        self.val_data = self.val_data.iloc[:, 1:]
        self.test_data = pd.read_csv(f"{folder_name}/at_miss{miss_rate}_test_od_matrix.csv", low_memory=False)
        self.test_data.index = self.test_data.iloc[:, 0]
        self.test_data = self.test_data.iloc[:, 1:]

    def _make_tensor(self):
        """Turn data to tensor format and send them to the right device."""
        self.x_train = torch.tensor(self.x_train.values, dtype=torch.float32, device=self._device)
        self.x_val = torch.tensor(self.x_val.values, dtype=torch.float32, device=self._device)
        self.x_test = torch.tensor(self.x_test.values, dtype=torch.float32, device=self._device)
        self.y_train = torch.tensor(self.y_train.astype(float).values, dtype=torch.float32,
                                    device=self._device).reshape(-1, 1)
        self.y_val = torch.tensor(self.y_val.astype(float).values, dtype=torch.float32, device=self._device).reshape(-1,
                                                                                                                     1)
        self.y_test = torch.tensor(self.y_test.astype(float).values, dtype=torch.float32, device=self._device).reshape(
            -1, 1)

        self.train = list(zip(self.x_train, self.y_train))
        self.val = list(zip(self.x_val, self.y_val))
        self.test = list(zip(self.x_test, self.y_test))
        self.testt = self.test.copy()
        self.train = DataLoader(self.train, batch_size=self.batch_size, shuffle=False, drop_last=self.dr)
        self.val = DataLoader(self.val, batch_size=self.batch_size, shuffle=False, drop_last=self.dr)
        self.test = DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=self.dr)


def train_model(model: torch.nn.Module, train: torch.utils.data.DataLoader, val: torch.utils.data.DataLoader,
                epochs: int, learning_rate: float) -> [List, List, List]:
    """This will train the model completely...
    :return: three lists including epochs, train_losses, validation_losses
    """
    count_epoch = []
    loss_values = []
    val_loss_values = []
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    patient = 5
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        for batch, (data_train, y_train) in enumerate(train):
            model.train()
            y_pred = model(data_train)
            loss = criterion(y_pred, y_train)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            for data, y in val:
                val_pred = model(data)
                val_loss += criterion(val_pred, y)

            train_loss /= len(train)
            val_loss /= len(val)
            print(f"___{train_loss:.6f}___{val_loss:.6f}___")

            if val_loss <= best_loss:
                best_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            count_epoch.append(epoch)
            loss_values.append(train_loss)
            val_loss_values.append(val_loss)

            if early_stop_counter >= patient:
                print(f"Early_Stop_at_ {epoch} Epoch")
                break

    return count_epoch, loss_values, val_loss_values


def evaluate_model(model: torch.nn.Module, val: torch.utils.data.DataLoader) -> [List, List]:
    """Evaluate model"""
    model.eval()
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()
    val_mae, val_mse, val_r2 = 0, 0, 0
    predict, real = [], []
    with torch.inference_mode():
        for data, y in val:
            val_pred = model(data)
            val_mae += MAE(val_pred, y)
            val_mse += MSE(val_pred, y)
            val_r2 += r2_score(val_pred, y)
            predict.append(val_pred)
            real.append(y)
        val_mae /= len(val)
        val_mse /= len(val)
        val_rmse = val_mse ** 0.5
        data_nums = [np.array(predict).reshape(-1, 1), np.array(real).reshape(-1, 1)]
        val_r2 = r2_score(data_nums[1], data_nums[0])
    return [val_rmse, val_mae, val_r2], data_nums


def test_model(model: torch.nn.Module, test: torch.utils.data.DataLoader) -> [List, List]:
    """Test Model"""
    model.eval()
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()
    test_mae, test_mse, test_r2 = 0, 0, 0
    predict, real = [], []
    with torch.inference_mode():
        for data, y in test:
            test_pred = model(data)
            test_mae += MAE(test_pred, y)
            test_mse += MSE(test_pred, y)
            predict.append(test_pred)
            real.append(y)
        test_mae /= len(test)
        test_mse /= len(test)
        test_rmse = test_mse ** 0.5
        data_nums = [np.array(predict).reshape(-1, 1), np.array(real).reshape(-1, 1)]
        test_r2 = r2_score(data_nums[1], data_nums[0])
    return [test_rmse, test_mae, test_r2], data_nums


def plot_fn(data, save=True, show=True, model=None, bins=None) -> None:
    """Plot R2 and Histogram."""
    fig_r = plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(data[0].squeeze(), data[1].squeeze())
    b = max(max(data[0]), max(data[1]))
    plt.scatter([0, b[0]], [0, b[0]])
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(
        f'Scatter Plot of Actual vs Predicted Values and R2 is {r2_score(data[1].squeeze(), data[0].squeeze()):.2f}')
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.plot([x_min, x_max], [y_min, y_max], ls="--", c=".3")

    plt.subplot(1, 2, 2)
    plt.hist(data[0].squeeze() - data[1].squeeze(), bins=bins)
    plt.xlabel('Predicted Values - Real Values')
    plt.ylabel("#")
    plt.title(f'Histogram of Values')
    folder = "Plots"
    zone = model.split("_")[0]
    seed = model.split("_")[1]
    sub_folder = f"{zone}_plots/random_{seed}"
    if save:
        directory = f"{folder}/NN_plots/{sub_folder}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig_r.savefig(f"{folder}/NN_plots/{sub_folder}/{model}mis.png")

    if show:
        plt.show()
    else:
        plt.close()
