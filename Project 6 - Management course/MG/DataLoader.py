import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class My_Dataset_Time_Series(Dataset):
    def __init__(self, data_: pd.DataFrame, sequence_length: int):
        self.data = data_
        self.X = self.data[Dataloader.features].values
        self.y = self.data[Dataloader.target].values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, index):
        X = torch.tensor(self.X[index:index + self.sequence_length], dtype=torch.float)
        y = torch.tensor(self.y[index + self.sequence_length - 1], dtype=torch.float)
        return X, y


class My_Dataset_CNN(Dataset):
    def __init__(self, data_: pd.DataFrame, sequence_length: int):
        self.data = data_
        self.X = self.data[Dataloader.features].values
        self.y = self.data[Dataloader.target].values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = torch.tensor(self.X[index], dtype=torch.float)
        y = torch.tensor(self.y[index], dtype=torch.float)
        return X, y


class Dataloader:
    halls = {"Jaber": 0, "Theater": 1, "Mechanic": 2, "Kahroba": 3, "Rabiee": 4, "Sabz": 5, "Borgeii": 6, "Jabari": 7}
    features = ["Duration(hr)", "International", "Renting(millionToman)", "Expenses(millionToman)", "count",
                "Sound_System_quality", "Capacity"]
    target = ["Number_of_Attendance"]
    all_features = features + target

    def __init__(self, data_path: str, approach: str, sequence_length: int, seed: int = 42):
        self.path = Path(data_path)
        self.approach = approach
        self.data = pd.read_csv(data_path)
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self._setting()
        self.seed = seed
        self._split_data(train_size=0.8, val_size=0.1, test_size=0.1)
        self.test = self.train_data
        self._scale()
        self.seq = sequence_length
        self._create_datasets(sequence_length)

    def _setting(self):
        self.data["Place_index"] = self.data["Place"].apply(lambda x: Dataloader.halls[x])
        self.data["International"] = self.data["International"].apply(lambda x: 1 if x else 0)
        self.data["month_year"] = self.data["Date"].apply(lambda x: x.strftime("%m-%Y"))
        self.data = self.data.drop(["Event_name", "Place", "Date"], axis=1)
        count = self.data.groupby("month_year")["Duration(hr)"].count().values
        self.data = self.data.groupby(["month_year"]).mean()
        self.data["count"] = count
        self.month_year = self.data.index

        self.data = self.data[Dataloader.all_features]
        self.data.index = [i for i in range(len(self.data.index))]
        self.columns = self.data.columns

    def _split_data(self, train_size, val_size, test_size):
        # self.train_data, self.val_test_data = train_test_split(self.data, train_size=train_size, random_state=self.seed)
        # self.val_data, self.test_data = train_test_split(self.val_test_data,
        #                                                  test_size=test_size / (test_size + val_size),
        #                                                  random_state=self.seed)
        n = len(self.data)
        self.train_data = self.data.iloc[:int(n * train_size), :]
        self.val_data = self.data.iloc[int(n * train_size):int(n * train_size + n * val_size), :]
        self.test_data = self.data.iloc[int(n * train_size + n * val_size):, :]

    def _scale(self):
        self.scaler_inputs = MinMaxScaler()
        self.scaler_outputs = MinMaxScaler()
        self.train_data.loc[:, Dataloader.features] = self.scaler_inputs.fit_transform(
            self.train_data[Dataloader.features])
        self.train_data.loc[:, Dataloader.target] = self.scaler_outputs.fit_transform(
            self.train_data[Dataloader.target])
        self.val_data.loc[:, Dataloader.features] = self.scaler_inputs.transform(self.val_data[Dataloader.features])
        self.val_data.loc[:, Dataloader.target] = self.scaler_outputs.transform(self.val_data[Dataloader.target])
        self.test_data.loc[:, Dataloader.features] = self.scaler_inputs.transform(self.test_data[Dataloader.features])
        self.test_data.loc[:, Dataloader.target] = self.scaler_outputs.transform(self.test_data[Dataloader.target])

    def _create_datasets(self, seq):
        if self.approach == "Time_series":
            self.train_dataset = My_Dataset_Time_Series(self.train_data, seq)
            self.val_dataset = My_Dataset_Time_Series(self.val_data, seq)
            self.test_dataset = My_Dataset_Time_Series(self.test_data, seq)
        elif self.approach == "CNN":
            self.train_dataset = My_Dataset_CNN(self.train_data, seq)
            self.val_dataset = My_Dataset_CNN(self.val_data, seq)
            self.test_dataset = My_Dataset_CNN(self.test_data, seq)

    def creat_dataloaders(self, batch_size=32):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True)

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True)

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True)

        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def predict(self, model: torch.nn.Module, x: np.ndarray, device: str, if_scale_input: bool = True) -> pd.DataFrame:
        x = pd.DataFrame(x, columns=Dataloader.features)
        if if_scale_input:
            x_scaled = self.scaler_inputs.transform(x)
        else:
            x_scaled = x.values.copy()
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(dim=0)
        y_scaled = model(x_scaled.to(device)).cpu().detach().numpy()

        # x_predict = self.scaler_inputs.inverse_transform(np.array(x_scaled).reshape(1, -1))
        x_predict = x.values[-1].reshape(1, -1)
        y_predict = self.scaler_outputs.inverse_transform(np.array(y_scaled).reshape(1, -1))

        predict = pd.DataFrame(np.concatenate((x_predict, y_predict), axis=1),
                               columns=Dataloader.all_features)
        return predict
