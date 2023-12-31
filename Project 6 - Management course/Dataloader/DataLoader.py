import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class My_Dataset(Dataset):
    features = ['Duration', 'Number_of_Attendance', 'International', 'Renting']
    target = ['Number_of_Attendance', 'International', 'Renting']

    def __init__(self, data_: pd.DataFrame, sequence_length: int):
        self.data = data_
        self.X = self.data[My_Dataset.features].values
        self.y = self.data[My_Dataset.target].values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, index):
        X = torch.tensor(self.X[index:index + self.sequence_length], dtype=torch.float)
        y = torch.tensor(self.y[index + self.sequence_length - 1], dtype=torch.float)
        return X, y


class Dataloader:
    halls = {"Jaber": 0, "Theater": 1, "Mechanic": 2, "Kahroba": 3, "Rabiee": 4, "Sabz": 5, "Borgeii": 6, "Jabari": 7}

    def __init__(self, data_path: str, sequence_length: int, seed: int = 42):
        self.path = Path(data_path)
        self.data = pd.read_csv(data_path)
        self._setting()
        self.seed = seed
        self._split_data(train_size=0.8, val_size=0.1, test_size=0.1)
        self._scale()
        self.seq = sequence_length
        self._create_datasets(sequence_length)

    def _setting(self):
        self.data["Place_index"] = self.data["Place"].apply(lambda x: Dataloader.halls[x])
        self.data["International"] = self.data["International"].apply(lambda x: 1 if x else 0)
        self.data = self.data.drop(["Event_name", "Date", "Place"], axis=1)
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
        scaler_ = MinMaxScaler()
        self.train_data = pd.DataFrame(scaler_.fit_transform(self.train_data), columns=self.columns)
        self.val_data = pd.DataFrame(scaler_.transform(self.val_data), columns=self.columns)
        self.test_data = pd.DataFrame(scaler_.transform(self.test_data), columns=self.columns)

    def _create_datasets(self, seq):
        self.train_dataset = My_Dataset(self.train_data, seq)
        self.val_dataset = My_Dataset(self.val_data, seq)
        self.test_dataset = My_Dataset(self.test_data, seq)

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
