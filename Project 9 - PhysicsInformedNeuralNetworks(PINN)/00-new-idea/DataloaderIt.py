import torch

import pandas as pd
import Information

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split


class DatasetIt(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.x = data.iloc[:, 0].values
        self.y = data.iloc[:, 1].values
        self.x = torch.tensor(self.x, dtype=torch.float)
        self.y = torch.tensor(self.y, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]
        return x, y


class DataLoaderIt:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.train_data, self.val_data, self.test_data = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.train_dataloader, self.val_dataloader, self.test_dataloader = None, None, None
        self.scaler_x = None
        self.scaler_y = None
        self.random_state = None

    def fit(self, train_size=None, val_size=None, test_size=None, batch_size=None, drop_last=True):
        # make randomness ready.
        torch.manual_seed(Information.random_state_data)
        self.random_state = Information.random_state_data

        # split data.
        self.train_data, self.val_data, self.test_data = self._split_data(self.data, train_size, val_size, test_size)

        # scale data.
        self._scale_data()

        # get dataloaders.
        if batch_size is None:
            batch_size = len(self.train_data)
            drop_last = False
            print("batch_size set to", batch_size)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._get_dataloaders(self.train_data,
                                                                                                 self.val_data,
                                                                                                 self.test_data,
                                                                                                 batch_size=batch_size,
                                                                                                 drop_last=drop_last)
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def _scale_data(self):
        max_val = self.train_data.iloc[:, 0].max()
        shape_train = self.train_data.iloc[:, 0].shape
        shape_val = self.val_data.iloc[:, 0].shape
        shape_test = self.test_data.iloc[:, 0].shape
        if max_val == 0:
            raise ValueError("Please provide at least one training data more than 0 to scale it or change scaler.")

        def transformer(x):
            return x / max_val

        def inverse_transformer(x):
            return x * max_val

        self.scaler_x = FunctionTransformer(func=transformer, inverse_func=inverse_transformer, validate=True)
        self.train_data.iloc[:, 0] = self.scaler_x.fit_transform(self.train_data.values[:, 0].reshape(-1, 1)).reshape(
            shape_train)
        self.val_data.iloc[:, 0] = self.scaler_x.transform(self.val_data.values[:, 0].reshape(-1, 1)).reshape(shape_val)
        self.test_data.iloc[:, 0] = self.scaler_x.transform(self.test_data.values[:, 0].reshape(-1, 1)).reshape(
            shape_test)

    def _split_data(self, data, train: float, val: float, test: float, random_state=None):
        if train is not None and val is not None and test is not None:
            check = train + val + test == 1
            if not check:
                raise ValueError("The sum of train, validation, and test sizes must be equal to 1.")
        elif train is not None and val is not None:
            test = 1 - train - val
        elif train is not None and test is not None:
            val = 1 - train - test
        elif val is not None and test is not None:
            train = 1 - val - test
        else:
            raise ValueError("Please provide at lease two of (train, validation, and test) sizes.")

        train_data, valtest_data = train_test_split(data, train_size=train, random_state=self.random_state)
        val_data, test_data = train_test_split(valtest_data, train_size=val / (val + test),
                                               random_state=self.random_state)
        return train_data, val_data, test_data

    @staticmethod
    def _get_datasets(train_data, val_data, test_data):
        train_dataset = DatasetIt(train_data)
        val_dataset = DatasetIt(val_data)
        test_dataset = DatasetIt(test_data)
        return train_dataset, val_dataset, test_dataset

    def _get_dataloaders(self, train_data, val_data, test_data, batch_size, drop_last):
        # get datasets.
        train_dataset, val_dataset, test_dataset = self._get_datasets(train_data, val_data, test_data)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
        return train_dataloader, val_dataloader, test_dataloader
