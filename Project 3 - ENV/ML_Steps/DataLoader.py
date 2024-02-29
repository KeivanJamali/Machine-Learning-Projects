import torch
import pandas as pd
import Information as info

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        """
        Initializes a new instance of the class.

        Args:
            data (pd.DataFrame): The data to be used for initialization.

        Returns:
            None
        """
        self.data = data
        self.sequence = info.sequence
        self.X = self.data[info.features].values
        self.y = self.data[info.target].values

    def __len__(self):
        """You most probably need to change the slices."""
        if info.model == "RNN":
            return len(self.data) - self.sequence
        elif info.model == "CNN":
            return len(self.data)
        # else:
        #     return ?

    def __getitem__(self, item):
        """You most probably need to change the slices."""
        slicing_x = [item, item + 1] if info.model == "CNN" else [item, item + self.sequence]
        slicing_y = [item, item + 1] if info.model == "CNN" else [item + self.sequence, item + self.sequence + 1]

        # If you need a unique slice, you can use the following code
        # slicing_x = ?
        # slicing_y = ?

        x = torch.tensor(self.X[slicing_x[0]:slicing_x[1]], dtype=torch.float)
        y = torch.tensor(self.y[slicing_y[0]:slicing_y[1]], dtype=torch.float)
        return x, y


class MyDataloader:
    def __init__(self, file_path: str, train_percent: float, val_percent: float, batch_size: int):
        """
        Initializes the object with the specified parameters.

        Parameters:
            file_path (str): The path to the file containing the data.
            train_percent (float): The percentage of data to use for training.
            val_percent (float): The percentage of data to use for validation.
            batch_size (int): The batch size for training.
        """
        self.train_dataloader, self.val_dataloader, self.test_dataloader = None, None, None
        file_path = Path(file_path)
        self.data = pd.read_csv(file_path)
        self.random_state = info.random_seed
        self.batch_size = batch_size
        train_data, val_data, test_data = self._split_data(train_percent=train_percent, val_percent=val_percent,
                                                           test_percent=(1 - train_percent - val_percent))
        self.train_data, self.val_data, self.test_data = self._scale_data(train_data, val_data, test_data)
        self.train_dataset, self.val_dataset, self.test_dataset = self._make_datasets(self.train_data, self.val_data,
                                                                                      self.test_data)

    def _setting(self):
        """
        A function that performs some setting operation.
        """
        self.data.drop("date", axis=1, inplace=True)

    def _split_data(self, train_percent, val_percent, test_percent) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training, validation, and testing sets.

        Parameters:
            train_percent (float): The percentage of data to be used for training.
            val_percent (float): The percentage of data to be used for validation.
            test_percent (float): The percentage of data to be used for testing.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and testing dataframes.
        """
        train_data, val_test_data = train_test_split(self.data, train_size=train_percent,
                                                     random_state=self.random_state)
        val_data, test_data = train_test_split(val_test_data, train_size=val_percent / (val_percent + test_percent),
                                               random_state=self.random_state)

        return train_data, val_data, test_data

    def _scale_data(self, train_data, val_data, test_data) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale the data data using MinMaxScaler.

        Parameters:
            train_data (pd.DataFrame): The training data.
            val_data (pd.DataFrame): The validation data.
            test_data (pd.DataFrame): The test data.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The scaled training, validation, and test data.
        """
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_data.loc[:, info.features] = self.scaler_x.fit_transform(train_data[info.features])
        train_data.loc[:, info.target] = self.scaler_y.fit_transform(train_data[info.target])
        val_data.loc[:, info.features] = self.scaler_x.transform(val_data[info.features])
        val_data.loc[:, info.target] = self.scaler_y.transform(val_data[info.target])
        test_data.loc[:, info.features] = self.scaler_x.transform(test_data[info.features])
        test_data.loc[:, info.target] = self.scaler_y.transform(test_data[info.target])
        return train_data, val_data, test_data

    def _make_datasets(self, train_data, val_data, test_data) -> tuple:
        """
        Generates the datasets for training, validation, and testing.

        Args:
            train_data (np.ndarray): The training data.
            val_data (np.ndarray): The validation data.
            test_data (np.ndarray): The testing data.

        Returns:
            tuple: A tuple containing the train dataset, val dataset, and test dataset.
        """
        train_dataset = MyDataset(train_data)
        val_dataset = MyDataset(val_data)
        test_dataset = MyDataset(test_data)
        return train_dataset, val_dataset, test_dataset

    def _make_dataloader(self, train_dataset, val_dataset, test_dataset) -> tuple:
        """
        Create dataloaders for the given datasets.

        Parameters:
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            test_dataset (Dataset): The testing dataset.

        Returns:
            tuple: A tuple containing train_dataloader, val_dataloader, and test_dataloader.
        """
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        return train_dataloader, val_dataloader, test_dataloader

    def fit(self):
        """
        Fits the model by creating dataloaders for the train, validation, and test datasets.

        Returns:
            train_dataloader (torch.utils.data.DataLoader): Dataloader for the train dataset.
            val_dataloader (torch.utils.data.DataLoader): Dataloader for the validation dataset.
            test_dataloader (torch.utils.data.DataLoader): Dataloader for the test dataset.
        """
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._make_dataloader(self.train_dataset,
                                                                                                 self.val_dataset,
                                                                                                 self.test_dataset)
        return self.train_dataloader, self.val_dataloader, self.test_dataloader
