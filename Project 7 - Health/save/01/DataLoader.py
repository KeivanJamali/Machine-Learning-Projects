import torch
import Information
import pandas as pd
import numpy as np
import sklearn

from pathlib import Path
from torch.utils.data import DataLoader, Dataset



class MyDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        """
        Initializes a new instance of the class

        Args:
            data (pd.DataFrame): The data to be used for initialization.

        Returns:
            None
        """
        self.data = data
        self.sequence = Information.sequence
        self.X = self.data[Information.features].values
        self.y = self.data[Information.target].values

    def __len__(self):
        """You most probably need to change the slices."""
        return len(self.data) - self.sequence

    def __getitem__(self, item):
        """You most probably need to change the slices."""
        x = torch.tensor(self.X[item:item + self.sequence+1], dtype=torch.float)
        y = torch.tensor(self.y[item + self.sequence], dtype=torch.float)
        return x, y


class MyDataloader:
    def __init__(self, file_path: str, 
                 train_percent: float, 
                 val_percent: float, 
                 test_percent: float, 
                 batch_size: int, 
                 scalers: list=False, 
                 func=None,
                 func_inverse=None,
                 suppress_warnings=True):
        """
        Initializes the object with the specified parameters.

        Parameters:
            file_path (str): The path to the file containing the data.
            train_percent (float): The percentage of data to use for training.
            val_percent (float): The percentage of data to use for validation.
            batch_size (int): The batch size for training.
        """
        self.train_dataloader, self.val_dataloader, self.test_dataloader = None, None, None
        self.features, self.target = None, None
        self.scaler_x, self.scaler_y = [], []
        file_path = Path(file_path)
        self.data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        self.random_state = Information.random_seed
        self.batch_size = batch_size
        self.train_data, self.val_data, self.test_data = self._split_data(train_percent=train_percent, val_percent=val_percent,
                                                                          test_percent=test_percent)
        self.train_datas, self.val_datas, self.test_datas = self.train_data.copy(), self.val_data.copy(), self.test_data.copy()
        self._setting()
        for i in range(len(scalers)):
            self._select_scaler(scaler=scalers[i], func=func, func_inverse=func_inverse)
            self.train_datas, self.val_datas, self.test_datas = self._scale_data(self.train_datas, self.val_datas, self.test_datas, scaler=i)
            print(f"[INFO] Scaled with {scalers[i]}.")
            
            
        self.train_dataset, self.val_dataset, self.test_dataset = self._make_datasets(self.train_datas, self.val_datas,
                                                                                      self.test_datas)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._make_dataloader(self.train_dataset,
                                                                                                 self.val_dataset,
                                                                                                 self.test_dataset)
                                                                                                 
        if suppress_warnings:
            from IPython.display import clear_output
            clear_output(wait=True)


    def _setting(self):
        """
        A function that performs some setting operation.
        """
        self.features = Information.features[1:]
        self.target = Information.target

    def _split_data(self, train_percent, val_percent, test_percent=None) -> tuple:
        """
        Splits the data into training, validation, and testing sets.

        Parameters:
            train_percent (float): The percentage of data to be used for training.
            val_percent (float): The percentage of data to be used for validation.
            test_percent (float): The percentage of data to be used for testing.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and testing dataframes.
        """
        train_data = self.data.iloc[:int(train_percent * len(self.data)), :]
        val_test_data = self.data.iloc[
                        int(train_percent * len(self.data)):int((val_percent + train_percent) * len(self.data)), :]
        if test_percent:
            val_data = val_test_data
            test_data = self.data.iloc[int((val_percent + train_percent) * len(self.data)):, :]
            return train_data, val_data, test_data
        else:
            return train_data, val_test_data, []

    def _scale_data(self, train_data, val_data, test_data, scaler) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale the data data using MinMaxScaler.

        Parameters:
            train_data (pd.DataFrame): The training data.
            val_data (pd.DataFrame): The validation data.
            test_data (pd.DataFrame): The test data.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The scaled training, validation, and test data.
        """
        # shape_train = train_data2.loc[:, Information.features].shape
        # shape_val = val_data2.loc[:, Information.features].shape

        train_data.loc[:, self.features] = self.scaler_x[scaler].fit_transform(train_data[self.features].values)
        train_data.loc[:, self.target] = self.scaler_y[scaler].fit_transform(train_data[self.target].values)
        val_data.loc[:, self.features] = self.scaler_x[scaler].transform(val_data[self.features].values)
        val_data.loc[:, self.target] = self.scaler_y[scaler].transform(val_data[self.target].values)
        if test_data:
            test_data.loc[:, self.features] = self.scaler_x[scaler].transform(test_data[self.features].values)
            test_data.loc[:,self.target] = self.scaler_y[scaler].transform(test_data[self.target].values)

        return train_data, val_data, test_data

    @staticmethod
    def _make_datasets(train_data, val_data, test_data) -> tuple:
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
        if test_data:
            test_dataset = MyDataset(test_data)
            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, val_dataset, None

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

    def _select_scaler(self, scaler, func=None, func_inverse=None):
        if scaler == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler_x.append(MinMaxScaler(feature_range=(0, 1)))
            self.scaler_y.append(MinMaxScaler(feature_range=(0, 1)))
        elif scaler == "StandardScaler":
            from sklearn.preprocessing import StandardScaler
            self.scaler_x.append(StandardScaler())
            self.scaler_y.append(StandardScaler())
        elif scaler == "RobustScaler":
            from sklearn.preprocessing import RobustScaler
            self.scaler_x.append(RobustScaler())
            self.scaler_y.append(RobustScaler())
        elif scaler == "MaxAbsScaler":
            from sklearn.preprocessing import MaxAbsScaler
            self.scaler_x.append(MaxAbsScaler())
            self.scaler_y.append(MaxAbsScaler())
        elif scaler == "PowerTransformer":
            from sklearn.preprocessing import PowerTransformer
            self.scaler_x.append(PowerTransformer(method="yeo-johnson"))
            self.scaler_y.append(PowerTransformer(method="yeo-johnson"))
        elif scaler == "Custom" and func and func_inverse:
            from sklearn.preprocessing import FunctionTransformer
            self.scaler_x.append(FunctionTransformer(func=func, inverse_func=func_inverse, validate=True))
            self.scaler_y.append(FunctionTransformer(func=func, inverse_func=func_inverse, validate=True))
        elif not scaler:
            from sklearn.preprocessing import FunctionTransformer
            self.scaler_x.append(FunctionTransformer(func=lambda x:x, inverse_func=lambda x:x, validate=True))
            self.scaler_y.append(FunctionTransformer(func=lambda x:x, inverse_func=lambda x:x, validate=True))
        else:
            raise ValueError("[ERROR] Invalid scaler. [MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, Custom(need functions.)].")

            