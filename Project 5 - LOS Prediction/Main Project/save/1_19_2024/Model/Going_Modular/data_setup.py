"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from pathlib import Path


class Dataloader:

    def __init__(self):
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.class_names = None
        self.class_to_idx = None
        # self.create_dataloaders(data_dir=data_dir, city_code=city_code, batch_size=batch_size)

    def create_dataloaders(self,
                           data_dir: Path,
                           city_code: str,
                           batch_size: int,
                           ):
        """Creates training and testing DataLoaders.

        Takes in a training directory and testing directory path and turns
        them into PyTorch Datasets and then into PyTorch DataLoaders.

        Args:
          data_dir: Path to data directory.
          batch_size: Number of samples per batch in each of the DataLoaders.
          city_code: name of the city for loading the data from.

        Returns:
          A tuple of (train_dataloader, test_dataloader, class_names).
          Where class_names is a list of the target classes.
          Example usage:
            train_dataloader, test_dataloader, class_names = \
              = create_dataloaders(train_dir=path/to/train_dir,
                                   test_dir=path/to/test_dir,
                                   batch_size=32)
        """
        train_data = My_Dataset(data_dir=data_dir, city_code=city_code, group="train")
        val_data = My_Dataset(data_dir=data_dir, city_code=city_code, group="val")
        test_data = My_Dataset(data_dir=data_dir, city_code=city_code, group="test")

        # Get class names
        self.class_names = ["A", "B", "C", "D", "E", "F"]
        self.class_to_idx = [0, 1, 2, 3, 4, 5]

        # Turn DataFrame into data loaders
        self.train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        self.val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        self.test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        return self.train_dataloader, self.val_dataloader, self.test_dataloader


class My_Dataset(Dataset):
    def __init__(self, data_dir: Path, city_code: str, group: str):
        # Use pd to create data(s)
        self.data = pd.read_csv(data_dir / f"{group}_data_{city_code}.csv")
        self.data.index = self.data.iloc[:, 0]
        self.data = self.data.iloc[:, 1:]
        self.data = self.data[["flow", "occ", "rainfall", "visibility", "windspeed", "feelslike", "LOS_index"]]
        self.data.dropna(inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = torch.tensor(
            self.data[["flow", "occ", "rainfall", "visibility", "windspeed", "feelslike"]].values[index],
            dtype=torch.float)  # Select all columns for data
        class_index = self.data["LOS_index"].values[index]  # Select the column as the label
        return inputs, class_index
