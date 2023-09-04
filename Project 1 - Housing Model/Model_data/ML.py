import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Housing_Model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(Housing_Model, self).__init__()
        self.hid1 = nn.Linear(input_size, hidden_size1)
        self.act1 = nn.ReLU()
        self.hid2 = nn.Linear(hidden_size1, hidden_size2)
        self.act2 = nn.ReLU()
        self.out_layer = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = self.act1(self.hid1(x))
        x = self.act2(self.hid2(x))
        x = self.out_layer(x)
        return x


class DataLoader:
    def __init__(self, data):
        self.data = data

    def preparation(self,random_state):
        scale = MinMaxScaler(feature_range=(0, 1))
        data_x = self.data.iloc[:, 1:].values
        data_y = self.data.iloc[:, 0].values
        data_x = scale.fit_transform(data_x)
        data_x = torch.tensor(data_x, dtype=torch.float32)
        data_y = torch.tensor(data_y, dtype=torch.float32).reshape(-1, 1)
        data_train, data_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=random_state)
        data_test, data_val, y_test, y_val, = train_test_split(data_test, y_test, test_size=0.5, random_state=random_state)
        return data_train, data_val, data_test, y_train, y_val, y_test

