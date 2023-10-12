import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


class Dataloader:
    def __init__(self, data_file_name: str, coordinate_file: str) -> None:
        """
        Here we are preparing data for GM model and GNN model.
        :param data_file_name: in this order: origin, dest, demand
        :param coordinate_file: name of nodal_components.(id, x, y : in degree coordinate)
        """

        self.production_scaled = None
        self.attraction_scaled = None
        self.random_state = None
        self._read_data(data_file_name)
        self.coordinate_file = coordinate_file

    def _read_data(self, data_file_name: str) -> None:
        """
        Read the file.
        :param data_file_name: Name of the file
        :return: data
        """
        if "dat" in data_file_name:
            self.data = pd.read_csv(data_file_name, sep="\s+")
        else:
            self.data = pd.read_csv(data_file_name)

        self.data.columns = [0, 1, 2]
        self.data_no_zero = self.data[self.data[2] != 0]
        self.data_zero = self.data[self.data[2] == 0]

    def _od_matrix_train_produce(self, scaled: bool = False) -> None:
        """
        it will produce the train od_matrix
        :param scaled: If you want to work with scaled data.
        :return: od_matrix
        """
        if scaled:
            od_matrix = pd.DataFrame(index=self.od_zones_real, columns=self.od_zones_real).fillna("False")
            for _, row, column, value in self.train_data_scaled.itertuples():
                od_matrix.at[row, column] = value
            for _, row, column, __ in self.data_zero.itertuples():
                od_matrix.at[row, column] = "No_connection"

            for i in range(self.od_n_real):
                od_matrix.iloc[i, i] = "No_connection"
            self.od_matrix_train_scaled = od_matrix
        else:
            od_matrix = pd.DataFrame(index=self.od_zones_real, columns=self.od_zones_real).fillna("False")
            for _, row, column, value in self.train_data.itertuples():
                od_matrix.at[row, column] = value
            for _, row, column, __ in self.data_zero.itertuples():
                od_matrix.at[row, column] = "No_connection"

            for i in range(self.od_n_real):
                od_matrix.iloc[i, i] = "No_connection"
            self.od_matrix_train = od_matrix

    def _od_matrix_test_produce(self, scaled: bool = False) -> None:
        """
        it will produce the test od_matrix for GNN
        :param scaled: If you want to work with scaled data.
        :return: od_matrix
        """
        if scaled:
            od_matrix = pd.DataFrame(index=self.od_zones_real, columns=self.od_zones_real).fillna("False")
            for _, row, column, value in self.test_data_scaled.itertuples():
                od_matrix.at[row, column] = value

            for i in range(self.od_n_real):
                od_matrix.iloc[i, i] = "No_connection"

            self.od_matrix_test_scaled = od_matrix
        else:
            od_matrix = pd.DataFrame(index=self.od_zones_real, columns=self.od_zones_real).fillna("False")
            for _, row, column, value in self.test_data.itertuples():
                od_matrix.at[row, column] = value

            for i in range(self.od_n_real):
                od_matrix.iloc[i, i] = "No_connection"

            self.od_matrix_test = od_matrix

    def _od_matrix_validation_produce(self, scaled: bool = False) -> None:
        """
        it will produce the validation od_matrix
        :param scaled: If you want to work with scaled data.
        :return: od_matrix
        """
        if scaled:
            od_matrix = pd.DataFrame(index=self.od_zones_real, columns=self.od_zones_real).fillna("False")
            for _, row, column, value in self.val_data_scaled.itertuples():
                od_matrix.at[row, column] = value

            for i in range(self.od_n_real):
                od_matrix.iloc[i, i] = "No_connection"
            self.od_matrix_val_scaled = od_matrix
        else:
            od_matrix = pd.DataFrame(index=self.od_zones_real, columns=self.od_zones_real).fillna("False")
            for _, row, column, value in self.val_data.itertuples():
                od_matrix.at[row, column] = value

            for i in range(self.od_n_real):
                od_matrix.iloc[i, i] = "No_connection"
            self.od_matrix_val = od_matrix

    @staticmethod
    def _distance_calculator(x1, y1, x2, y2, scale) -> float:
        """
        Calculate the distance between two points.
        :param x1: component x of point 1
        :param y1: component y of point 1
        :param x2: component x of point 2
        :param y2: component y of point 2
        :param scale: if you need, you can scale the data(multiple all cells by "scale").
        :return: Distance between point 1 and 2
        """
        return scale * (np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    @staticmethod
    def _haversine(x, y) -> float:
        """
        first we change degree to radian. then the formulation. and at last returning the distance in Km.
        :param x: component x of points
        :param y: component y of points
        :return: distance (km)
        """
        d_x = np.radians(x) - np.radians(x).reshape(-1, 1)
        d_y = np.radians(y) - np.radians(y).reshape(-1, 1)
        a = np.sin(d_x / 2) ** 2 + np.cos(np.radians(x)) * np.cos(
            np.radians(x).reshape(-1, 1)) * np.sin(d_y / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # radius of the Earth in kilometers
        return c * r

    def _data_split(self, missing_rate: float, val_rate: float, scaled: bool = False) -> None:
        """
        It will split the data into train and missed.
        :param missing_rate: What percent of data should be missed.
        """
        self.train_data, self.missed_data = train_test_split(self.data_no_zero, test_size=missing_rate,
                                                             random_state=self.random_state)
        if (1 / int(missing_rate * 10)) == 1:
            if scaled:
                self._data_scale()
                self.val_data_scaled, self.test_data_scaled = train_test_split(self.missed_data_scaled,
                                                                               train_size=val_rate,
                                                                               random_state=self.random_state)
            else:
                self.val_data, self.test_data = train_test_split(self.missed_data, train_size=val_rate,
                                                                 random_state=self.random_state)
        else:
            if scaled:
                self._data_scale()
                self.lost_data_scaled, self.test_val_scaled = train_test_split(self.missed_data_scaled,
                                                                               test_size=(1 / int(missing_rate * 10)),
                                                                               random_state=self.random_state)
                self.val_data_scaled, self.test_data_scaled = train_test_split(self.test_val_scaled,
                                                                               train_size=val_rate,
                                                                               random_state=self.random_state)
            else:
                self.lost_data, self.test_val = train_test_split(self.missed_data,
                                                                 test_size=(1 / int(missing_rate * 10)),
                                                                 random_state=self.random_state)
                self.val_data, self.test_data = train_test_split(self.test_val, train_size=val_rate,
                                                                 random_state=self.random_state)

    def _data_scale(self) -> None:
        """
        It scales the data to normal => mean = 0 and var = 1
        """
        self.train_data_scaled = self.train_data.copy()
        self.missed_data_scaled = self.missed_data.copy()
        self.data_scaled = self.data.copy()

        def fx(x):
            if type(x) == str:
                return 0
            else:
                return x

        max_data = max(self.train_data_scaled.values.reshape(
            len(self.train_data_scaled.columns) * len(self.train_data_scaled.index)), key=fx)

        self.train_data_scaled.iloc[:, 2] = self.train_data_scaled.iloc[:, 2] / max_data
        self.missed_data_scaled.iloc[:, 2] = self.missed_data_scaled.iloc[:, 2] / max_data
        self.data_scaled.iloc[:, 2] = self.data_scaled.iloc[:, 2] / max_data

    def _travel_time_matrix_produce(self, coordinate_file_name: str, scaled: bool = False) -> None:
        """
        It will produce a travel_time_matrix from file of point coordinates.
        :param coordinate_file_name: name of the file.
        :param scaled: If you are working with scaled data.
        :return: travel_time_matrix
        """
        if scaled:
            coordinates = pd.read_csv(coordinate_file_name)
            coordinates.columns = [0, 1, 2]

            x = coordinates.loc[self.od_zones_real - 1, 1].values
            y = coordinates.loc[self.od_zones_real - 1, 2].values
            self.travel_time_matrix_scaled = pd.DataFrame(self._haversine(x, y), index=self.od_zones_real,
                                                          columns=self.od_zones_real)
            scaler = MinMaxScaler()
            self.travel_time_matrix_scaled = pd.DataFrame(scaler.fit_transform(self.travel_time_matrix_scaled),
                                                          index=self.od_zones_real,
                                                          columns=self.od_zones_real)
        else:
            coordinates = pd.read_csv(coordinate_file_name)
            coordinates.columns = [0, 1, 2]

            x = coordinates.loc[self.od_zones_real - 1, 1].values
            y = coordinates.loc[self.od_zones_real - 1, 2].values

            self.travel_time_matrix = pd.DataFrame(self._haversine(x, y), index=self.od_zones_real,
                                                   columns=self.od_zones_real)

    def _od_matrix_complete_produce(self, scaled: bool = False) -> None:
        """
        It will produce the whole real od_matrix.
        :param scaled: If you want to work with scaled data.
        :return: od_matrix_real
        """
        self.od_zones_real = pd.unique(self.data.iloc[:, [0, 1]].values.ravel('K'))
        self.od_zones_real.sort()
        self.od_n_real = len(self.od_zones_real)

        if scaled:
            od_matrix_real = self.data_scaled.pivot_table(index=self.data_scaled.columns[0],
                                                          columns=self.data_scaled.columns[1],
                                                          values=self.data_scaled.columns[2], fill_value=0)
            od_matrix_real.columns = self.od_zones_real
            od_matrix_real.index = self.od_zones_real

            od_matrix_real = od_matrix_real.loc[:, (od_matrix_real != 0).any(axis=0)]
            od_matrix_real = od_matrix_real.loc[(od_matrix_real != 0).any(axis=1), :]

            self.rows_to_remove = np.setdiff1d(self.od_zones_real, od_matrix_real.index)
            self.cols_to_remove = np.setdiff1d(self.od_zones_real, od_matrix_real.columns)

            self.od_matrix_real_scaled = od_matrix_real

        else:
            od_matrix_real = self.data.pivot_table(index=self.data.columns[0], columns=self.data.columns[1],
                                                   values=self.data.columns[2], fill_value=0)
            od_matrix_real.columns = self.od_zones_real
            od_matrix_real.index = self.od_zones_real

            od_matrix_real = od_matrix_real.loc[:, (od_matrix_real != 0).any(axis=0)]
            od_matrix_real = od_matrix_real.loc[(od_matrix_real != 0).any(axis=1), :]

            self.rows_to_remove = np.setdiff1d(self.od_zones_real, od_matrix_real.index)
            self.cols_to_remove = np.setdiff1d(self.od_zones_real, od_matrix_real.columns)

            self.od_matrix_real = od_matrix_real

    def _static_data_preparation(self, folder: str, coordinate_file_name: str) -> None:
        """
        This function prepare not changeable with missing_rate data.
        :param coordinate_file_name: name of nodal_components.(id, x, y : in degree coordinate)
        :param folder: Where the data will be saved.
        :return: [attraction, production, travel_time_matrix, od_matrix, od_matrix_real]
        """
        directory = f"{folder}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        self._od_matrix_complete_produce(False)
        self.od_matrix_real.to_csv(f"{folder}/real_od_matrix.csv")
        print("done real_od")
        self.production = pd.DataFrame(self.od_matrix_real.sum(axis=1))
        self.production.to_csv(f"{folder}/production.csv")
        print("done production")
        self.attraction = pd.DataFrame(self.od_matrix_real.sum(axis=0))
        self.attraction.to_csv(f"{folder}/attraction.csv")
        print("done attraction")
        self._travel_time_matrix_produce(coordinate_file_name, scaled=False)
        self.travel_time_matrix.drop(self.rows_to_remove, axis=0, inplace=True)
        self.travel_time_matrix.drop(self.cols_to_remove, axis=1, inplace=True)
        self.travel_time_matrix.to_csv(f"{folder}/travel_time_matrix.csv")
        print("done travel_time")

    def _dynamic_data_preparation(self, folder_main: str, missing_rate: float,
                                  random_state: int = None) -> None:
        """
        This function prepare train data.
        :param folder_main: Where the data will be saved.
        :param missing_rate: missing_rate
        :param random_state: random_state
        :return: it will store train data in folder.
        """
        directory = f"{folder_main}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = f"{folder_main}/train_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = f"{folder_main}/test_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = f"{folder_main}/val_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.random_state = random_state
        self._data_split(missing_rate=missing_rate, val_rate=0.5, scaled=False)
        self._od_matrix_train_produce(scaled=False)
        self.od_matrix_train.drop(self.rows_to_remove, axis=0, inplace=True)
        self.od_matrix_train.drop(self.cols_to_remove, axis=1, inplace=True)
        self.od_matrix_train.to_csv(f"{folder_main}/train_data/at_miss{missing_rate:.2f}_train_od_matrix.csv")
        print(f"done train {missing_rate:.2f}")

        self._od_matrix_test_produce(scaled=False)
        self.od_matrix_test.drop(self.rows_to_remove, axis=0, inplace=True)
        self.od_matrix_test.drop(self.cols_to_remove, axis=1, inplace=True)
        self.od_matrix_test.to_csv(f"{folder_main}/test_data/at_miss{missing_rate:.2f}_test_od_matrix.csv")
        print(f"done test {missing_rate:.2f}")

        self._od_matrix_validation_produce(scaled=False)
        self.od_matrix_val.drop(self.rows_to_remove, axis=0, inplace=True)
        self.od_matrix_val.drop(self.cols_to_remove, axis=1, inplace=True)
        self.od_matrix_val.to_csv(f"{folder_main}/val_data/at_miss{missing_rate:.2f}_val_od_matrix.csv")
        print(f"done validation {missing_rate:.2f}")

    def not_scaled_data_preparation(self, main_folder: str, missing_rates: list,
                                    random_state: int = None) -> None:
        """
        This function will produce all the data and make them ready for both GM and GNN approaches.
        :param main_folder: Where the data will be saved.
        :param missing_rates: a list of missing rates you want to get data from.
        :param random_state: randon_state
        :return: it will store the data in folder.
        """
        self._static_data_preparation(main_folder, self.coordinate_file)
        for i in missing_rates:
            self._dynamic_data_preparation(main_folder, i, random_state)

    def scaled_data_preparation(self, folder: str, missing_rates: list,
                                random_state: int = None) -> None:
        """
        This function will produce all the data which are scaled and ready for both GM and GNN approaches.
        :param folder: Where the data will be saved.
        :param missing_rates: a list of missing rates you want to get data from.
        :param random_state: random_state
        :return: it will store the data in folder.
        """

        for i in missing_rates:
            sub_folder = int(i * 100)
            directory = f"{folder}/{sub_folder}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.random_state = random_state
            self._data_split(missing_rate=i, val_rate=0.5, scaled=True)

            self._od_matrix_complete_produce(True)
            self.od_matrix_real_scaled.to_csv(f"{folder}/{sub_folder}/real_od_matrix.csv")
            print("done real_od")
            self.attraction_scaled = pd.DataFrame(self.od_matrix_real_scaled.sum(axis=0))
            self.attraction_scaled.to_csv(f"{folder}/{sub_folder}/attraction.csv")
            print("done attraction")
            self.production_scaled = pd.DataFrame(self.od_matrix_real_scaled.sum(axis=1))
            self.production_scaled.to_csv(f"{folder}/{sub_folder}/production.csv")
            print("done production")
            self._travel_time_matrix_produce(self.coordinate_file, scaled=True)
            self.travel_time_matrix_scaled.drop(self.rows_to_remove, axis=0, inplace=True)
            self.travel_time_matrix_scaled.drop(self.cols_to_remove, axis=1, inplace=True)
            self.travel_time_matrix_scaled.to_csv(f"{folder}/{sub_folder}/travel_time_matrix.csv")
            print("done travel_time")
            self._od_matrix_train_produce(scaled=True)
            self.od_matrix_train_scaled.drop(self.rows_to_remove, axis=0, inplace=True)
            self.od_matrix_train_scaled.drop(self.cols_to_remove, axis=1, inplace=True)
            self.od_matrix_train_scaled.to_csv(f"{folder}/{sub_folder}/at_miss{i:.2f}_train_od_matrix.csv")
            print(f"done train {i:.2f}")
            self._od_matrix_test_produce(scaled=True)
            self.od_matrix_test_scaled.drop(self.rows_to_remove, axis=0, inplace=True)
            self.od_matrix_test_scaled.drop(self.cols_to_remove, axis=1, inplace=True)
            self.od_matrix_test_scaled.to_csv(f"{folder}/{sub_folder}/at_miss{i:.2f}_test_od_matrix.csv")
            print(f"done test {i:.2f}")
            self._od_matrix_validation_produce(scaled=True)
            self.od_matrix_val_scaled.drop(self.rows_to_remove, axis=0, inplace=True)
            self.od_matrix_val_scaled.drop(self.cols_to_remove, axis=1, inplace=True)
            self.od_matrix_val_scaled.to_csv(f"{folder}/{sub_folder}/at_miss{i:.2f}_val_od_matrix.csv")
            print(f"done validation {i:.2f}")
