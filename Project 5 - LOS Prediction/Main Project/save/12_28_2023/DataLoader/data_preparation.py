import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


class DataLoader_MultiCity:
    car_length = 0.0049

    def __init__(self, data: pd.DataFrame, detectors_data: pd.DataFrame, weather_data: pd.DataFrame,
                 full_data: bool = False):
        """
        params:
            data: pd.DataFrame: Full data include all cities or one city.
            full_data: bool: if you want to get information about unique days in the city data.
        """
        self.merged_data = None
        self.data = data
        self.sub_data = {}
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.transformed_data = None
        self.path = Path("D:\All Python\All_Big_raw_Data\LOS prediction\Traffic Dataset\DataLoader")

        self.information = data.groupby("city").count()
        self.cities = self.information.index
        self.features = self.information.columns

        if full_data:
            self._full_information()

        self.detectors_data = detectors_data
        self.weather_data = weather_data

    def fit(self):
        """
        It will return the train_data, val_data, test_data to be used in further modeling.
        The data is one-hour data of occupancy, rain intensity, and flow as the result.
        """
        self._transform_hourly_data()
        print(f"[INFO] Data #transformed# successfully.")
        ########################### should change
        self.weather_data["hour"] = self.weather_data["interval"] // 3600
        self.weather_data["date"] = self.weather_data["day"] + pd.to_timedelta(self.weather_data["hour"],
                                                                               unit="h")
        ###########################
        self.merged_data = self.transformed_data.merge(
            self.weather_data[
                ["date", "temp", "feelslike", "dew", "humidity", "rainfall", "precipprob", "preciptype", "snow",
                 "snowdepth", "windgust", "windspeed", "winddir", "sealevelpressure", "cloudcover", "visibility",
                 "solarradiation", "solarenergy", "uvindex", "severerisk", "conditions", "icon"]], on=["date"],

            how="left")  # data to be used as features.
        print(f"[INFO] Data merged with weather successfully.")
        self._LOS()
        print(f"[INFO] LOS founded and added to data!")
        self.train_data, self.val_data, self.test_data = self._split_scale(self.merged_data, train_size=0.8,
                                                                           val_size=0.1, test_size=0.1)
        print(f"[INFO] Split successful.")

        self._save()
        print(f"[INFO] Saved successfully.")
        return self.train_data, self.val_data, self.test_data

    @staticmethod
    def _split_scale(data: pd.DataFrame, train_size=0.8, val_size=0.1, test_size=0.1, seed=42) -> (
            pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        This function will prepare train, validation and test data from an data data.
        params:
            seed: int: random seed or random state.
        """
        if train_size + val_size + test_size != 1:
            raise ValueError("train_rate + val_rate + test_rate must be 1.")

        train_data, test_val = train_test_split(data, train_size=train_size, random_state=seed)
        val_data, test_data = train_test_split(test_val, train_size=val_size / (1 - train_size), random_state=seed)
        return train_data, val_data, test_data

    def pick_city(self, city: list):
        """
        params:
            data: pd.DataFrame: Full data include all cities.
            city: str: the city or cities you want to have in your data.
        """
        chosen_city = []
        merged_data = pd.DataFrame()
        for c in city:
            if c in self.cities:
                self.sub_data[c] = self.data[self.data["city"] == c]
                merged_data = pd.concat([merged_data, self.sub_data[c]], ignore_index=True)
                chosen_city.append(c)
            else:
                raise KeyError(f"There is no ##{c}## in the list. please pick one of these cities: {self.cities}")

        merged_data.drop(merged_data[merged_data["error"] == 1].index, inplace=True)  # drop Error = 1
        new_data = DataLoader_MultiCity(data=merged_data, detectors_data=self.detectors_data,
                                        weather_data=self.weather_data, full_data=True)
        print(f"[INFO] {chosen_city} was/were picked successfully.")
        return new_data

    def _full_information(self):
        date_cities_count = []
        if len(self.cities) > 1:
            for city in self.cities:
                if city not in self.data.keys():
                    self.sub_data[city] = self.data[self.data["city"] == city]
                date_cities_count.append(len(self.sub_data[city].groupby("day")))

        elif len(self.cities) == 1:
            date_cities_count.append(self.data.groupby("day"))
            del self.sub_data

        else:
            raise KeyError(f"There is no city. please check [_full_information] function.")

        self.information["num_days"] = date_cities_count

    def _transform_hourly_data(self):
        self.transformed_data = self.data.copy()  # make a copy.
        self.transformed_data["day"] = pd.to_datetime(self.transformed_data["day"])  # date will be datetime now.
        self.transformed_data["hour"] = self.transformed_data["interval"] // 3600
        self.transformed_data = self.transformed_data[["day", "hour", "flow", "occ"]]  # get columns
        self.transformed_data["date"] = self.transformed_data["day"] + pd.to_timedelta(self.transformed_data["hour"],
                                                                                       unit="h")
        self.transformed_data = self.transformed_data.groupby(["day", "hour"]).mean()  # merge data in hour unit
        print(f"[INFO] data loaded successfully.")
        self._occupancy_to_speed(car_length=self.car_length)  # find density and speed and add columns.
        print(f"[INFO] speed and density produced successfully.")
        self.transformed_data.index = range(len(self.transformed_data))  # reorder indexes.

    def _occupancy_to_speed(self, car_length=0.0049):
        detector_length = self.detectors_data[self.detectors_data["detid"] == self.data["detid"][0]].length
        self.transformed_data["density"] = self.transformed_data["occ"].apply(
            lambda x: (x * 1) / (car_length + detector_length))
        self.transformed_data["speed"] = self.transformed_data["flow"] / self.transformed_data["density"]

    def _save(self):
        self.train_data.to_csv(self.path / "train_data_luzern.csv")
        self.val_data.to_csv(self.path / "val_data_luzern.csv")
        self.test_data.to_csv(self.path / "test_data_luzern.csv")

    def _LOS(self):
        max_flow = max(self.merged_data["flow"])
        self.merged_data["LOS"] = self.merged_data["flow"].apply(
            lambda x: self._classification(x / max_flow, type="name"))
        self.merged_data["LOS_index"] = self.merged_data["flow"].apply(
            lambda x: self._classification(x / max_flow, type="index"))

    @staticmethod
    def _classification(x, type: str):
        if type == "name":
            if 0 < x <= 0.6:
                return "A"
            elif 0.6 < x <= 0.7:
                return "B"
            elif 0.7 < x <= 0.8:
                return "C"
            elif 0.8 < x <= 0.9:
                return "D"
            elif 0.9 < x <= 1:
                return "E"
            elif x > 1:
                return "F"
        elif type == "index":
            if 0 < x <= 0.6:
                return 0
            elif 0.6 < x <= 0.7:
                return 1
            elif 0.7 < x <= 0.8:
                return 2
            elif 0.8 < x <= 0.9:
                return 3
            elif 0.9 < x <= 1:
                return 4
            elif x > 1:
                return 5
