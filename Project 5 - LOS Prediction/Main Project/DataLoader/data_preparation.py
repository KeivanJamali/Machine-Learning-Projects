import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os


class DataLoader_MultiCity:
    car_length = 0.0049
    path = Path("D:\All Python\All_Big_raw_Data\LOS prediction\Traffic Dataset\DataLoader")

    def __init__(self, data: pd.DataFrame, detectors_data: pd.DataFrame, weather_data: pd.DataFrame,
                 full_data: bool = False):
        """
        params:
            data: pd.DataFrame: Full data include all cities or one city.
            full_data: bool: if you want to get information about unique days in the city data.

            self.data: store the data.
            self.sub_data: store the data of each city.
            self.merged_data: store the merged version of the traffic data and weather data together.
            self.transformed_data: store the data which the time changes into hourly.
            self.weather_data_ready: store the weather data with specific features that we want.
            self.path: store the path which we will store data.
            self.information: DataFrame: give a brief describe of data for each city.
            self.cities: store the name of all cities.
            self.features: store all the features in the traffic data. Only columns.
            self.detectors_data: store the detectors data.
            self.weather_data: store the weather data.
            self.train: store the train data.
            self.val: store the validation data.
            self.test: store the test data.

        """
        self.merged_data = None
        self.data = data
        self.sub_data = {}
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.transformed_data = None
        self.weather_data_ready = None

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
        self.transformed_data = self._transform_hourly_data()
        print(f"[INFO] Take the mean of data in each hour successfully.")

        self._occupancy_to_speed(car_length=DataLoader_MultiCity.car_length)  # find density and speed and add columns.
        print(f"[INFO] Speed and density produced successfully and added to the dataframe of transform_hourly_data.")

        features = ["date", "temp", "feelslike", "dew", "humidity", "rainfall", "precipprob", "preciptype", "snow",
                    "snowdepth", "windgust", "windspeed", "winddir", "sealevelpressure", "cloudcover", "visibility",
                    "solarradiation", "solarenergy", "uvindex", "severerisk", "conditions", "icon"]
        self.merged_data, self.weather_data_ready = self._weather_preparation(features=features)
        print(f"[INFO] Data merged with weather successfully.")

        self._LOS()
        print(f"[INFO] LOS founded and added to data! we have now columns of 'LOS' and 'LOS_index'.")

        # self.make_numerical()
        # print(f"[INFO] columns of 'preciptype' and 'conditions' and 'icon' become numerical.")

        self.setting_of_data()
        print(f"[INFO] Setting of data finished.")

        self._add_features()
        print(f"[INFO] add new features!")

        self.train_data, self.val_data, self.test_data = self._split_scale(self.merged_data, train_size=0.8,
                                                                           val_size=0.1, test_size=0.1)
        print(f"[INFO] Split successful.")

        self._save()
        print(f"[INFO] Saved successfully.")

        print(f"number of features: {len(self.merged_data.columns) - 5}")
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

    def _transform_hourly_data(self) -> pd.DataFrame:
        data = self.data.copy()  # make a copy.
        data["day"] = pd.to_datetime(data["day"])  # date will be datetime now.
        data["hour"] = data["interval"] // 3600
        data = data[["day", "hour", "flow", "occ"]]  # get columns
        data["date"] = data["day"] + pd.to_timedelta(data["hour"], unit="h")
        data = data.groupby(["day", "hour"]).mean()  # merge data in hour unit
        data.index = range(len(data))  # reorder indexes.
        return data

    def _occupancy_to_speed(self, car_length=0.0049):
        self.detector = self.detectors_data[self.detectors_data["detid"] == self.data["detid"][0]]
        self.detector_length = self.detectors_data[self.detectors_data["detid"] == self.data["detid"][0]].length
        self.transformed_data["density"] = self.transformed_data["occ"].apply(
            lambda x: (x * 1) / (car_length + self.detector_length))
        self.transformed_data["speed"] = self.transformed_data["flow"] / self.transformed_data["density"]

    def _weather_preparation(self, features: list) -> tuple[pd.DataFrame, pd.DataFrame]:
        """I need 'interval' column and day column to build the weather.
        so weather data need to have these two columns. Merged data will produce at the end.
        param:
            features: list: all features you want to get from weather data."""
        weather_data = self.weather_data.copy()
        weather_data["hour"] = weather_data["interval"] // 3600
        weather_data["date"] = weather_data["day"] + pd.to_timedelta(weather_data["hour"], unit="h")
        weather_data = weather_data[features]
        data = self.transformed_data.merge(weather_data, on=["date"], how="left")
        return data, weather_data

    def _save(self):
        if not os.path.exists(DataLoader_MultiCity.path / f"{'-'.join(list(self.cities))}"):
            os.makedirs(DataLoader_MultiCity.path / f"{'-'.join(list(self.cities))}")
        self.train_data.to_csv(DataLoader_MultiCity.path / f"{'-'.join(list(self.cities))}/train_data_luzern.csv")
        self.val_data.to_csv(DataLoader_MultiCity.path / f"{'-'.join(list(self.cities))}/val_data_luzern.csv")
        self.test_data.to_csv(DataLoader_MultiCity.path / f"{'-'.join(list(self.cities))}/test_data_luzern.csv")
        self.merged_data.to_csv(DataLoader_MultiCity.path / f"{'-'.join(list(self.cities))}/full_data.csv")

    def _LOS(self):
        """In this function we will make columns of 'LOS' and 'LOS_index'."""
        max_flow = max(self.merged_data["flow"])
        self.merged_data["LOS"] = self.merged_data["flow"].apply(
            lambda x: self._classification(x / max_flow, type_="name"))
        self.merged_data["LOS_index"] = self.merged_data["flow"].apply(
            lambda x: self._classification(x / max_flow, type_="index"))

    @staticmethod
    def _classification(x, type_: str):
        if type_ == "name":
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
        elif type_ == "index":
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

    def make_numerical(self):
        def preciptype_to_numeric(x):
            if type(x) != str:
                return pd.NA
            l = x.split(",")
            if "rain" in l and "snow" in l:
                return 3
            elif "rain" in l:
                return 2
            elif "snow" in l:
                return 1
            else:
                return 0

        def conditions_to_numeric(x):
            if type(x) != str:
                return pd.NA
            l = x.split(", ")
            result = 0
            if "Rain" in l:
                result += 1
            if "Snow" in l:
                result += 1
            if "Overcast" in l:
                result += 1
            if "Partially cloudy" in l:
                result += 1
            if "Clear" in l:
                result += 0

            return result

        def icon_to_numeric(x):
            if type(x) != str:
                return pd.NA
            if "partly-cloudy-night" in x:
                return 0
            elif "cloudy" in x:
                return 0
            elif "partly-cloudy-day" == x:
                return 0
            elif "rain" == x:
                return 1
            elif "snow" == x:
                return 1
            elif "clear-night" == x:
                return 0
            elif "clear-day" == x:
                return 0

        self.merged_data["preciptype"] = self.merged_data["preciptype"].apply(lambda x: preciptype_to_numeric(x))
        self.merged_data["conditions"] = self.merged_data["conditions"].apply(lambda x: conditions_to_numeric(x))
        self.merged_data["icon"] = self.merged_data["icon"].apply(lambda x: icon_to_numeric(x))

    def setting_of_data(self):
        self.merged_data = self.merged_data.drop("severerisk", axis=1)  # completely NA
        self.merged_data = self.merged_data.drop("density", axis=1)  # not for nn now. and not accurate
        self.merged_data = self.merged_data.drop("speed", axis=1)  # not for nn now. and not accurate
        self.merged_data = self.merged_data.drop("preciptype", axis=1)  # lot of NA
        self.merged_data = self.merged_data.drop("visibility", axis=1)  # lot of NA
        self.merged_data = self.merged_data.drop("dew", axis=1)
        self.merged_data = self.merged_data.drop("precipprob", axis=1)
        self.merged_data = self.merged_data.drop("snow", axis=1)
        self.merged_data = self.merged_data.drop("windgust", axis=1)
        self.merged_data = self.merged_data.drop("winddir", axis=1)
        self.merged_data = self.merged_data.drop("uvindex", axis=1)
        self.merged_data = self.merged_data.drop("conditions", axis=1)
        self.merged_data = self.merged_data.drop("icon", axis=1)

    def _add_features(self):
        # adding the feature of month.
        self.merged_data["month"] = self.merged_data["date"].apply(lambda x: x.month)

        # adding the feature of time of raining.
        data = self.merged_data.copy()
        data["index"] = data.index

        def time_of_raining(x):
            hours = 0
            while x >= 0:
                if self.merged_data["rainfall"][x] > 0:
                    hours += 1
                    x -= 1
                else:
                    break
            return hours

        self.merged_data["time_of_raining"] = data["index"].apply(lambda x: time_of_raining(x))
