import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing
from torch.utils.data import DataLoader


class DataLoader_MultiCity:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.data = {}
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.information = raw_data.groupby("city").count()
        self.cities = self.information.index
        self.features = self.information.columns

    def flow_predict(self, city: str):
        self.train_data, self.val_data, self.test_data = self._split_scale(self.data[city])

    @staticmethod
    def _split_scale(data: pd.DataFrame, seed=42) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        train_data, test_val = train_test_split(data, test_size=0.2, random_state=seed)
        val_data, test_data = train_test_split(test_val, test_size=0.2, random_state=seed)
        return train_data, val_data, test_data

    def pick_city(self, city: list):
        """
        params:
            data: pd.DataFrame: Full data include all cities.
            city: str: the city or cities you want to have in your data.
        """
        chosen_city = []

        for c in city:
            if c in self.cities:
                self.data[c] = self.raw_data[self.raw_data["city"] == c]
                chosen_city.append(c)
            else:
                raise KeyError(f"There is no ##{c}## in the list. please pick one of these cities: {self.cities}")

        print(f"[INFO] {chosen_city} was/were picked successfully.")

    def _full_information(self):
        date_cities_count = []
        for city in self.cities:
            self.data[city] = self.raw_data[self.raw_data["city"] == city]
            date_cities_count.append(len(self.data[city]))

        self.information["num_days"] = date_cities_count
