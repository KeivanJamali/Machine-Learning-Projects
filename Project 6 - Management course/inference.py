from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class Dataloader:
    halls = {"Jaber": 0, "Theater": 1, "Mechanic": 2, "Kahroba": 3, "Rabiee": 4, "Sabz": 5, "Borgeii": 6, "Jabari": 7}
    features = ["Total_Profit"]
    target = ["month_year"]
    data_name = ["Systemquality_predicted.csv", "newPlaces_predicted.csv", "credit_predicted.csv", "TrainingData.csv"]
    all_features = features + target

    def __init__(self):
        self.data_system = Dataloader._setting(pd.read_csv(Dataloader.data_name[0]))
        self.data_new = Dataloader._setting(pd.read_csv(Dataloader.data_name[1]))
        self.data_credit = Dataloader._setting(pd.read_csv(Dataloader.data_name[2]))
        self.data_train = Dataloader._setting(pd.read_csv(Dataloader.data_name[3]))
        self.data = pd.merge(self.data_system, self.data_new, on="month_year")
        self.data = pd.merge(self.data, self.data_credit, on="month_year")
        self.data.columns = ["Systemquality", "newPlaces", "credit"]
    @staticmethod
    def _setting(data):
        data["Date"] = pd.to_datetime(data["Date"])
        data["month_year"] = data["Date"].apply(lambda x: x.strftime("%m-%Y"))
        data["month_year"] = pd.to_datetime(data["month_year"], format="mixed")
        data = data[Dataloader.all_features]
        data = data.groupby(["month_year"]).mean()
        return data

    def plot(self):
        plt.scatter(self.data.index, self.data["Systemquality"], c="blue", label="Systemquality")
        plt.scatter(self.data.index, self.data["newPlaces"], c="red", label="newPlaces")
        plt.scatter(self.data.index, self.data["credit"], c="green", label="credit")
        plt.title("Systemquality, newPlaces, credit alternatives.")
        plt.xlabel("Date")
        plt.ylabel("Profit(MillionToman)")
        plt.legend()
        plt.savefig("plot_for_3_alter.png")
        plt.clf()
        plt.scatter(self.data_train.index, self.data_train, c="b", label="Training")
        plt.title("Training data.")
        plt.xlabel("Date")
        plt.ylabel("Profit(MillionToman)")
        plt.legend()
        plt.savefig("plot_for_training.png")


