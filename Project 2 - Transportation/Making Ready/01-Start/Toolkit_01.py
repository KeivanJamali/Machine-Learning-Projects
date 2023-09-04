import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Regression:
    def __init__(self, df: pd.DataFrame):
        """
        :param df: data. no need to drop its NAN. this function does it for you.
        """
        self.df = df.dropna()
        self.line = self.find_m_c()

    def hypothesis(self) -> pd.DataFrame:
        """
        function of line
        :return: y of the line
        """
        return self.line[1] + self.line[0] * self.df.iloc[:, 0]

    def cost(self, temp=0) -> float:
        """
        the cost of this line... error...
        :return: total cost
        """
        n = len(self.df)
        for i in range(len(self.df)):
            temp += (self.hypothesis().iloc[i] - self.df.iloc[i, 1]) ** 2
        return (1 / (2 * n)) * temp

    def find_m_c(self, up=0, down=0) -> list:
        """
        it will find a great line to minimize the cost.
        :return: a list that has m(slope) and c(constant)
        """
        mean_x = np.mean(self.df.iloc[:, 0])
        mean_y = np.mean(self.df.iloc[:, 1])
        for i in range(len(self.df)):
            x = self.df.iloc[i, 0]
            y = self.df.iloc[i, 1]
            up += (x - mean_x) * (y - mean_y)
            down += (x - mean_x) ** 2
        m = up / down
        c = mean_y - m * mean_x
        return [m, c]

    def plot(self):
        """
        you will get the plot of it
        """
        plt.clf()
        plt.plot(self.df.iloc[:, 0], self.hypothesis(), "red", label="line")
        plt.scatter(self.df.iloc[:, 0], self.df.iloc[:, 1], marker=".", label="data")
        plt.legend()
        plt.show()


class K_means:
    def __init__(self, data: pd.DataFrame, n_cluster, max_iteration):
        """
        it will do K_Means methods with a step and plotting.
        :param data: my data to classify
        :param n_cluster: number of cluster
        :param max_iteration: max iterations to try
        """
        self.data = data
        self.n_cluster = n_cluster
        self.max_iteration = max_iteration

    def kmeans(self, iteration=1):
        """
        this is the main function to work with!
        """
        centroids = self.random_centroid()
        old_centroid = pd.DataFrame()
        while iteration < self.max_iteration and not centroids.equals(old_centroid):
            old_centroid = centroids
            labels = self.get_labels(centroids)
            centroids = self.new_centroid(labels)
            self.plot_clusters(labels, centroids, iteration)
            iteration += 1

    def random_centroid(self) -> pd.DataFrame:
        """
        it gives you centroid randomly from your data...
        :return: centroid
        """
        centroids = []
        for i in range(self.n_cluster):
            centroid = self.data.apply(lambda x: float(x.sample()))
            centroids.append(centroid)
        return pd.concat(centroids, axis=1)

    def get_labels(self, centroids: pd.DataFrame) -> pd.DataFrame:
        """
        it will label all data according to the centroid you give to it
        :param centroids: centroid that you try to label data with
        :return: labeled data
        """
        distance = centroids.apply(lambda x: np.sqrt(((self.data - x) ** 2).sum(axis=1)))
        return distance.idxmin(axis=1)

    def new_centroid(self, labels):
        """
        it would find a new center according to the labeled data
        :param labels: data which is labeled
        :return: again a centroid which fit in new data
        """
        return self.data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

    def plot_clusters(self, labels: pd.DataFrame, centroids: pd.DataFrame, iterations: int) -> None:
        """
        it will plot for you every step.
        """
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(self.data)
        centroids_2d = pca.transform(centroids.T)
        clear_output(wait=True)
        plt.title(f"Iteration {iterations}")
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="o", s=60)
        plt.show()


def scale_data(data: pd.DataFrame, start=1, end=10) -> pd.DataFrame:
    """
    it scales data from 1 to 10 by default.
    :param data: Entering data
    :param start: default 1
    :param end: default 10
    :return:
    """
    data = ((data - data.min()) / (data.max() - data.min())) * (end - start) + start
    return data


def split_scale(data_x: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2, random_state: int = None):
    """
    it split data to train and test
    :param data_x: input x
    :param y: input y
    :param test_size: test size
    :param random_state: random state
    :return: Data_train, Data_test, y,train, y_test
    """
    data_train, data_test, y_train, y_test = train_test_split(data_x, y, test_size=test_size, random_state=random_state)
    sc = StandardScaler()
    data_train_scaled = sc.fit_transform(data_train)
    data_test_scaled = sc.transform(data_test)
    return data_train_scaled, data_test_scaled, y_train, y_test
