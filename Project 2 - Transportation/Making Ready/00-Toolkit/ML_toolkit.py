import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output


class K_means:
    def __init__(self, data: pd.DataFrame, n_cluster, max_iteration):
        self.data = data
        self.n_cluster = n_cluster
        self.max_iteration = max_iteration

    def kmeans(self, iteration=1):
        centroids = self.random_centroid()
        old_centroid = pd.DataFrame()
        while iteration < self.max_iteration and not centroids.equals(old_centroid):
            old_centroid = centroids
            labels = self.get_labels(centroids)
            centroids = self.new_centroid(labels)
            self.plot_clusters(labels, centroids, iteration)
            iteration += 1

    def random_centroid(self) -> pd.DataFrame:
        centroids = []
        for i in range(self.n_cluster):
            centroid = self.data.apply(lambda x: float(x.sample()))
            centroids.append(centroid)
        return pd.concat(centroids, axis=1)

    def get_labels(self, centroids: pd.DataFrame) -> pd.DataFrame:
        distance = centroids.apply(lambda x: np.sqrt(((self.data - x) ** 2).sum(axis=1)))
        return distance.idxmin(axis=1)
        # l2 = []
        # for i in range(len(self.data)):
        #     s2 = 10 ** 100
        #     la = None
        #     for k in range(self.n_cluster):
        #         s = 0
        #         for j in range(len(self.data.columns)):
        #             s += np.sqrt((self.data.iloc[i, j] - centroids.iloc[j, k]) ** 2)
        #         if s2 > s:
        #             la = k
        #             s2 = s
        #     l2.append(la)
        # label = pd.Series(l2)
        # return label

    def new_centroid(self, labels):
        return self.data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

    def plot_clusters(self, labels: pd.DataFrame, centroids: pd.DataFrame, iterations: int) -> None:
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
    :param start: default 1
    :param end: default 10
    :return:
    """
    data = ((data - data.min()) / (data.max() - data.min())) * (end - start) + start
    return data
