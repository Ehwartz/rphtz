import numpy as np
import matplotlib.pyplot as plt
from data import create_clusters
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class KMeans:
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.cluster_centers = list()

    def fit(self, x):

        i = 0
        init_centers = list()
        # for i in range(self.n_clusters):
        #     index = random.randint(0, x.shape[0] - 1)
        #     self.cluster_centers.append(x[index])
        while i < self.n_clusters:
            index = random.randint(0, x.shape[0] - 1)
            if index not in init_centers:
                init_centers.append(index)
                self.cluster_centers.append(x[index])
                i += 1

        for iter in range(self.max_iter):
            # print('Iteration: %d' % iter)
            centers_temp = np.expand_dims(
                np.array([self.cluster_centers]), axis=0).repeat(
                x.shape[0], axis=0).reshape(
                [x.shape[0], self.n_clusters, x.shape[1]]).transpose([1, 0, 2])

            x_temp = np.array([x]).repeat(self.n_clusters, axis=0).reshape([self.n_clusters, x.shape[0], x.shape[1]])
            # dists = np.dot(x_temp - centers_temp, x_temp - centers_temp).sum(axis=2)

            dists = ((x_temp - centers_temp) * (x_temp - centers_temp)).sum(axis=2)

            # dists = (x_temp - centers_temp) * (x_temp - centers_temp)
            # dists[:][:][3:-1] = sigmoid(dists[:][:][3:-1])
            # dists = dists.sum(axis=2)
            # dists = sigmoid(dists)
            # print('dists\n', dists)
            # print(dists.sum(axis=2))
            # print(dists.sum(axis=2).transpose([1, 0]))

            indices = np.argmin(dists, axis=0)
            # print('centers temp:\n ', centers_temp)
            # print('x_temp:\n ', x_temp)
            #
            # print('indices\n', indices)
            # print(np.where(indices==1))
            # print(x[np.where(indices==1)])

            for n in range(self.n_clusters):
                # self.cluster_centers[n] = np.array(self.clusters[n]).sum(axis=0) / len(self.clusters[n])
                self.cluster_centers[n] = np.mean(x[np.where(indices == n)], axis=0)

    def constrained_seed_fit(self, x, cluster: list):
        self.n_clusters = len(clusters)
        for i in range(self.n_clusters):
            self.cluster_centers.append(cluster[i].sum(axis=0) / cluster[i].shape[0])

        for iter in range(self.max_iter):
            # print('Iteration: %d' % iter)
            centers_temp = np.expand_dims(np.array([self.cluster_centers]), axis=0).repeat(x.shape[0], axis=0).reshape(
                [x.shape[0], self.n_clusters, x.shape[1]]).transpose([1, 0, 2])
            x_temp = np.array([x]).repeat(self.n_clusters, axis=0).reshape(
                [self.n_clusters, x.shape[0], x.shape[1]])
            dists = ((x_temp - centers_temp) * (x_temp - centers_temp)).sum(axis=2)
            indices = np.argmin(dists, axis=0)
            for n in range(self.n_clusters):
                self.cluster_centers[n] = np.mean(x[np.where(indices == n)], axis=0)

    def predict(self, x):
        centers_temp = np.expand_dims(np.array([self.cluster_centers]), axis=0).repeat(x.shape[0], axis=0).reshape(
            [x.shape[0], self.n_clusters, x.shape[1]]).transpose([1, 0, 2])
        x_temp = np.array([x]).repeat(self.n_clusters, axis=0).reshape(
            [self.n_clusters, x.shape[0], x.shape[1]])

        dists = ((x_temp - centers_temp) * (x_temp - centers_temp)).sum(axis=2)
        indices = np.argmin(dists, axis=0)
        return indices


if __name__ == '__main__':
    KM = KMeans(3, 1000)
    centers = np.array([[1, 3],
                        [4, 2],
                        [5, 6]])
    x_, _ = create_clusters(centers, radius=1, n=200)

    KM.fit(x_)
    for i in range(KM.n_clusters):
        print('Cluster Center%d: ' % i, KM.cluster_centers[i])
    pred = KM.predict(x_)
    print(pred)
    for n in range(KM.n_clusters):
        clusters = x_[np.where(pred == n)].transpose()
        plt.scatter(clusters[0], clusters[1])

    plt.show()
