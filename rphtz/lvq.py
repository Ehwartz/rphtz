import numpy as np
import random
import matplotlib.pyplot as plt
from data import create_clusters
import time


def dist(x1, x2):
    return np.dot(x1 - x2, x1 - x2)# .sum(axis=0)


class LVQ:
    def __init__(self, n_clusters, n_iter, lr):
        self.protoVecs = list()
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.lr = lr

    def fit(self, x, label):
        x_size = x.shape[0]
        for q in range(self.n_clusters):
            index = random.randint(0, x_size - 1)
            self.protoVecs.append(x[index])
        for iter in range(self.n_iter):
            dists = list()
            index = random.randint(0, x_size - 1)
            for i in range(self.n_clusters):
                dists.append(dist(self.protoVecs[i], x[index]))
            minID = dists.index(min(dists))
            if minID == label[index]:
                self.protoVecs[minID] += self.lr * (x[index] - self.protoVecs[minID])
            else:
                self.protoVecs[minID] -= self.lr * (x[index] - self.protoVecs[minID])


if __name__ == '__main__':
    centers = np.array([[1, 3],
                        [4, 2],
                        [5, 6]])
    x_, label = create_clusters(centers, radius=1, n=200)
    model = LVQ(3, 1000, 0.01)
    model.fit(x_, label)
    print(model.protoVecs)
    dists = np.zeros(shape=(model.n_clusters, x_.shape[0], x_.shape[1]))
    for c in range(model.n_clusters):
        for i in range(x_.shape[0]):
            dists[c][i] = dist(model.protoVecs[c], x_[i])
    print(dists)
    min_dist = np.min(dists, axis=0)
    clusters = (np.array([min_dist]) == dists).sum(axis=2).transpose()
    # print(min_dist)
    print(clusters)
    colors = ['red', 'blue', 'yellow']
    for n in range(3):
        for i in range(clusters.shape[0]):
            if clusters[i][n]:
                plt.scatter(x_[:, 0], x_[:, 1], c=colors[n])

    plt.show()
