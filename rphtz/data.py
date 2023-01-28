import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


class DataSet:
    def __init__(self, x, y):
        self.data = x, y
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class DataLoader:
    def __init__(self, dataset: DataSet, batch_size: int, shuffle=False):
        self.dataset = dataset
        self.n = dataset.data[0].shape[0]
        self.batch_size = batch_size
        self.i = 0
        self.rand_indices = np.arange(0, self.n)
        np.random.shuffle(self.rand_indices)
        self.counter = 0
        self.length = self.n // batch_size + 1 * (self.n % batch_size != 0)
        self.indices = np.arange(0, self.length + 1) * batch_size
        self.indices[-1] = self.n
        self.shuffle = shuffle
        if self.shuffle:
            def gi(item):
                if self.counter >= self.length:
                    np.random.shuffle(self.rand_indices)
                    self.counter = 0
                x = self.dataset.x[self.rand_indices[self.indices[item]:self.indices[item + 1]]]
                y = self.dataset.y[self.rand_indices[self.indices[item]:self.indices[item + 1]]]
                self.counter += 1
                return x, y

            self.gi = gi
        else:
            def gi(item):

                x = self.dataset.x[self.rand_indices[self.indices[item]:self.indices[item + 1]]]
                y = self.dataset.y[self.rand_indices[self.indices[item]:self.indices[item + 1]]]

                return x, y

            self.gi = gi

    def __next__(self):
        if self.i >= self.length:
            raise StopIteration
        idx = self.i
        self.i += 1
        return idx

    def __iter__(self):
        return self

    def __getitem__(self, item):
        return self.gi(item)
        # if self.counter >= self.length:
        #     np.random.shuffle(self.rand_indices)
        #     self.counter = 0
        # x = self.dataset.x[self.rand_indices[self.indices[item]:self.indices[item + 1]]]
        # y = self.dataset.y[self.rand_indices[self.indices[item]:self.indices[item + 1]]]
        # self.counter += 1
        # return x, y


def create_dataset(func, start: float, stop: float, n: int, err: float):
    x = np.expand_dims(np.array([np.linspace(start, stop, n)]).transpose(), axis=2)
    y = func(x) + err * (2 * np.random.random(x.shape) - 1)
    return DataSet(x, y)


def load_images(file: str, start: int, end: int, imgsz=(28, 28), mode='train'):
    ftxt = open(file + '/' + mode + '.txt', 'r')
    # if mode == 'train':
    #     ftxt = open(file + '/train.txt', 'r')
    # elif mode == 'test':
    #     ftxt = open(file + '/test.txt', 'r')
    lines = ftxt.readlines()
    num = end - start
    imgs = np.empty(shape=[num, 1, imgsz[0] * imgsz[1]])
    labels = np.zeros(shape=[num, 1, 10])
    pbar = tqdm(total=num, desc='Loading Images', leave=True)
    for i in range(num):
        # print('Loading Image %d' % (start + i))
        line = lines[start + i].split()
        imgs[i] = cv2.imread(file + '/' + mode + '/' + line[0], flags=0).reshape([1, 28 * 28]) / 255
        labels[i][0][int(line[1])] = 1
        pbar.update()

    return imgs, labels


def create_clusters(centers, radius: float, n: int):
    if isinstance(centers, list):
        centers = np.array(centers)
    ndim = centers.shape[1]
    y = np.random.randint(0, centers.shape[0], size=n)
    # x = np.empty(shape=[n, ndim])
    x = centers[y] + (2 * np.random.random(size=[n, ndim]) - 1) * radius
    return x, y


def create_svm_data(centers, radius: float, n: int, unlabeled:float=0):
    if isinstance(centers, list):
        centers = np.array(centers)
    ndim = centers.shape[1]
    if unlabeled:
        y = np.random.randint(0, 2, size=n)

        x = centers[y] + (2 * np.random.random(size=[n, ndim]) - 1) * radius
        mask = np.random.random(size=n) < unlabeled
        y = (2 * y - 1) * mask

        return x, y
    else:
        y = np.random.randint(0, 2, size=n)

        x = centers[y] + (2 * np.random.random(size=[n, ndim]) - 1) * radius
        y = (2 * y - 1)
        return x, y


if __name__ == '__main__':
    pass
    centers = np.random.randint(0, 4, size=[3, 2])
    print(centers)
    create_clusters(centers, radius=0.5, n=5)
    a = np.array([0, 1])
    print(a[-1])
    x_, y_ = create_svm_data(np.array([[1, 3], [3, 1]]), radius=0, n=5, unlabeled=0.5)
    print(x_)
    print(y_)
