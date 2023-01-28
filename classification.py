import numpy as np
import rphtz.nn as nn
import rphtz.data as data
import rphtz.optim as optim
from rphtz.data import DataSet, DataLoader
import time

import matplotlib.pyplot as plt


def accuracy(m, dataloader):
    correct = 0
    for i in range(dataloader.length):
        x, y = dataloader[i]
        # print(softmax(m(x)))
        pred = np.argmax(m(x), axis=2)
        label = np.argmax(y, axis=2)
        correct += (pred == label).sum(axis=0)
    accur = correct / dataloader.n

    return accur


if __name__ == '__main__':
    structure = {'Linear.1': (28 * 28, 512),
                 'Tanh.1': None,
                 'Linear.2': (512, 256),
                 'Tanh.2': None,
                 'Linear.3': (256, 10)}

    model = nn.Module().create_from_dict(structure).load('./weights/classifier1.npy')
    criterion = nn.CrossEntropy()
    optimizer = optim.Adam(parameters=model.parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    x_train, y_train = data.load_images('D:/PythonProjects/mnist', 0, 512)
    dataset = DataSet(x_train, y_train)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    epoch = 10
    softmax = nn.Softmax()
    losses = nn.train(model, criterion, dataloader, optimizer, epoch, print_info=True)
    model.save('./weights/classifier.npy')

    plt.plot(losses)
    plt.show()
    print(accuracy(model, dataloader))
