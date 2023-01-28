import numpy as np
import rphtz.nn as nn
import rphtz.optim as optim
import rphtz.data as data
from rphtz.data import DataLoader
import matplotlib.pyplot as plt

import os


class Net(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(1, 64)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(64, 64)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(64, 1)
        super().__init__()

    def forward(self):
        self.m = self.linear1.forward(self.m)
        self.m = self.tanh1.forward(self.m)
        self.m = self.linear2.forward(self.m)
        self.m = self.tanh2.forward(self.m)
        self.m = self.linear3.forward(self.m)
        return self.m


def create_models(Module, num: int):
    M = Module()
    Ms = list()
    for n in range(num):
        m = Module()
        for i in range(len(M.parameters)):
            m.parameters[i].weight = M.parameters[i].weight.copy()
            m.parameters[i].bias = M.parameters[i].bias.copy()
        Ms.append(m)
    return Ms


def create_optimizers(models):
    optimizers = list()
    optim_names = ['SGD', 'Momentum', 'Nesterov momentum', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']
    optimizers.append(optim.SGD(models[0].parameters, 0.001))
    optimizers.append(optim.SGD(models[1].parameters, 0.001, momentum=0.5))
    optimizers.append(optim.SGD(models[2].parameters, 0.001, momentum=0.5, nesterov=True))
    optimizers.append(optim.Adagrad(models[3].parameters, 0.01))
    optimizers.append(optim.Adadelta(models[4].parameters, lr=0.01, alpha=0.7))
    optimizers.append(optim.RMSprop(models[5].parameters, lr=0.005, alpha=0.9))
    optimizers.append(optim.Adam(models[6].parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8))
    return optimizers, optim_names


def get_losses():
    models = create_models(Net, 7)
    criterion = nn.MSELoss(reduction='sum')
    optimizers, optim_names = create_optimizers(models)
    epoch = 1000
    losses = list()
    dataset = data.create_dataset(np.sin, start=-16, stop=16, n=256, err=0)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    for i in range(7):
        losses.append(nn.train(models[i], criterion, dataloader, optimizers[i], epoch, print_info=False))
    losses = np.array(losses)
    return losses


def display(losses, optim_names, start=None, end=None, step: int = 1, figure=0):
    plt.figure(figure)
    for i in range(len(losses)):
        plt.plot(losses[i][start:end:step], label=optim_names[i])
    plt.legend(optim_names)
    plt.savefig('./pics/loss%d' % figure + '.jpg')
    plt.show()


def display_folder(file, optim_names, start=None, end=None, step=1):
    fs = os.listdir(file)
    for i in range(len(fs)):
        l = np.load(file + '/' + fs[i])
        display(l, optim_names, start=start, end=end, step=step, figure=i)


if __name__ == '__main__':
    # for i in range(0, 10):
    #     losses = get_losses()
    #     np.save('./losses/losses%d' % i + '.npy', losses)
    #     print(i)

    optim_names = ['SGD', 'Momentum', 'Nesterov momentum', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']
    # losses = np.load('./losses/losses1.npy')
    # display(losses, optim_names, start=10, end=None)
    display_folder('./losses', optim_names, start=5, end=None, step=1)
