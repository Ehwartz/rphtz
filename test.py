import numpy as np
import rphtz.nn as nn
import rphtz.data as data
import rphtz.optim as optim
from rphtz.data import DataSet
from rphtz.data import DataLoader
import time
import matplotlib.pyplot as plt


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


structure = {'Linear.1': (1, 128),
             'Tanh.1': None,
             'Linear.2': (128, 64),
             'Tanh.2': None,
             'Linear.3': (64, 1)}

if __name__ == '__main__':
    # model = Net()  # Method. 1
    model = nn.Module().create_from_dict(structure)  # Method. 2
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    dataset = data.create_dataset(np.sin, start=-16, stop=16, n=128, err=0.02)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    epoch = 1000

    t0 = time.time()

    losses = nn.train(model, criterion, dataloader, optimizer, epoch, print_info=True)
    t1 = time.time()
    print('Training time: %.5f' % (t1 - t0))

    pred = model(dataset.x)
    plt.figure(1)
    plt.plot(dataset.x[:, 0, 0], pred[:, 0, 0])
    plt.scatter(dataset.x[:, 0, 0], dataset.y[:, 0, 0])

    plt.figure(2)
    plt.plot(losses)

    plt.show()
