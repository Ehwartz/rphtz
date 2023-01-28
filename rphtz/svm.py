import numpy as np
import matplotlib.pyplot as plt
from data import create_svm_data


class SVC:
    def __init__(self, dim, C, lr, n_iters, optim):
        self.w = np.zeros(shape=[dim])
        self.alphas = self.w
        self.alphas_grad = self.alphas
        # self.w = np.mat(self.w)
        self.b = np.zeros(shape=[1])
        # self.b = np.mat(self.b)
        self.w_grad = self.w.copy()
        self.b_grad = self.b.copy()
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.optim = optim
        self.supVecNums = list()

    def dist(self, x, y):
        return y * np.dot(self.w, x)

    def Loss(self, x, y):
        loss = 0.0
        loss += np.dot(self.w, self.w) / 2
        dist = (1 - ((self.w * x).sum(axis=1) + self.b) * y)
        mask = np.array([(dist > 0) * 1]).transpose()
        # print(mask)
        loss += (mask * dist).sum()
        self.w_grad = self.w - (mask * x * np.array([y]).transpose()).sum(axis=0)
        self.b_grad = -(mask.reshape(y.shape) * y).sum(axis=0)
        return loss

    def dual(self, x, y, alphas):
        argmax = 0.0
        argmax += self.alphas.sum(axis=0)
        for i in range(y.shape[0]):
            for j in range(y.shape[0]):
                argmax -= alphas[i] * alphas[j] * y[i] * y[j] * np.dot(x[i], x[j]) / 2

        return argmax

    def fit(self, x, y):
        if self.optim == 'GD':
            self.gradient_descent(x, y)
        if self.optim == 'CD':
            self.coordinate_descent(x, y, 0.01)

    def gradient_descent(self, x, y):

        loss_pre = self.Loss(x, y) + 1
        for n in range(self.n_iters):
            loss = self.Loss(x, y)
            # if loss > loss_pre:
            #     break
            loss_pre = loss
            print('iter: ', n, 'Loss:', loss)
            self.w -= self.lr * self.w_grad
            self.b -= self.lr * self.b_grad

    def coordinate_descent(self, x, y, delta):
        self.alphas = np.ones(y.shape)
        self.alphas_grad = np.zeros(y.shape)
        for n in range(self.n_iters):
            loss_sum = 0.0
            for i in range(self.alphas.shape[0]):
                argmax = self.dual(x, y, self.alphas)
                alphas_ = np.zeros(self.alphas.shape)
                alphas_[i] = delta
                argmax_ = self.dual(x, y, self.alphas + alphas_)
                self.alphas_grad[i] = (argmax_ - argmax) / delta
                self.alphas[i] = (self.alphas[i] + self.lr * self.alphas_grad[i]) * (self.alphas[i] > 0)
                loss_sum += argmax
            print('iter: ', n, '    Sum: ', loss_sum / y.shape[0])

        self.w = (np.array([self.alphas]).transpose() * x * np.array([y]).transpose()).sum(axis=0)
        self.supVecNums = list()
        for i in range(y.shape[0]):
            if self.alphas[i] > 0:
                self.supVecNums.append(i)

        print(self.supVecNums)


class S3VC(SVC):
    def __init__(self, dim, C, lr, n_iters, optim='GD'):
        super().__init__(dim, C, lr, n_iters, optim)
        self.unlabeledX = list()
        self.unlabeledY = list()
        self.labeledX = list()
        self.labeledY = list()

    def fit(self, x, y):
        for i in range(y.shape[0]):
            if y[i] == 0:
                self.unlabeledX.append(x[i])
                self.unlabeledY.append(y[i])
            else:
                self.labeledX.append(x[i])
                self.labeledY.append(y[i])

        self.unlabeledX = np.array(self.unlabeledX)
        self.unlabeledY = np.array(self.unlabeledY)
        self.labeledX = np.array(self.labeledX)
        self.labeledY = np.array(self.labeledY)

        self.gradient_descent(self.labeledX, self.labeledY)
        print(self.w)
        print(self.b)
        for n in range(self.n_iters):
            loss = 0.0
            dist = (1 - ((self.w * self.labeledX).sum(axis=1) + self.b) * self.labeledY)
            mask = np.array([(dist > 0) * 1]).transpose()
            loss += np.dot(self.w, self.w) / 2
            loss += (mask * dist).sum()
            maskF = ((((self.w * self.unlabeledX).sum(axis=1) + self.b) > 0) * 1 -
                     (((self.w * self.unlabeledX).sum(axis=1) + self.b) < 0) * 1)
            maskU = ((1 - np.fabs(((self.w * self.unlabeledX).sum(axis=1) + self.b))) > 0) * 1

            loss += (maskU * (1 - np.fabs(((self.w * self.unlabeledX).sum(axis=1) + self.b)))).sum()
            print('iter: ', n, 'Loss:', loss)
            self.w_grad = (self.w -
                           (mask * self.labeledX * np.array([self.labeledY]).transpose()).sum(axis=0) -
                           (np.array([maskF * maskU]).transpose() * self.unlabeledX).sum(axis=0))
            self.b_grad = -((mask.reshape(self.labeledY.shape) * self.labeledY).sum(axis=0) +
                            (maskF.reshape(self.unlabeledY.shape) * maskU.reshape(self.unlabeledY.shape)).sum(axis=0))

            self.w -= self.w_grad * self.lr
            self.b -= self.b_grad * self.lr


if __name__ == '__main__':
    x, y = create_svm_data(np.array([[1, 3], [3, 1]]),
                           radius=1,
                           n=200,
                           unlabeled=0.5)
    model = S3VC(2, 1, 0.0003, 20000, 'GD')

    model.fit(x, y)
    print(model.w, '\n', model.b)

    x_max, x_min = 0, 0
    for i in range(y.shape[0]):
        if x[i][0] > x_max:
            x_max = x[i][0]
        if x[i][1] < x_min:
            x_min = x[i][1]
        if np.dot(model.w, x[i]) + model.b > 0:
            y[i] = 1
        if np.dot(model.w, x[i]) + model.b < 0:
            y[i] = -1

    x_ = np.linspace(x_min, x_max)
    k = -model.w[0] / model.w[1]
    b = -model.b / model.w[1]
    y_ = k * x_ + b
    plt.plot(x_, y_)

    # print(model.alphas)
    for i in range(y.shape[0]):
        if y[i] > 0:
            plt.plot(x[i][0], x[i][1], marker='o', color='r')
        else:
            plt.plot(x[i][0], x[i][1], marker='o', color='b')

    plt.show()
