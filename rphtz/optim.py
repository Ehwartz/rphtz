from . import data
import numpy as np
import math


class SGD:
    def __init__(self, parameters: list, lr: float, momentum=0, nesterov=False):
        self.parameters = parameters
        self.lr = lr
        assert momentum < 1, 'Momentum out of range'
        # self.momentum = momentum
        self.beta = momentum
        if not momentum:
            for param in parameters:
                del param.w_momentum
                del param.b_momentum

            def s():
                for i in range(len(self.parameters)):
                    delta_w = self.lr * self.parameters[i].w_grad
                    delta_b = self.lr * self.parameters[i].b_grad
                    self.parameters[i].update(delta_w, delta_b)

            self.step = s

        elif not nesterov:
            def s():
                for i in range(len(self.parameters)):
                    delta_w = self.lr * ((1 - self.beta) * self.parameters[i].w_grad +
                                         self.beta * self.parameters[i].w_momentum)
                    delta_b = self.lr * ((1 - self.beta) * self.parameters[i].b_grad +
                                         self.beta * self.parameters[i].b_momentum)
                    self.parameters[i].update(delta_w, delta_b)
                    self.parameters[i].w_momentum = delta_w
                    self.parameters[i].b_momentum = delta_b

            self.step = s

        elif nesterov:
            def s():
                for i in range(len(self.parameters)):
                    delta_w = self.lr * ((1 - self.beta) * self.parameters[i].w_grad +
                                         self.beta * self.parameters[i].w_momentum)
                    delta_b = self.lr * ((1 - self.beta) * self.parameters[i].b_grad +
                                         self.beta * self.parameters[i].b_momentum)
                    self.parameters[i].update((1 + self.lr) * delta_w, (1 + self.lr) * delta_b)
                    self.parameters[i].w_momentum = delta_w
                    self.parameters[i].b_momentum = delta_b
                    # self.parameters[i].update(self.lr * self.parameters[i].w_momentum,
                    #                           self.lr * self.parameters[i].b_momentum)

            self.step = s


class Adagrad:
    def __init__(self, parameters: list, lr: float, eps=1e-10):
        self.parameters = parameters
        self.lr = lr
        self.eps = eps
        self.r_w = list()
        self.r_b = list()
        for param in parameters:
            del param.w_momentum
            del param.b_momentum
            self.r_w.append(np.zeros(shape=param.weight.shape))
            self.r_b.append(np.zeros(shape=param.bias.shape))

    def step(self):
        for i in range(len(self.parameters)):
            self.r_w[i] += np.square(self.parameters[i].w_grad)
            delta_w = (self.lr / (np.sqrt(self.r_w[i]) + self.eps)) * self.parameters[i].w_grad
            self.r_b[i] += np.square(self.parameters[i].b_grad)
            delta_b = (self.lr / (np.sqrt(self.r_b[i]) + self.eps)) * self.parameters[i].b_grad
            self.parameters[i].update(delta_w, delta_b)


class Adadelta:
    def __init__(self, parameters: list, lr: float, alpha: float, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.rho = alpha
        self.eps = eps
        self.r_w = list()
        self.r_b = list()
        for param in parameters:
            self.r_w.append(np.zeros(shape=param.weight.shape))
            self.r_b.append(np.zeros(shape=param.bias.shape))

    def step(self):
        for i in range(len(self.parameters)):
            self.r_w[i] = self.rho * self.r_w[i] + (1 - self.rho) * np.square(self.parameters[i].w_grad)

            delta_w = np.sqrt((self.parameters[i].w_momentum + self.eps) /
                              (self.r_w[i] + self.eps)) * self.parameters[i].w_grad

            self.r_b[i] = self.rho * self.r_b[i] + (1 - self.rho) * np.square(self.parameters[i].b_grad)
            delta_b = np.sqrt((self.parameters[i].b_momentum + self.eps) /
                              (self.r_b[i] + self.eps)) * self.parameters[i].b_grad
            self.parameters[i].update(delta_w, delta_b)
            self.parameters[i].w_momentum = self.rho * self.parameters[i].w_momentum + (
                    1 - self.rho) * delta_w * delta_w
            self.parameters[i].b_momentum = self.rho * self.parameters[i].b_momentum + (
                    1 - self.rho) * delta_b * delta_b


class RMSprop:
    def __init__(self, parameters: list, lr: float, alpha: float, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.rho = alpha
        self.eps = eps
        self.r_w = list()
        self.r_b = list()
        for param in parameters:
            del param.w_momentum
            del param.b_momentum
            self.r_w.append(np.zeros(shape=param.weight.shape))
            self.r_b.append(np.zeros(shape=param.bias.shape))

    def step(self):
        for i in range(len(self.parameters)):
            self.r_w[i] = self.rho * self.r_w[i] + (1 - self.rho) * np.square(self.parameters[i].w_grad)
            delta_w = (self.lr / (np.sqrt(self.r_w[i]) + self.eps)) * self.parameters[i].w_grad
            self.r_b[i] = self.rho * self.r_b[i] + (1 - self.rho) * np.square(self.parameters[i].b_grad)
            delta_b = (self.lr / (np.sqrt(self.r_b[i]) + self.eps)) * self.parameters[i].b_grad
            self.parameters[i].update(delta_w, delta_b)


class Adam:
    def __init__(self, parameters: list, lr: float, betas: tuple, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.r_w = list()
        self.r_b = list()
        self.t = 0
        for param in parameters:
            self.r_w.append(np.zeros(shape=param.weight.shape))
            self.r_b.append(np.zeros(shape=param.bias.shape))

    def step(self):
        self.t += 1
        for i in range(len(self.parameters)):
            self.parameters[i].w_momentum = (self.beta1 * self.parameters[i].w_momentum
                                             + (1 - self.beta1) * self.parameters[i].w_grad)
            self.r_w[i] = self.beta2 * self.r_w[i] + (1 - self.beta2) * np.square(self.parameters[i].w_grad)
            wm_hat = self.parameters[i].w_momentum / (1 - math.pow(self.beta1, self.t))
            wr_hat = self.r_w[i] / (1 - math.pow(self.beta2, self.t))
            delta_w = (self.lr * wm_hat / (np.sqrt(wr_hat) + self.eps))

            self.parameters[i].b_momentum = (self.beta1 * self.parameters[i].b_momentum
                                             + (1 - self.beta1) * self.parameters[i].b_grad)
            self.r_b[i] = self.beta2 * self.r_b[i] + (1 - self.beta2) * np.square(self.parameters[i].b_grad)
            bm_hat = self.parameters[i].b_momentum / (1 - math.pow(self.beta1, self.t))
            br_hat = self.r_b[i] / (1 - math.pow(self.beta2, self.t))
            delta_b = (self.lr * bm_hat / (np.sqrt(br_hat) + self.eps))

            self.parameters[i].update(delta_w, delta_b)
