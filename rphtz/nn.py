import numpy as np
from .data import DataSet, DataLoader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class Tensor:
    def __init__(self, shape: tuple):
        self.shape = shape
        self.data = np.random.random(size=shape)
        self.grad = np.zeros(self.shape)
        self.layer_record = modules()

        # self.shape = shape

    def __add__(self, other):
        ret = Tensor(shape=self.shape)


class modules:
    def __init__(self):
        self.M = None
        self.data = None
        self.parameters = list()
        self.parameters_grad = list()
        self.chain = list()
        self.input = None
        self.output = None
        self.layer_in = None
        self.layer_out = None
        self.layer_record = None
        self.grad_in = None
        pass

    def forward(self, input: np.ndarray):
        return input

    def __call__(self, input: np.ndarray, *args, **kwargs):
        return self.forward(input=input)

    def backward(self):
        if self.layer_in:
            self.layer_in.backward()

    def __str__(self):
        return self.data


class Linear(modules):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super(Linear, self).__init__()
        # self.weight = np.empty(shape=(in_features, out_features))
        self.has_bias = bias
        self.weight = np.random.normal(size=(in_features, out_features))
        self.bias = np.random.normal(size=(1, out_features))
        self.w_grad = np.empty(shape=(in_features, out_features))
        self.b_grad = np.empty(shape=(1, out_features))
        self.w_momentum = np.zeros(shape=self.weight.shape)
        self.b_momentum = np.zeros(shape=self.bias.shape)
        self.grad_in = np.empty(shape=(1, out_features))

        del self.parameters
        del self.parameters_grad
        del self.layer_record
        del self.data
        del self.chain

        if bias:
            self.params = {'weight': self.weight, 'bias': self.bias}
            self.grads = {'weight': self.w_grad, 'bias': self.b_grad}

            def c(input):
                assert input.ndim == 3, "Dimension of input Error"

                self.input = input
                self.output = np.matmul(input, self.weight) + self.bias
                return self.output

            def b():
                self.grad_in = np.matmul(self.layer_out.grad_in, self.weight.transpose())

                self.w_grad = (np.matmul(self.input.transpose([0, 2, 1]), self.layer_out.grad_in).sum(axis=0))
                self.b_grad = self.layer_out.grad_in.sum(axis=0)
                if self.layer_in:
                    self.layer_in.backward()

            def u(delta_w, delta_b):
                self.weight -= delta_w
                self.bias -= delta_b

            self.call = c
            self.backward = b
            self.update = u

        else:
            del self.bias
            del self.b_grad
            self.params = {'weight': self.weight}
            self.grads = {'weight': self.w_grad}

            def c(input):
                assert input.ndim == 3, "Dimension of input Error"

                self.input = input
                self.output = np.matmul(input, self.weight)
                return self.output

            def b():
                self.grad_in = np.matmul(self.layer_out.grad_in, self.weight.transpose())

                self.w_grad = (np.matmul(self.input.transpose([0, 2, 1]), self.layer_out.grad_in).sum(axis=0))
                if self.layer_in:
                    self.layer_in.backward()

            def u(delta_w, delta_b):
                self.weight -= delta_w

            self.call = c
            self.backward = b
            self.update = u

    def forward(self, m: modules):
        if m.layer_record:
            self.layer_in = m.layer_record
            m.layer_record.layer_out = self
        m.layer_record = self
        m.chain.append(self)
        m.parameters.append(self)

        return m

    def __call__(self, input: np.ndarray, *args, **kwargs):
        return self.call(input)
        # assert input.ndim == 3, "Dimension of input Error"
        #
        # self.input = input
        # self.output = np.matmul(input, self.weight) + self.bias
        # return self.output


class Tanh(modules):
    def __init__(self):
        super().__init__()
        del self.parameters
        del self.parameters_grad
        del self.layer_record
        del self.data
        del self.chain

    def forward(self, m: modules):
        if m.layer_record:
            self.layer_in = m.layer_record
            m.layer_record.layer_out = self
        m.layer_record = self
        m.chain.append(self)
        return m

    def __call__(self, input: np.ndarray, *args, **kwargs):
        self.input = input
        self.output = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        return self.output

    def backward(self):
        self.grad_in = self.layer_out.grad_in * (1 - self.output * self.output)
        if self.layer_in:
            self.layer_in.backward()


class Sigmoid(modules):
    def __init__(self):
        super().__init__()
        del self.parameters
        del self.parameters_grad
        del self.layer_record
        del self.data
        del self.chain

    def forward(self, m: modules):
        if m.layer_record:
            self.layer_in = m.layer_record
            m.layer_record.layer_out = self
        m.layer_record = self
        m.chain.append(self)
        return m

    def __call__(self, input: np.ndarray, *args, **kwargs):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self):
        self.grad_in = self.layer_out.grad_in * self.output * (1 - self.output)
        if self.layer_in:
            self.layer_in.backward()


class ReLU(modules):
    def __init__(self):
        super().__init__()
        del self.parameters
        del self.parameters_grad
        del self.layer_record
        del self.data
        del self.chain

    def forward(self, m: modules):
        if m.layer_record:
            self.layer_in = m.layer_record
            m.layer_record.layer_out = self
        m.layer_record = self
        m.chain.append(self)
        return m

    def __call__(self, input: np.ndarray, *args, **kwargs):
        self.input = input
        self.output = (input > 0) * input
        return self.output

    def backward(self):
        self.grad_in = self.layer_out.grad_in * (self.input > 0) * 1
        if self.layer_in:
            self.layer_in.backward()


class LeakyReLU(modules):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        del self.parameters
        del self.parameters_grad
        del self.layer_record
        del self.data
        del self.chain
        self.mask = None
        self.alpha = np.asarray(alpha, dtype=np.float64)

    def forward(self, m: modules):
        if m.layer_record:
            self.layer_in = m.layer_record
            m.layer_record.layer_out = self
        m.layer_record = self
        m.chain.append(self)
        return m

    def __call__(self, input: np.ndarray, *args, **kwargs):
        self.input = input
        self.mask = np.ones(shape=self.input.shape)
        self.mask[np.where(self.input < 0)] *= self.alpha
        self.output = self.mask * input
        return self.output

    def backward(self):

        self.grad_in = self.layer_out.grad_in * self.mask
        if self.layer_in:
            self.layer_in.backward()


class ELU(modules):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        del self.parameters
        del self.parameters_grad
        del self.layer_record
        del self.data
        del self.chain
        self.mask = None
        self.alpha = np.asarray(alpha, dtype=np.float64)

    def forward(self, m: modules):
        if m.layer_record:
            self.layer_in = m.layer_record
            m.layer_record.layer_out = self
        m.layer_record = self
        m.chain.append(self)
        return m

    def __call__(self, input: np.ndarray, *args, **kwargs):
        self.input = input
        self.mask = input > 0
        self.output = self.input * self.mask + self.alpha * (np.exp(self.input) - 1) * ~self.mask
        return self.output

    def backward(self):
        grad = self.mask * 1 + self.alpha * np.exp(self.input * ~self.mask)
        self.grad_in = self.layer_out.grad_in * grad
        if self.layer_in:
            self.layer_in.backward()


class Swish(modules):
    def __init__(self, beta: float = 1):
        super().__init__()
        del self.parameters
        del self.parameters_grad
        del self.layer_record
        del self.data
        del self.chain
        self.beta = np.asarray(beta, dtype=np.float64)
        self.sig = None

    def forward(self, m: modules):
        if m.layer_record:
            self.layer_in = m.layer_record
            m.layer_record.layer_out = self
        m.layer_record = self
        m.chain.append(self)
        return m

    def __call__(self, input: np.ndarray, *args, **kwargs):
        self.input = input
        self.sig = sigmoid(self.beta * self.input)
        self.output = input * self.sig
        return self.output

    def backward(self):
        self.grad_in = self.layer_out.grad_in * (self.beta * self.output + self.sig * (1 - self.beta * self.output))
        if self.layer_in:
            self.layer_in.backward()


class Softmax(modules):
    def __init__(self):
        super().__init__()
        del self.parameters
        del self.parameters_grad
        del self.layer_record
        del self.data
        del self.chain

    def forward(self, m: modules):
        if m.layer_record:
            self.layer_in = m.layer_record
            m.layer_record.layer_out = self
        m.layer_record = self
        m.chain.append(self)
        return m

    def __call__(self, input: np.ndarray, *args, **kwargs):
        self.input = input
        exp_arr = np.exp(input)
        self.output = exp_arr / np.expand_dims(exp_arr.sum(axis=2), axis=2)
        return self.output

    def backward(self):
        self.grad_in = self.layer_out.grad_in
        if self.layer_in:
            self.layer_in.backward()


class Loss:
    def __init__(self):
        self.input = None
        self.output = None
        self.layer_in = None
        self.layer_out = None
        self.grad_in = None
        self.data = None
        self.label = None
        self.m = modules()

    def forward(self, m: modules):
        self.layer_in = m.layer_record

    # def __call__(self, output: np.ndarray, label: np.ndarray, *args, **kwargs):
    #     return self.m

    def backward(self):
        pass


class MSELoss(Loss):
    def __init__(self, reduction='sum'):
        super(MSELoss, self).__init__()
        if reduction == 'sum' or reduction != 'mean':
            def c(output: np.ndarray, label: np.ndarray):
                self.input = output
                self.label = label
                self.output = (output - label) * (output - label)

                self.data = self.output.sum(axis=0).sum(axis=0).sum(axis=0)
                self.m.data = self.data
                return self.m

            def b():
                self.grad_in = 2 * (self.input - self.label)
                if self.layer_in:
                    self.layer_in.backward()

            self.call = c
            self.backward = b

        if reduction == 'mean':
            def c(output: np.ndarray, label: np.ndarray):
                self.input = output
                self.label = label
                self.output = (output - label) * (output - label)

                self.data = self.output.mean()
                self.m.data = self.data
                return self.m

            def b():
                self.grad_in = 2 * (self.input - self.label) / self.input.shape[0]
                if self.layer_in:
                    self.layer_in.backward()

            self.call = c
            self.backward = b

    def forward(self, m: modules):
        m.layer_record.layer_out = self
        self.layer_in = m.layer_record
        m.layer_record = self

        m.data = self.output
        m.layer_in = m.layer_record
        self.m = m
        return self.m

    def __call__(self, output: np.ndarray, label: np.ndarray, *args, **kwargs):
        return self.call(output, label)
        # self.input = output
        # self.label = label
        # self.output = (output - label) * (output - label)
        #
        # self.data = self.output.sum(axis=0).sum(axis=0).sum(axis=0)
        # self.m.data = self.data
        # return self.m

    def backward(self):
        self.grad_in = 2 * (self.input - self.label)
        if self.layer_in:
            self.layer_in.backward()


class CrossEntropy(Loss):
    def __init__(self):
        self.softmax = None
        super(CrossEntropy, self).__init__()

    def forward(self, m: modules):
        m.layer_record.layer_out = self
        self.layer_in = m.layer_record
        m.layer_record = self

        m.data = self.output
        m.layer_in = m.layer_record
        self.m = m
        return self.m

    def __call__(self, output: np.ndarray, label: np.ndarray, *args, **kwargs):
        self.input = output
        self.label = label
        exp_arr = np.exp(output)
        self.softmax = exp_arr / np.expand_dims(exp_arr.sum(axis=2), axis=2)
        self.output = - np.log(self.softmax) * self.label

        self.data = self.output.sum(axis=0).sum(axis=0).sum(axis=0)
        self.m.data = self.data
        return self.m

    def backward(self):
        # print('MSELoss: backward')
        self.grad_in = self.softmax - self.label
        if self.layer_in:
            self.layer_in.backward()


class Module(object):
    def __init__(self):

        self.head = None
        self.ms = modules()
        self.chain = list()
        self.m = modules()
        self.forward()
        self.chain = self.m.chain
        self.parameters = self.m.parameters
        # self.parameters_grad = self.m.parameters_grad
        self.param_dict = dict()
        for i in range(len(self.parameters)):
            self.param_dict['Linear.%d' % i] = {'w': self.parameters[i].weight,
                                                'b': self.parameters[i].bias}

    def forward(self):
        for module in self.chain:
            self.m = module.forward(self.m)

    def __call__(self, x: np.ndarray, *args, **kwargs):
        for module in self.chain:
            x = module(x)
        return x

    def update(self, delta_params: list):
        for i in range(len(self.parameters)):
            self.parameters[i].update(delta_params[i]['w'], delta_params[i]['b'])

    def __new__(cls, *args, **kwargs):
        return super(Module, cls).__new__(cls, *args, **kwargs)

    def save(self, file):
        for i in range(len(self.parameters)):
            self.param_dict['Linear.%d' % i] = {'w': self.parameters[i].weight,
                                                'b': self.parameters[i].bias}
        np.save(file, self.param_dict)

    def load(self, file):
        params = np.load(file, allow_pickle=True).item()
        for i in range(len(self.parameters)):
            self.parameters[i].weight = params['Linear.%d' % i]['w']  # .copy()
            self.parameters[i].bias = params['Linear.%d' % i]['b']  # .copy()
        for i in range(len(self.parameters)):
            self.param_dict['Linear.%d' % i] = {'w': self.parameters[i].weight,
                                                'b': self.parameters[i].bias}

        return self

    def create_from_dict(self, structure: dict):
        """

        :param structure:
            A dictionary that describe the structure of the model
            For example:
            structure =  {'Linear.0': (1, 4),
                          'Tanh.1': None,
                          'Linear.1': (4, 4),
                          'Tanh.2': None,
                          'Linear.2': (4, 1)}
            model = Module().create_from_dict(structure)
        :return: A  created model with the input structure
        """
        param_index = 0
        for s in structure:
            _st = s.split('.')
            if _st[0] == 'Linear':
                self.m = Linear(in_features=structure[s][0], out_features=structure[s][1]).forward(self.m)
                self.param_dict['Linear.%d' % param_index] = self.chain[-1]
                param_index += 1
            elif _st[0] == 'Tanh':
                self.m = Tanh().forward(self.m)
            elif _st[0] == 'Sigmoid':
                self.m = Sigmoid().forward(self.m)
            elif _st[0] == 'ReLU':
                self.m = ReLU().forward(self.m)
            elif _st[0] == 'LeakyReLU':
                if _st[1]:
                    self.m = LeakyReLU(alpha=_st[1]).forward(self.m)
                else:
                    self.m = LeakyReLU().forward(self.m)
            elif _st[0] == 'ELU':
                if _st[1]:
                    self.m = ELU(alpha=_st[1]).forward(self.m)
                else:
                    self.m = ELU().forward(self.m)
            elif _st[0] == 'Swish':
                if _st[1]:
                    self.m = Swish(beta=_st[1]).forward(self.m)
                else:
                    self.m = Swish().forward(self.m)
        return self


def train(model: Module, criterion, dataloader: DataLoader, optimizer, epoch, print_info=True):
    if print_info:
        def tr(model: Module, criterion, dataloader: DataLoader, optimizer, epoch):

            losses = list()
            criterion.forward(model.m)
            for ep in range(epoch):
                loss_sum = 0
                logging.info('Epoch: %d\t' % ep)
                for i in tqdm(range(dataloader.length), desc='Epoch %d' % ep, leave=True, position=0):
                    x, y = dataloader[i]
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.data
                losses.append(loss_sum)
                # print('Loss sum: %.5f' % loss_sum)
            return np.array(losses)

        return tr(model, criterion, dataloader, optimizer, epoch)

    else:
        def tr(model: Module, criterion, dataloader: DataLoader, optimizer, epoch):
            losses = list()
            criterion.forward(model.m)
            for ep in range(epoch):
                loss_sum = 0
                for i in range(dataloader.length):
                    x, y = dataloader[i]
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.data
                losses.append(loss_sum)
            return np.array(losses)

        return tr(model, criterion, dataloader, optimizer, epoch)


class SA:
    def __init__(self, T0: float, r: float):
        self.T0 = T0
        self.r = r
        self.T = T0
        self.delta_params = list()
        self.count = 0
        self.record = list()

    def generate_new(self, model: Module):
        delta_params = list()
        for i in range(len(model.parameters)):
            delta_params.append({'w': (2 * np.random.random(size=model.parameters[i].weight.shape) - 1) * self.T,
                                 'b': (2 * np.random.random(size=model.parameters[i].bias.shape) - 1) * self.T})

        return delta_params

    def update_params(self, model: Module):
        for i in range(len(model.parameters)):
            model.parameters[i].update(delta_w=self.delta_params[i]['w'],
                                       delta_b=self.delta_params[i]['b'])

    def revert(self, model: Module):
        for i in range(len(model.parameters)):
            model.parameters[i].update(delta_w=-self.delta_params[i]['w'],
                                       delta_b=-self.delta_params[i]['b'])

    def Metropolis(self, cur_loss, pre_loss):
        if cur_loss < pre_loss:
            return True
        else:
            return False

    def run(self, model, criterion, dataloader, optimizer, epoch, print_info=False):
        if print_info:
            def r(model, criterion, dataloader, optimizer, epoch):
                losses = list()
                loss = criterion.forward(model.m)

                for ep in range(epoch):
                    for i in range(dataloader.length):
                        x, y = dataloader[i]
                        output = model(x)
                        loss = criterion(output, y)
                        loss.backward()
                        optimizer.step()
                        print('Epoch: {}\tBatch: {}\tLoss: {}'.format(ep, i, loss.data))
                    pre_loss = criterion(model(dataloader.dataset.x), dataloader.dataset.y).data
                    self.delta_params = self.generate_new(model)
                    self.update_params(model)
                    cur_loss = criterion(model(dataloader.dataset.x), dataloader.dataset.y).data
                    if not self.Metropolis(cur_loss, pre_loss):
                        self.revert(model)
                        self.delta_params.clear()
                        # losses.append(pre_loss)
                        print('Loss sum: %.5f' % pre_loss)
                    else:
                        # losses.append(cur_loss)
                        self.record.append(ep)
                        print('Loss sum: %.5f' % cur_loss)
                        self.count += 1
                        self.T *= self.r
                        print('\tSA Worked--- ---')
                        self.delta_params.clear()
                    loss_sum = criterion(model(dataloader.dataset.x), dataloader.dataset.y).data
                    losses.append(loss_sum)
                return np.array(losses)

            return r(model, criterion, dataloader, optimizer, epoch)

        else:
            def r(model, criterion, dataloader, optimizer, epoch):
                losses = list()
                loss = criterion.forward(model.m)

                for ep in range(epoch):
                    for i in range(dataloader.length):
                        x, y = dataloader[i]
                        output = model(x)
                        loss = criterion(output, y)
                        loss.backward()
                        optimizer.step()
                    pre_loss = criterion(model(dataloader.dataset.x), dataloader.dataset.y).data
                    self.delta_params = self.generate_new(model)
                    self.update_params(model)
                    cur_loss = criterion(model(dataloader.dataset.x), dataloader.dataset.y).data
                    if not self.Metropolis(cur_loss, pre_loss):
                        self.revert(model)
                        self.delta_params.clear()
                        # losses.append(pre_loss)
                    else:
                        # losses.append(cur_loss)
                        self.record.append(ep)
                        self.count += 1
                        self.T *= self.r
                        self.delta_params.clear()
                    loss_sum = criterion(model(dataloader.dataset.x), dataloader.dataset.y).data
                    losses.append(loss_sum)
                return np.array(losses)

            return r(model, criterion, dataloader, optimizer, epoch)


if __name__ == '__main__':
    pass
