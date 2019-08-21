"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import random
import time

import numpy as np
import os
import pickle

class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0

class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        # Might we need to store something before returning?
        self.state = 1/(1+np.exp(-x))
        return self.state

    def derivative(self):

        # Maybe something we need later in here...
        return self.state * (1 - self.state)

class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return 1 - np.square(self.state)

class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        x[x < 0] = 0
        self.state = x
        return self.state

    def derivative(self):
        # return np.greater(self.state, 0).astype(int)
        return 1. * (self.state > 0)

class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None
        self.yhats = []

    def old_forward(self, x, y):

        self.logits = x
        self.labels = y

        # #max for each instance
        # max = np.array([np.max(xi) for xi in x])
        # #denom for each instance
        # denoms = np.array([max_i + np.log(np.exp(batch_i - max_i).sum()) for batch_i, max_i in zip(x, max)])

        losses = []
        for i in range(x.shape[0]):
            xi = x[i]
            yi = y[i]
            if np.count_nonzero(yi) == 9:
                yi = 1 - yi
            # denom = denoms[i]
            # sm = np.array([np.exp(xij) / denom for xij in xi])
            denom = np.exp(xi).sum()
            sm = np.array([np.exp(xij) / denom for xij in xi])
            if self.sm is None:
                self.sm = sm
            else:
                self.sm = np.vstack([self.sm, sm])
            #pick, from the 10 possible values for this example, the softmax at the index where y==1
            loss = -1 * np.log(sm[yi == 1])
            losses.append(loss)

        self.loss = np.array(losses)
        self.loss = self.loss.reshape(self.loss.shape[0],)

        return self.loss

    def forward(self, x, y):
        self.logits = x
        self.labels = y

        losses = []
        self.sm = None
        for i in range(x.shape[0]):
            xi = x[i]
            yi = y[i]
            if np.count_nonzero(yi) == 9:
                yi = 1 - yi
            m = np.max(xi)
            yhat = np.exp(xi - m)
            self.yhats.append(yhat)
            sm = np.log(yhat / yhat.sum())
            # sm = m + np.log(np.exp(xi).sum())

            if self.sm is None:
                self.sm = sm
            else:
                self.sm = np.vstack([self.sm, sm])
            loss = -1 * sm[yi == 1]
            losses.append(loss)

        self.loss = np.array(losses)
        self.loss = np.reshape(self.loss, self.loss.shape[0],)

        return self.loss

    def derivative(self):

        # self.sm might be useful here...
        batch_grad = []
        for i in range(self.logits.shape[0]):
            # grad = self.sm[i] - self.labels[i]
            grad = self.yhats[i] / self.yhats[i].sum()
            grad[self.labels[i]==1] -=1
            batch_grad.append(grad)
        return np.vstack(batch_grad)

#MLP, then BatchNorm, then Activation
class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.fan_in = fan_in

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))


        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, train=True):

        if train == False:
            self.x = x
            self.norm = (x - self.running_mean) / (np.sqrt(self.running_var + self.eps))
            self.out = self.gamma * self.norm + self.beta
            return self.out

        self.x = x

        # self.mean = # ??? (1,fan_in)
        self.mean = np.sum(x, axis=0) / x.shape[0]
        # self.var = # ???
        square_diff = np.square(x - self.mean.T)
        self.var = np.sum(square_diff, axis=0) / x.shape[0]

        # self.norm = # ???
        self.norm = (x - self.mean) / (np.sqrt(self.var + self.eps))
        # self.out = # ???
        self.out = self.gamma * self.norm + self.beta

        # if self.out.mean() > 0+1e-6 or self.out.mean() < 0-1e-6:
        if self.out.mean() != 0:
            print("Batchnorm'd values don't have mean of 0; instead, they have mean of " + repr(self.out.mean()))
        # if self.out.var() > 1+1e-6 or self.out.var() < 1-1e-6:
        if self.out.var() != 0:
            print("Batchnorm'd values don't have var of 1; instead, they have var of " + repr(self.out.var()))

        # update running batch statistics
        # self.running_mean = # ???
        self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
        # self.running_var = # ???
        self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var

        return self.out

    def backward(self, delta):
        m = len(self.x)
        dL_dY = delta
        dL_dNorm = dL_dY * self.gamma
        dL_dBeta = np.sum(dL_dY, axis=0)
        dL_dGamma = np.sum(dL_dY * self.norm, axis=0)

        # dNorm_dVar = np.matmul(self.x - self.mean, -1/2 * np.power((self.var + self.eps),-3/2))
        # dL_dVar = np.inner(dL_dNorm, dNorm_dVar)
        dL_dVar = -1/2 * np.sum(dL_dNorm * (self.x - self.mean) * np.power((self.var + self.eps), -3/2) , axis=0)

        # dL_dMean = np.sum(dL_dNorm * -1/np.sqrt(self.var + self.eps)) + np.matmul(dL_dVar, (-2 * (self.x - self.mean)).sum() / m)
        # dL_dMean = -1 * np.sum(dL_dNorm * np.power((self.var + self.eps), -1/2), axis=0) - 1/2 * np.sum((dL_dNorm * (self.x - self.mean) * np.power((self.var - self.eps), -3/2))  , axis=0) * (-2 / m * np.sum((self.x - self.mean), axis=0))
        dL_dMean = -1 * np.sum(dL_dNorm * np.power((self.var + self.eps), -1 / 2), axis=0) + dL_dVar * (-2 / m * np.sum((self.x - self.mean), axis=0))

        dL_dX = dL_dNorm * np.power((self.var + self.eps), -1 / 2) + dL_dVar * (2 / m)*(self.x - self.mean) + (dL_dMean * 1 / m)

        self.dbeta = dL_dBeta
        self.dgamma = dL_dGamma

        return dL_dX

# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.array(np.random.normal(0, 1, (d0, d1)))

def zeros_bias_init(d):
    return np.zeros(d)

class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------
        self.W = []
        self.b = []

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        if self.nlayers == 1:
            self.W.append(weight_init_fn(input_size, output_size))
            self.b.append(bias_init_fn(output_size))
        else:
            self.W.append(weight_init_fn(input_size, hiddens[0]))
            self.b.append(bias_init_fn(hiddens[0]))
            for i in range(0, len(hiddens)-1):
                self.W.append(weight_init_fn(hiddens[i], hiddens[i+1]))
                self.b.append(bias_init_fn(hiddens[i+1]))
            self.W.append(weight_init_fn(hiddens[-1], output_size))
            self.b.append(bias_init_fn(output_size))
        self.dW = []
        self.db = []
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = [BatchNorm(w.shape[0]) for w in self.W[1:self.num_bn_layers+1]]

        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.x = []
        self.y = []
        self.z = []
        # self.dW_updates = np.zeros_like(self.W)
        # self.db_updates = np.zeros_like(self.b)
        self.dW_updates = [np.zeros_like(self.W[i]) for i in range(len(self.W))]
        self.db_updates = [np.zeros_like(self.b[i]) for i in range(len(self.b))]


    def forward(self, x):
        self.x = x
        self.y.append(x)
        # if self.nlayers == 1:
        #     self.output = self.activations[0].forward(np.matmul(x, self.W[0]) + self.b[0])
        # else:
        for i in range(self.nlayers-1):
            zk = np.matmul(self.y[i], self.W[i]) + self.b[i]
            self.z.append(zk)
            # shape = (len(self.y[i]), len(self.W[i+1][1]))
            # yk = np.reshape(self.activations[i].forward(zk), shape)
            if self.num_bn_layers >= i + 1:
                zk = self.bn_layers[i].forward(zk, self.train_mode)
            yk = self.activations[i].forward(zk)
            self.y.append(yk)

        i = self.nlayers-1
        zk = np.matmul(self.y[i], self.W[i]) + self.b[i]
        self.z.append(zk)
        shape = (len(self.y[i]), self.output_size)
        yk = np.reshape(self.activations[i].forward(zk), shape)
        self.y.append(yk)

        # the output of the layer immediately before the criterion layer
        return self.y[-1]

    def zero_grads(self):
        self.dW = []
        self.db = []
        # self.dW_updates = [np.zeros_like(self.W[i]) for i in range(len(self.W))]
        # self.db_updates = [np.zeros_like(self.b[i]) for i in range(len(self.b))]
        if self.bn:
            for i in range(0, self.num_bn_layers):
                self.bn_layers[i].dbeta = np.zeros((1, self.bn_layers[i].fan_in))
                self.bn_layers[i].dgamma = np.zeros((1, self.bn_layers[i].fan_in))

    def zero_intermediates(self):
        self.x = []
        self.y = []
        self.z = []

    def step(self):
        # if self.momentum == 0.0:
        #     for i in range(len(self.W)):
        #         self.W[i] -= self.dW[i] * self.lr
        #         self.b[i] -= self.db[i] * self.lr
        # else:
        for i in range(len(self.W)):
            # if len(self.dW_updates) < i+1:
            #     self.dW_updates.append(np.zeros_like(self.W[i]))
            #     self.db_updates.append(np.zeros_like(self.b[i]))

            this_dW_update = self.lr * self.dW[i]
            this_db_update = self.lr * self.db[i]

            last_dW_update = self.dW_updates[i] * self.momentum
            last_db_update = self.db_updates[i] * self.momentum

            momentum_dW_update = last_dW_update - this_dW_update
            momentum_db_update = last_db_update - this_db_update

            self.dW_updates[i] = momentum_dW_update
            self.db_updates[i] = momentum_db_update

            self.W[i] += momentum_dW_update
            self.b[i] += momentum_db_update

        if self.bn:
            for i in range(0, self.num_bn_layers):
                self.bn_layers[i].gamma += self.lr * self.bn_layers[i].dgamma
                self.bn_layers[i].beta += self.lr * self.bn_layers[i].dbeta

    def momentum(self):
        self.step()

    def backward(self, labels):
        loss = self.criterion.forward(self.y[-1], labels)
        dDiv_dY = self.criterion.derivative()
        # if self.nlayers - 2 <= self.num_bn_layers:
        #     dDiv_dY = self.bn_layers[-1].backward(dDiv_dY)
        dDiv_dZk = dDiv_dY * self.activations[-1].derivative()

        if(len(self.activations)) > 1:
            yk = self.activations[-2].state
            dDiv_dWk = np.matmul(yk.T, dDiv_dZk) / len(yk)
            self.dW.append(dDiv_dWk)
            dDiv_dbk = np.sum(dDiv_dZk, axis=0) / len(yk)
            self.db.append(dDiv_dbk)
        # linear classifier
        else:
            yk = self.x
            dDiv_dWk = np.matmul(yk.T, dDiv_dZk) / len(yk)
            # dDiv_dWk = np.matmul(self.y[-2].T, dDiv_dZk) / len(self.y[-2])
            self.dW.append(dDiv_dWk)
            dDiv_dbk = np.sum(dDiv_dZk, axis=0) / len(yk)
            self.db.append(dDiv_dbk)

        dDiv_dYk = np.matmul(dDiv_dZk, self.W[-1].T)

        # for loop
        for k in range(len(self.activations) - 2, -1, -1):
            dDiv_dZk = dDiv_dYk * self.activations[k].derivative()
            if self.num_bn_layers >= k + 1:
                dDiv_dZk = self.bn_layers[k].backward(dDiv_dZk)
            # self.activations[k - 1].state
            if k == 0:
                yk = self.x
            else:
                yk = self.activations[k-1].state
            dDiv_dWk = np.matmul(yk.T, dDiv_dZk) / len(yk)
            # dDiv_dWk = np.matmul(self.y[k].T, dDiv_dZk) / len(self.y[k])
            self.dW.append(dDiv_dWk)
            dDiv_dbk = np.sum(dDiv_dZk, axis=0) / len(yk)
            self.db.append(dDiv_dbk)

            dDiv_dYk = np.matmul(dDiv_dZk, self.W[k].T)

        self.dW.reverse()
        self.db.reverse()

        return loss

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def weight_init(x, y):
    return np.random.randn(x, y)

def bias_init(x):
    return np.zeros((1, x))

# mlp = MLP(784, 10, [], [Identity()], weight_init, bias_init,
#                   SoftmaxCrossEntropy(), 0.008, momentum=0.0,
#                   num_bn_layers=0)
# mlp = MLP(784, 10, [32], [Sigmoid(), Identity()], weight_init, bias_init,
#                   SoftmaxCrossEntropy(), 0.008, momentum=0.008,
#                   num_bn_layers=0)
# mlp = MLP(784, 10, [64,32], [Sigmoid(), Sigmoid(), Identity()], weight_init, bias_init,
#                   SoftmaxCrossEntropy(), 0.008, momentum=0.0,
#                   num_bn_layers=1)
# saved_data = pickle.load(open("data.pkl", 'rb'))
# data = saved_data[3]
# x = data[0]
# y = data[1]
# mlp.train()
# mlp.zero_grads()
# mlp.forward(x)
# mlp.backward(y)
# mlp.step()
#
# mlp.eval()
# mlp.forward(x)


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):
        # batch_training_losses = []
        # batch_validation_losses = []
        # batch_training_errors = []
        # batch_validation_errors = []

        batch_training_loss = 0
        batch_validation_loss = 0
        batch_training_error = 0
        batch_validation_error = 0

        num_training_batches = 0
        num_validation_batches = 0

        start_time = time.time()

        # Per epoch setup ...
        np.random.shuffle(idxs)
        train_data = trainx[idxs]
        train_labels = trainy[idxs]

        mlp.train()
        for b in range(0, len(trainx), batch_size):

            forward = mlp.forward(train_data[b:b+batch_size])
            # training_predictions = softmax(forward)
            training_predictions = np.argmax(forward, axis=1)
            # batch_training_errors.append((training_predictions != np.reshape([np.nonzero(y) for y in trainy[idxs[b:b+batch_size]]], batch_size)).mean())
            training_labels = np.argmax(train_labels[b:b+batch_size], axis=1)
            training_correct = (training_predictions == training_labels).sum()
            batch_training_error += (1.0 - training_correct / batch_size)
            mlp.zero_grads()
            batch_losses = mlp.backward(train_labels[b:b+batch_size])
            batch_training_loss += np.mean(batch_losses)
            mlp.step()
            mlp.zero_intermediates()
            num_training_batches += 1

        mlp.eval()
        for b in range(0, len(valx), batch_size):
            forward = mlp.forward(valx[b:b+batch_size])
            # validation_predictions = softmax(forward)
            validation_predictions = np.argmax(forward, axis=1)
            validation_labels = np.argmax(valy[b:b+batch_size], axis=1)
            validation_correct = (validation_predictions == validation_labels).sum()
            batch_validation_error += (1.0 - validation_correct/batch_size)
            batch_validation_loss += np.mean(mlp.criterion.forward(forward, valy[b:b+batch_size]))

            num_validation_batches += 1

        # Accumulate data...
        training_errors.append(batch_training_error / num_training_batches)
        training_losses.append(batch_training_loss / num_training_batches)
        validation_errors.append(batch_validation_error / num_validation_batches)
        validation_losses.append(batch_training_loss / num_validation_batches)

        if e % 1 == 0:
            print('After ' + repr(e) + ' epochs, training loss is ' + repr(training_losses[e]) + ' and validation loss is ' + repr(validation_losses[e]))
            print('After ' + repr(e) + ' epochs, training error is ' + repr(training_errors[e]) + ' and validation error is ' + repr(validation_errors[e]))
            print('Epoch ' + repr(e) + ' took ' + repr(time.time() - start_time) + ' seconds')

    # Cleanup ...
    # Return results ...

    return training_losses, training_errors, validation_losses, validation_errors

def softmax(x):
    yhats = []
    sms = None
    for i in range(x.shape[0]):
        xi = x[i]
        m = np.max(xi)
        yhat = np.exp(xi - m)
        yhats.append(yhat)
        sm = np.argmax(np.log(yhat / yhat.sum()))

        if sms is None:
            sms = sm
        else:
            sms = np.vstack([sms, sm])
    return sms

def make_one_hot(labels_idx):
    labels = np.zeros((labels_idx.shape[0], 10))
    labels[np.arange(labels_idx.shape[0]), labels_idx] = 1
    return labels

def shuffle_together(x, y):
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]

def process_dset_partition(dset_partition, normalize=True):
    data, labels_idx = dset_partition
    mu, std = data.mean(), data.std() if normalize else (0, 1)
    return (data - mu) / std, make_one_hot(labels_idx)
#
# epochs = 200
# batch_size = 100
# thisdir = os.path.dirname(__file__)
# train_data_path = os.path.join(thisdir, "../data/train_data.npy")
# train_labels_path = os.path.join(thisdir, "../data/train_labels.npy")
#
# val_data_path = os.path.join(thisdir, "../data/val_data.npy")
# val_labels_path = os.path.join(thisdir, "../data/val_labels.npy")
#
# test_data_path = os.path.join(thisdir, "../data/test_data.npy")
# test_labels_path = os.path.join(thisdir, "../data/test_labels.npy")
#
# dset = (
#     process_dset_partition((np.load(train_data_path), np.load(train_labels_path))),
#     process_dset_partition((np.load(val_data_path), np.load(val_labels_path))),
#     process_dset_partition((np.load(test_data_path), np.load(test_labels_path))))
#
# mlp = MLP(784, 10, [32, 32, 32], [Sigmoid(), Sigmoid(), Sigmoid(), Identity()],
#              random_normal_weight_init, zeros_bias_init,
#              SoftmaxCrossEntropy(),
#              1e-3)
# training_losses, training_errors, validation_losses, validation_errors = get_training_stats(mlp, dset, epochs, batch_size)
