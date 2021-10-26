from loss import LOSS_FUNCS
from activation import ACTIVATION_FUNCS
from optimiser import gradient, OPTIMISER_REQUIREMENTS, OPTIMISER_FUNCS

import numpy as np
np.random.seed(14)

class Layer:

    def __init__(self, units, last_units, optimiser, activation):

        self.units = units

        self.a = np.zeros((units, 1))
        self.last_a = None
        self.z = None
        self.w = np.random.rand(units, last_units)
        self.b = np.random.rand(units, 1)

        self.R, self.d_R = ACTIVATION_FUNCS[activation]
        self.optimise = OPTIMISER_FUNCS[optimiser]
        self.x = OPTIMISER_REQUIREMENTS[optimiser].copy()

    def feedforward(self, last_a):


        self.last_a = last_a

        self.z = self.w @ self.last_a + self.b

        self.a = self.R(self.z)

        return self.a

    def backpropagate(self, d_a=None, y=None, d_L=None):

        if d_a is None:

            d_a = d_L(y, self.a)

        d_b, d_w = gradient(self.z, self.last_a, d_a, self.d_R)

        delta_b, delta_w, last_d_a = self.optimise(self.w, self.z, d_b, d_w, d_a, self.d_R, self.x)

        self.b -= delta_b
        self.w -= delta_w

        return last_d_a

class Network:

    def __init__(self, input_dim, loss):
        
        self.layers = []
        self.units = [input_dim]
        [self.L, self.d_L] = LOSS_FUNCS[loss]

    def add_layer(self, units, optimizer, activation):

        self.layers.append(Layer(units, self.units[-1], optimizer, activation))
        self.units.append(units)

    def train(self, X, y, epochs=10000, batch_size=8, update=None):
        
        X_len = X.shape[1]

        batch_start = 0

        for _ in range(epochs):

            batch_end = batch_start + batch_size

            if batch_end >= X_len:

                batch_X = X[:, batch_start:]
                batch_y = y[:, batch_start:]
                batch_start = 0
                
                rand_idxs = np.arange(X_len)
                np.random.shuffle(rand_idxs)
                X = X[:, rand_idxs]
                y = y[:, rand_idxs]

            else:

                batch_X = X[:, batch_start:batch_end]
                batch_y = y[:, batch_start:batch_end]
                batch_start = batch_end

            #batch_X = np.array([[int(input())] for i in range(0,3)]).reshape(-1, 1)
            #batch_y = np.array([[int(input())] for i in range(0,1)]).reshape(-1, 1)
            
            a = self.predict(batch_X)
            
            #WARNING!
            #self.layers[-1].a = np.around(self.layers[-1].a)

            self.backpropagate(batch_y)

            if update is not None:   
                update()


    def predict(self, X):

        y_hat = X
        for layer in self.layers:
            y_hat = layer.feedforward(y_hat)

        return y_hat

    def backpropagate(self, y):

        d_a = self.layers[-1].backpropagate(y=y, d_L=self.d_L)

        for layer in self.layers[-2::-1]:
            d_a = layer.backpropagate(d_a=d_a)

#if __name__ == 'main': 

network = Network(input_dim=2, loss='binary_ce')
network.add_layer(4, 'rmsprop', 'sigmoid')
network.add_layer(1, 'rmsprop', 'sigmoid')

X = np.array([
    [0, 1, 0, 1],
    [0, 0, 1, 1],
])

y = np.array([
    [0, 1, 1, 0]
])

X_test = np.array([
    [0],
    [1]
])

network.train(X, y, epochs=1000, batch_size=1)
print(network.predict(X_test))
