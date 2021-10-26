import numpy as np
epsilon = 1e-6

sum_of_squares = lambda y, y_hat: (y_hat - y) ** 2
d_sum_of_squares = lambda y, y_hat: 2 * (y_hat - y)

binary_ce = lambda y, y_hat: -1 * (y * np.log(y_hat) + (1 - y) * log(1 - y_hat))
d_binary_ce = lambda y, y_hat: (1 - y) / (1 - y_hat + epsilon) - y / (y_hat + epsilon)

LOSS_FUNCS = {
    'binary_ce': [binary_ce, d_binary_ce],
    'sum_of_squares': [sum_of_squares, d_sum_of_squares]
}