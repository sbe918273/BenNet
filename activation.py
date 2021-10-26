import numpy as np

sigmoid = lambda x: 1.0 /( 1 + np.exp(-x))
d_sigmoid = lambda x: np.exp(-x) / ((np.exp(-x) + 1) ** 2)

relu = lambda x: np.maximum(x, 0)
d_relu = lambda x: (x > 0) * 1

ACTIVATION_FUNCS = {
    'sigmoid': [sigmoid, d_sigmoid],
    'relu': [relu, d_relu]
}