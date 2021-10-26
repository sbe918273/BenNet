import numpy as np

def gradient(z, last_a, d_a, d_R):

    d_b = d_R(z) * d_a

    mean_d_b = (d_b @ np.ones((d_b.shape[1], 1))) / d_b.shape[1]
    mean_d_w = (d_b @ last_a.T) / d_b.shape[1]

    return mean_d_b, mean_d_w

def adam(w, z, d_b, d_w, d_a, d_R, x):

        x['v_b'] = d_b if (x['v_b'] is None) else (x['v_b'] * x['beta_1']) + (d_b * (1. - x['beta_1']))
        x['s_b'] = (d_b ** 2) if (x['s_b'] is None) else (x['s_b'] * x['beta_2']) + ((d_b ** 2) * (1. - x['beta_2']))

        delta_b = (x['alpha'] * x['v_b']) / ((x['s_b'] ** 0.5) + x['epsilon'])

        x['v_w'] = d_w if (x['v_w'] is None) else (x['v_w'] * x['beta_1']) + (d_w * (1. - x['beta_1']))
        x['s_w'] = (d_w ** 2) if (x['s_w'] is None) else (x['s_w'] * x['beta_2']) + ((d_w ** 2) * (1. - x['beta_2']))

        delta_w = (x['alpha'] * x['v_w']) / (np.sqrt(x['s_w']) + x['epsilon'])

        last_d_a = w.T @ (d_R(z) * d_a)

        return delta_b, delta_w, last_d_a


def momentum(w, z, d_b, d_w, d_a, d_R, x):


    x['v_b'] = d_b if (x['v_b'] is None) else (x['v_b'] * x['beta']) + (d_b * (1. - x['beta']))

    delta_b = x['alpha'] * x['v_b']

    x['v_w'] = d_w if (x['v_w'] is None) else (x['v_w'] * x['beta']) + (d_w * (1. - x['beta']))

    delta_w = x['alpha'] * x['v_w']

    last_d_a = w.T @ (d_R(z) * d_a)

    return delta_b, delta_w, last_d_a


def rmsprop(w, z, d_b, d_w, d_a, d_R, x):


    x['s_b'] = (d_b ** 2) if (x['s_b'] is None) else (x['s_b'] * x['beta']) + ((d_b ** 2) * (1. - x['beta']))

    delta_b = (x['alpha'] * d_b) / ((x['s_b'] ** 0.5) + x['epsilon'])

    x['s_w'] = (d_w ** 2) if (x['s_w'] is None) else (x['s_w'] * x['beta']) + ((d_w ** 2) * (1. - x['beta']))

    delta_w = (x['alpha'] * d_w) / ((x['s_w'] ** 0.5) + x['epsilon'])

    last_d_a = w.T @ (d_R(z) * d_a)

    return delta_b, delta_w, last_d_a


def sgd(w, z, d_b, d_w, d_a, d_R, x):

    delta_b = x['alpha'] * d_b

    delta_w = x['alpha'] * d_w

    last_d_a = w.T @ (d_R(z) * d_a)

    return delta_b, delta_w, last_d_a


OPTIMISER_REQUIREMENTS = {
    'sgd': {
        'alpha': 1
    },

    'momentum': {
        'alpha': 1, 
        'beta': 0.9, 
        'v_w': None, 
        'v_b': None,
    },

    'rmsprop': {
        'alpha': 1e-1, 
        'beta': 0.9, 
        'epsilon': 1e-7, 
        's_w': None, 
        's_b': None
    },

    'adam': {
        'alpha': 1e-1, 
        'beta_1': 0.9, 
        'beta_2': 0.99, 
        'epsilon': 1e-10, 
        'v_w': None, 
        'v_b': None, 
        's_w': None, 
        's_b': None
    }
}

OPTIMISER_FUNCS = {
    'sgd': sgd,
    'adam': adam,
    'momentum': momentum,
    'rmsprop': rmsprop

}
