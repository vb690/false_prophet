import os

import pickle


def find_factors(x):
    """
    """
    factors = []
    for i in range(1, x + 1):

        if x % i == 0:
            factors.append(i)

    print(factors)
    return factors


def generate_dir(path):
    """
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_obj(obj, path):
    """
    """
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    """
    """
    with open(f'{path}.pkl', 'rb') as f:
        return pickle.load(f)
