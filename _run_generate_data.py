import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import pickle
from itertools import product
from _utils_functions import functions

import warnings
warnings.filterwarnings('ignore')

def main(data_path, f_sigma_list, generate_data=False):
    ###
    replications = 100
    n = 100
    ###

    # Generating data
    if generate_data:
        # print(f_sigma_list)
        data = {(f, sigma): {} for f, sigma in f_sigma_list}
        np.random.seed(42)

        x = np.random.uniform(-3, 3, (replications, n))
        eps = np.random.normal(0, 1.0, (replications, n))
        for f, sigma in f_sigma_list:
            if f in ['f5', 'f5b', 'f4b']:
                continue
            for i in range(replications):
                if i % 100 == 0:
                    print(f'Generating data for f={f}, sigma={sigma}, replication={i}')
                x_i = x[i, :]
                eps_i = eps[i, :]
                f_evaluated = functions[f](x_i)
                y_i = f_evaluated + sigma * eps_i

                data[(f, sigma)][i] = (x_i, y_i)
                if f == 'f4':
                    for f_ in ['f5', 'f5b', 'f4b']:
                        if f_ in [f_ for f_, _ in f_sigma_list]:
                            if f_ == 'f5':
                                data[(f_, sigma)][i] = (-x_i, y_i)
                            elif f_ == 'f5b':
                                data[('f5b', sigma)][i] = (-x_i, y_i*200.0)
                            elif f_ == 'f4b':
                                data[('f4b', sigma)][i] = (x_i, y_i*200.0)

        with open(data_path, "wb") as f:
            pickle.dump(data, f)
    else:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    return data
