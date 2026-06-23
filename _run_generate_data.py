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
from _run_model_keys._run_model_keys import data_name

import warnings
warnings.filterwarnings('ignore')

###
replications = 500
n = 100

f_value = [functions['f1'], functions['f2'], functions['f3'], functions['f4'], functions['f5']]
sigma_value = [1.0, 0.5, 0.33]
f_sigma_value = list(product(f_value, sigma_value))
###

# Generating data
data_folder = 'data/'
os.makedirs(data_folder, exist_ok=True)
data_path = os.path.join(data_folder, f'{data_name}.pkl')
generate_data = False
if generate_data:
    data = {(f.__name__, sigma): {} for f, sigma in f_sigma_value}
    np.random.seed(42)

    x = np.random.uniform(-3, 3, (replications, n))
    eps = np.random.normal(0, 1.0, (replications, n))
    for f, sigma in f_sigma_value:
        if f.__name__ == 'f5':
            continue
        for i in range(replications):
            x_i = x[i, :]
            eps_i = eps[i, :]
            f_evaluated = f(x_i)
            y_i = f_evaluated + sigma * eps_i

            data[(f.__name__, sigma)][i] = (x_i, y_i)
            if f.__name__ == 'f4':
                data[('f5', sigma)][i] = (-x_i, y_i)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
else:
    with open(data_path, "rb") as f:
        data = pickle.load(f)
