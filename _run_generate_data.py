import numpy as np
import pickle
from itertools import product

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

###
replications = 500
def f1(x):
    return x/1.758
def f2(x):
    return x**2/2.75 - 1.5
def f3(x):
    return np.sin(x)/0.72

f_value = [f1, f2, f3]
functions = {f.__name__: f for f in f_value}

sigma_value = [1.0, 0.5, 0.33]
f_sigma_value = list(product(f_value, sigma_value))
###

# Generating data
generate_data = True
if generate_data:
    data = {(f.__name__, sigma): {} for f, sigma in f_sigma_value}

    n = 100
    for f, sigma in f_sigma_value:
        for i in range(replications):
            x = np.random.uniform(-3, 3, n)
            f_evaluated = f(x)
            f_evaluated /= np.std(f_evaluated)
            y = f_evaluated + np.random.normal(0, sigma, n)
            data[(f.__name__, sigma)][i] = (x, y)

    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)
else:
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
