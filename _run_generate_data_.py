import numpy as np
import pickle
from itertools import product
from _utils_spline import eval_spline_basis_equispaced_numeric

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

def f4(x):
    spline_degree = 3
    n_internal_knots = 5
    eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x), np.max(x), n_internal_knots, x)
    B = eval_B['B']
    return B[:, -1] # last column
def f5(x):
    spline_degree = 3
    n_internal_knots = 5
    eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(-x), np.max(-x), n_internal_knots, -x)
    B = eval_B['B']
    return B[:, 0] # first column

f_value = [f1, f2, f3]
f_value = [f4, f5]
functions = {f.__name__: f for f in f_value}

sigma_value = [1.0, 0.5, 0.33]
f_sigma_value = list(product(f_value, sigma_value))
###

# Generating data
data_name = 'data45s.pkl'
# data_name = 'data.pkl'
generate_data = False
if generate_data:
    data = {(f.__name__, sigma): {} for f, sigma in f_sigma_value}
    if f4 in f_value:
        for sigma in sigma_value:
            data[('f5', sigma)] = {}
    n = 100
    for f, sigma in f_sigma_value:
        for i in range(replications):
            x = np.random.uniform(-3, 3, n)
            f_evaluated = f(x)
            # f_evaluated /= np.std(f_evaluated)
            # f_evaluated *= 10.0
            y = f_evaluated + np.random.normal(0, sigma, n)#*10.0
            data[(f.__name__, sigma)][i] = (x, y)
            if f.__name__ == 'f4':
                data[('f5', sigma)][i] = (-x, y)

    with open(data_name, "wb") as f:
        pickle.dump(data, f)
else:
    with open(data_name, "rb") as f:
        data = pickle.load(f)
