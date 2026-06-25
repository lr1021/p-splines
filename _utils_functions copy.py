import numpy as np

def f1(x):
    return 2 * (x/6) + 6.0
def f2(x):
    return 1 * (x**2/9-1/2) - 1.5
def f3(x):
    return (-3) * (np.sin(x)/2) + 0.0

from _utils_spline import eval_spline_basis_equispaced_numeric

def f4(x):
    spline_degree = 3
    n_internal_knots = 5
    eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x), np.max(x), n_internal_knots, x)
    B = eval_B['B']
    c = B[:, -1]
    c = c/(np.max(c) - np.min(c))*5.0
    return c

def f5(x):
    spline_degree = 3
    n_internal_knots = 5
    eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x), np.max(x), n_internal_knots, x)
    B = eval_B['B']
    c = B[:, 0]
    c = c/(np.max(c) - np.min(c))*5.0
    return c

def f4b(x):
    spline_degree = 3
    n_internal_knots = 5
    eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x), np.max(x), n_internal_knots, x)
    B = eval_B['B']
    c = B[:, -1]
    c = c/(np.max(c) - np.min(c))*5.0
    return c*200.0

def f5b(x):
    spline_degree = 3
    n_internal_knots = 5
    eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x), np.max(x), n_internal_knots, x)
    B = eval_B['B']
    c = B[:, 0]
    c = c/(np.max(c) - np.min(c))*5.0
    return c*200.0

f_list = [f1, f2, f3, f4, f5, f4b, f5b]
functions = {f.__name__: f for f in f_list}