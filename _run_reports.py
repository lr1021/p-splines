import numpy as np
import pickle
from multiprocessing import Pool

from _utils_reports import html_report
from _run_model_keys import model_keys, data_path, reports_path, idatas_path

import warnings
warnings.filterwarnings('ignore')


######################
def init_worker():
    """Initialize worker process"""
    global data_path, reports_path, idatas_path, functions, implementation_var_names, a, b, order, spline_degree, n_internal_knots, data, html_report_args

    def f1(x):
        return x/1.758
    def f2(x):
        return x**2/2.75 - 1.5
    def f3(x):
        return np.sin(x)/0.72

    functions = {f.__name__: f for f in [f1, f2, f3]}

    implementation_var_names = {'standard': [['w']],
             'centring+dropping': [['w']],
             'conditioning': [['theta']],
             'spectral': [['w'], ['w0', 'wp']],
             'svd': [['w']]}
    
    a = 1
    b = 0.005
    order = 2
    spline_degree = 3
    n_internal_knots = 20
    with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)
    
    html_report_args = {'reports_path': reports_path, 'idatas_path': idatas_path, 'functions': functions, 'implementation_var_names': implementation_var_names, 'a': a, 'b': b, 'order': order, 'spline_degree': spline_degree, 'n_internal_knots': n_internal_knots, 'data': data}

def worker(task):
    print(task, ' report')
    html_report(task, **html_report_args, replace_existing=True)

# Create tasks list
tasks = list(model_keys)
np.random.seed(42)
np.random.shuffle(tasks)

# Number of workers
N_WORKERS = 50
if __name__ == "__main__":
    with Pool(N_WORKERS, initializer=init_worker) as p:
        results = p.map(worker, tasks)

