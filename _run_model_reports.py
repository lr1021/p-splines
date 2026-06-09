import numpy as np
import pickle
from multiprocessing import Pool

from _utils_reports import html_report
from _run_model_keys import a, b, order, spline_degree, n_internal_knots, functions, model_keys, directory_path, data_path, reports_path, idatas_path, builder

import warnings
warnings.filterwarnings('ignore')

######################
with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)

def init_worker():
    """Initialize worker process"""
    #implementation_var_names = {'standard': [['w']],
             #'centring+dropping': [['w']],
             #'conditioning': [['theta']],
             #'spectral': [['w'], ['w0', 'wp']],
             #'svd': [['w']]}
    
    

def worker(task):
    print(task, ' report')

    html_report_args = {'reports_path': reports_path, 'idatas_path': idatas_path, 'functions': functions,
                        'a': a, 'b': b,
                        'order': order, 'spline_degree': spline_degree,
                        'n_internal_knots': n_internal_knots, 'data': data}
    
    html_report(task, **html_report_args, replace_existing=True)

# Create tasks list
tasks = list(model_keys)
np.random.seed(42)
np.random.shuffle(tasks)

# Number of workers
N_WORKERS = 50
def main():
    with Pool(N_WORKERS, initializer=init_worker) as p:
        results = p.map(worker, tasks)

if __name__ == "__main__":
    main()

