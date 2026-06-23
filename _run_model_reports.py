from importlib import import_module
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import pickle
from multiprocessing import Pool

from _utils_reports import html_report
from _utils_functions import functions

import warnings
warnings.filterwarnings('ignore')

# from _run_model_keys._run_model_keys import a, b, order, spline_degree, n_internal_knots, report_model_keys, data_path, directory_path, reports_path, idatas_path, builder, reports_workers, replace_reports
def init_worker(k, d):
    global keys, data
    keys = k
    data = d

def worker(task):
    global keys, data
    print(task, ' report')
    html_report_args = {'reports_path': keys.reports_path, 'idatas_path': keys.idatas_path, 'functions': functions,
                        'a': keys.a, 'b': keys.b,
                        'order': keys.order, 'spline_degree': keys.spline_degree,
                        'n_internal_knots': keys.n_internal_knots, 'data': data}
    
    html_report(task, **html_report_args, replace_existing=keys.replace_reports)
######################
def main(keys_path):
    keys_loc = keys_path.replace('/', '.').removesuffix('.py')
    keys = import_module(keys_loc)
    
    # print("Running reports")
    with open(keys.data_path, "rb") as data_file:
            data = pickle.load(data_file)

    # Create tasks list
    tasks = list(keys.report_model_keys)
    np.random.seed(42)
    np.random.shuffle(tasks)

    # Number of workers
    N_WORKERS = keys.reports_workers

    with Pool(N_WORKERS, initializer=init_worker, initargs=(keys, data)) as p:
        results = p.map(worker, tasks)

if __name__ == "__main__":
    main(sys.argv[1])

