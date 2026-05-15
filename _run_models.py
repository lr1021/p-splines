import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
import io
import base64
import pickle
import multiprocessing
from multiprocessing import Pool

from _utils_models import build_model
from _utils_reports import html_report
from _run_model_keys import model_keys

import warnings
warnings.filterwarnings('ignore')


for model_key in model_keys:
    print(model_key)
    (f, sigma, implementation, penalised, replication) = model_key
    
    
    if os.path.exists(f"idata/idata_{model_key[0]}_{model_key[1]}_{model_key[2]}_{model_key[3]}_{model_key[4]}.nc"):
        idata = az.from_netcdf(f"idata/idata_{model_key[0]}_{model_key[1]}_{model_key[2]}_{model_key[3]}_{model_key[4]}.nc")
    else:
        model_data = data[(f, sigma)][replication]
        x_data = model_data[0]
        y_data = model_data[1]

        model, X, X_plot = build_model_MvN(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order)
        model_dict[(f, sigma, implementation, penalised, replication)] = model
        with model: 
            idata = pm.sample(tune = n_tune,
                                    draws = n_draws,
                                    chains = n_chains,
                                    random_seed=42,
                                    cores = n_cores,
                                    discard_tuned_samples=True,
                                    progressbar=True)
            idata.to_netcdf(f"idata/idata_{model_key[0]}_{model_key[1]}_{model_key[2]}_{model_key[3]}_{model_key[4]}.nc")
        idata_dict[model_key] = idata
    
    row = pd.DataFrame([{
        "key1": model_key[0],
        "key2": model_key[1],
        "key3": model_key[2],
        "key4": model_key[3],
        "key5": model_key[4],
        "runtime_seconds": idata.sample_stats.attrs["sampling_time"],
    }])

    # append or create csv
    if os.path.exists('idata/timings.csv'):
        row.to_csv('idata/timings.csv', mode="a", header=False, index=False)
    else:
        row.to_csv('idata/timings.csv', index=False)

    

######################
def init_worker():
    """Initialize worker process"""
    global directory_path, functions, implementation_var_names, a, b, order, spline_degree, n_internal_knots, data, html_report_args
    directory_path = "../p-splines/"

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
    with open(os.path.join(directory_path, "data.pkl"), "rb") as data_file:
        data = pickle.load(data_file)
    
    html_report_args = {'directory_path': directory_path, 'functions': functions, 'implementation_var_names': implementation_var_names, 'a': a, 'b': b, 'order': order, 'spline_degree': spline_degree, 'n_internal_knots': n_internal_knots, 'data': data}

def worker(task):
    print(task, ' report')
    html_report(task, **html_report_args, replace_existing=True)

# Create tasks list
tasks = list(model_keys)
np.random.seed(42)  # for reproducibility
np.random.shuffle(tasks)

# Number of workers (adjust based on your server)
# Each model uses n_chains, so N_WORKERS * n_chains = total cores used
N_WORKERS = 50
if __name__ == "__main__":
    with Pool(N_WORKERS, initializer=init_worker) as p:
        results = p.map(worker, tasks)

