import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
import pymc as pm
import pandas as pd
import pickle
from multiprocessing import Pool
import time

from _utils_models import builder_dict
from _utils_reports import html_report, extract_sampling_time, write_idata_path, write_idata_path
from _run_model_keys import functions, model_keys, directory_path, data_path, reports_path, idatas_path, builder

import warnings
warnings.filterwarnings('ignore')

n_tune = 1000
n_draws = 1000
n_chains = 4
n_cores = 4

######################
with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)

def init_worker():
    """Initialize worker process"""
    global a, b, order, spline_degree, n_internal_knots
    a = 1
    b = 0.005
    order = 2
    spline_degree = 3
    n_internal_knots = 20

def worker(task):
    print(task, ' idata')
    (f, sigma, implementation, penalised, replication, builder) = task
    idata_path = os.path.join(idatas_path, f"idata_{f}_{sigma}_{implementation}_{penalised}_{replication}({builder}).nc")
    if os.path.exists(idata_path):
        print(task, ' idata exists')
        # idata = az.from_netcdf(idata_path)
    else:
        model_data = data[(f, sigma)][replication]
        x_data = model_data[0]
        y_data = model_data[1]
        model, X, X_plot = builder_dict[builder](x_data, y_data, a, b,
                            spline_degree, n_internal_knots,
                            implementation, penalised, order)
        with model:
            s0 = time.time()
            idata = pm.sample(tune = n_tune,
                              draws = n_draws,
                              chains = n_chains,
                              random_seed=42,
                              cores = n_cores,
                              nuts_sampler="nutpie",
                              store_divergences=True,
                              discard_tuned_samples=True,
                              progressbar=True)
            s1 = time.time()
        idata.to_netcdf(idata_path)

        row = pd.DataFrame([{
            "f": f,
            "sigma": sigma,
            "implementation": implementation,
            "penalised": penalised,
            "replication": replication,
            "runtime_seconds": s1 - s0,
        }])

        # append or create csv
        timings_path = os.path.join(idatas_path, 'timings.csv')
        if os.path.exists(timings_path):
            row.to_csv(timings_path, mode="a", header=False, index=False)
        else:
            row.to_csv(timings_path, index=False)

# Create tasks list
tasks = list(model_keys)
np.random.seed(42)  # for reproducibility
np.random.shuffle(tasks)

# Number of workers (adjust based on your server)
# Each model uses n_chains, so N_WORKERS * n_chains = total cores used
N_WORKERS = 8
if __name__ == "__main__":
    with Pool(N_WORKERS, initializer=init_worker) as p:
        p.map(worker, tasks)

timings_df = pd.read_csv(os.path.join(idatas_path, 'timings.csv'))
timings_df.sort_values('runtime_seconds', ascending=True, inplace=True)
timings_df.to_csv(os.path.join(idatas_path, 'timings.csv'), index=False)

###
import _run_model_reports
_run_model_reports.main()

### Summary
model_keys_df = pd.DataFrame(model_keys, columns=['f', 'sigma', 'implementation', 'penalised', 'replication', 'builder'])
# summary data across replications
unique_keys = model_keys_df.drop(columns='replication').drop_duplicates()
for _, row in unique_keys.iterrows():
    f, sigma, implementation, penalised, builder = row
    key_replications = model_keys_df[
        (model_keys_df['f'] == f) &
        (model_keys_df['sigma'] == sigma) &
        (model_keys_df['implementation'] == implementation) &
        (model_keys_df['penalised'] == penalised) &
        (model_keys_df['builder'] == builder)
    ]['replication'].tolist()

    sampling_times = [] # take mean and median
    divergences = [] # mean and median and max
    summaries = [] # mean of summary stats (ess_bulk, ess_tail, r_hat, etc.) + min ess + max rhat
    for replication in key_replications:
        model_key = (f, sigma, implementation, penalised, replication, builder)
        sampling_times.append(extract_sampling_time(model_key, idatas_path))
        idata = az.from_netcdf(write_idata_path(model_key, idatas_path))
        divergences.append(int(idata.sample_stats["diverging"].sum()))