# Copy of _run_model_idatas.py that takes tau from the keys file (tau_exp / tau_val), see _utils/_utils_tau.py
from importlib import import_module
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import signal
import itertools

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import pandas as pd
import pickle
from multiprocessing import Pool, Lock
import time
import xarray as xr

from _utils._utils_models import builder_dict, stratified_shuffle
from _utils._utils_reports import extract_sampling_time, write_idata_path, write_idata_path, extract_w_post

from _utils._utils_tau import builder_tau_kwargs, check_idata_tau, nudge_tau

# from _run_model_keys._run_model_keys import a, b, order, spline_degree, n_internal_knots, model_keys, data_path, directory_path, reports_path, idatas_path, builder, idatas_workers, run_idatas, run_reports, run_replications_report, replace_idatas, n_tune, n_draws, n_chains, n_cores, target_accept, max_treedepth, report_model_keys

import warnings
warnings.filterwarnings('ignore')

######################

def init_worker(csv_l, net_cdf_l, k, d):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    """Initialize worker process"""
    global csv_lock
    global net_cdf_lock
    global keys, data
    keys = k
    data = d
    csv_lock = csv_l
    net_cdf_lock = net_cdf_l

def build_and_sample(task, x_data, y_data, tau_kwargs):
    """Warm-up sample (to exclude compilation from the timing) + real sample. Raises on failure."""
    (f, sigma, implementation, penalised, replication, builder) = task
    model, _, _, _ = builder_dict[builder](x_data, y_data, keys.a, keys.b,
                    keys.spline_degree, keys.n_internal_knots,
                    implementation, penalised, keys.order, **tau_kwargs)
    # first sample
    with model:
        pm.sample(tune = 1,
                  draws = 1,
                  chains = keys.n_chains,
                  random_seed=42,
                  cores = keys.n_cores,
                  nuts_sampler="nutpie",
                  store_divergences=True,
                  discard_tuned_samples=True,
                  progressbar=False,
                  quiet=True,
                  target_accept=keys.target_accept,
                  max_treedepth=keys.max_treedepth)
    gc.collect()

    model, _, _, _ = builder_dict[builder](x_data, y_data, keys.a, keys.b,
                    keys.spline_degree, keys.n_internal_knots,
                    implementation, penalised, keys.order, **tau_kwargs)
    with model:
        s0 = time.time()
        idata = pm.sample(tune = keys.n_tune,
                          draws = keys.n_draws,
                          chains = keys.n_chains,
                          random_seed=42,
                          cores = keys.n_cores,
                          nuts_sampler="nutpie",
                          store_divergences=True,
                          discard_tuned_samples=True,
                          progressbar=not keys.quiet,
                          quiet=keys.quiet,
                          target_accept=keys.target_accept,
                          max_treedepth=keys.max_treedepth)
        s1 = time.time()
    return idata, model, s0, s1

def worker(task):
    global first_sample
    print(task, ' idata')
    (f, sigma, implementation, penalised, replication, builder) = task
    idata_path = os.path.join(keys.idatas_path, f"idata_{f}_{sigma}_{implementation}_{penalised}_{replication}({builder}).nc")
    if os.path.exists(idata_path) and not keys.replace_idatas:
        print(task, ' idata exists')
        check_idata_tau(idata_path, getattr(keys, 'tau_val', None))
        # idata = az.from_netcdf(idata_path)
    else:
        model_data = data[(f, sigma)][replication]
        x_data = model_data[0]
        y_data = model_data[1]

        tau_kwargs = builder_tau_kwargs(keys)
        try:
            idata, model, s0, s1 = build_and_sample(task, x_data, y_data, tau_kwargs)
        except Exception as e:
            if 'tau_val' not in tau_kwargs:
                print(f"Error in sampling for task {task}: {e}")
                return
            # retry once with a tiny relative nudge of tau (pytensor/numba compile bug at some exact values)
            nudged = {'tau_val': nudge_tau(tau_kwargs['tau_val'])}
            print(f"WARNING: task {task} failed at tau_val={tau_kwargs['tau_val']!r} ({type(e).__name__}: {str(e)[:120]}); "
                  f"retrying with tau_val={nudged['tau_val']!r}")
            try:
                idata, model, s0, s1 = build_and_sample(task, x_data, y_data, nudged)
            except Exception as e2:
                print(f"Error in sampling for task {task} (also with nudged tau): {e2}")
                return
        with net_cdf_lock:
            idata.to_netcdf(idata_path)
        del idata, model
        gc.collect()    

        row = pd.DataFrame([{
            "f": f,
            "sigma": sigma,
            "implementation": implementation,
            "penalised": penalised,
            "replication": replication,
            "runtime_seconds": s1 - s0,
        }])

        # append or create csv
        timings_path = os.path.join(keys.idatas_path, '../timings.csv')
        with csv_lock:
            if os.path.exists(timings_path):
                    row.to_csv(timings_path, mode="a", header=False, index=False)
            else:
                row.to_csv(timings_path, index=False)

def main(keys_path):
    keys_loc = keys_path.replace('/', '.').removesuffix('.py')
    keys = import_module(keys_loc)
    
    with open(keys.data_path, "rb") as data_file:
            data = pickle.load(data_file)

    # Create tasks list
    np.random.seed(42)  # for reproducibility
    tasks = list(keys.model_keys)
    tasks_df = pd.DataFrame(tasks, columns=['f', 'sigma', 'implementation', 'penalised', 'replication', 'builder'])
    tasks_df = stratified_shuffle(tasks_df, ['f', 'sigma', 'implementation', 'penalised'], ['replication'], random_state=42)
    tasks = list(tasks_df.itertuples(index=False, name=None))

    # Number of workers (adjust based on your server)
    # Each model uses n_chains, so N_WORKERS * n_chains = total cores used
    N_WORKERS = keys.idatas_workers
    print(f"tau_val = {getattr(keys, 'tau_val', None)} (None = builder default)")

    csv_l = Lock()
    net_cdf_l = Lock()

    if keys.run_idatas:
        # print("Running idatas")
        try:
            with Pool(N_WORKERS, initializer=init_worker, initargs=(csv_l, net_cdf_l, keys, data), maxtasksperchild=1) as p:
                # chunksize=1 renews worker after each task, prevents sampler cache problems,
                # (DO NOT REMOVE maxtasksperchild=1: nutpie reuses compiled models within a process,
                #  so a later task can silently sample an earlier task's model/data)
                # to not count structure building as part of sampling I run
                # a first void sample for each task
                p.map(worker, tasks, chunksize=1)
        except KeyboardInterrupt:
            print("Keyboard interrupt received, terminating workers.")
            p.terminate()
            p.join()
    else:
        print("Skipping idatas")
    
    timings_df = pd.read_csv(os.path.join(keys.idatas_path, '../timings.csv'))
    timings_df.sort_values('runtime_seconds', ascending=True, inplace=True)
    timings_df.to_csv(os.path.join(keys.idatas_path, '../timings_sorted.csv'), index=True)

if __name__ == "__main__":
    main(sys.argv[1])