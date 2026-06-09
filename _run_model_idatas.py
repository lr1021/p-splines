import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
import pymc as pm
import pandas as pd
import pickle
from multiprocessing import Pool
import time
import xarray as xr

from _utils_models import builder_dict
from _utils_reports import html_report, extract_sampling_time, write_idata_path, write_idata_path, extract_w_post
from _run_model_keys import a, b, order, spline_degree, n_internal_knots, functions, model_keys, directory_path, data_path, reports_path, idatas_path, builder

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
        model, X, X_plot, var_names = builder_dict[builder](x_data, y_data, a, b,
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
unique_keys.sort_values(['f', 'sigma', 'penalised', 'builder', 'implementation'], inplace=True)
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
    f_metrics = {   "ess_bulk_min/s": [],
                    "ess_bulk_p5/s": [],
                    "ess_bulk_median/s": [],
                    "ess_tail_min/s": [],
                    "ess_tail_p5/s": [],
                    "ess_tail_median/s": [],
                    "r_hat_max": [],
                    "r_hat_95": []}
    
    for replication in key_replications:
        model_key = (f, sigma, implementation, penalised, replication, builder)
        model_data = data[(f, sigma)][replication]
        x_data = model_data[0]
        y_data = model_data[1]
        model, X, X_plot, var_names = builder_dict[builder](x_data, y_data, a, b,
                            spline_degree, n_internal_knots,
                            implementation, penalised, order)
        idata = az.from_netcdf(write_idata_path(model_key, idatas_path))

        # sampling times
        sampling_time = extract_sampling_time(model_key, idatas_path)
        sampling_times.append(sampling_time)
        # divergences
        divergences.append(int(idata.sample_stats["diverging"].sum()))

        # posterior summaries
        # 'mean', 'sd', 'hdi_3%', 'hdi_97%', 'mcse_mean', 'mcse_sd', 'ess_bulk', 'ess_tail', 'r_hat'
        summary_array = az.summary(idata, var_names=var_names).to_numpy()
        summary_array[:, [6, 7]] /= sampling_time
        summaries.append(summary_array)

        # f posterior summaries
        w_post = extract_w_post(idata)
        f_plot_post = X_plot @ w_post.T
        f_idata = az.convert_to_inference_data(xr.DataArray(f_plot_post,
                                                            dims=["chain", "draw", "x"]))
        f_rhat = az.rhat(f_idata)["x"].values
        f_ess_bulk = az.ess(f_idata, method="bulk")["x"].values / sampling_time
        f_ess_tail = az.ess(f_idata, method="tail")["x"].values / sampling_time
        # f metrics
        f_metrics['ess_bulk_min/s'].append(np.min(f_ess_bulk))
        f_metrics['ess_bulk_p5/s'].append(np.percentile(f_ess_bulk, 5))
        f_metrics['ess_bulk_median/s'].append(np.median(f_ess_bulk))
        f_metrics['ess_tail_min/s'].append(np.min(f_ess_tail))
        f_metrics['ess_tail_p5/s'].append(np.percentile(f_ess_tail, 5))
        f_metrics['ess_tail_median/s'].append(np.median(f_ess_tail))
        f_metrics['r_hat_max'].append(np.max(f_rhat))
        f_metrics['r_hat_95'].append(np.percentile(f_rhat, 95))
    summaries = np.stack(summaries, axis=0)

    ###
    n_replications = len(key_replications)
    mean_sampling_time = np.mean(sampling_times)
    median_sampling_time = np.median(sampling_times)
    portion_divergent = np.mean(np.array(divergences)>0)
    total_divergences = np.sum(divergences)
    ###

    ###
    replication_summary = pd.DataFrame()
    replication_summary['mean_sd'] = summaries[:, :, 1].mean(axis=0)
    replication_summary['mean_hdi_range'] = (summaries[:, :, 3] - summaries[:, :, 2]).mean(axis=0)

    replication_summary['mean_mcse_mean'] = summaries[:, :, 4].mean(axis=0)
    replication_summary['mean_mcse_sd'] = summaries[:, :, 5].mean(axis=0)

    replication_summary['mean_ess_bulk/s'] = summaries[:, :, 6].mean(axis=0)
    replication_summary['min_ess_bulk/s'] = summaries[:, :, 6].min(axis=0)
    replication_summary['mean_ess_tail/s'] = summaries[:, :, 7].mean(axis=0)
    replication_summary['min_ess_tail/s'] = summaries[:, :, 7].min(axis=0)

    replication_summary['mean_r_hat'] = summaries[:, :, 8].mean(axis=0)
    replication_summary['max_r_hat'] = summaries[:, :, 8].max(axis=0)
    ###
    f_summary = {k+"_mean": [np.mean(v)] for k, v in f_metrics.items()}
    f_summary_df = pd.DataFrame(f_summary)