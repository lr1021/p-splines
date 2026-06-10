import sys

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
from _utils_reports import extract_sampling_time, write_idata_path, write_idata_path, extract_w_post
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
            try:
                idata = pm.sample(tune = n_tune,
                                draws = n_draws,
                                chains = n_chains,
                                random_seed=42,
                                cores = n_cores,
                                nuts_sampler="nutpie",
                                store_divergences=True,
                                discard_tuned_samples=True,
                                progressbar=True)
            except Exception as e:
                print(f"Error in sampling for task {task}: {e}")
                return
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

#############################################################################################
# Reports
import _run_model_reports
_run_model_reports.main()

############################################################################################# 
sys.exit()
# Summary
model_keys_df = pd.DataFrame(model_keys, columns=['f', 'sigma', 'implementation', 'penalised', 'replication', 'builder'])
# summary data across replications
unique_keys = model_keys_df.drop(columns='replication').drop_duplicates()
unique_keys.sort_values(['f', 'sigma', 'penalised', 'builder', 'implementation'], inplace=True)
unique_keys.reset_index(drop=True, inplace=True)
unique_keys['parts'] = [[] for _ in range(len(unique_keys))]

unique_keys['general summary'] = [[] for _ in range(len(unique_keys))]
unique_keys['replication summary'] = [[] for _ in range(len(unique_keys))]
unique_keys['f summary'] = [[] for _ in range(len(unique_keys))]

for i, row in unique_keys.iterrows():
    f, sigma, implementation, penalised, builder, _, _, _, _ = row
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
                    "ess_bulk_mean/s": [],
                    "ess_tail_min/s": [],
                    "ess_tail_p5/s": [],
                    "ess_tail_mean/s": [],
                    "r_hat_max": [],
                    "r_hat>1.01_mean": []}
    
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
        summary_az = az.summary(idata, var_names=var_names)
        variables = summary_az.index.tolist()
        summary_array = summary_az.to_numpy().copy()
        summary_array[:, [6, 7]] /= sampling_time
        summaries.append(summary_array)

        # f posterior summaries
        w_post, _ = extract_w_post(idata)
        #f_plot_post = np.einsum('ij,cdj->cdi', X_plot, w_post)
        f_plot_post = w_post @ X_plot.T
        f_idata = az.convert_to_inference_data(xr.DataArray(f_plot_post,
                                                            dims=["chain", "draw", "x"]))
        f_rhat = az.rhat(f_idata)["x"].values
        f_ess_bulk = az.ess(f_idata, method="bulk")["x"].values / sampling_time
        f_ess_tail = az.ess(f_idata, method="tail")["x"].values / sampling_time
        # f metrics
        f_metrics['ess_bulk_min/s'].append(np.min(f_ess_bulk))
        f_metrics['ess_bulk_p5/s'].append(np.percentile(f_ess_bulk, 5))
        f_metrics['ess_bulk_mean/s'].append(np.mean(f_ess_bulk))
        f_metrics['ess_tail_min/s'].append(np.min(f_ess_tail))
        f_metrics['ess_tail_p5/s'].append(np.percentile(f_ess_tail, 5))
        f_metrics['ess_tail_mean/s'].append(np.mean(f_ess_tail))
        f_metrics['r_hat_max'].append(np.max(f_rhat))
        f_metrics['r_hat>1.01_mean'].append(np.mean(f_rhat > 1.01))
    summaries = np.stack(summaries, axis=0)

    ###
    general_summary = {
        "n_replications": [len(key_replications)],
        "mean_sampling_time": [np.mean(sampling_times)],
        "median_sampling_time": [np.median(sampling_times)],
        "portion_divergent": [np.mean(np.array(divergences)>0)],
        "total_divergences": [np.sum(divergences)]
    }
    unique_keys.at[i, 'general summary'].append(pd.DataFrame(general_summary))
    unique_keys.at[i, 'parts'].append('<h2>General Summary</h2>')
    unique_keys.at[i, 'parts'].append(pd.DataFrame(general_summary).to_html(index=False))
    ###

    ###
    replication_summary = pd.DataFrame()
    replication_summary['variable'] = variables

    replication_summary['mean_ess_bulk/s'] = summaries[:, :, 6].mean(axis=0)
    #replication_summary['min_ess_bulk/s'] = summaries[:, :, 6].min(axis=0)
    replication_summary['mean_ess_tail/s'] = summaries[:, :, 7].mean(axis=0)
    #replication_summary['min_ess_tail/s'] = summaries[:, :, 7].min(axis=0)

    replication_summary['mean_r_hat'] = summaries[:, :, 8].mean(axis=0)
    replication_summary['r_hat>1.01'] = (summaries[:, :, 8] > 1.01).mean(axis=0)

    replication_summary['mean_sd'] = summaries[:, :, 1].mean(axis=0)
    replication_summary['mean_hdi_range'] = (summaries[:, :, 3] - summaries[:, :, 2]).mean(axis=0)

    replication_summary['mean_mcse_mean'] = summaries[:, :, 4].mean(axis=0)
    replication_summary['mean_mcse_sd'] = summaries[:, :, 5].mean(axis=0)

    mean_w_row = replication_summary.loc[
        [v.startswith('w') for v in replication_summary['variable']],
        ['mean_ess_bulk/s', 'mean_ess_tail/s', 'mean_r_hat', 'r_hat>1.01',
         'mean_sd', 'mean_hdi_range', 'mean_mcse_mean', 'mean_mcse_sd']].mean()
    #mean_w_row['r_hat>1.01'] = replication_summary.loc[
        #[v.startswith('w') for v in replication_summary['variable']],
        #'r_hat>1.01'].mean()
    mean_w_row['variable'] = 'w (summary)'
    mean_w_df = mean_w_row.to_frame().T
    replication_summary = pd.concat([mean_w_df[
    ['variable'] + [c for c in mean_w_df.columns if c != 'variable']],
    replication_summary], ignore_index=True)

    unique_keys.at[i, 'replication summary'].append(replication_summary)
    unique_keys.at[i, 'parts'].append('<h2>Replication Summary</h2>')
    unique_keys.at[i, 'parts'].append(replication_summary.to_html(index=False))
    ###
    f_summary = {k+"_mean": [np.mean(v)] for k, v in f_metrics.items()}
    f_summary['r_hat_max'] = [np.max(f_metrics['r_hat_max'])]
    f_summary_df = pd.DataFrame(f_summary)
    unique_keys.at[i, 'f summary'].append(f_summary_df)
    unique_keys.at[i, 'parts'].append('<h2>f Summary</h2>')
    unique_keys.at[i, 'parts'].append(f_summary_df.to_html(index=False))
html_parts = []
title = f"Replication Summary Report:"
html_parts = [f"<html><head><title>{title}</title>",
                    "<style>",
                    "body { font-family: Arial; font-size: 10px; line-height: 1.2; margin: 8px; text-align:center; }",
                    "h1, h2 { margin: 4px 0 8px 0; font-weight: normal; }",
                    "table { border-collapse: collapse; font-size: 15px; margin: 0 auto 12px auto; width: 80%; }",
                    "table th, table td { border: 1px solid #aaa; padding: 4px 6px; text-align: center; }",
                    "img { max-width: 80%; margin: 8px auto; display: block; }",
                    "</style></head><body>",
                    f"<h1>{title}</h1>"]


unique_keys_tasks = model_keys_df.drop(columns=['penalised', 'implementation', 'replication']).drop_duplicates()
for i, task in unique_keys_tasks.iterrows():
    f, sigma, builder = task
    task_summary_df = {'Penalised': [],
                       'Implementation': [],

                       'w (summary) mean_ess_bulk/s_mean': [],
                       'w (summary) mean_ess_tail/s_mean': [],
                       'w (summary) r_hat>1.01': [],

                       'f (summary) mean_ess_bulk/s_mean': [],
                       'f (summary) mean_ess_tail/s_mean': [],
                       'f (summary) r_hat>1.01': [],
                       'mean_sampling_time': [],
                       'n_replications': [],
                       'portion_divergent': []}
    for i, row in unique_keys.iterrows():
        if row['f'] == f and row['sigma'] == sigma and row['builder'] == builder:
            task_summary_df['Penalised'].append(row['penalised'])
            task_summary_df['Implementation'].append(row['implementation'])

            replication_summary = row['replication summary'][0]
            f_summary = row['f summary'][0]
            general_summary = row['general summary'][0]

            task_summary_df['w (summary) mean_ess_bulk/s_mean'].append(
                replication_summary.loc[replication_summary['variable'] == 'w (summary)', 'mean_ess_bulk/s'].values[0])
            task_summary_df['w (summary) mean_ess_tail/s_mean'].append(
                replication_summary.loc[replication_summary['variable'] == 'w (summary)', 'mean_ess_tail/s'].values[0])
            task_summary_df['w (summary) r_hat>1.01'].append(
                replication_summary.loc[replication_summary['variable'] == 'w (summary)', 'r_hat>1.01'].values[0])

            task_summary_df['f (summary) mean_ess_bulk/s_mean'].append(f_summary['ess_bulk_mean/s_mean'].values[0])
            task_summary_df['f (summary) mean_ess_tail/s_mean'].append(f_summary['ess_tail_mean/s_mean'].values[0])
            task_summary_df['f (summary) r_hat>1.01'].append(f_summary['r_hat>1.01_mean_mean'].values[0])

            task_summary_df['mean_sampling_time'].append(general_summary['mean_sampling_time'].values[0])
            task_summary_df['n_replications'].append(general_summary['n_replications'].values[0])
            task_summary_df['portion_divergent'].append(general_summary['portion_divergent'].values[0])
    html_parts.append(f"<h2>f={f}, sigma={sigma}, builder={builder}</h2>")
    task_summary_df = pd.DataFrame(task_summary_df)
    task_summary_df.sort_values(['Penalised', 'w (summary) mean_ess_bulk/s_mean'], ascending=[True, False], inplace=True)
    html_parts.append(pd.DataFrame(task_summary_df).to_html(index=False))
html_parts.append("<hr>")
for i, row in unique_keys.iterrows():
    f, sigma, implementation, penalised, builder = row[['f', 'sigma', 'implementation', 'penalised', 'builder']]
    html_parts.append(f"<h2>Model: f={f}, sigma={sigma}, implementation={implementation}, penalised={penalised}, builder={builder}</h2>")
    for part in row['parts']:
        html_parts.append(part)
    html_parts.append("<hr>")
html_parts.append("</body></html>")
with open(os.path.join(reports_path, "../replication_summary_report.html"), "w") as f:
    f.write("\n".join(html_parts))