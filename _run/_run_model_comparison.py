from importlib import import_module
import os
import sys
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from _run._run_model_reports import worker
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import pandas as pd
import pickle
from multiprocessing import Pool
import time
import xarray as xr

from _utils._utils_functions import functions
from _utils._utils_models import builder_dict
from _utils._utils_reports import extract_sampling_time, f_plot_post, fig_to_base64, function_plot, write_idata_path, write_idata_path, extract_w_post

import warnings
warnings.filterwarnings('ignore')

# from _run_model_keys._run_model_keys import a, b, order, spline_degree, n_internal_knots, model_keys, data_path, directory_path, reports_path, idatas_path, builder, replications_report_workers

######################
def init_worker(data, keys_dict, builder_dict):
    global _worker_data, _worker_keys_dict, _worker_builder_dict

    _worker_data = data
    _worker_keys_dict = keys_dict
    _worker_builder_dict = builder_dict

def worker(args):
    f, sigma, r, i1, i2, penalised, builder = args

    model_data = _worker_data[(f, sigma)][r]
    x_data = model_data[0]
    y_data = model_data[1]

    result = {
        "replication": r,
        "x_data": x_data,
        "y_data": y_data
    }
    

    for i, implementation in enumerate([i1, i2]):
        model_key = (f, sigma, implementation, penalised, r, builder)
        
        idata_path = write_idata_path(model_key, _worker_keys_dict['idatas_path'])
        if os.path.exists(idata_path):
            idata = az.from_netcdf(idata_path)
        else:
            raise FileNotFoundError(f"{idata_path} not found")
                        
        model, X, X_plot, var_names = _worker_builder_dict[builder](
             x_data=x_data, y_data=y_data, a=_worker_keys_dict['a'], b=_worker_keys_dict['b'],
             spline_degree=_worker_keys_dict['spline_degree'], n_internal_knots=_worker_keys_dict['n_internal_knots'],
             implementation=implementation, penalised=penalised, order=_worker_keys_dict['order'])
        x_plot = np.linspace(np.min(x_data), np.max(x_data), X_plot.shape[0])

        _, w_post_flat = extract_w_post(idata)
        if "beta_0_post" in idata.posterior.data_vars:
            beta_0_var = "beta_0_post"
        else:
            beta_0_var = "beta_0"
        b_post = idata.posterior[beta_0_var].values.flatten()
        f_plot_mean, f_plot_median, f_plot_025, f_plot_975 = f_plot_post(X_plot, w_post_flat, b_post, builder)

        result[f"f_plot_mean_i{i+1}"] = f_plot_mean
        result[f"f_plot_median_i{i+1}"] = f_plot_median
        result[f"f_plot_025_i{i+1}"] = f_plot_025
        result[f"f_plot_975_i{i+1}"] = f_plot_975

    result['f_plot_mean_diff'] = np.sqrt(np.sum((result['f_plot_mean_i1'] - result['f_plot_mean_i2']) ** 2))
    result['f_plot_median_diff'] = np.sqrt(np.sum((result['f_plot_median_i1'] - result['f_plot_median_i2']) ** 2))
    result['x_plot'] = x_plot

    return result

######################
def main(keys_path):
    keys_loc = keys_path.replace('/', '.').removesuffix('.py')
    keys = import_module(keys_loc)
    
    with open(keys.data_path, "rb") as data_file:
        data = pickle.load(data_file)
    try:
        comparison_list = keys.comparison_list
    except AttributeError:
        comparison_list = []
    
    comparison_report_folder = os.path.join(keys.reports_path, "../")
    for comparison_pair in comparison_list:
        print(f"Generating comparison report for {comparison_pair}...")
        i1, i2 = sorted(list(comparison_pair))

        model_keys_df = pd.DataFrame(keys.model_keys, columns=['f', 'sigma', 'implementation', 'penalised', 'replication', 'builder'])
        # print(model_keys_df['implementation'].unique())
        model_keys_df = model_keys_df[model_keys_df['implementation'].isin([i1, i2])]

        model_keys_df_f_sigma_penalised_builder = model_keys_df.drop(columns=['implementation', 'replication']).drop_duplicates()
        model_keys_df_f_sigma_penalised_builder.sort_values(['f', 'sigma', 'penalised', 'builder'], inplace=True)
        # print(model_keys_df_f_sigma_penalised_builder)
        for i, row in model_keys_df_f_sigma_penalised_builder.iterrows():
            f, sigma, penalised, builder = row

            comparison_report_path = os.path.join(comparison_report_folder, f"comparison_report({i1}_{i2})({f}_{sigma}_{penalised}).html")
            if os.path.exists(comparison_report_path) and not keys.replace_comparison_report:
                print(f"Comparison report for {i1} vs {i2} already exists. Skipping...")
                continue
            else:   
                task_df = model_keys_df[
                    (model_keys_df['f'] == f) &
                    (model_keys_df['sigma'] == sigma) &
                    (model_keys_df['penalised'] == penalised) &
                    (model_keys_df['builder'] == builder)
                ].copy()

                task_df1 = task_df[task_df['implementation'] == i1]
                task_df2 = task_df[task_df['implementation'] == i2]
                shared_replications = set(task_df1['replication']).intersection(set(task_df2['replication']))
                if len(shared_replications) == 0:
                    print(f"No shared replications for {f}, {sigma}, {penalised}, {builder}. Skipping...")
                    continue
                task_df = task_df[task_df['replication'].isin(shared_replications)]


                ###
                tasks = [(f, sigma, r, i1, i2, penalised, builder) for r in shared_replications]
                N_WORKERS = keys.replications_report_workers

                keys_dict = {
                    'a': keys.a,
                    'b': keys.b,
                    'order': keys.order,
                    'spline_degree': keys.spline_degree,
                    'n_internal_knots': keys.n_internal_knots,
                    'idatas_path': keys.idatas_path
                }

                with Pool(N_WORKERS, initializer=init_worker, initargs=(data, keys_dict, builder_dict)) as p:
                    results = p.map(worker, tasks)
                
                comparison_df = pd.DataFrame(results)
                comparison_df.sort_values(['f_plot_mean_diff'], inplace=True, ascending=False)
                top_n = 10
                top_comparison_df = comparison_df.head(top_n)

                html_parts = [f"<html><head><title>Comparison Report: {i1} vs {i2}</title>",
                                "<style>",
                                "body { font-family: Arial; font-size: 10px; line-height: 1.2; margin: 8px; text-align:center; }",
                                "h1, h2 { margin: 4px 0 8px 0; font-weight: normal; }",
                                "table { border-collapse: collapse; font-size: 15px; margin: 0 auto 12px auto; width: auto; }",
                                "table th, table td { border: 1px solid #aaa; padding: 4px 6px; text-align: center; }",
                                "img { max-width: 80%; margin: 8px auto; display: block; }",
                                "</style></head><body>",
                                f"<h1>Comparison Report: {i1} vs {i2}</h1>"]
                
                
                for _, row in top_comparison_df.iterrows():
                    r = row['replication']
                    x_data = row['x_data']
                    y_data = row['y_data']
                    x_plot = row['x_plot']
                    f_plot_mean_i1 = row['f_plot_mean_i1']
                    f_plot_mean_i2 = row['f_plot_mean_i2']
                    f_plot_median_i1 = row['f_plot_median_i1']
                    f_plot_median_i2 = row['f_plot_median_i2']
                    f_plot_025_i1 = row['f_plot_025_i1']
                    f_plot_025_i2 = row['f_plot_025_i2']
                    f_plot_975_i1 = row['f_plot_975_i1']
                    f_plot_975_i2 = row['f_plot_975_i2']

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(x_data, y_data, marker='o', label='Data', alpha=0.5, s=1, color='blue')
                    c1 = 'red'
                    c2 = 'green'
                    ax.plot(x_plot, f_plot_mean_i1, label=f'Posterior Mean {i1}', color=c1)
                    ax.plot(x_plot, f_plot_mean_i2, label=f'Posterior Mean {i2}', color=c2)
                    ax.fill_between(x_plot, f_plot_025_i1, f_plot_975_i1, color=c1, alpha=0.05, label=f'95% Credible Interval {i1}', linewidth=0.5)
                    ax.fill_between(x_plot, f_plot_025_i2, f_plot_975_i2, color=c2, alpha=0.05, label=f'95% Credible Interval {i2}', linewidth=0.5)

                    if not ("dengue" in f):
                        f_plot, x_pl = function_plot(f, x_plot, functions, r, x_data)
                        ax.plot(x_pl, f_plot, label='True Function', color='black', linestyle='--', linewidth=0.5)
                    
                    ax.set_title(f"Replication: {r}")
                    ax.legend(bbox_to_anchor=(1.05, 1))
                    img_base64 = fig_to_base64(fig)
                    html_parts.append(f"<h2>Replication: {r}</h2><img src='data:image/png;base64,{img_base64}' />")
                html_parts.append("</body></html>")
                with open(comparison_report_path, "w") as o:
                    o.write("\n".join(html_parts))

if __name__ == "__main__":
    main(sys.argv[1])