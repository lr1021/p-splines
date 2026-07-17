from importlib import import_module
import os
import re
import sys
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from _run._run_model_reports import worker
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arviz as az
import pymc as pm
import pandas as pd
import pickle
from multiprocessing import Pool
import time
import xarray as xr
from weasyprint import HTML

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
    i1, i2, tau_vals, f, sigma, r, quality, penalised, builder = args

    distortion_report_path = os.path.join(_worker_keys_dict['distortion_report_path'], f"{quality}_{f}_{sigma}_{r}_{penalised}({builder})({i1}_v_{i2}).html")
    html_parts = [f"<html><head><title> Distortion Report: {i1} vs {i2} ({f}, {sigma}, {r}, {quality})</title>",
                                "<style>",
                                "body { font-family: Arial; font-size: 10px; line-height: 1.2; margin: 8px; text-align:center; }",
                                "h1 { margin: 4px 0 8px 0; font-weight: normal; font-size: 80px; }",
                                "h2 { margin: 2px 0 4px 0; font-weight: normal; font-size: 60px; }",
                                "table { border-collapse: collapse; font-size: 15px; margin: 0 auto 12px auto; width: auto; }",
                                "table th, table td { border: 1px solid #aaa; padding: 4px 6px; text-align: center; }",
                                "img { max-width: 80%; margin: 8px auto; display: block; }",
                                "</style></head><body>",
                                f"<h1>{i1} vs {i2}</h1>",
                                f"<h1>{f}, {sigma}, {r}, {quality}</h1>",
                                f"<h1>Replication {r}, {quality}</h1>",
                                f"<h1>{tau_vals}, {penalised}, {builder}</h1>"]
    for tau in tau_vals:
        result = {}
        for i, implementation in enumerate([i1, i2]):
            model_key = (f, sigma, implementation, penalised, r, builder)
            idata_path = write_idata_path(model_key, _worker_keys_dict['idatas_path'])
            idata_path = re.sub(r'tau[-+eE.\d]+', f'tau{tau}', idata_path)
            if os.path.exists(idata_path):
                idata = az.from_netcdf(idata_path)
            else:
                raise FileNotFoundError(f"{idata_path} not found")
            model_data = _worker_data[(f, sigma)][r]
            x_data = model_data[0]
            y_data = model_data[1]
            result['replication'] = r
            result['quality'] = quality
            result['x_data'] = x_data
            result['y_data'] = y_data
                        
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
            result[f"f_plot_range95CI_i{i+1}"] = np.mean(f_plot_975 - f_plot_025)
        result['f_plot_mean_diff'] = np.sqrt(np.sum((result['f_plot_mean_i1'] - result['f_plot_mean_i2']) ** 2))
        result['f_plot_median_diff'] = np.sqrt(np.sum((result['f_plot_median_i1'] - result['f_plot_median_i2']) ** 2))
        result['x_plot'] = x_plot

        

        fig, ax = plt.subplots(figsize=(12, 4))
        c1 = 'red'
        c2 = 'green'
        ax.plot(x_plot, result['f_plot_mean_i1'], label=f'Posterior Mean {i1}', color=c1)
        ax.plot(x_plot, result['f_plot_mean_i2'], label=f'Posterior Mean {i2}', color=c2)
        ax.fill_between(x_plot, result['f_plot_025_i1'], result['f_plot_975_i1'], color=c1, alpha=0.08, label=f'95% Credible Interval {i1}', linewidth=0.5)
        ax.fill_between(x_plot, result['f_plot_025_i2'], result['f_plot_975_i2'], color=c2, alpha=0.08, label=f'95% Credible Interval {i2}', linewidth=0.5)

        

        if not (("dengue" in f) or ("cherry" in f)):
            f_plot, x_pl = function_plot(f, x_plot, functions, r, x_data)
            ax.plot(x_pl, f_plot, label='True Function', color='black', linestyle='--', linewidth=0.5)

        

        if not (builder in ['nb_ortho_diag', 'p_ortho_diag']):
            ax.scatter(x_data, y_data, marker='x', label='Data', alpha=1.0, s=20, color='blue')

       
                    
        ax.set_title(f"tau 1e-{tau}", fontsize=20)
        img_base64 = fig_to_base64(fig)
        html_parts.append(f"<img src='data:image/png;base64,{img_base64}' />")
        html_parts.append(f"<h2>95CI size: {i1}: {result['f_plot_range95CI_i1']:.2f}</h2>")
        html_parts.append(f"<h2>95CI size: {i2}: {result['f_plot_range95CI_i2']:.2f}</h2>")
    
    
    legend_fig, legend_ax = plt.subplots(figsize=(4, 2))
    legend_ax.axis("off")

    
    handles = [
        Line2D([0], [0], color='red', label=f'Posterior Mean {i1}'),
        Line2D([0], [0], color='green', label=f'Posterior Mean {i2}'),
        Line2D([0], [0], color='red', alpha=0.3, lw=8, label=f'95% CI {i1}'),
        Line2D([0], [0], color='green', alpha=0.3, lw=8, label=f'95% CI {i2}'),
        Line2D([0], [0], color='black', linestyle='--', label='True Function'),
        Line2D([0], [0], marker='x', color='blue', linestyle='', label='Data'),
    ]
    
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=1,
        fontsize=20,     # larger text
    handlelength=3,  # longer line samples
    handletextpad=1,
    labelspacing=1.0,
    frameon=False,
    )

    legend_base64 = fig_to_base64(legend_fig)

    
    html_parts.append(
    f"<img src='data:image/png;base64,{legend_base64}' />"
    )

    html_parts.append("</body></html>")
    print(distortion_report_path)
    
    with open(distortion_report_path, "w") as o:
        o.write("\n".join(html_parts))

######################
def main(keys_path, distortion_report_key_path):
    keys_loc = keys_path.replace('/', '.').removesuffix('.py')
    keys = import_module(keys_loc)

    d_keys_loc = distortion_report_key_path.replace('/', '.').removesuffix('.py')
    d_keys = import_module(d_keys_loc)
    
    with open(keys.data_path, "rb") as data_file:
        data = pickle.load(data_file)
    try:
        comparison_list = keys.comparison_list
    except AttributeError:
        comparison_list = []

    tasks = []
    for comparison_pair in keys.comparison_list:
        i1, i2 = sorted(list(comparison_pair))
        for f, sigma in d_keys.f_sigma_distortion.keys():
            for r, quality in d_keys.f_sigma_distortion[(f, sigma)]:
                tasks.append((i1, i2, d_keys.tau_vals, f, sigma, r, quality, keys.penalised, keys.builder))

    N_WORKERS = keys.replications_report_workers

    keys_dict = {
        'a': keys.a,
        'b': keys.b,
        'order': keys.order,
        'spline_degree': keys.spline_degree,
        'n_internal_knots': keys.n_internal_knots,
        'idatas_path': keys.idatas_path,
        'distortion_report_path': d_keys.distortion_report_path,
    }

    if not os.path.exists(d_keys.distortion_report_path):
        os.makedirs(d_keys.distortion_report_path)

    with Pool(N_WORKERS, initializer=init_worker, initargs=(data, keys_dict, builder_dict)) as p:
        results = p.map(worker, tasks)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])