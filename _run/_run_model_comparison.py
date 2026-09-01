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
from matplotlib.scale import FuncScale
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
        if _worker_keys_dict['loo']:
            if ("log_likelihood" not in idata.groups()):
                print(f"Computing log_likelihood for {model_key}")
                with model:
                    idata = pm.compute_log_likelihood(idata)
                # optionally save it back
                idata.load()
                tmp_path = idata_path + ".tmp"
                idata.to_netcdf(tmp_path)
                os.replace(tmp_path, idata_path)
            idata_path = write_idata_path(model_key, _worker_keys_dict['idatas_path'])
            
            loo_path = idata_path.replace(".nc", "_loo.pkl")
            if os.path.exists(loo_path):
                with open(loo_path, "rb") as l_path:
                    loo = pickle.load(l_path)
            else:
                loo = az.loo(idata, pointwise=True)
                with open(loo_path, "wb") as l_path:
                    pickle.dump(loo, l_path)
            result[f"elpd_loo_i{i+1}"] = loo.elpd_loo
            result[f"elpd_loo_pointwise_i{i+1}"] = loo.loo_i
            result[f"elpd_loo_se_i{i+1}"] = loo.se
            result[f"p_loo_i{i+1}"] = loo.p_loo
            result[f'n_data_points'] = loo.n_data_points
            result[f"loo_bad_k_i{i+1}"] = np.sum(loo.pareto_k > loo.good_k)

            idata.close()


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

            comparison_report_path = os.path.join(comparison_report_folder, f"comparison_report({i1}_{i2})({f}_{sigma}_{penalised}_{builder}).html")
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
                    'idatas_path': keys.idatas_path,
                    'loo': (keys.loo if hasattr(keys, 'loo') else False)
                }

                with Pool(N_WORKERS, initializer=init_worker, initargs=(data, keys_dict, builder_dict)) as p:
                    results = p.map(worker, tasks)
                
                comparison_df = pd.DataFrame(results)
                if hasattr(keys, 'spread') and keys.spread:
                    comparison_report_path = os.path.join(comparison_report_folder, f"comparison_report({i1}_{i2})({f}_{sigma}_{penalised}_{builder})_{keys.spread_start}spread{keys.spread_step}.html")
                    top_comparison_df = comparison_df[keys.spread_start::keys.spread_step]
                else:
                    comparison_df.sort_values(['f_plot_mean_diff'], inplace=True, ascending=False)
                    top_n = keys.top_n if hasattr(keys, 'top_n') else 10
                    top_comparison_df = comparison_df.head(top_n)

                html_parts = [f"<html><head><title>Comparison Report: {i1} vs {i2}</title>",
                                "<style>",
                                "body { font-family: Arial; font-size: 10px; line-height: 1.2; margin: 8px; text-align:center; }",
                                "h1, h2 { margin: 4px 0 8px 0; font-weight: normal; }",
                                "table { border-collapse: collapse; font-size: 15px; margin: 0 auto 12px auto; width: auto; }",
                                "table th, table td { border: 1px solid #aaa; padding: 4px 6px; text-align: center; }",
                                "img { max-width: 80%; margin: 8px auto; display: block; }",
                                "</style></head><body>",
                                f"<h1>Comparison Report: {i1} vs {i2}</h1>",
                                f"<h1>{f}, {sigma}, {penalised}, {builder}</h1>"]
                
                
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
                    f_plot_range95CI_i1 = row['f_plot_range95CI_i1']
                    f_plot_range95CI_i2 = row['f_plot_range95CI_i2']

                    fig, ax = plt.subplots(figsize=(12, 4))
                    c1 = 'red'
                    c2 = 'green'
                    ax.plot(x_plot, f_plot_mean_i1, label=f'Posterior Mean {i1}', color=c1)
                    ax.plot(x_plot, f_plot_mean_i2, label=f'Posterior Mean {i2}', color=c2)
                    ax.fill_between(x_plot, f_plot_025_i1, f_plot_975_i1, color=c1, alpha=0.08, label=f'95% Credible Interval {i1}', linewidth=0.5)
                    ax.fill_between(x_plot, f_plot_025_i2, f_plot_975_i2, color=c2, alpha=0.08, label=f'95% Credible Interval {i2}', linewidth=0.5)

                    if not (("dengue" in f) or ("cherry" in f) or ("weighted" in f) or ("df" in f)):
                        f_plot, x_pl = function_plot(f, x_plot, functions, r, x_data)
                        ax.plot(x_pl, f_plot, label='True Function', color='black', linestyle='--', linewidth=0.5)

                    if isinstance(y_data, tuple):
                        y_data = y_data[0]
                    if not (builder in ['nb_ortho_diag', 'p_ortho_diag', 'popnb_ortho_diag']):
                        ax.scatter(x_data, y_data, marker='x', label='Data', alpha=1.0, s=20, color='blue')
                    else:
                        ax.set_ylim(ymin=0)
                        knots = (np.max(x_data) - np.min(x_data))/(keys.n_internal_knots + 1) * np.arange(0, keys.n_internal_knots + 2) + np.min(x_data)
                        if hasattr(keys, 'show_knots') and keys.show_knots:
                            for knot in knots:
                                ax.axvline(x=knot, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                        if hasattr(keys, 'show_data_density') and keys.show_data_density:
                            ax_hist = ax.twinx()
                            ax_hist.hist(
                                x_data,
                                bins=knots,
                                density=True,
                                alpha=0.07,
                                color='blue',
                                label='Data Histogram'
                            )
                            ax_hist.hist(
                                x_data,
                                bins=knots,
                                density=True,
                                histtype='step',
                                color='black',
                                linewidth=1.0
                            )
                            ax_hist.set_ylabel("x Data Density")
                            ax_hist.set_zorder(0)          # optional: draw behind main axis
                            ax.patch.set_alpha(0)          # optional: keep histogram visible
                        elif hasattr(keys, 'show_pointwise_loo_diff') and keys.show_pointwise_loo_diff:
                            elpd_loo_diff = row['elpd_loo_pointwise_i1'] - row['elpd_loo_pointwise_i2']
                            pointwise_loo_threshold = 0
                            ax_loo = ax.twinx()
                            ax_loo.scatter(x_data[elpd_loo_diff < -pointwise_loo_threshold], y_data[elpd_loo_diff < -pointwise_loo_threshold], marker='o', label=f'Pointwise LOO: {i2} better', alpha=0.7, s=20, color='green')
                            ax_loo.scatter(x_data[elpd_loo_diff > pointwise_loo_threshold], y_data[elpd_loo_diff > pointwise_loo_threshold], marker='o', label=f'Pointwise LOO: {i1} better', alpha=0.7, s=20, color='red')
                            ax_loo.set_ylabel("elpd loo diff")
                            ax_loo.set_zorder(0)          # optional: draw behind main axis
                            ax.patch.set_alpha(0)          # optional: keep histogram visible
                
                    ax.set_title(f"Replication: {r}")
                    ax.legend(bbox_to_anchor=(1.05, 1))
                    img_base64 = fig_to_base64(fig)
                    html_parts.append(f"<h2>Replication: {r}</h2><img src='data:image/png;base64,{img_base64}' />")
                    html_parts.append(f"<h2>95CI size: {i1}: {f_plot_range95CI_i1:.2f}</h2>")
                    html_parts.append(f"<h2>95CI size: {i2}: {f_plot_range95CI_i2:.2f}</h2>")

                    if hasattr(keys, 'loo') and keys.loo:
                        elpd_loo_i1 = row['elpd_loo_i1']
                        elpd_loo_se_i1 = row['elpd_loo_se_i1']
                        p_loo_i1 = row['p_loo_i1']
                        loo_bad_k_i1 = row['loo_bad_k_i1']

                        elpd_loo_i2 = row['elpd_loo_i2']
                        elpd_loo_se_i2 = row['elpd_loo_se_i2']
                        p_loo_i2 = row['p_loo_i2']
                        loo_bad_k_i2 = row['loo_bad_k_i2']

                        elpd_loo_diff = row['elpd_loo_pointwise_i1'] - row['elpd_loo_pointwise_i2']
                        elpd_loo_diff_mean = np.mean(elpd_loo_diff)
                        elpd_loo_diff_mean_se = np.std(elpd_loo_diff) / np.sqrt(len(elpd_loo_diff))

                        n_data_points = row['n_data_points']
                        html_parts.append(f"<h2>Number of data points: {n_data_points}, elpd_loo_pointwise_diff: {elpd_loo_diff_mean:.2f} ± {elpd_loo_diff_mean_se:.2f}</h2>")

                        html_parts.append("""
                            <table border="1" style="border-collapse: collapse;">
                                <tr>
                                    <th>Model</th>
                                    <th>elpd_loo</th>
                                    <th>SE</th>
                                    <th>p_loo</th>
                                    <th>Bad k</th>
                                </tr>
                            """)

                        html_parts.append(
                            f"""
                            <tr>
                                <td>{i1}</td>
                                <td>{elpd_loo_i1:.2f}</td>
                                <td>{elpd_loo_se_i1:.2f}</td>
                                <td>{p_loo_i1:.2f}</td>
                                <td>{loo_bad_k_i1.values}</td>
                            </tr>
                            """
                        )

                        html_parts.append(
                            f"""
                            <tr>
                                <td>{i2}</td>
                                <td>{elpd_loo_i2:.2f}</td>
                                <td>{elpd_loo_se_i2:.2f}</td>
                                <td>{p_loo_i2:.2f}</td>
                                <td>{loo_bad_k_i2.values}</td>
                            </tr>
                            """
                        )

                        html_parts.append("</table>")

                        # html_parts.append(f"<h2>LOO: {i1}: elpd_loo: {elpd_loo_i1:.2f}, se: {elpd_loo_se_i1:.2f}, p_loo: {p_loo_i1:.2f}, bad k: {loo_bad_k_i1.values}</h2>")
                        # html_parts.append(f"<h2>LOO: {i2}: elpd_loo: {elpd_loo_i2:.2f}, se: {elpd_loo_se_i2:.2f}, p_loo: {p_loo_i2:.2f}, bad k: {loo_bad_k_i2.values}</h2>")
                html_parts.append("</body></html>")
                with open(comparison_report_path, "w") as o:
                    o.write("\n".join(html_parts))

if __name__ == "__main__":
    main(sys.argv[1])