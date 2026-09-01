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
from matplotlib.ticker import MultipleLocator


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
    full_report = False
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

    n_tau = len(tau_vals)

    # One row of plots
    fig, axes = plt.subplots(
        n_tau,
        1,
        figsize=(8, 4 * n_tau),
        squeeze=False,
    )

    axes = axes.flatten()

    for tau_idx, tau in enumerate(tau_vals):
        ax = axes[tau_idx]
        result = {}
        for i, implementation in enumerate([i1, i2]):
            model_key = (f, sigma, implementation, penalised, r, builder)
            idata_path = write_idata_path(model_key, _worker_keys_dict['idatas_path'])
            idata_path = re.sub(r'tau[-+eE.\d]+', f'tau{tau}', idata_path)
            if os.path.exists(idata_path):
                try:
                    idata = az.from_netcdf(idata_path)
                except Exception as e:
                    print(f"Error occurred while loading {idata_path}: {e}")
                    raise
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

            _, w_post_flat = extract_w_post(idata)
            if "beta_0_post" in idata.posterior.data_vars:
                beta_0_var = "beta_0_post"
            else:
                beta_0_var = "beta_0"
            b_post = idata.posterior[beta_0_var].values.flatten()
            f_plot_mean, f_plot_median, f_plot_025, f_plot_975 = f_plot_post(X_plot, w_post_flat, b_post, builder)
            f_data_mean, f_data_median, f_data_025, f_data_975 = f_plot_post(X, w_post_flat, b_post, builder)

            result[f"f_plot_mean_i{i+1}"] = f_plot_mean
            result[f"f_plot_median_i{i+1}"] = f_plot_median
            result[f"f_plot_025_i{i+1}"] = f_plot_025
            result[f"f_plot_975_i{i+1}"] = f_plot_975

            result[f"f_data_mean_i{i+1}"] = f_data_mean
            result[f"f_data_median_i{i+1}"] = f_data_median
            result[f"f_data_025_i{i+1}"] = f_data_025
            result[f"f_data_975_i{i+1}"] = f_data_975

            result[f"f_plot_range95CI_i{i+1}"] = np.mean(f_plot_975 - f_plot_025)
        result['f_plot_mean_diff'] = np.sqrt(np.sum((result['f_plot_mean_i1'] - result['f_plot_mean_i2']) ** 2))
        result['f_plot_median_diff'] = np.sqrt(np.sum((result['f_plot_median_i1'] - result['f_plot_median_i2']) ** 2))
        result['x_plot'] = x_plot

        okabe_ito_colors = np.array(["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
          "#0072B2","#D55E00", "#CC79A7", "#999999"])
        if i1 == 'svd':
            c1 = 'green' # okabe_ito_colors[3]  # Green
            c2 = 'red' # okabe_ito_colors[6]  # Orange
        elif i2 == 'svd':
            c1 = 'red' #okabe_ito_colors[6]  # Orange
            c2 = 'green' #okabe_ito_colors[3]  # Green
        else:
            c1 = 'red' #okabe_ito_colors[6]  # Orange
            c2 = 'green' #okabe_ito_colors[3]  # Green
        ax.plot(x_plot, result['f_plot_mean_i1'], label=f'Posterior Mean {i1}', color=c1)
        ax.plot(x_plot, result['f_plot_mean_i2'], label=f'Posterior Mean {i2}', color=c2)
        ax.fill_between(x_plot, result['f_plot_025_i1'], result['f_plot_975_i1'], color=c1, alpha=0.08, label=f'95% Credible Interval {i1}', linewidth=0.5)
        ax.fill_between(x_plot, result['f_plot_025_i2'], result['f_plot_975_i2'], color=c2, alpha=0.08, label=f'95% Credible Interval {i2}', linewidth=0.5)

        

        if not (("dengue" in f) or ("cherry" in f) or ("weighted" in f)):
            try:
                f_plot, x_pl = function_plot(f, x_plot, functions, r, x_data)
                ax.plot(x_pl, f_plot, label='True Function', color='black', linestyle='--', linewidth=0.5)
            except Exception as e:
                print(f"Error occurred while plotting true function for {f}: {e}")
                raise

        
        if isinstance(y_data, tuple):
            y_data = y_data[0]
        if not (builder in ['nb_ortho_diag', 'p_ortho_diag', 'popnb_ortho_diag']):
            ax.scatter(x_data, y_data, marker='x', label='Data', alpha=1.0, s=20, color='blue')
        else:
            ax.set_ylim(ymin=0)
            ax.grid(axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
            knots = (np.max(x_data) - np.min(x_data))/(_worker_keys_dict['n_internal_knots'] + 1) * np.arange(0, _worker_keys_dict['n_internal_knots'] + 2) + np.min(x_data)
            if _worker_keys_dict['show_knots']:
                for knot in knots:
                    ax.axvline(x=knot, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            if _worker_keys_dict['show_data_density']:
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
            elif _worker_keys_dict['show_pointwise_loo_diff']:
                elpd_loo_diff = result['elpd_loo_pointwise_i1'] - result['elpd_loo_pointwise_i2']
                f_plot_mean_i1, f_plot_mean_i2 = result['f_plot_mean_i1'], result['f_plot_mean_i2']
                f_scale = np.maximum(np.abs(np.max(f_plot_mean_i1)-np.min(f_plot_mean_i1)), np.abs(np.max(f_plot_mean_i2)-np.min(f_plot_mean_i2)))

                f_data_mean_i1, f_data_mean_i2 = result['f_data_mean_i1'], result['f_data_mean_i2']
                f_data_mean_diff = np.abs(result['f_data_mean_i1'] - result['f_data_mean_i2'])
                f_data_025_diff = np.abs(result['f_data_025_i1'] - result['f_data_025_i2'])
                f_data_975_diff = np.abs(result['f_data_975_i1'] - result['f_data_975_i2'])
                sig_p = 0.0
                f_data_sig_diff = ((f_data_mean_diff > sig_p * f_scale) + (f_data_025_diff > sig_p * f_scale) + (f_data_975_diff > sig_p * f_scale)) > 0
                pointwise_loo_threshold = 0.005
                elpd_loo_diff_i1_better = elpd_loo_diff > pointwise_loo_threshold
                elpd_loo_diff_i2_better = (elpd_loo_diff < -pointwise_loo_threshold)
                
                ax_loo = ax.twinx()
                ax_loo.scatter(x_data[(elpd_loo_diff_i2_better)&f_data_sig_diff], elpd_loo_diff[(elpd_loo_diff_i2_better)&f_data_sig_diff], marker='o', label=f'Pointwise LOO: {i2} better', alpha=0.7, s=10, color='green')
                ax_loo.scatter(x_data[elpd_loo_diff_i1_better & f_data_sig_diff], elpd_loo_diff[elpd_loo_diff_i1_better & f_data_sig_diff], marker='o', label=f'Pointwise LOO: {i1} better', alpha=0.7, s=10, color='red')
                ax_loo.set_ylabel("elpd loo diff")
                ax_loo.set_zorder(0)          # optional: draw behind main axis
                ax.patch.set_alpha(0)         # optional: keep histogram visible

                # elpd_loo_diff mean over distortion regions
                elpd_loo_diff_mean_distortion = np.mean(elpd_loo_diff[f_data_sig_diff])
                    
       
        ax.set_title(
            f"$\\tau = 10^{{{-tau}}}$",
            fontsize=14,
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        # y-axis: ticks every 0.5, covering the automatic y-limits
        ymin, ymax = ax.get_ylim()
        ymin = np.floor(ymin * 2) / 2
        ymax = np.ceil(ymax * 2) / 2

        ax.set_ylim(ymin, ymax)
        step = 0.5
        # Double step until fewer than 25 ticks
        while (ymax - ymin) / step + 1 >= 25:
            step *= 2
        ax.yaxis.set_major_locator(MultipleLocator(step))

        # ax.set_title(f"tau 1e{int(-tau)}", fontsize=20)
    handles = [
        Line2D(
            [0], [0],
            color=c1,
            label=f"Posterior Mean {i1}",
        ),
        Line2D(
            [0], [0],
            color=c2,
            label=f"Posterior Mean {i2}",
        ),
        Line2D(
            [0], [0],
            color=c1,
            alpha=0.3,
            lw=8,
            label=f"95% CI {i1}",
        ),
        Line2D(
            [0], [0],
            color=c2,
            alpha=0.3,
            lw=8,
            label=f"95% CI {i2}",
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            label="True Function",
        ),
        Line2D(
            [0], [0],
            marker="x",
            color=okabe_ito_colors[2],
            linestyle="",
            label="Data",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        fontsize=14,
        frameon=False,
        handlelength=2,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    # Leave room for legend underneath
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    # legend_fig, legend_ax = plt.subplots(figsize=(4, 2))
    # legend_ax.axis("off")

    
    # handles = [
    #     Line2D([0], [0], color='red', label=f'Posterior Mean {i1}'),
    #     Line2D([0], [0], color='green', label=f'Posterior Mean {i2}'),
    #     Line2D([0], [0], color='red', alpha=0.3, lw=8, label=f'95% CI {i1}'),
    #     Line2D([0], [0], color='green', alpha=0.3, lw=8, label=f'95% CI {i2}'),
    #     Line2D([0], [0], color='black', linestyle='--', label='True Function'),
    #     Line2D([0], [0], marker='x', color='blue', linestyle='', label='Data'),
    # ]
    
    # legend_ax.legend(
    #     handles=handles,
    #     loc="center",
    #     ncol=1,
    #     fontsize=20,     # larger text
    # handlelength=3,  # longer line samples
    # handletextpad=1,
    # labelspacing=1.0,
    # frameon=False,
    # )

    # legend_base64 = fig_to_base64(legend_fig)

    
    # html_parts.append(
    # f"<img src='data:image/png;base64,{legend_base64}' />"
    # )

    # html_parts.append("</body></html>")
    # print(distortion_report_path)
    
    # with open(distortion_report_path, "w") as o:
    #     o.write("\n".join(html_parts))
    base_path = os.path.join(
        _worker_keys_dict["distortion_report_path"],
        f"{quality}_{f}_{sigma}_{r}_{penalised}"
        f"({builder})({i1}_v_{i2})"
    )

    fig.savefig(
        base_path + ".png",
        dpi=300,
        bbox_inches="tight",
    )

    # If you also want PDF:
    # fig.savefig(
    #     base_path + ".pdf",
    #     bbox_inches="tight",
    # )

    plt.close(fig)

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
        'show_knots': keys.show_knots if hasattr(keys, 'show_knots') else False,
        'show_data_density': keys.show_data_density if hasattr(keys, 'show_data_density') else False,
        'loo': keys.loo if hasattr(keys, 'loo') else False,
        'show_pointwise_loo_diff': keys.show_pointwise_loo_diff if hasattr(keys, 'show_pointwise_loo_diff') else False
    }

    if not os.path.exists(d_keys.distortion_report_path):
        os.makedirs(d_keys.distortion_report_path)

    with Pool(N_WORKERS, initializer=init_worker, initargs=(data, keys_dict, builder_dict)) as p:
        results = p.map(worker, tasks)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])