# N-way version of _run_model_distortion_ms_mult.py: compares any number of implementations
# (e.g. ('svd', 'ortho_conditioning', 'opr_exact')) in one figure, one column per spline.
# Uses _utils_models_mult_opr (adds 'opr_exact'). Labels / colours / order come from
# _utils/_utils_comp_style.py. No LOO.
#
# Usage (from the repo root): python _run/_run_model_distortion_ms_mult_comp.py <keys.py> <_distortion_f_sigma_r/config.py>
# The implementations are read from `comparison_list` in the distortion config if present,
# otherwise from the keys file. Entries may be tuples of any length.
from importlib import import_module
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import arviz as az
import pickle
from multiprocessing import Pool

from _utils._utils_functions import functions
from _utils._utils_models_mult_opr import builder_dict
from _utils._utils_reports_mult import centred_f_plot_post, centred_function_plot, write_idata_path, extract_w_post
from _utils._utils_comp_style import (sort_impls, plot_impl, legend_handles, get_comparison_list,
                                      DATA_COLOR, TRUE_FUNCTION_COLOR)

import warnings
warnings.filterwarnings('ignore')

######################

def init_worker(data, keys_dict, builder_dict):
    global _worker_data, _worker_keys_dict, _worker_builder_dict
    _worker_data = data
    _worker_keys_dict = keys_dict
    _worker_builder_dict = builder_dict


def worker(args):
    impls, tau_vals, f, sigma, r, quality, penalised, builder = args
    impls = sort_impls(impls)
    n_tau = len(tau_vals)
    show_true = not (("dengue" in f) or ("cherry" in f) or ("weighted" in f))
    show_data = builder not in ['nb_ortho_diag', 'p_ortho_diag', 'popnb_ortho_diag']

    curve_dict_path = os.path.join("data", f"curve_dict_{_worker_keys_dict['n_internal_knots']}_01_mult.pkl")
    try:
        with open(curve_dict_path, "rb") as pl:
            curve_dict = pickle.load(pl)
    except Exception as e:
        print(f"Error loading curve_dict from {curve_dict_path}: {e}")
        curve_dict = None

    model_data = _worker_data[(f, sigma)][r]
    x_data = model_data[0]
    y_data = model_data[1]
    num_s = x_data.shape[1] if len(x_data.shape) > 1 else 1

    fig, axes = plt.subplots(n_tau, num_s, figsize=(6 * num_s, 4 * n_tau), squeeze=False)
    axes = axes.flatten()

    for tau_idx, tau in enumerate(tau_vals):
        for implementation in impls:
            model_key = (f, sigma, implementation, penalised, r, builder)
            idata_path = write_idata_path(model_key, _worker_keys_dict['idatas_path'])
            idata_path = re.sub(r'tau[-+eE.\d]+', f'tau{tau}', idata_path)
            if not os.path.exists(idata_path):
                raise FileNotFoundError(f"{idata_path} not found")
            idata = az.from_netcdf(idata_path)

            # model only rebuilt for the design matrices (independent of tau)
            model, X_list, X_plot_list, var_names = _worker_builder_dict[builder](
                x_data=x_data, y_data=y_data, a=_worker_keys_dict['a'], b=_worker_keys_dict['b'],
                spline_degree=_worker_keys_dict['spline_degree'], n_internal_knots=_worker_keys_dict['n_internal_knots'],
                implementation=implementation, penalised=penalised, order=_worker_keys_dict['order'])

            for s in range(num_s):
                X = X_list[s]
                X_plot = X_plot_list[s]
                x_plot = np.linspace(np.min(x_data[:, s]), np.max(x_data[:, s]), X_plot.shape[0])
                _, w_post_flat = extract_w_post(idata, s)
                f_plot_mean, f_plot_median, f_plot_025, f_plot_975 = centred_f_plot_post(X, X_plot, w_post_flat, builder)
                plot_impl(axes[tau_idx*num_s + s], x_plot, f_plot_mean, f_plot_025, f_plot_975, implementation)
            idata.close()

        for s in range(num_s):
            ax = axes[tau_idx*num_s + s]
            x_plot = np.linspace(np.min(x_data[:, s]), np.max(x_data[:, s]), X_plot_list[s].shape[0])

            if show_true:
                f_plot, x_pl = centred_function_plot(f, x_plot, functions, r, x_data, s)
                ax.plot(x_pl, f_plot, label='True Function', color=TRUE_FUNCTION_COLOR, linestyle='--', linewidth=0.5)

            y_scatter = y_data[0] if isinstance(y_data, tuple) else y_data
            if show_data:
                # partial residuals for spline s: remove the mean of curve s and the other curves
                scatter_y_s = y_scatter.copy()
                m = f.rsplit('_')
                function_name = m[0]
                n_val = int(m[2])
                scale = int(m[3])
                curves = curve_dict[(eval(function_name), n_val, r)].copy()*scale
                scatter_y_s -= np.mean(curves[:, s])
                for s_o in range(num_s):
                    if s_o != s:
                        scatter_y_s -= curves[:, s_o]
                ax.scatter(x_data[:, s], scatter_y_s, marker='x', label='Data', alpha=1.0, s=20, color=DATA_COLOR)
            else:
                ax.set_ylim(ymin=0)
                ax.grid(axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
                knots = (np.max(x_data) - np.min(x_data))/(_worker_keys_dict['n_internal_knots'] + 1) * np.arange(0, _worker_keys_dict['n_internal_knots'] + 2) + np.min(x_data)
                if _worker_keys_dict['show_knots']:
                    for knot in knots:
                        ax.axvline(x=knot, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                if _worker_keys_dict['show_data_density']:
                    ax_hist = ax.twinx()
                    ax_hist.hist(x_data, bins=knots, density=True, alpha=0.07, color=DATA_COLOR, label='Data Histogram')
                    ax_hist.hist(x_data, bins=knots, density=True, histtype='step', color='black', linewidth=1.0)
                    ax_hist.set_ylabel("x Data Density")
                    ax_hist.set_zorder(0)
                    ax.patch.set_alpha(0)

            ax.set_title(f"$\\tau = 10^{{{-tau}}}, s={s}$", fontsize=14)
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            # y-axis: ticks every 0.5, covering the automatic y-limits
            ymin, ymax = ax.get_ylim()
            ymin = np.floor(ymin * 2) / 2
            ymax = np.ceil(ymax * 2) / 2
            ax.set_ylim(ymin, ymax)
            step = 0.5
            while (ymax - ymin) / step + 1 >= 25:
                step *= 2
            ax.yaxis.set_major_locator(MultipleLocator(step))

    fig.tight_layout()
    # legend below the axes; bbox_inches="tight" grows the canvas to fit it, whatever the number of implementations
    fig.legend(handles=legend_handles(impls, show_true, show_data), loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=14, frameon=False, handlelength=2, handletextpad=0.6, columnspacing=1.2)

    base_path = os.path.join(
        _worker_keys_dict["distortion_report_path"],
        f"{quality}_{f}_{sigma}_{r}_{penalised}"
        f"({builder})({'_v_'.join(sorted(impls))})"
    )
    fig.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

######################

def main(keys_path, distortion_report_key_path):
    keys = import_module(keys_path.replace('/', '.').removesuffix('.py'))
    d_keys = import_module(distortion_report_key_path.replace('/', '.').removesuffix('.py'))

    with open(keys.data_path, "rb") as data_file:
        data = pickle.load(data_file)

    tasks = []
    for impls in get_comparison_list(keys, d_keys):
        for f, sigma in d_keys.f_sigma_distortion.keys():
            for r, quality in d_keys.f_sigma_distortion[(f, sigma)]:
                tasks.append((impls, d_keys.tau_vals, f, sigma, r, quality, keys.penalised, keys.builder))

    keys_dict = {
        'a': keys.a,
        'b': keys.b,
        'order': keys.order,
        'spline_degree': keys.spline_degree,
        'n_internal_knots': keys.n_internal_knots,
        'idatas_path': keys.idatas_path,
        'distortion_report_path': d_keys.distortion_report_path,
        'show_knots': getattr(keys, 'show_knots', False),
        'show_data_density': getattr(keys, 'show_data_density', False),
    }

    os.makedirs(d_keys.distortion_report_path, exist_ok=True)

    with Pool(keys.replications_report_workers, initializer=init_worker, initargs=(data, keys_dict, builder_dict)) as p:
        p.map(worker, tasks)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
