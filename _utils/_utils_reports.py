# from copyreg import pickle
import pickle
from re import match
import re
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
import io
import base64
import pandas as pd
import xarray as xr

from _utils._utils_models import builder_dict

import warnings
warnings.filterwarnings('ignore')

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

def write_idata_path(model_key, idatas_path):
    (f, sigma, implementation, penalised, replication, builder) = model_key
    return os.path.join(idatas_path, f"idata_{f}_{sigma}_{implementation}_{penalised}_{replication}({builder}).nc")

def write_report_path(model_key, reports_path):
    (f, sigma, implementation, penalised, replication, builder) = model_key
    return os.path.join(reports_path, f"report_{f}_{sigma}_{implementation}_{penalised}_{replication}({builder}).html")

def extract_w_post(idata):
    w_vars = sorted([v for v in idata.posterior.data_vars if v.startswith("w")], reverse=True)
    w_post = np.concatenate([idata.posterior[v].values for v in w_vars], axis=2)
    w_post_flat = w_post.reshape(-1, w_post.shape[-1])
    return w_post, w_post_flat

def extract_sampling_time(model_key, idatas_path):
    (f, sigma, implementation, penalised, replication, builder) = model_key
    timings_path = os.path.join(idatas_path, '../timings.csv')
    timings_df = pd.read_csv(timings_path)
    match = timings_df.loc[
    (timings_df["f"] == f)
    & (timings_df["sigma"] == sigma)
    & (timings_df["implementation"] == implementation)
    & (timings_df["penalised"] == penalised)
    & (timings_df["replication"] == replication),
    "runtime_seconds",
    ]
    if not match.empty:
        sampling_time = np.mean(match)
    else:
        print(f"Warning: No timing found for model_key {model_key} in timings.csv. Setting sampling_time to NaN.")
        sampling_time = np.nan
        #sampling_time = idata.sample_stats.attrs["sampling_time"]
    return sampling_time

def function_plot(f, x_plot, functions, replication, x_data):
        if f in functions:
            return functions[f](x_plot), x_plot
        elif np.any([f_name_start in f for f_name_start in ['uniform', 'exponential', 'uni_normal', 'bi_normal']]):
            if np.any([f_name_start in f for f_name_start in ['uni_normal', 'bi_normal']]) or (len(f.split('_')) == 4):
                # print(f)
                m = re.fullmatch(r'([A-Za-z_]+)_(\d+)_(\d+)_(\d+)', f)
                function_name = m.group(1)
                num_knots = int(m.group(2))
                n_val = int(m.group(3))
                scale = int(m.group(4))
                # curve_plot_dict_path = os.path.join("data", f"curve_plot_dict_{num_knots}.pkl")
                curve_plot_dict_path = os.path.join("data", f"curve_plot_dict_{num_knots}_01.pkl")
                # curve_plot_dict_path = os.path.join("data", f"curve_plot_dict_{num_knots}_01_alt.pkl")
                # curve_plot_dict_path = os.path.join("data", f"curve_plot_dict_{num_knots}_01_alt_bi.pkl")
                try:
                    with open(curve_plot_dict_path, "rb") as pl:
                        curve_plot_dict = pickle.load(pl)
                    try:
                        f_plot = curve_plot_dict[(function_name, n_val, replication)]*scale
                    except Exception as e:
                        print(f"Error occurred while accessing curve plot for {f}: {e}")
                        raise
                    x_pl = np.linspace(np.min(x_data), np.max(x_data), len(f_plot))
                except Exception as e:
                    print(f"Error occurred while loading curve plot for {f}: {e}")
                    raise
            else:
                m = re.fullmatch(
                    r'([A-Za-z_]+)_(\d+)_(\d+)_([0-9.]+)_([0-9.]+)_(\d+)',
                    f
                )
                function_name = m.group(1)
                num_knots = int(m.group(2))
                n_val = int(m.group(3))
                lim = float(m.group(4))
                sparse = float(m.group(5))
                scale = int(m.group(6))
                curve_plot_dict_path = os.path.join("data", f"curve_plot_dict_CD{num_knots}_01.pkl")
                try:
                    with open(curve_plot_dict_path, "rb") as pl:
                        curve_plot_dict = pickle.load(pl)
                    f_plot = curve_plot_dict[(function_name, n_val, lim, sparse, replication)]*scale
                    x_pl = np.linspace(np.min(x_data), np.max(x_data), len(f_plot))
                except Exception as e:
                    print(f"Error occurred while loading curve plot for {f}: {e}")
                    raise
            

        return f_plot, x_pl

def f_plot_post(X, w_post_flat, b_post, builder):
        f_p = X @ w_post_flat.T
        
        # x_plot = x_data[x_data_order]
        f_p += b_post
        if builder in ['ortho_diag_softplus']:
            f_p = np.log(1 + np.exp(f_p))  # softplus transformation for non-negative outputs
        if builder in ['popnb_ortho_diag']:
            f_p -= -10.0 #np.mean(b_post)
            f_p = np.exp(f_p)  # exponential transformation for non-negative outputs
        if builder in ['nb_ortho_diag', 'p_ortho_diag']:
            # f_p = np.exp(f_p)
            f_p = np.log(1 + np.exp(f_p))  # softplus transformation for non-negative outputs
        f_plot_mean = np.mean(f_p, axis=1)
        f_plot_median = np.median(f_p, axis=1)
        f_plot_975 = np.percentile(f_p, 97.5, axis=1)
        f_plot_025 = np.percentile(f_p, 2.5, axis=1)
        return f_plot_mean, f_plot_median, f_plot_025, f_plot_975

def html_report(model_key, reports_path, idatas_path, functions,
                a, b, order, spline_degree, n_internal_knots, data,
                replace_existing=False):
    print(model_key, ' report')
    (f, sigma, implementation, penalised, replication, builder) = model_key
    report_path = write_report_path(model_key, reports_path)
    if os.path.exists(report_path) and not replace_existing:
        print(f"Report already exists for model key: {model_key}")
        return
    idata_path = write_idata_path(model_key, idatas_path)
    if os.path.exists(idata_path):
        idata = az.from_netcdf(idata_path)
    else:
        print(f"{idata_path} not found")
        return
    
    model_data = data[(f, sigma)][replication]
    x_data = model_data[0]
    x_data_order = np.argsort(x_data)
    y_data = model_data[1]
    model, X, X_plot, var_names = builder_dict[builder](x_data=x_data, y_data=y_data, a=a, b=b, spline_degree=spline_degree, n_internal_knots=n_internal_knots, implementation=implementation, penalised=penalised, order=order)
    x_plot = np.linspace(np.min(x_data), np.max(x_data), X_plot.shape[0])
    if isinstance(y_data, tuple):
        y_data = y_data[0]

    # base and title
    title = f"Model Report: {model_key}"
    report_parts = [f"<html><head><title>{title}</title>",
                        "<style>",
                        "body { font-family: Arial; font-size: 12px; line-height: 1.2; margin: 8px; text-align:center; }",
                        "h1, h2 { margin: 4px 0 8px 0; font-weight: normal; }",
                        "table { border-collapse: collapse; font-size: 15px; margin: 0 auto 12px auto; width: 80%; }",
                        "table th, table td { border: 1px solid #aaa; padding: 4px 6px; text-align: center; }",
                        "img { max-width: 80%; margin: 8px auto; display: block; }",
                        "</style></head><body>",
                        f"<h1>{title}</h1>"]
    
    # sampling time
    sampling_time = extract_sampling_time(model_key, idatas_path)
    report_parts.append(f"<h2>Sampling Time: {sampling_time:.2f}</h2>")

    # divergences
    n_divergent = int(idata.sample_stats["diverging"].sum())
    report_parts.append(f"<h2>Divergences: {n_divergent}</h2>")

    # variables
    #var_names = ['beta_0', 'sigma_2', 'tau']
    #if penalised:
        #var_names.append('tau_p')
    #var_names_list = implementation_var_names[implementation]
    #var_names += var_names_list[int(penalised) * (len(var_names_list)>1)]

    # summary table
    summary_df = az.summary(idata, var_names=var_names)
    summary_html = summary_df.to_html()
    report_parts.append("<h2>Posterior Summary</h2>")
    report_parts.append(summary_html)

    f_metrics = {   "ess_bulk_min/s": [],
                    "ess_bulk_p5/s": [],
                    "ess_bulk_mean/s": [],
                    "ess_tail_min/s": [],
                    "ess_tail_p5/s": [],
                    "ess_tail_mean/s": [],
                    "r_hat_max": [],
                    "r_hat>1.01_mean": []}
    # f posterior summaries
    w_post, _ = extract_w_post(idata)
    #f_p = np.einsum('ij,cdj->cdi', X_plot, w_post)
    f_p = w_post @ X_plot.T
    if "beta_0_post" in idata.posterior.data_vars:
        beta_0_var = "beta_0_post"
    else:
        beta_0_var = "beta_0"
    # beta_0_var = [v for v in idata.posterior.data_vars if v.startswith("beta_0")][0]
    b_post = idata.posterior[beta_0_var].values
    f_p = f_p + b_post[:, :, None]
    # f_plot_post = f_plot_post - np.mean(f_plot_post, axis=2, keepdims=True)
    f_idata = az.convert_to_inference_data(xr.DataArray(f_p,
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
    report_parts.append("<h2>f Summary</h2>")
    report_parts.append(pd.DataFrame(f_metrics).to_html(index=False))

    # traces
    trace_axes = az.plot_trace(idata, var_names=var_names)
    fig = trace_axes.ravel()[0].figure
    img_base64 = fig_to_base64(fig)
    report_parts.append(f"<h2>Trace</h2><img src='data:image/png;base64,{img_base64}' />")

    # pair plot
    az.rcParams["plot.max_subplots"] = 100
    pair_axes = az.plot_pair(idata, var_names=var_names, textsize=14, divergences=True)
    for i in range(pair_axes.shape[0]):
        pair_axes[i, 0].yaxis.label.set_rotation(0)
        pair_axes[i, 0].yaxis.label.set_ha('right')
    for j in range(pair_axes.shape[1]):
        pair_axes[0, j].xaxis.label.set_rotation(45)
        pair_axes[0, j].xaxis.label.set_ha('right')
    fig = pair_axes.ravel()[0].figure
    img_base64 = fig_to_base64(fig)
    report_parts.append(f"<h2>Pair Plot</h2><img src='data:image/png;base64,{img_base64}' />")

    # spline plot
    #if "w" in idata.posterior:
        #w_post = idata.posterior["w"].values.reshape(-1, X.shape[1])
    #elif "wp" in idata.posterior and "w0" in idata.posterior:
        #wp_post = idata.posterior["wp"].values.reshape(-1, X.shape[1] - (order-1))
        #w0_post = idata.posterior["w0"].values.reshape(-1, order-1)
        #w_post = np.hstack([wp_post, w0_post])

    _, w_post_flat = extract_w_post(idata)
    #w_post_flat[:, :-1] = 0
    if "beta_0_post" in idata.posterior.data_vars:
        beta_0_var = "beta_0_post"
    else:
        beta_0_var = "beta_0"
    # beta_0_var = [v for v in idata.posterior.data_vars if v.startswith("beta_0")][0]
    b_post = idata.posterior[beta_0_var].values.flatten()

    X = X[x_data_order, :]
    ###
    f_plot_mean, f_plot_median, f_plot_025, f_plot_975 = f_plot_post(X_plot, w_post_flat, b_post, builder)
    f_min = np.nanmin(f_plot_025)
    f_max = np.nanmax(f_plot_975)
    
    f_range = f_max - f_min
    ###
    
    fig, ax = plt.subplots(figsize=(12, 4))
    if not builder in ['nb_ortho_diag', 'p_ortho_diag']:
        ax.scatter(x_data, y_data, marker='o', label='Data', alpha=0.5, s=1, color='blue')
    else:
        if 'ax2' not in locals():
            ax2 = ax.twinx()
        y_min = np.min(y_data)
        y_max = np.max(y_data)
        y_range = y_max - y_min

        ax2.scatter(x_data, y_data, marker='o', label='Data', alpha=0.5, s=1, color='blue')
        ax2.set_ylim((f_min/f_range - 0.1)*y_range, (f_max/f_range + 0.1)*y_range)

    #ax.plot(x_data[x_data_order], f_mean, label='Posterior Mean data', color='red', linestyle='dashed')
    ax.set_ylim((f_min/f_range - 0.1)*f_range, (f_max/f_range + 0.1)*f_range)
    ax.set_xlim(np.min(x_data), np.max(x_data))
    ax.plot(x_plot, f_plot_mean, label='Posterior Mean plot', color='red')
    ax.plot(x_plot, f_plot_median, label='Posterior Median', color='orange', linestyle='dashed')
    ax.fill_between(x_plot, f_plot_025, f_plot_975, color='red', alpha=0.3, label='95% Credible Interval')
    
    if not (('dengue' in f) or ('cherry' in f) or ('weighted' in f) or ('df3' in f)):
        f_plot, x_pl = function_plot(f, x_plot, functions, replication, x_data)
        ax.plot(x_pl, f_plot, label='True Function', color='green')

    # if f in functions:
        # ax.plot(x_plot, functions[f](x_plot) - np.mean(functions[f](x_plot)), label='True Function', color='green')
        # ax.plot(x_plot, functions[f](x_plot), label='True Function', color='green')
    # elif np.any([f_name_start in f for f_name_start in ['uniform', 'exponential', 'uni_normal', 'bi_normal']]):
        # m = re.fullmatch(r'([A-Za-z_]+)_(\d+)_(\d+)_(\d+)', f)
        # function_name = m.group(1)
        # num_knots = int(m.group(2))
        # n_val = int(m.group(3))
        # scale = int(m.group(4))

        # curve_plot_dict_path = os.path.join("data", f"curve_plot_dict_{num_knots}.pkl")
        # with open(curve_plot_dict_path, "rb") as pl:
            # curve_plot_dict = pickle.load(pl)
        # f_plot = curve_plot_dict[(function_name, n_val, replication)]*scale
        # x_pl = np.linspace(np.min(x_data), np.max(x_data), len(f_plot))
        # ax.plot(x_pl, f_plot, label='True Function', color='green')
    # ax.plot(x_plot, functions[f](x_plot), label='True Function', color='green')

    
    ax.set_title(f"Spline Fit: {model_key}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    img_base64 = fig_to_base64(fig)
    report_parts.append(f"<h2>Spline Fit</h2><img src='data:image/png;base64,{img_base64}' />")

    plot_basis = False
    if plot_basis:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_plot, X[:, -1])
        img_base64 = fig_to_base64(fig)
        report_parts.append(f"<h2>Spline Basis</h2><img src='data:image/png;base64,{img_base64}' />")

    report_parts.append("</body></html>")

    # --- Write HTML files ---
    with open(report_path, "w") as report_file:
        report_file.write("\n".join(report_parts))
