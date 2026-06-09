from re import match
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
import io
import base64
import pandas as pd

from _utils_models import builder_dict

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
    w_post = np.hstack([
        idata.posterior[v].values.reshape(idata.posterior[v].sizes["chain"]
                                        * idata.posterior[v].sizes["draw"], -1)
        for v in w_vars])
    return w_post

def extract_sampling_time(model_key, idatas_path):
    (f, sigma, implementation, penalised, replication, builder) = model_key
    timings_path = os.path.join(idatas_path, 'timings.csv')
    timings_df = pd.read_csv(timings_path)
    match = timings_df.loc[
    (timings_df["f"] == f)
    & (timings_df["sigma"] == sigma)
    & (timings_df["implementation"] == implementation)
    & (timings_df["penalised"] == penalised)
    & (timings_df["replication"] == replication),
    "runtime_seconds",
    ]
    sampling_time = np.mean(match) if not match.empty else np.nan
    #sampling_time = idata.sample_stats.attrs["sampling_time"]
    return sampling_time

def html_report(model_key, reports_path, idatas_path, functions,
                a, b, order, spline_degree, n_internal_knots, data,
                replace_existing=False):
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

    w_post = extract_w_post(idata)

    X = X[x_data_order, :]
    f_post = X @ w_post.T
    f_mean = np.mean(f_post, axis=1)
    f_median = np.median(f_post, axis=1)
    f_975 = np.percentile(f_post, 97.5, axis=1)
    f_025 = np.percentile(f_post, 2.5, axis=1)

    f_plot_post = X_plot @ w_post.T
    f_plot_mean = np.mean(f_plot_post, axis=1)
    f_plot_median = np.median(f_plot_post, axis=1)
    f_plot_975 = np.percentile(f_plot_post, 97.5, axis=1)
    f_plot_025 = np.percentile(f_plot_post, 2.5, axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x_data, y_data, marker='o', label='Data', alpha=0.5, s=1, color='blue')
    #ax.plot(x_data[x_data_order], f_mean, label='Posterior Mean data', color='red', linestyle='dashed')
    ax.plot(x_plot, f_plot_mean, label='Posterior Mean plot', color='red')
    ax.plot(x_plot, f_plot_median, label='Posterior Median', color='orange', linestyle='dashed')
    ax.plot(x_plot, functions[f](x_plot) - np.mean(functions[f](x_plot)), label='True Function', color='green')
    ax.fill_between(x_plot, f_plot_025, f_plot_975, color='red', alpha=0.3, label='95% Credible Interval')
    ax.set_title(f"Spline Fit: {model_key}")
    ax.legend()
    img_base64 = fig_to_base64(fig)
    report_parts.append(f"<h2>Spline Fit</h2><img src='data:image/png;base64,{img_base64}' />")

    report_parts.append("</body></html>")

    # --- Write HTML files ---
    with open(report_path, "w") as report_file:
        report_file.write("\n".join(report_parts))
