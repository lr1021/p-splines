from importlib import import_module
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

from _utils._utils_models import builder_dict
from _utils._utils_reports import extract_sampling_time, write_idata_path, write_idata_path, extract_w_post
from _utils._utils_reports import fig_to_base64

import warnings
warnings.filterwarnings('ignore')

# from _run_model_keys._run_model_keys import a, b, order, spline_degree, n_internal_knots, model_keys, data_path, directory_path, reports_path, idatas_path, builder, replications_report_workers

######################
metric_list = {'metric':[], 'val_col':[], 'p025_col':[], 'p975_col':[], 'val_curve':[], 'p025_curve':[], 'p975_curve':[]}

# f_ESS_bulk
# metric_list['metric'].append('f_umean_ess_bulk')
# metric_list['val_col'].append('f (summary) umean_ess_bulk_rmean')
# metric_list['sd_col'].append('f (summary) umean_ess_bulk_rsd')

# f_ESS_bulk/s
metric_list['metric'].append('f_umean_ess_bulk/s')
metric_list['val_col'].append('f (summary) umean_ess_bulk/s_rmean')
# metric_list['sd_col'].append('f (summary) umean_ess_bulk/s_rsd')
metric_list['p025_col'].append('f (summary) umean_ess_bulk/s_rp025')
metric_list['p975_col'].append('f (summary) umean_ess_bulk/s_rp975')

# f_ESS_tail
# metric_list['metric'].append('f_umean_ess_tail')
# metric_list['val_col'].append('f (summary) umean_ess_tail_rmean')
# metric_list['sd_col'].append('f (summary) umean_ess_tail_rsd')

# f_ESS_tail/s
metric_list['metric'].append('f_umean_ess_tail/s')
metric_list['val_col'].append('f (summary) umean_ess_tail/s_rmean')
# metric_list['sd_col'].append('f (summary) umean_ess_tail/s_rsd')
metric_list['p025_col'].append('f (summary) umean_ess_tail/s_rp025')
metric_list['p975_col'].append('f (summary) umean_ess_tail/s_rp975')

# f any rhat
# metric_list['metric'].append('f_uany_rhat>=1.01')
# metric_list['val_col'].append('f (summary) uany_r_hat>=1.01_rmean')
# metric_list['sd_col'].append(None)

# sampling time
# metric_list['metric'].append('sampling_time')
# metric_list['val_col'].append('sampling_time_rmean')
# metric_list['sd_col'].append('sampling_time_rsd')

# divergences > 0 % 
metric_list['metric'].append('div>0%')
metric_list['val_col'].append('div>0_rmean')
# metric_list['sd_col'].append(None)
metric_list['p025_col'].append(None)
metric_list['p975_col'].append(None)

# divergences %
metric_list['metric'].append('div/samples%')
metric_list['val_col'].append('div/samples_rmean')
# metric_list['sd_col'].append('div/samples_rsd')
metric_list['p025_col'].append('div/samples_rp025')
metric_list['p975_col'].append('div/samples_rp975')

# divergences > 1%
metric_list['metric'].append('div>1%samples%')
metric_list['val_col'].append('div>1%samples_rmean')
# metric_list['sd_col'].append(None)
metric_list['p025_col'].append(None)
metric_list['p975_col'].append(None)

######################
def main(keys_path):
    # base and title
    print(keys_path)
    full_report = False

    keys_loc = keys_path.replace('/', '.').removesuffix('.py')
    keys = import_module(keys_loc)

    task_summary_folder = os.path.join(keys.reports_path, "../replication_reports/task_summaries")
    if (not os.path.exists(task_summary_folder)) or (len(os.listdir(task_summary_folder)) == 0):
        print("No task summaries found.")
    if full_report:
        title = f"Task Summaries Report"
        parts = [f"<html><head><title>{title}</title>",
                            "<style>",
                            "body { font-family: Arial; font-size: 12px; line-height: 1.2; margin: 8px; text-align:center; }",
                            "h1, h2 { margin: 4px 0 8px 0; font-weight: normal; }",
                            "table { border-collapse: collapse; font-size: 15px; margin: 0 auto 12px auto; width: 80%; }",
                            "table th, table td { border: 1px solid #aaa; padding: 4px 6px; text-align: center; }",
                            "img { max-width: 80%; margin: 8px auto; display: block; }",
                            "</style></head><body>",
                            f"<h1>{title}</h1>"]
        parts.append(f"<h2>Points = mean across replications, Intervals = Standard Deviation of the Mean</h2>")
        
        
        # print(keys.implementation_list)
        # keys.implementation_list = ['svd', 'post_centring', 'ortho_post_centring'][:]
        task_summary_report_path = os.path.join(keys.reports_path, "../task_summary_report.html")

        task_summary_files = [f for f in os.listdir(task_summary_folder) if f.startswith("task_summary(") and f.endswith(".csv")]
        cols=['f', 'sigma', 'penalised']
        for task_file in task_summary_files:
            task_summary_df = pd.read_csv(os.path.join(task_summary_folder, task_file))
            task_summary_df = task_summary_df.drop('Penalised', axis=1)
            metric_cols = [c for c in task_summary_df.columns if c not in cols]
            break

        tasks_df = pd.DataFrame(columns=cols + metric_cols)

        for task_file in task_summary_files:
            # "task_summary({f}_{sigma}_{penalised}_{builder}).csv"
            # print(task_file)
            task_summary_df = pd.read_csv(os.path.join(task_summary_folder, task_file))
            task_summary_df = task_summary_df.drop('Penalised', axis=1)
            f, sigma, penalised = (task_file.replace("task_summary(", "").replace(").csv", "").split("_"))[:3]
            for c in cols:
                task_summary_df[c] = eval(c)
            tasks_df = pd.concat([tasks_df, task_summary_df], ignore_index=True)

        tasks_df.sort_values(by=['penalised', 'f', 'sigma', 'Implementation'], inplace=True)

        for penalised in sorted(tasks_df['penalised'].unique()):
        # for penalised in sorted(keys.penalised_list):
            #print(penalised)
            current_df = tasks_df[tasks_df['penalised'] == penalised].copy()

            # Define task ordering once
            tasks = (
                current_df[['f', 'sigma']]
                .drop_duplicates()
                .sort_values(['f', 'sigma'])
            )

            task_labels = [
                f"({f}, {sigma})"
                for f, sigma in zip(tasks['f'], tasks['sigma'])
            ]

            x = np.arange(len(tasks))

            for metric, val_col, p025_col, p975_col in zip(
                metric_list['metric'],
                metric_list['val_col'],
                metric_list['p025_col'],
                metric_list['p975_col']
            ):

                fig, ax = plt.subplots(figsize=(12, 6))

                for implementation in sorted(current_df['Implementation'].unique()):
                # print(current_df['Implementation'].unique())
                # for implementation in sorted(keys.implementation_list):
                    # print(implementation)

                    impl_df = current_df[
                        current_df['Implementation'] == implementation
                    ]

                    # align to task order
                    plot_df = tasks.merge(
                        impl_df,
                        on=['f', 'sigma'],
                        how='left'
                    )

                    y = plot_df[val_col].astype(float).values
                    

                    if p025_col is not None:
                        # se = (
                        #     plot_df[sd_col].astype(float).values
                        #     / np.sqrt(plot_df['n_replications'].astype(float).values)
                        # )
                        p025 = plot_df[p025_col].astype(float).values
                        p975 = plot_df[p975_col].astype(float).values
                    else:
                        p025 = np.zeros_like(y)
                        p975 = np.zeros_like(y)

                    if metric[-1] == '%':
                        y = y * 100  # convert to percentage
                        p025 = p025 * 100  # convert to percentage
                        p975 = p975 * 100  # convert to percentage
                    ax.plot(x, y, marker='o', label=implementation)

                    if p025_col is not None:
                        ax.fill_between(
                            x,
                            p025,
                            p975,
                            alpha=0.15
                        )

                ax.set_xticks(x)
                ax.set_xticklabels(task_labels, rotation=45, ha='right')

                ax.set_xlabel("(f, sigma)")
                ax.set_ylabel(metric)
                ax.set_title(
                    f"{metric}  |  penalised={penalised}"
                )

                ax.legend(
                    title="Implementation",
                    bbox_to_anchor=(1.05, 1)
                )
                #plt.tight_layout()
                img_base64 = fig_to_base64(fig)
                parts.append(f"<h2>{metric}</h2><img src='data:image/png;base64,{img_base64}' />")

        parts.append("</body></html>")
        # --- Write HTML files ---
        with open(task_summary_report_path, "w") as report_file:
            report_file.write("\n".join(parts))

    else:
        task_summary_report_path_folder = os.path.join(keys.reports_path, "../task_summary_report")
        if not os.path.exists(task_summary_report_path_folder):
            os.makedirs(task_summary_report_path_folder)

        task_summary_files = [f for f in os.listdir(task_summary_folder) if f.startswith("task_summary(") and f.endswith(".csv")]
        cols=['f', 'sigma', 'penalised']
        for task_file in task_summary_files:
            task_summary_df = pd.read_csv(os.path.join(task_summary_folder, task_file))
            task_summary_df = task_summary_df.drop('Penalised', axis=1)
            metric_cols = [c for c in task_summary_df.columns if c not in cols]
            break

        tasks_df = pd.DataFrame(columns=cols + metric_cols)

        for task_file in task_summary_files:
            # "task_summary({f}_{sigma}_{penalised}_{builder}).csv"
            # print(task_file)
            task_summary_df = pd.read_csv(os.path.join(task_summary_folder, task_file))
            task_summary_df = task_summary_df.drop('Penalised', axis=1)
            f, sigma, penalised = (task_file.replace("task_summary(", "").replace(").csv", "").split("_"))[:3]
            for c in cols:
                task_summary_df[c] = eval(c)
            tasks_df = pd.concat([tasks_df, task_summary_df], ignore_index=True)

        tasks_df.sort_values(by=['penalised', 'f', 'sigma', 'Implementation'], inplace=True)

        for penalised in sorted(tasks_df['penalised'].unique()):
        # for penalised in sorted(keys.penalised_list):
            #print(penalised)
            current_df = tasks_df[tasks_df['penalised'] == penalised].copy()

            # Define task ordering once
            tasks = (
                current_df[['f', 'sigma']]
                .drop_duplicates()
                .sort_values(['f', 'sigma'])
            )

            task_labels = [
                f"({f}, {sigma})"
                for f, sigma in zip(tasks['f'], tasks['sigma'])
            ]

            x = np.arange(len(tasks))

            for metric, val_col, p025_col, p975_col in zip(
                metric_list['metric'],
                metric_list['val_col'],
                metric_list['p025_col'],
                metric_list['p975_col']
            ):

                fig, ax = plt.subplots(figsize=(12, 6))

                for implementation in sorted(current_df['Implementation'].unique()):
                # print(current_df['Implementation'].unique())
                # for implementation in sorted(keys.implementation_list):
                    # print(implementation)
                    okabe_ito_colors = np.array(["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
          "#0072B2","#D55E00", "#CC79A7", "#999999"])
                    if implementation == 'svd':
                        color = okabe_ito_colors[6]
                    elif implementation == 'ortho_post_centring':
                        color = okabe_ito_colors[5]
                    else:
                        color = okabe_ito_colors[0]
                    impl_df = current_df[
                        current_df['Implementation'] == implementation
                    ]

                    # align to task order
                    plot_df = tasks.merge(
                        impl_df,
                        on=['f', 'sigma'],
                        how='left'
                    )

                    y = plot_df[val_col].astype(float).values
                    

                    if p025_col is not None:
                        # se = (
                        #     plot_df[sd_col].astype(float).values
                        #     / np.sqrt(plot_df['n_replications'].astype(float).values)
                        # )
                        p025 = plot_df[p025_col].astype(float).values
                        p975 = plot_df[p975_col].astype(float).values
                    else:
                        p025 = np.zeros_like(y)
                        p975 = np.zeros_like(y)

                    if metric[-1] == '%':
                        y = y * 100  # convert to percentage
                        p025 = p025 * 100  # convert to percentage
                        p975 = p975 * 100  # convert to percentage
                    ax.plot(x, y, marker='o', label=implementation, color=color)

                    if p025_col is not None:
                        ax.fill_between(
                            x,
                            p025,
                            p975,
                            alpha=0.15,
                            color=color
                        )

                ax.set_xticks(x)
                ax.set_xticklabels(task_labels, rotation=45, ha='right')

                ax.set_xlabel("(f, sigma)")
                ax.set_ylabel(metric)
                ax.set_title(
                    f"{metric}  |  penalised={penalised}"
                )

                # ax.legend(
                #     title="Implementation",
                #     loc="upper left",
                #     bbox_to_anchor=(1.02, 1),
                # )
                safe_metric = metric.replace("/", "_")
                safe_metric = safe_metric.replace("%", "prc")
                fig.savefig(
                    os.path.join(task_summary_report_path_folder, f"figure_{safe_metric}_penalised_{penalised}.png"),
                    bbox_inches="tight",
                    dpi=300,
                )

if __name__ == "__main__":
    main(sys.argv[1])