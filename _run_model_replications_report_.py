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
from _run_model_keys._run_model_keys import a, b, order, spline_degree, n_internal_knots, model_keys, directory_path, data_path, reports_path, idatas_path, builder, replications_report_workers

import warnings
warnings.filterwarnings('ignore')

######################
task_summary_round =  {'f (summary) umean_ess_bulk_rmean': 2,
                        'f (summary) umean_ess_bulk_rsd': 2,
                        'f (summary) umean_ess_bulk/s_rmean': 2,
                        'f (summary) umean_ess_bulk/s_rsd': 2,

                        'f (summary) umean_ess_tail_rmean': 2,
                        'f (summary) umean_ess_tail_rsd': 2,
                        'f (summary) umean_ess_tail/s_rmean': 2,
                        'f (summary) umean_ess_tail/s_rsd': 2,

                        'w (summary) umean_ess_bulk/s_rmean': 2,
                        'w (summary) umean_ess_bulk/s_rsd': 2,
                        'w (summary) umean_ess_bulk_rmean': 2,
                        'w (summary) umean_ess_bulk_rsd': 2,

                        'w (summary) umean_ess_tail/s_rmean': 2,
                        'w (summary) umean_ess_tail/s_rsd': 2,
                        'w (summary) umean_ess_tail_rmean': 2,
                        'w (summary) umean_ess_tail_rsd': 2,

                        'sampling_time_rmean': 3,
                        'sampling_time_rsd': 3
                        }
        
######################
with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)

def replication_compute(model_key):
    f, sigma, implementation, penalised, replication, builder = model_key
    model_data = data[(f, sigma)][replication]
    x_data = model_data[0]
    y_data = model_data[1]
    model, X, X_plot, var_names = builder_dict[builder](x_data, y_data, a, b,
                        spline_degree, n_internal_knots,
                        implementation, penalised, order)
    try:
        idata = az.from_netcdf(write_idata_path(model_key, idatas_path))
    except Exception as e:
        print(f"Error loading idata for model_key {model_key}: {e}")
        return None, None, None, None, None

    # sampling times
    sampling_time = extract_sampling_time(model_key, idatas_path)
    # divergences
    try:
        n_divergences = int(idata.sample_stats["diverging"].sum())
        n_samples = idata.sample_stats["diverging"].size
    except Exception as e:
        print(f"Error extracting divergences for model_key {model_key}: {e}")
        n_divergences = 0
        n_samples = 1

    # posterior summaries
    # 'mean', 'sd', 'hdi_3%', 'hdi_97%', 'mcse_mean', 'mcse_sd', 'ess_bulk', 'ess_tail', 'r_hat'
    summary_az = az.summary(idata, var_names=var_names)
    variables = summary_az.index.tolist()
    summary_array = summary_az.to_numpy().copy()
    summary_array = np.concatenate([summary_array, summary_array[:, [6, 7]] / sampling_time], axis=1)

    # f posterior summaries
    w_post, _ = extract_w_post(idata)
    f_plot_post = w_post @ X_plot.T
    f_idata = az.convert_to_inference_data(xr.DataArray(f_plot_post,
                                                        dims=["chain", "draw", "x"]))
    f_rhat = az.rhat(f_idata)["x"].values
    f_ess_bulk = az.ess(f_idata, method="bulk")["x"].values
    f_ess_tail = az.ess(f_idata, method="tail")["x"].values
    # f metrics
    f_m = {}
    f_m['umin_ess_bulk'] = np.min(f_ess_bulk)
    f_m['up5_ess_bulk'] = np.percentile(f_ess_bulk, 5)
    f_m['umean_ess_bulk'] = np.mean(f_ess_bulk)

    f_m['umin_ess_bulk/s'] = np.min(f_ess_bulk / sampling_time)
    f_m['up5_ess_bulk/s'] = np.percentile(f_ess_bulk / sampling_time, 5)
    f_m['umean_ess_bulk/s'] = np.mean(f_ess_bulk / sampling_time)

    f_m['umin_ess_tail'] = np.min(f_ess_tail)
    f_m['up5_ess_tail'] = np.percentile(f_ess_tail, 5)
    f_m['umean_ess_tail'] = np.mean(f_ess_tail)

    f_m['umin_ess_tail/s'] = np.min(f_ess_tail / sampling_time)
    f_m['up5_ess_tail/s'] = np.percentile(f_ess_tail / sampling_time, 5)
    f_m['umean_ess_tail/s'] = np.mean(f_ess_tail / sampling_time)

    f_m['umax_r_hat'] = np.max(f_rhat)
    f_m['umean_r_hat>=1.01'] = np.mean(f_rhat >= 1.01)
    f_m['uany_r_hat>=1.01'] = np.any(f_rhat >= 1.01)
    return sampling_time, n_divergences, n_samples, summary_array, variables, f_m


# Summary
def main_iter(model_keys_df):
    #model_keys_df = pd.DataFrame(mk, columns=['f', 'sigma', 'implementation', 'penalised', 'replication', 'builder'])
    # summary data across replications
    unique_keys = model_keys_df.drop(columns='replication').drop_duplicates()
    unique_keys.sort_values(['f', 'sigma', 'penalised', 'builder', 'implementation'], inplace=True)
    unique_keys.reset_index(drop=True, inplace=True)
    unique_keys['parts'] = [[] for _ in range(len(unique_keys))]

    unique_keys['general summary'] = [[] for _ in range(len(unique_keys))]
    unique_keys['variables'] = [[] for _ in range(len(unique_keys))]
    unique_keys['summaries'] = [[] for _ in range(len(unique_keys))]
    unique_keys['replication summary'] = [[] for _ in range(len(unique_keys))]
    unique_keys['f summary'] = [[] for _ in range(len(unique_keys))]

    s0 = time.time()

    for i, row in unique_keys.iterrows():
        f, sigma, implementation, penalised, builder, _, _, _, _, _, _ = row

        p_ = os.path.join(reports_path, f"../replication_reports/replication_report({f}_{sigma}_{penalised}_{builder})")
        os.makedirs(p_, exist_ok=True)
        general_summary_path = os.path.join(p_, "general_summary.csv")
        replication_summary_path = os.path.join(p_, "replication_summary.csv")
        f_summary_path = os.path.join(p_, "f_summary.csv")

        # read variables
        model_key = (f, sigma, implementation, penalised, 0, builder)
        try:
            idata = az.from_netcdf(write_idata_path(model_key, idatas_path))
            variables_list = az.summary(idata).index.tolist()
        except Exception as e:
            print(f"Error loading idata for model_key {model_key}: {e}")

        if not os.path.exists(general_summary_path) or not os.path.exists(replication_summary_path) or not os.path.exists(f_summary_path):

            key_replications = model_keys_df[
                (model_keys_df['f'] == f) &
                (model_keys_df['sigma'] == sigma) &
                (model_keys_df['implementation'] == implementation) &
                (model_keys_df['penalised'] == penalised) &
                (model_keys_df['builder'] == builder)
            ]['replication'].tolist()

            variables_list = []
            sampling_times = [] # take mean and median
            divergences = [] # mean and median and max
            draws = []
            summaries = [] # mean of summary stats (ess_bulk, ess_tail, r_hat, etc.) + min ess + max rhat
            f_metrics = {
                "umin_ess_bulk": [],
                "up5_ess_bulk": [],
                "umean_ess_bulk": [],

                "umin_ess_bulk/s": [],
                "up5_ess_bulk/s": [],
                "umean_ess_bulk/s": [],

                "umin_ess_tail": [],
                "up5_ess_tail": [],
                "umean_ess_tail": [],

                "umin_ess_tail/s": [],
                "up5_ess_tail/s": [],
                "umean_ess_tail/s": [],

                "umax_r_hat": [],
                "umean_r_hat>=1.01": [],
                "uany_r_hat>=1.01": []}
            
            model_keys_rep = [(f, sigma, implementation, penalised, replication, builder) 
                        for replication in key_replications]
            with Pool(processes=int(min(len(key_replications), replications_report_workers))) as pool:
                results = pool.map(replication_compute, model_keys_rep)
            for sampling_time, n_divergences, n_samples, summary_array, variables, f_m in results:
                sampling_times.append(sampling_time)
                divergences.append(n_divergences)
                draws.append(n_samples)
                summaries.append(summary_array)
                variables_list.append(variables)

                for key in f_metrics:
                    f_metrics[key].append(f_m[key])

            summaries = np.stack(summaries, axis=0)

            ###
            general_summary = {
                "n_replications": [len(key_replications)],
                "sampling_time_rmean": [np.mean(sampling_times)],
                "sampling_time_rsd": [np.std(sampling_times)],
                "div>0_rmean": [np.mean(np.array(divergences)>0)],
                "div/samples_rmean": [np.mean(np.array(divergences)/np.array(draws))],
                "div>1%samples_rmean": [np.mean(np.array(divergences)/np.array(draws) > 0.01)],
                "total_divergences": [np.sum(divergences)]
            }
            general_summary_round = {
                "sampling_time_rmean": 3,
                "sampling_time_rsd": 3
            }
            unique_keys.at[i, 'general summary'].append(pd.DataFrame(general_summary))
            unique_keys.at[i, 'parts'].append('<h2>General Summary</h2>')
            general_summary_df = pd.DataFrame(general_summary)


            general_summary_df.to_csv(general_summary_path, index=False)

            general_summary_df = general_summary_df.round(general_summary_round)
            unique_keys.at[i, 'parts'].append(general_summary_df.to_html(index=False))
            ###

            ###
            replication_summary = pd.DataFrame()
            replication_summary['variable'] = variables_list[0]

            replication_summary['ess_bulk_rmean'] = summaries[:, :, 6].mean(axis=0).round(2)
            replication_summary['ess_bulk/s_rmean'] = summaries[:, :, -2].mean(axis=0).round(2)
            #replication_summary['sd_ess_bulk'] = summaries[:, :, 6].std(axis=0)
            #replication_summary['sd_ess_bulk/s'] = summaries[:, :, -2].std(axis=0)
            #replication_summary['min_ess_bulk/s'] = summaries[:, :, 6].min(axis=0)
            replication_summary['ess_tail_rmean'] = summaries[:, :, 7].mean(axis=0).round(2)
            replication_summary['ess_tail/s_rmean'] = summaries[:, :, -1].mean(axis=0).round(2)
            #replication_summary['sd_ess_tail'] = summaries[:, :, 7].std(axis=0)
            #replication_summary['sd_ess_tail/s'] = summaries[:, :, -1].std(axis=0)
            #replication_summary['min_ess_tail/s'] = summaries[:, :, 7].min(axis=0)

            replication_summary['r_hat_rmean'] = summaries[:, :, 8].mean(axis=0)
            replication_summary['r_hat>=1.01_rmean'] = (summaries[:, :, 8] >= 1.01).mean(axis=0)

            replication_summary['sd_rmean'] = summaries[:, :, 1].mean(axis=0)
            replication_summary['hdi_range_rmean'] = (summaries[:, :, 3] - summaries[:, :, 2]).mean(axis=0).round(2)

            replication_summary['mcse_mean_rmean'] = summaries[:, :, 4].mean(axis=0)
            replication_summary['mcse_sd_rmean'] = summaries[:, :, 5].mean(axis=0)

            mean_w_row = replication_summary.loc[
                [v.startswith('w') for v in replication_summary['variable']],
                replication_summary.columns != 'variable'].mean()
            
            mean_w_row['variable'] = 'w (summary)'
            mean_w_df = mean_w_row.to_frame().T
            replication_summary = pd.concat([mean_w_df[
            ['variable'] + [c for c in mean_w_df.columns if c != 'variable']],
            replication_summary], ignore_index=True)

            unique_keys.at[i, 'variables'].append(variables_list[0])
            unique_keys.at[i, 'summaries'].append(summaries)
            unique_keys.at[i, 'replication summary'].append(replication_summary)
            unique_keys.at[i, 'parts'].append('<h2>Replication Summary</h2>')
            unique_keys.at[i, 'parts'].append(replication_summary.to_html(index=False))

            replication_summary.to_csv(replication_summary_path, index=False)
            ###
            f_summary = {k+"_rmean": [np.mean(v)] for k, v in f_metrics.items()}
            for key in f_metrics:
                if 'ess' in key:
                    f_summary[key+"_rsd"] = [np.std(f_metrics[key])]
            f_summary['umax_r_hat_rmax'] = [np.max(f_metrics['umax_r_hat'])]
            f_summary_df = pd.DataFrame(f_summary)
            unique_keys.at[i, 'f summary'].append(f_summary_df)
            unique_keys.at[i, 'parts'].append('<h2>f Summary</h2>')
            unique_keys.at[i, 'parts'].append(f_summary_df.to_html(index=False))

            f_summary_df.to_csv(f_summary_path, index=False)
        else:
            general_summary_df = pd.read_csv(general_summary_path)
            replication_summary = pd.read_csv(replication_summary_path)
            f_summary_df = pd.read_csv(f_summary_path)

            unique_keys.at[i, 'variables'].append(variables_list[0])
            # unique_keys.at[i, 'summaries'].append(summaries)

            unique_keys.at[i, 'general summary'].append(general_summary_df)
            unique_keys.at[i, 'parts'].append('<h2>General Summary</h2>')
            unique_keys.at[i, 'parts'].append(general_summary_df.to_html(index=False))

            unique_keys.at[i, 'replication summary'].append(replication_summary)
            unique_keys.at[i, 'parts'].append('<h2>Replication Summary</h2>')
            unique_keys.at[i, 'parts'].append(replication_summary.to_html(index=False))
            
            unique_keys.at[i, 'f summary'].append(f_summary_df)
            unique_keys.at[i, 'parts'].append('<h2>f Summary</h2>')
            unique_keys.at[i, 'parts'].append(f_summary_df.to_html(index=False))

    s1 = time.time()
    print(f"Summary report generated in {s1 - s0:.2f} seconds.")

    html_parts = []
    title = f"Replication Summary Report:"
    html_parts = [f"<html><head><title>{title}</title>",
                        "<style>",
                        "body { font-family: Arial; font-size: 10px; line-height: 1.2; margin: 8px; text-align:center; }",
                        "h1, h2 { margin: 4px 0 8px 0; font-weight: normal; }",
                        "table { border-collapse: collapse; font-size: 15px; margin: 0 auto 12px auto; width: auto; }",
                        "table th, table td { border: 1px solid #aaa; padding: 4px 6px; text-align: center; }",
                        "img { max-width: 80%; margin: 8px auto; display: block; }",
                        "</style></head><body>",
                        f"<h1>{title}</h1>"]


    unique_keys_tasks = model_keys_df.drop(columns=['penalised', 'implementation', 'replication']).drop_duplicates()
    penalised = model_keys_df['penalised'].iloc[0]
    for i, task in unique_keys_tasks.iterrows():
        p_ = os.path.join(reports_path, f"../replication_reports/replication_report({f}_{sigma}_{penalised}_{builder})")
        os.makedirs(p_, exist_ok=True)
        task_summary_path = os.path.join(p_, "task_summary.csv")
        
        if not os.path.exists(task_summary_path):
            f, sigma, builder = task
            task_summary_df = {'Penalised': [],
                            'Implementation': [],
                            
                            'f (summary) umean_ess_bulk_rmean': [],
                            'f (summary) umean_ess_bulk_rsd': [],
                            'f (summary) umean_ess_bulk/s_rmean': [],
                            'f (summary) umean_ess_bulk/s_rsd': [],

                            'f (summary) umean_ess_tail_rmean': [],
                            'f (summary) umean_ess_tail_rsd': [],
                            'f (summary) umean_ess_tail/s_rmean': [],
                            'f (summary) umean_ess_tail/s_rsd': [],

                            'f (summary) umean_r_hat>=1.01_rmean': [],
                            'f (summary) uany_r_hat>=1.01_rmean': [],
                            
                            'Implementation_': [],

                            'w (summary) umean_ess_bulk/s_rmean': [],
                            'w (summary) umean_ess_bulk/s_rsd': [],
                            'w (summary) umean_ess_bulk_rmean': [],
                            'w (summary) umean_ess_bulk_rsd': [],

                            'w (summary) umean_ess_tail/s_rmean': [],
                            'w (summary) umean_ess_tail/s_rsd': [],
                            'w (summary) umean_ess_tail_rmean': [],
                            'w (summary) umean_ess_tail_rsd': [],

                            'w (summary) umean_r_hat>=1.01_rmean': [],
                            'w (summary) uany_r_hat>=1.01_rmean': [],

                            'other_bad_rhat': [],

                            'sampling_time_rmean': [],
                            'sampling_time_rsd': [],
                            'n_replications': [],
                            'div>0_rmean': [],
                            'div/samples_rmean': [],
                            'div>1%samples_rmean': [],
                            }
            
            for i, row in unique_keys.iterrows():
                if row['f'] == f and row['sigma'] == sigma and row['builder'] == builder:
                    task_summary_df['Penalised'].append(row['penalised'])
                    task_summary_df['Implementation'].append(row['implementation'])
                    task_summary_df['Implementation_'].append(row['implementation'])

                    print(row['variables'])
                    variables = row['variables'][0]
                    replication_summary = row['replication summary'][0]
                    f_summary = row['f summary'][0]
                
                    general_summary = row['general summary'][0]
                    ### u_w_r
                    summaries = row['summaries'][0]

                    w_summaries = summaries[:, [v.startswith('w') for v in variables], :]
                    w_rhat = w_summaries[:, :, 8]
                    w_bad_rhat = (w_rhat >= 1.01)
                    uany_bad_rhat = np.any(w_bad_rhat, axis=1)
                    uany_bad_rhat_rmean = np.mean(uany_bad_rhat)
                    umean_bad_rhat_rmean = np.mean(w_bad_rhat)
                    w_mean_summaries = w_summaries.mean(axis=1) # umean

                    other_bad_rhat = {}
                    other_summaries = summaries[:, [not v.startswith('w') for v in variables], :]
                    other_variables = [v for v in variables if not v.startswith('w')]
                    for i, v in enumerate(other_variables):
                        v_summaries = other_summaries[:, i, :].copy()
                        v_summaries_rhat = v_summaries[:, 8]
                        v_bad_rhat = (v_summaries_rhat >= 1.01)
                        if np.any(v_bad_rhat):
                            other_bad_rhat[v] = np.arange(len(v_bad_rhat))[v_bad_rhat]
                    
                    ###
                    ### w
                    task_summary_df['w (summary) umean_ess_bulk/s_rmean'].append(w_mean_summaries[:, -2].mean())
                    task_summary_df['w (summary) umean_ess_bulk/s_rsd'].append(w_mean_summaries[:, -2].std())
                    task_summary_df['w (summary) umean_ess_bulk_rmean'].append(w_mean_summaries[:, 6].mean())
                    task_summary_df['w (summary) umean_ess_bulk_rsd'].append(w_mean_summaries[:, 6].std())

                    task_summary_df['w (summary) umean_ess_tail/s_rmean'].append(w_mean_summaries[:, -1].mean())
                    task_summary_df['w (summary) umean_ess_tail/s_rsd'].append(w_mean_summaries[:, -1].std())
                    task_summary_df['w (summary) umean_ess_tail_rmean'].append(w_mean_summaries[:, 7].mean())
                    task_summary_df['w (summary) umean_ess_tail_rsd'].append(w_mean_summaries[:, 7].std())

                    # task_summary_df['w (summary) umean_r_hat>=1.01_rmean'].append((w_mean_summaries[:, 8] >= 1.01).mean())
                    task_summary_df['w (summary) umean_r_hat>=1.01_rmean'].append(umean_bad_rhat_rmean)
                    task_summary_df['w (summary) uany_r_hat>=1.01_rmean'].append(uany_bad_rhat_rmean)

                    task_summary_df['other_bad_rhat'].append(other_bad_rhat)

                    ### f
                    task_summary_df['f (summary) umean_ess_bulk/s_rmean'].append(f_summary['umean_ess_bulk/s_rmean'].values[0])
                    task_summary_df['f (summary) umean_ess_bulk/s_rsd'].append(f_summary['umean_ess_bulk/s_rsd'].values[0])
                    task_summary_df['f (summary) umean_ess_bulk_rmean'].append(f_summary['umean_ess_bulk_rmean'].values[0])
                    task_summary_df['f (summary) umean_ess_bulk_rsd'].append(f_summary['umean_ess_bulk_rsd'].values[0])

                    task_summary_df['f (summary) umean_ess_tail/s_rmean'].append(f_summary['umean_ess_tail/s_rmean'].values[0])
                    task_summary_df['f (summary) umean_ess_tail/s_rsd'].append(f_summary['umean_ess_tail/s_rsd'].values[0])
                    task_summary_df['f (summary) umean_ess_tail_rmean'].append(f_summary['umean_ess_tail_rmean'].values[0])
                    task_summary_df['f (summary) umean_ess_tail_rsd'].append(f_summary['umean_ess_tail_rsd'].values[0])

                    task_summary_df['f (summary) umean_r_hat>=1.01_rmean'].append(f_summary['umean_r_hat>=1.01_rmean'].values[0])
                    task_summary_df['f (summary) uany_r_hat>=1.01_rmean'].append(f_summary['uany_r_hat>=1.01_rmean'].values[0])

                    ### general
                    task_summary_df['sampling_time_rmean'].append(general_summary['sampling_time_rmean'].values[0])
                    task_summary_df['sampling_time_rsd'].append(general_summary['sampling_time_rsd'].values[0])
                    task_summary_df['n_replications'].append(general_summary['n_replications'].values[0])
                    task_summary_df['div>0_rmean'].append(general_summary['div>0_rmean'].values[0])
                    task_summary_df['div/samples_rmean'].append(general_summary['div/samples_rmean'].values[0])
                    task_summary_df['div>1%samples_rmean'].append(general_summary['div>1%samples_rmean'].values[0])

            html_parts.append(f"<h2>f={f}, sigma={sigma}, builder={builder}</h2>")
            task_summary_df = pd.DataFrame(task_summary_df)
            task_summary_df.sort_values(['Penalised', 'f (summary) umean_ess_bulk_rmean'], ascending=[True, False], inplace=True)

            pd.DataFrame(task_summary_df).to_csv(task_summary_path, index=False)
        else:
            task_summary_df = pd.read_csv(task_summary_path)

        task_summary_df = task_summary_df.round(task_summary_round)
        html_parts.append(pd.DataFrame(task_summary_df).to_html(index=False))
        ##
        
        

    s2 = time.time()
    print(f"Task summary tables generated in {s2 - s1:.2f} seconds.")

    html_parts.append("<hr>")
    for i, row in unique_keys.iterrows():
        f, sigma, implementation, penalised, builder = row[['f', 'sigma', 'implementation', 'penalised', 'builder']]
        html_parts.append(f"<h2>Model: f={f}, sigma={sigma}, implementation={implementation}, penalised={penalised}, builder={builder}</h2>")
        for part in row['parts']:
            html_parts.append(part)
        html_parts.append("<hr>")
    
    s3 = time.time()
    print(f"Full report generated in {s3 - s2:.2f} seconds>.")

    html_parts.append("</body></html>")
    with open(os.path.join(reports_path, f"../replication_reports/replication_report({f}_{sigma}_{penalised}_{builder}).html"), "w") as o:
        o.write("\n".join(html_parts))

def main():
    model_keys_df = pd.DataFrame(model_keys, columns=['f', 'sigma', 'implementation', 'penalised', 'replication', 'builder'])
    model_keys_df_f_sigma_penalised_builder = model_keys_df.drop(columns=['implementation', 'replication']).drop_duplicates()
    model_keys_df_f_sigma_penalised_builder.sort_values(['f', 'sigma', 'penalised', 'builder'], inplace=True)
    for i, row in model_keys_df_f_sigma_penalised_builder.iterrows():
        f, sigma, penalised, builder = row
        print(f"Processing f={f}, sigma={sigma}, penalised={penalised}, builder={builder}...")
        main_iter(model_keys_df[
            (model_keys_df['f'] == f) &
            (model_keys_df['sigma'] == sigma) &
            (model_keys_df['penalised'] == penalised) &
            (model_keys_df['builder'] == builder)
        ])
if __name__ == "__main__":
    main()