# Task-summary figures in the "dodge" style (copy of _run_model_task_summaries_ms_comp.py, PNG mode only):
#   - implementations offset horizontally around each (f, sigma) tick, so equal values do not overlap
#   - one marker shape per implementation (from _utils/_utils_comp_style.py), no connecting lines
#   - 95% replication interval (2.5-97.5% quantiles across replications) as capped error bars
#   - no legend in the figures; one separate legend.png per folder
# Writes to <run>/task_summary_report/ (figures + legend.png).
# Usage (from the repo root): python _run/_run_model_task_summaries_ms_comp_dodge.py <keys file>
from importlib import import_module
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # repo root, for _run/_utils imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _run._run_model_task_summaries_ms_comp import metric_list
from _utils._utils_comp_style import impl_style, sort_impls

import warnings
warnings.filterwarnings('ignore')

DODGE_WIDTH = 0.36           # total horizontal spread of the implementations around a tick
MARKER_SIZE = 7
INTERVAL_LW = 1.2
INTERVAL_CAPSIZE = 3
INTERVAL_LABEL = '95% replication interval'
INTERVAL_COLOR = 'gray'      # legend entry only; in the figures the bars take the implementation colour


def load_tasks_df(task_summary_folder):
    """All task_summary(...) csvs in one frame, with f, sigma, penalised columns (as in the comp script)."""
    cols = ['f', 'sigma', 'penalised']
    frames = []
    for task_file in sorted(os.listdir(task_summary_folder)):
        if not (task_file.startswith("task_summary(") and task_file.endswith(".csv")):
            continue
        # "task_summary({f}_{sigma}_{penalised}_{builder}).csv"
        df = pd.read_csv(os.path.join(task_summary_folder, task_file)).drop('Penalised', axis=1)
        f, sigma, penalised = task_file.replace("task_summary(", "").replace(").csv", "").split("_")[:3]
        df['f'], df['sigma'], df['penalised'] = f, sigma, penalised
        frames.append(df)
    tasks_df = pd.concat(frames, ignore_index=True)
    return tasks_df.sort_values(by=cols + ['Implementation'])


def dodge_offsets(n):
    if n == 1:
        return np.zeros(1)
    return np.linspace(-DODGE_WIDTH / 2, DODGE_WIDTH / 2, n)


def plot_metric(current_df, tasks, metric, val_col, p025_col, p975_col, penalised):
    x = np.arange(len(tasks))
    impls = sort_impls(current_df['Implementation'].unique())
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, implementation in zip(dodge_offsets(len(impls)), impls):
        style = impl_style(implementation)
        plot_df = tasks.merge(current_df[current_df['Implementation'] == implementation], on=['f', 'sigma'], how='left')
        scale = 100 if metric[-1] == '%' else 1  # convert to percentage
        y = plot_df[val_col].astype(float).values * scale
        xi = x + offset
        if p025_col is not None:
            lo = plot_df[p025_col].astype(float).values * scale
            hi = plot_df[p975_col].astype(float).values * scale
            # bar drawn from the quantiles themselves (centre/half-width), independent of where the mean lies
            ax.errorbar(xi, (lo + hi) / 2, yerr=(hi - lo) / 2, fmt='none', ecolor=style['color'],
                        elinewidth=INTERVAL_LW, capsize=INTERVAL_CAPSIZE, capthick=INTERVAL_LW, zorder=2)
        ax.plot(xi, y, linestyle='none', marker=style['marker'], markersize=MARKER_SIZE,
                color=style['color'], label=style['label'], zorder=3)

    task_labels = [f"({f}, {sigma})" for f, sigma in zip(tasks['f'], tasks['sigma'])]
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=45, ha='right')
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xlabel("(f, sigma)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}  |  penalised={penalised}")
    return fig


def legend_figure(impls, show_interval=True):
    """Stand-alone horizontal legend: implementations (colour + marker) and the interval symbol."""
    handles = [Line2D([0], [0], linestyle='none', marker=impl_style(i)['marker'], markersize=MARKER_SIZE,
                      color=impl_style(i)['color'], label=impl_style(i)['label']) for i in impls]
    if show_interval:
        # an errorbar container renders as a capped vertical bar in the legend
        tmp_fig, tmp_ax = plt.subplots()
        eb = tmp_ax.errorbar([0], [0], yerr=[1], fmt='none', ecolor=INTERVAL_COLOR, elinewidth=INTERVAL_LW,
                             capsize=INTERVAL_CAPSIZE, capthick=INTERVAL_LW, label=INTERVAL_LABEL)
        handles.append(eb)
    fig = plt.figure(figsize=(12, 0.6))
    fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc='center',
               ncol=len(handles), frameon=False, fontsize=12, handlelength=1.0, handletextpad=0.5, columnspacing=2.0)
    if show_interval:
        plt.close(tmp_fig)
    return fig


def main(keys_path):
    print(keys_path)
    keys = import_module(keys_path.replace('/', '.').removesuffix('.py'))

    task_summary_folder = os.path.join(keys.reports_path, "../replication_reports/task_summaries")
    if (not os.path.exists(task_summary_folder)) or (len(os.listdir(task_summary_folder)) == 0):
        print("No task summaries found.")
        return
    out_folder = os.path.join(keys.reports_path, "../task_summary_report")
    os.makedirs(out_folder, exist_ok=True)

    tasks_df = load_tasks_df(task_summary_folder)
    for penalised in sorted(tasks_df['penalised'].unique()):
        current_df = tasks_df[tasks_df['penalised'] == penalised].copy()
        tasks = current_df[['f', 'sigma']].drop_duplicates().sort_values(['f', 'sigma'])
        for metric, val_col, p025_col, p975_col in zip(metric_list['metric'], metric_list['val_col'],
                                                       metric_list['p025_col'], metric_list['p975_col']):
            fig = plot_metric(current_df, tasks, metric, val_col, p025_col, p975_col, penalised)
            safe_metric = metric.replace("/", "_").replace("%", "prc")
            fig.savefig(os.path.join(out_folder, f"figure_{safe_metric}_penalised_{penalised}.png"),
                        bbox_inches="tight", dpi=300)
            plt.close(fig)

    fig = legend_figure(sort_impls(tasks_df['Implementation'].unique()))
    fig.savefig(os.path.join(out_folder, "legend.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
