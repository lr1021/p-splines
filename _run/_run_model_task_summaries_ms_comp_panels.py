# Task-summary figures as 2x2 panels: one figure per metric, rows = runs (e.g. k = 5, k = 20),
# columns = unpenalised / penalised. Same encoding as _run_model_task_summaries_ms_comp_dodge.py
# (dodged points, one marker per implementation, 95% replication interval as error bars), plus:
#   - two-level x labels (sigma under the ticks, f centred below), alternating grey bands per f
#   - shared y-axis within each row, light horizontal grid, no titles, readable axis labels
#   - no legend in the figures; one separate legend.png
# Usage (from the repo root):
#   python _run/_run_model_task_summaries_ms_comp_panels.py <out_folder> <keys file row 1> <keys file row 2> ...
from importlib import import_module
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # repo root, for _run/_utils imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from _run._run_model_task_summaries_ms_comp import metric_list
from _run._run_model_task_summaries_ms_comp_dodge import load_tasks_df, legend_figure
from _utils._utils_comp_style import impl_style, sort_impls

import warnings
warnings.filterwarnings('ignore')

FIG_SIZE = (6.5, 4.4)     # inches, full text width
FONT_SIZE = 9
DODGE_WIDTH = 0.54        # total horizontal spread of the implementations around a tick
MARKER_SIZE = 4.5
INTERVAL_LW = 1.0
INTERVAL_CAPSIZE = 2
BAND_COLOR = '0.93'       # alternating background per f
GRID_COLOR = '0.85'
PENALISED_ORDER = [('False', 'unpenalised'), ('True', 'penalised')]

YLABELS = {
    'f_umean_ess_bulk/s': 'Bulk ESS / s',
    'f_umean_ess_tail/s': 'Tail ESS / s',
    'div>0%': 'Fits with divergences (%)',
    'div/samples%': 'Divergent draws (%)',
    'div>1%samples%': 'Fits with >1% divergent draws (%)',
}


def panel(ax, df, penalised, val_col, p025_col, p975_col, scale, f_labels):
    d = df[df['penalised'] == penalised]
    tasks = d[['f', 'sigma']].drop_duplicates().sort_values(['f', 'sigma'])
    x = np.arange(len(tasks))
    impls = sort_impls(d['Implementation'].unique())

    # alternating background band for every other f (f values are consecutive in the sorted tasks)
    f_vals = tasks['f'].values
    f_groups = [(f, np.where(f_vals == f)[0]) for f in dict.fromkeys(f_vals)]
    for gi, (f, idx) in enumerate(f_groups):
        if gi % 2 == 1:
            ax.axvspan(idx.min() - 0.5, idx.max() + 0.5, color=BAND_COLOR, zorder=0, lw=0)

    offsets = np.zeros(1) if len(impls) == 1 else np.linspace(-DODGE_WIDTH / 2, DODGE_WIDTH / 2, len(impls))
    for offset, implementation in zip(offsets, impls):
        style = impl_style(implementation)
        p = tasks.merge(d[d['Implementation'] == implementation], on=['f', 'sigma'], how='left')
        y = p[val_col].astype(float).values * scale
        if p025_col is not None:
            lo = p[p025_col].astype(float).values * scale
            hi = p[p975_col].astype(float).values * scale
            ax.errorbar(x + offset, (lo + hi) / 2, yerr=(hi - lo) / 2, fmt='none', ecolor=style['color'],
                        elinewidth=INTERVAL_LW, capsize=INTERVAL_CAPSIZE, capthick=INTERVAL_LW, zorder=2)
        ax.plot(x + offset, y, linestyle='none', marker=style['marker'], markersize=MARKER_SIZE,
                color=style['color'], zorder=3)

    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'σ={s}' for s in tasks['sigma'].values], fontsize=FONT_SIZE - 2)
    ax.tick_params(axis='x', length=0)
    if f_labels:
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        for f, idx in f_groups:
            ax.text(idx.mean(), -0.17, f, transform=tr, ha='center', va='top', fontsize=FONT_SIZE)
    ax.grid(axis='y', color=GRID_COLOR, lw=0.5)
    ax.set_axisbelow(True)


def main(out_folder, keys_paths):
    plt.rcParams.update({'font.size': FONT_SIZE, 'axes.spines.top': False, 'axes.spines.right': False})
    rows = []
    for keys_path in keys_paths:
        keys = import_module(keys_path.replace('/', '.').removesuffix('.py'))
        task_summary_folder = os.path.join(keys.reports_path, "../replication_reports/task_summaries")
        rows.append((f"k = {keys.n_internal_knots}", load_tasks_df(task_summary_folder)))
    os.makedirs(out_folder, exist_ok=True)

    for metric, val_col, p025_col, p975_col in zip(metric_list['metric'], metric_list['val_col'],
                                                   metric_list['p025_col'], metric_list['p975_col']):
        scale = 100 if metric[-1] == '%' else 1  # convert to percentage
        fig, axes = plt.subplots(len(rows), len(PENALISED_ORDER), figsize=FIG_SIZE, sharex=True, sharey='row',
                                 squeeze=False)
        for r, (row_label, df) in enumerate(rows):
            for c, (penalised, col_label) in enumerate(PENALISED_ORDER):
                ax = axes[r, c]
                panel(ax, df, penalised, val_col, p025_col, p975_col, scale, f_labels=(r == len(rows) - 1))
                if r == 0:
                    ax.set_title(col_label, fontsize=FONT_SIZE)
                if c == 0:
                    ax.set_ylabel(row_label)
        fig.supylabel(YLABELS.get(metric, metric), fontsize=FONT_SIZE)  # metric once, row labels per row
        fig.tight_layout(h_pad=1.6)
        safe_metric = metric.replace("/", "_").replace("%", "prc")
        fig.savefig(os.path.join(out_folder, f"figure_{safe_metric}.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    impls = sort_impls(set().union(*[set(df['Implementation'].unique()) for _, df in rows]))
    fig = legend_figure(impls)
    fig.savefig(os.path.join(out_folder, "legend.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
