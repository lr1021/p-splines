"""Display names, colours and ordering of implementations in comparison plots.

Variants are grouped by rules (e.g. 'conditioning' and 'ortho_conditioning' are
both shown as 'Conditioning'), so one entry covers all variants in _utils_models*.
"""

DATA_COLOR = 'blue'
TRUE_FUNCTION_COLOR = 'black'
FALLBACK_COLOR = 'gray'

# (match rule, style). First matching rule wins, so keep the most specific rules first.
#   ('exact', name)     -> implementation == name
#   ('contains', text)  -> text in implementation
IMPL_STYLE_RULES = [
    (('exact', 'opr_exact'),             dict(label='OPR exact (SVD)',              color='#6A3D9A', order=2, marker='s')),
    (('exact', 'svd'),                   dict(label='OPR (SVD)',                    color='green',   order=0, marker='o')),
    (('contains', 'centring+dropping'),  dict(label='Centring and Dropping (Last)', color='red',     order=1, marker='^')),
    (('contains', 'conditioning'),       dict(label='Conditioning',                 color='red',     order=1, marker='^')),
    (('contains', 'post_centring'),      dict(label='Post-centring',                color='red',     order=1, marker='^')),
]


def impl_style(implementation):
    """Return dict(label, color, order, marker) for an implementation name (unknown names: raw name, grey, last)."""
    for (kind, text), style in IMPL_STYLE_RULES:
        if (kind == 'exact' and implementation == text) or (kind == 'contains' and text in implementation):
            return dict(style)
    return dict(label=implementation, color=FALLBACK_COLOR, order=99, marker='D')


def sort_impls(implementations):
    """Plot/legend order: by style order, then by name."""
    return sorted(implementations, key=lambda i: (impl_style(i)['order'], i))


######################
# shared matplotlib helpers for comparison plots

CI_FILL_ALPHA = 0.08
CI_LINE_WIDTH = 0.6


def plot_impl(ax, x_plot, f_plot_mean, f_plot_025, f_plot_975, implementation):
    """Posterior mean line + 95% credible band (light fill, thin edge lines) in the implementation colour."""
    c = impl_style(implementation)['color']
    ax.fill_between(x_plot, f_plot_025, f_plot_975, color=c, alpha=CI_FILL_ALPHA, linewidth=0)
    ax.plot(x_plot, f_plot_025, color=c, linewidth=CI_LINE_WIDTH)
    ax.plot(x_plot, f_plot_975, color=c, linewidth=CI_LINE_WIDTH)
    ax.plot(x_plot, f_plot_mean, color=c, label=f"Posterior Mean {impl_style(implementation)['label']}")


def legend_handles(impls, show_true_function=True, show_data=True):
    """Handles for fig.legend(..., ncol=2): left column posterior means (+ true function),
    right column credible intervals (+ data), one row per implementation."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    means = [Line2D([0], [0], color=impl_style(i)['color'], label=f"Posterior Mean {impl_style(i)['label']}") for i in impls]
    cis = [Patch(facecolor=to_rgba(impl_style(i)['color'], 0.15), edgecolor=impl_style(i)['color'], linewidth=CI_LINE_WIDTH,
                 label=f"95% Credible Interval {impl_style(i)['label']}") for i in impls]
    left = means + ([Line2D([0], [0], color=TRUE_FUNCTION_COLOR, linestyle="--", label="True Function")] if show_true_function else [])
    right = cis + ([Line2D([0], [0], marker="x", color=DATA_COLOR, linestyle="", label="Data")] if show_data else [])
    while len(left) < len(right):
        left.append(Line2D([], [], linestyle='', label=' '))
    while len(right) < len(left):
        right.append(Line2D([], [], linestyle='', label=' '))
    return left + right


def get_comparison_list(keys, d_keys):
    """Implementations to compare: `comparison_list` of the distortion config, else of the keys file.
    Entries may be tuples of any length."""
    comparison_list = getattr(d_keys, 'comparison_list', None)
    if comparison_list is None:
        comparison_list = getattr(keys, 'comparison_list', [])
    return [tuple(c) for c in comparison_list]
