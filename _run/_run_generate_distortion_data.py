import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os

from _utils._utils_spline import eval_spline_basis_equispaced_numeric

###

def I(n):
    return np.eye(n)
def e(n, j):
    e = np.zeros((n, 1))
    e[j-1] = 1
    return e
def R(n, j):
    return I(n) - np.ones((n, 1)) @ e(n, j).T
def Q(B):
    n, k = B.shape
    return I(k) - np.ones((k, 1)) @ (np.ones((n,1)).T @ B)/n

###

def max_ratio_cd(k):
    v = (np.ones((k, 1)).T @  R(k, k)).T
    return v / np.linalg.norm(v, 2)

def max_ratio_cond(B):
    k = B.shape[1]
    v = (np.ones((k, 1)).T @  Q(B)).T
    return v / np.linalg.norm(v, 2), np.linalg.norm(v, 2)

###

def generate_data_dict(distribution_list, n_list, replication_list, x_range=[-3.0, 3.0], seed=0):
    data_dict = {}
    for dist in distribution_list:
        np.random.seed(seed)
        n_max = max(n_list)
        if dist == 'uniform':
            x = np.random.uniform(-3, 3, (len(replication_list), n_max))
        elif dist == 'uni_normal':
            x = np.random.normal(0.0, 1.0, (len(replication_list), n_max))
        elif dist == 'bi_normal':
            x = np.concatenate([np.random.normal(-2.0, 0.5, (len(replication_list), n_max//2)), np.random.normal(0.5, 0.5, (len(replication_list), n_max//2))], axis=1)
        elif dist == 'exponential':
            x = np.random.exponential(1, (len(replication_list), n_max))
        x_min = np.min(x, axis=1).reshape(-1, 1)
        x_max = np.max(x, axis=1).reshape(-1, 1)
        x = (x - x_min) / (x_max - x_min) * 6 - 3

        for n_val in n_list:
            selected_indices = np.random.choice(n_max, n_val-2, replace=False)
            x_n = x[:, selected_indices]

            for i, r in enumerate(replication_list):
                x_n_r = x_n[i].copy()
                x_n_r = np.concatenate([[x_range[0]], x_n_r, [x_range[1]]])
                x_n_r.sort()
                data_dict[(dist, n_val, r)] = x_n_r
    return data_dict

def generate_curve_dict(distribution_list, n_list, replication_list, num_knots, data_dict, plot=False):
    degree = 3
    curve_dict = {}
    pen_dict = {}
    curve_plot_dict = {}

    for dist in distribution_list:
        for n_val in n_list:
            for r in replication_list:
                x_vals = data_dict[(dist, n_val, r)].copy()
                
                if len(x_vals)>num_knots:
                    B = eval_spline_basis_equispaced_numeric(degree, np.min(x_vals), np.max(x_vals), num_knots, x_vals)['B']
                    v, pen = max_ratio_cond(B)
                    curve = B@v
                    curve = curve/(np.max(curve) - np.min(curve))
                    curve_dict[(dist, n_val, r)] = curve
                    pen_dict[(dist, n_val, r)] = pen
                    if plot:
                        x_plot = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                        B_plot = eval_spline_basis_equispaced_numeric(degree, np.min(x_plot), np.max(x_plot), num_knots, x_plot)['B']
                        curve_plot = B_plot@v
                        curve_plot = curve_plot/(np.max(curve) - np.min(curve))
                        curve_plot_dict[(dist, n_val, r)] = curve_plot
    if plot:           
        return curve_dict, curve_plot_dict, pen_dict
    else:
        return curve_dict, pen_dict
    
def generate_data(distribution_list, n_list, sigma_list, replication_list, num_knots, data_dict):
    f_sigma_list = []
    for dist in distribution_list:
        for n in n_list: 
            for sigma in sigma_list:  
                f_sigma_list.append((f'{dist}_{n}', sigma))
    data = {(f, sigma): {} for f, sigma in f_sigma_list}

    scale = 1000
    for dist in distribution_list:
        np.random.seed(0)
        eps = np.random.normal(0.0, 1.0, (len(replication_list), np.max(n_list)))
        for n_val in n_list:
            for i, r in enumerate(replication_list):
                x_n_r = data_dict[(dist, n_val, r)].copy()
                eps_n_r = eps[i, :]
                data_dict[(dist, n_val, r)] = x_n_r
                degree = 3
                x_vals = x_n_r.copy()
                x_plot = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                if num_knots > len(x_vals):
                    B = eval_spline_basis_equispaced_numeric(degree, np.min(x_vals), np.max(x_vals), num_knots, x_vals)['B']
                    B_plot = eval_spline_basis_equispaced_numeric(degree, np.min(x_vals), np.max(x_vals), num_knots, x_plot)['B']
                    v = max_ratio_cond(B)

                    curve = B@v
                    curve = curve/(np.max(curve) - np.min(curve))
                    for sigma in sigma_list:
                        y = (curve + sigma * eps_n_r/5.0) * scale
                        data[(f'{dist}_{n_val}', sigma)][r] = (x_n_r, y)
    return data

###

n_list = [10, 50, 100, 500]
distribution_list = ['uniform', 'uni_normal', 'bi_normal', 'exponential']
n_replications = 100
replication_list = list(range(n_replications))

###

gen_data_dict = False
data_dict_path = os.path.join("../data", "data_dictQ.pkl")
if gen_data_dict:
    data_dict = generate_data_dict(distribution_list, n_list, replication_list)
    with open(data_dict_path, "wb") as f:
        pickle.dump(data_dict, f)
with open(data_dict_path, "rb") as f:
    data_dict = pickle.load(f)

###

gen_curve_dict = False
num_knots = 5

curve_dict_path = os.path.join("../data", f"curve_dict_{num_knots}.pkl")
curve_plot_dict_path = os.path.join("../data", f"curve_plot_dict_{num_knots}.pkl")

if gen_curve_dict:
    curve_dict, curve_plot_dict, pen_dict = generate_curve_dict(distribution_list, n_list, replication_list, num_knots, data_dict, plot=True)
    with open(curve_dict_path, "wb") as f:
        pickle.dump(curve_dict, f)
    with open(curve_plot_dict_path, "wb") as f:
        pickle.dump(curve_plot_dict, f)

with open(curve_dict_path, "rb") as f:
    curve_dict = pickle.load(f)
with open(curve_plot_dict_path, "rb") as f:
    curve_plot_dict = pickle.load(f)

###

sigma_list = [0.1, 0.5, 1.0]
scale_list = [100, 200, 500, 1000, 2000]

gen_data = False
data_path = os.path.join("../data", f"dataQ_{num_knots}.pkl")

if gen_data:
    data = generate_data(scale_list, distribution_list, n_list, sigma_list, replication_list, num_knots, data_dict, curve_dict)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
with open(data_path, "rb") as f:
    data = pickle.load(f)
    
###

def plot_max_ratio_cond(data_dict, n_list, replication_list, distribution, num_knots):
    fig, ax = plt.subplots(figsize=(6, 4))

    for dist in distribution:
        for n in n_list:
            for r in replication_list:
                x = data_dict[(dist, n, r)]

                degree = 3
                x_vals = x.copy()
                x_plot = np.linspace(np.min(x_vals), np.max(x_vals), 100)

                B = eval_spline_basis_equispaced_numeric(
                    degree, np.min(x_vals), np.max(x_vals), num_knots, x_vals
                )['B']
                B_plot = eval_spline_basis_equispaced_numeric(
                    degree, np.min(x_vals), np.max(x_vals), num_knots, x_plot
                )['B']

                v, pen = max_ratio_cond(B)

                ax.plot(x_plot, B_plot @ v, label=f'{dist}, {n}, {r}')

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    return fig, ax

def plot_max_ratio_cond_implementation(data_dict, n_list, replication_list,
                                       distribution, num_knots):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Assign one color per distribution
    cmap = plt.get_cmap("tab10")
    colors = {dist: cmap(i) for i, dist in enumerate(distribution)}

    for dist in distribution:
        color = colors[dist]

        for n in n_list:
            for r in replication_list:
                x = data_dict[(dist, n, r)]

                degree = 3
                x_vals = x.copy()
                x_plot = np.linspace(np.min(x_vals), np.max(x_vals), 100)

                B = eval_spline_basis_equispaced_numeric(
                    degree, np.min(x_vals), np.max(x_vals), num_knots, x_vals
                )['B']
                B_plot = eval_spline_basis_equispaced_numeric(
                    degree, np.min(x_vals), np.max(x_vals), num_knots, x_plot
                )['B']

                v, pen = max_ratio_cond(B)

                # Only label the first line for each distribution
                label = dist if (n == n_list[0] and r == replication_list[0]) else None

                ax.plot(
                    x_plot,
                    B_plot @ v,
                    color=color,
                    alpha=0.5,
                    label=label,
                )

    ax.legend(title="Distribution", loc="upper left", bbox_to_anchor=(1.05, 1))

    return fig, ax
