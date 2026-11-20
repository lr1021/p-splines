from math import tau
import os

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.math import softplus
from scipy.linalg import null_space
from _utils._utils_spline import eval_spline_basis_equispaced_numeric, difference_matrix

import warnings
warnings.filterwarnings('ignore')

###

###
# print(os.listdir("data"))
try:
    dengue_data = pd.read_csv(os.path.join("data", "dengue.csv.gz"), usecols=['geocode', 'date', 'casos', 'epiweek', 'uf', 'macroregional_geocode', 'regional_geocode', 'uf_code'])
    map_regional_health = pd.read_csv(os.path.join("data", "map_regional_health.csv"))
    pop_data = pd.read_csv(os.path.join("data", "datasus_population_2001_2025.csv.gz"))
    map_regional_health = map_regional_health.merge(pop_data, how='left', on='geocode')
    mr_code_to_pop = map_regional_health.groupby(['macroregional_geocode', 'year'])['population'].sum().to_dict()
    macroregional_geocode = 3521
    year = 2025
    pop = mr_code_to_pop[(macroregional_geocode, year)]
except:
    print('Failed Data')
###

def stratified_shuffle(df, stratify_cols, random_cols, random_state=None):
    df = df.sort_values(stratify_cols+random_cols).reset_index(drop=True)
    
    unique_r = df[random_cols].drop_duplicates()
    n_unique_r = len(unique_r)
    unique_g = df[stratify_cols].drop_duplicates()
    n_unique_g = len(unique_g)

    new_df = pd.DataFrame()
    for g in range(n_unique_g):
        new_df = pd.concat([new_df, df[g*n_unique_r:(g+1)*n_unique_r].sample(frac=1).reset_index(drop=True)], ignore_index=True)
    df = new_df
    new_df = pd.DataFrame()
    for r in range(n_unique_r):
        new_df = pd.concat([new_df, df[r::n_unique_r]], ignore_index=True)
    df = new_df
    new_df = pd.DataFrame()
    for r in range(n_unique_r):
        new_df = pd.concat([new_df, df[r*n_unique_g:(r+1)*n_unique_g].sample(frac=1).reset_index(drop=True)], ignore_index=True)
    df = new_df
    return df

def build_model_ortho_diag_mult(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order, tau_val=1e-2):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        var_names = ['beta_0', 'sigma_2']

        # tau = pm.Gamma("tau", alpha=a, beta=b)#+1e-2 #, initval=1.0)
        num_splines = x_data.shape[-1]
        X_list = []
        X_plot_list = []
        f_sum = np.zeros(y_data.shape)
        beta_0_post = pm.Deterministic("beta_0_post", beta_0)

        for s in range(num_splines):
            tau_s = pm.Deterministic(f"tau_{s}", pt.as_tensor_variable(tau_val))
            var_names += [f'tau_{s}']
            if penalised:
                tau_p_s = pm.Gamma(f"tau_p_{s}", alpha=a, beta=b)#+1e-2#, initval=1.0)
                #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0)) 
                var_names += [f'tau_p_{s}']

            # Spline implementation
            x_data_s = x_data[:, s]
            eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data_s), np.max(x_data_s), n_internal_knots, x_data_s)
            B = eval_B['B']
            k = B.shape[1]

            x_plot = np.linspace(np.min(x_data_s), np.max(x_data_s), 100)
            eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data_s), np.max(x_data_s), n_internal_knots, x_plot)
            B_plot = eval_B_plot['B'][:, :]

            if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
                X = B.copy()
                X_plot = B_plot.copy()
                pre = pt.eye(k)

                model.add_coord(f"sample_w_dim_{s}", range(k))
                if not penalised:
                    sample_w = pm.Normal(f"sample_w_{s}",mu=0, tau=tau_s, shape=k, dims=f"sample_w_dim_{s}")
                else:
                    K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    sample_w = pm.Normal(f"sample_w_{s}", mu=0, tau=Q, shape=k, dims=f"sample_w_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U
                    pre = pre @ K_U

                true_sample_w = pre @ sample_w
                w_post = pm.Deterministic(f"w_post_{s}", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims=f"sample_w_dim_{s}")
                beta_0_post += true_sample_w.mean(axis=0)
                if 'beta_0' in var_names:
                    var_names[var_names.index('beta_0')] = 'beta_0_post'
                var_names += [f'w_post_{s}']
                f = pm.math.dot(X, sample_w)
            elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
                X = B.copy()
                X_plot = B_plot.copy()
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                X = U * S
                X_plot = X_plot @ Vt.T
                pre = Vt.T

                model.add_coord(f"sample_w_ortho_dim_{s}", range(k))
                if not penalised:
                    sample_w_ortho = pm.Normal(f"sample_w_ortho_{s}",mu=0, tau=tau_s, shape=k, dims=f"sample_w_ortho_dim_{s}")
                else:
                    K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    sample_w_ortho = pm.Normal(f"sample_w_ortho_{s}", mu=0, tau=Q, shape=k, dims=f"sample_w_ortho_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U
                    pre = pre @ K_U

                true_sample_w = pre @ sample_w_ortho
                w_post_ortho = pm.Deterministic(f"w_post_ortho_{s}", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims=f"sample_w_ortho_dim_{s}")
                beta_0_post += true_sample_w.mean(axis=0)
                if 'beta_0' in var_names:
                    var_names[var_names.index('beta_0')] = 'beta_0_post'
                var_names += [f'w_post_ortho_{s}']
                f = pm.math.dot(X, sample_w_ortho)
            elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
                X = B.copy()
                X_plot = B_plot.copy()
                X_plot = X_plot - X.mean(axis=0)
                X = X - X.mean(axis=0)  # centre the basis functions
                pre = pt.eye(k)

                model.add_coord(f"sample_w_dim_{s}", range(k))
                if not penalised:
                    sample_w = pm.Normal(f"sample_w_{s}",mu=0, tau=tau_s, shape=k, dims=f"sample_w_dim_{s}")
                else:
                    K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    sample_w = pm.Normal(f"sample_w_{s}", mu=0, tau=Q, shape=k, dims=f"sample_w_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U
                    pre = pre @ K_U

                true_sample_w = pre @ sample_w
                w_post = pm.Deterministic(f"w_post_{s}", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims=f"sample_w_dim_{s}")
                var_names += [f'w_post_{s}']
                f = pm.math.dot(X, sample_w)
            elif implementation=='ortho_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
                X = B.copy()
                X_plot = B_plot.copy()
                X_plot = X_plot - X.mean(axis=0)
                X = X - X.mean(axis=0)  # centre the basis functions

                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                X = U * S
                X_plot = X_plot @ Vt.T
                pre = Vt.T

                # return U, S, Vt, X, X_plot

                model.add_coord(f"sample_w_ortho_dim_{s}", range(k))
                if not penalised:
                    sample_w_ortho = pm.Normal(f"sample_w_ortho_{s}",mu=0, tau=tau_s, shape=k, dims=f"sample_w_ortho_dim_{s}")
                else:
                    K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    sample_w_ortho = pm.Normal(f"sample_w_ortho_{s}", mu=0, tau=Q, shape=k, dims=f"sample_w_ortho_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U
                    pre = pre @ K_U

                true_sample_w = pre @ sample_w_ortho
                w_post_ortho = pm.Deterministic(f"w_post_ortho_{s}", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims=f"sample_w_ortho_dim_{s}")
                var_names += [f'w_post_ortho_{s}']
                f = pm.math.dot(X, sample_w_ortho)
            elif implementation=='ortho_centring_T': # standard = full B splines, unidentifiable, use for posterior centring evaluation
                X = B.copy()
                X_plot = B_plot.copy()
                X_plot = X_plot - X.mean(axis=0)
                X = X - X.mean(axis=0)  # centre the basis functions

                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                X = U * S
                X_plot = X_plot @ Vt.T

                # return U, S, Vt, X, X_plot

                model.add_coord(f"w_oc_dim_{s}", range(k))
                if not penalised:
                    sample_w_ortho = pm.Normal(f"w_oc_{s}",mu=0, tau=tau_s, shape=k, dims=f"w_oc_dim_{s}")
                else:
                    K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    sample_w_ortho = pm.Normal(f"w_oc_{s}", mu=0, tau=Q, shape=k, dims=f"w_oc_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U
                # return K, K_V, K_U, Q, X, X_plot
                var_names += [f'w_oc_{s}']
                f = pm.math.dot(X, sample_w_ortho)
            elif implementation=='ortho_centring_T_exact': # standard = full B splines, unidentifiable, use for posterior centring evaluation
                X = B.copy()
                X_plot = B_plot.copy()
                X_plot = X_plot - X.mean(axis=0)
                X = X - X.mean(axis=0)  # centre the basis functions

                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                S[-1] = 0.0
                X = U * S
                X_plot = X_plot @ Vt.T

                # return U, S, Vt, X, X_plot

                model.add_coord(f"w_oc_dim_{s}", range(k))
                if not penalised:
                    sample_w_ortho = pm.Normal(f"w_oc_{s}",mu=0, tau=tau_s, shape=k, dims=f"w_oc_dim_{s}")
                else:
                    K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    sample_w_ortho = pm.Normal(f"w_oc_{s}", mu=0, tau=Q, shape=k, dims=f"w_oc_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U

                var_names += [f'w_oc_{s}']
                f = pm.math.dot(X, sample_w_ortho)
            elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
                X = B.copy()
                X_plot = B_plot.copy()
                X_plot = X_plot - X.mean(axis=0)
                X = X - X.mean(axis=0)  # centre the basis functions
                X = X[:, :-1]  # drop the last column to ensure identifiability
                X_plot = X_plot[:, :-1]

                model.add_coord(f"w_cd_dim_{s}", range(k-1))
                if not penalised:
                    w_cd = pm.Normal(f"w_cd_{s}",mu=0, tau=tau_s, shape=k-1, dims=f"w_cd_dim_{s}")
                else:
                    K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    w_cd = pm.Normal(f"w_cd_{s}", mu=0, tau=Q, shape=k-1, dims=f"w_cd_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U

                f = pm.math.dot(X, w_cd)
                var_names += [f'w_cd_{s}']
            elif implementation=='ortho_centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
                X = B.copy()
                X_plot = B_plot.copy()
                X_plot = X_plot - X.mean(axis=0)
                X = X - X.mean(axis=0)  # centre the basis functions
                X = X[:, :-1]  # drop the last column to ensure identifiability
                X_plot = X_plot[:, :-1]

                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                X = U * S
                X_plot = X_plot @ Vt.T

                model.add_coord(f"w_cd_ortho_dim_{s}", range(k-1))
                if not penalised:
                    w_cd_ortho = pm.Normal(f"w_cd_ortho_{s}",mu=0, tau=tau_s, shape=k-1, dims=f"w_cd_ortho_dim_{s}")
                else:
                    K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    w_cd_ortho = pm.Normal(f"w_cd_ortho_{s}", mu=0, tau=Q, shape=k-1, dims=f"w_cd_ortho_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U

                f = pm.math.dot(X, w_cd_ortho)
                var_names += [f'w_cd_ortho_{s}']
            elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
                X = B.copy()
                X_plot = B_plot.copy()
                Z = null_space(np.ones((X.shape[0], 1)).T @ X)
                X = X @ Z
                X_plot = X_plot @ Z

                model.add_coord(f"w_c_dim_{s}", range(k-1))
                if not penalised:
                    w_c = pm.Normal(f"w_c_{s}", mu=0, tau=tau_s, shape=k-1, dims=f"w_c_dim_{s}")
                else:
                    K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    w_c = pm.Normal(f"w_c_{s}", mu=0, tau=Q, shape=k-1, dims=f"w_c_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U

                f = pm.math.dot(X, w_c)
                var_names += [f'w_c_{s}']
            elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
                X = B.copy()
                X_plot = B_plot.copy()
                Z = null_space(np.ones((X.shape[0], 1)).T @ X)
                X = X @ Z
                X_plot = X_plot @ Z

                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                X = U * S
                X_plot = X_plot @ Vt.T

                model.add_coord(f"w_c_ortho_dim_{s}", range(k-1))
                if not penalised:
                    w_c_ortho = pm.Normal(f"w_c_ortho_{s}", mu=0, tau=tau_s, shape=k-1, dims=f"w_c_ortho_dim_{s}")
                else:
                    K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    w_c_ortho = pm.Normal(f"w_c_ortho_{s}", mu=0, tau=Q, shape=k-1, dims=f"w_c_ortho_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U

                f = pm.math.dot(X, w_c_ortho)
                var_names += [f'w_c_ortho_{s}']
            elif implementation=='spectral': # spectral = scheipl implementation, decompose improper prior
                X = B.copy()
                X_plot = B_plot.copy()
                
                # if not penalised this is the standard implementation, needs improper prior for decomposition
                if penalised:
                    # Penalised
                    r = k - order
                    K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                    K_pinv = np.linalg.pinv(K)
                    
                    # VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                    # X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                    V, U = np.linalg.eigh(X@K_pinv@X.T)
                    Vp = V[-r:]
                    Up = U[:, -r:]
                    # U0 = U[:, :-r]
                    Xp = Up*np.sqrt(Vp)

                    Phi_p = K_pinv @ X.T @ Up / np.sqrt(Vp)
                    X_plot_p = X_plot @ Phi_p

                    # Unpenalised
                    w_trend = np.arange(k)
                    w_trends = np.array([w_trend**d for d in range(1, order)]).T
                    w_trends = w_trends - w_trends.mean(axis=0)
                    X0 = X@w_trends
                    X0_plot = B_plot@w_trends
                    X0 = X0 / (np.max(X0_plot, axis=0) - np.min(X0_plot, axis=0))

                    X_plot_0 = X_plot@w_trends / (np.max(X0_plot, axis=0) - np.min(X0_plot, axis=0))

                    w_p = pm.Normal(f"w_p_{s}",mu=0, tau=tau_p_s, shape=r, dims=f"w_p_dim_{s}")
                    w_0 = pm.Normal(f"w_0_{s}", mu=0, tau=tau_s, shape=(order-1), dims=f"w_0_dim_{s}")
                    X = np.hstack([Xp, X0])
                    X_plot = np.hstack([X_plot_p, X_plot_0])
                    f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
                else:
                    raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
                var_names += [f'w_p_{s}', f'w_0_{s}'] 
            elif implementation=='ortho_spectral': # spectral = scheipl implementation, decompose improper prior
                X = B.copy()
                X_plot = B_plot.copy()
                
                # if not penalised this is the standard implementation, needs improper prior for decomposition
                if penalised:
                    # Unpenalised trends (can construct with spline weight trends)
                    w_trend = np.arange(k)
                    w_trends = np.array([w_trend**d for d in range(0, order)]).T
                    X0 = X@w_trends
                    X0_plot = X_plot@w_trends

                    for c in range(X0.shape[1]):
                        col = X0[:, c]
                        col_plot = X0_plot[:, c]

                        for c0 in range(c):
                            col0 = X0[:, c0]
                            col0_plot = X0_plot[:, c0]
                            col = col - (col0.T@col/np.linalg.norm(col0)**2)*col0
                            col_plot = col_plot - (col0.T@col/np.linalg.norm(col0)**2)*col0_plot

                        X0[:, c] = col
                        X0_plot[:, c] = col_plot

                    X0_ortho = X0.copy()
                    X0 = X0[:, 1:]
                    X0_ortho_plot = X0_plot.copy()
                    X0_plot = X0_plot[:, 1:]

                    X0 = X0 / (np.max(X0_plot, axis=0) - np.min(X0_plot, axis=0))
                    X0_plot = X0_plot / (np.max(X0_plot, axis=0) - np.min(X0_plot, axis=0))
                    #X0 = X0/np.linalg.norm(X0, axis=0)
                    #X0_plot = X0_plot/np.linalg.norm(X0_plot, axis=0)

                    # ortho
                    for c in range(X0_ortho.shape[1]):
                        col = X0_ortho[:, c].reshape(-1, 1)
                        col_plot = X0_ortho_plot[:, c].reshape(-1, 1)

                        X = X - (col.T@X/np.linalg.norm(col)**2).reshape(-1, 1).T*col
                        X_plot = X_plot - (col.T@X/np.linalg.norm(col)**2)*col_plot
                    # print(X.shape, np.linalg.matrix_rank(X))

                    # Penalised
                    r = k - order
                    K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                    K_pinv = np.linalg.pinv(K)

                    # VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                    # X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                    V, U = np.linalg.eigh(X@K_pinv@X.T)
                    Vp = V[-r:]
                    Up = U[:, -r:]
                    # U0 = U[:, :-r]
                    Xp = Up*np.sqrt(Vp)
                    
                    Phi_p = K_pinv @ X.T @ Up / np.sqrt(Vp)
                    X_plot_p = X_plot @ Phi_p
                    X = np.hstack([Xp, X0])
                    X_plot = np.hstack([X_plot_p, X0_plot])

                    w_p = pm.Normal(f"w_p_{s}",mu=0, tau=tau_p_s, shape=r, dims=f"w_p_dim_{s}")
                    w_0 = pm.Normal(f"w_0_{s}", mu=0, tau=tau_s, shape=(order-1), dims=f"w_0_dim_{s}")
                    f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
                else:
                    raise ValueError("Unpenalised Orthogonal Spectral = standard implementation, not implemented separately")
                var_names += [f'w_p_{s}', f'w_0_{s}']
            elif implementation=='w_centring': # svd = svd decomposition, decompose f_ construction
                X = B.copy()
                X_plot = B_plot.copy()

                A = np.eye(k)
                A[-1, :] = -1
                A[-1, -1] = 0
                U, S, Vt = np.linalg.svd(A)
                A_Q = U[:, :-1]
                X = X @ A_Q
                X_plot = X_plot @ A_Q

                # return U, S, Vt, Up, Sp, Vp, X, X_plot

                model.add_coord(f"w_wc_dim_{s}", range(k-1))
                if not penalised:
                    w_wc = pm.Normal(f"w_wc_{s}", mu=0, tau=tau_s, shape=k-1, dims=f"w_wc_dim_{s}")
                else:
                    K = A_Q.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ A_Q
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    w_wc = pm.Normal(f"w_wc_{s}", mu=0, tau=Q, shape=k-1, dims=f"w_wc_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U
                # return K, K_V, K_U, Q, X, X_plot
                f = pm.math.dot(X, w_wc)
                var_names += [f'w_wc_{s}']
            elif implementation=='ortho_w_centring': # svd = svd decomposition, decompose f_ construction
                X = B.copy()
                X_plot = B_plot.copy()

                A = np.eye(k)
                A[-1, :] = -1
                A[-1, -1] = 0
                U, S, Vt = np.linalg.svd(A)
                A_Q = U[:, :-1]
                X = X @ A_Q
                X_plot = X_plot @ A_Q

                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                X = U * S
                X_plot = X_plot @ Vt.T

                # return U, S, Vt, Up, Sp, Vp, X, X_plot

                model.add_coord(f"w_wc_ortho_dim_{s}", range(k-1))
                if not penalised:
                    w_wc_ortho = pm.Normal(f"w_wc_ortho_{s}", mu=0, tau=tau_s, shape=k-1, dims=f"w_wc_ortho_dim_{s}")
                else:
                    K = Vt @ A_Q.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ A_Q @ Vt.T
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    w_wc_ortho = pm.Normal(f"w_wc_ortho_{s}", mu=0, tau=Q, shape=k-1, dims=f"w_wc_ortho_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U
                # return K, K_V, K_U, Q, X, X_plot
                f = pm.math.dot(X, w_wc_ortho)
                var_names += [f'w_wc_ortho_{s}']
            elif implementation=='svd': # svd = svd decomposition, decompose f_ construction
                X = B.copy()
                X_plot = B_plot.copy()
                X_plot = X_plot - X.mean(axis=0)
                
                X_ = X - X.mean(axis=0)
                U, S, Vt = np.linalg.svd(X_, full_matrices=False)
                Up = U[:, :-1]
                Sp = S[:-1]
                Vp = Vt[:-1, :].T

                X_plot = X_plot @ Vp

                X = Up * Sp

                # return U, S, Vt, Up, Sp, Vp, X, X_plot

                model.add_coord(f"w_svd_dim_{s}", range(k-1))
                if not penalised:
                    w_svd = pm.Normal(f"w_svd_{s}", mu=0, tau=tau_s, shape=k-1, dims=f"w_svd_dim_{s}")
                else:
                    K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                    K_V, K_U = np.linalg.eigh(K)
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    w_svd = pm.Normal(f"w_svd_{s}", mu=0, tau=Q, shape=k-1, dims=f"w_svd_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U
                # return K, K_V, K_U, Q, X, X_plot
                f = pm.math.dot(X, w_svd)
                var_names += [f'w_svd_{s}']     
            elif implementation=='svd_aligned': # svd = svd decomposition, decompose f_ construction
                X = B.copy()
                X_plot = B_plot.copy()
                X_plot = X_plot - X.mean(axis=0)
                
                X_ = X - X.mean(axis=0)
                U, S, Vt = np.linalg.svd(X_, full_matrices=False)
                Up = U[:, :-1]
                Sp = S[:-1]
                Vp = Vt[:-1, :].T

                X_plot = X_plot @ Vp

                X = Up * Sp

                # return U, S, Vt, Up, Sp, Vp, X, X_plot

                model.add_coord(f"w_svd_dim_{s}", range(k-1))
                if not penalised:
                    raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
                else:
                    K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                    K_V, K_U = np.linalg.eigh(K)
                    e_k = np.zeros(k)
                    e_k[-1] = 1
                    l = np.argmax(np.abs(K_U.T @ e_k))
                    new_order = np.arange(k)
                    new_order[l:] = np.arange(l+1, k+1)
                    new_order[-1] = l
                    K_U = K_U[:-1, new_order[:-1]]
                    K_V = K_V[new_order[:-1]]
                    
                    Q = tau_p_s * K_V + tau_s
                    # Q = pt.maximum(Q, 1e-8)
                    w_svd = pm.Normal(f"w_svd_{s}", mu=0, tau=Q, shape=k-1, dims=f"w_svd_dim_{s}")
                    X = X @ K_U
                    X_plot = X_plot @ K_U
                # return K, K_V, K_U, Q, X, X_plot
                f = pm.math.dot(X, w_svd)
                var_names += [f'w_svd_{s}']
            else:
                raise ValueError("Implementation not recognised")  

            X_list.append(X)
            X_plot_list.append(X_plot)
            f_sum += f

        eta = beta_0 + f_sum
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X_list, X_plot_list, var_names

####
builder_dict = {'ortho_diag_mult': build_model_ortho_diag_mult}
####