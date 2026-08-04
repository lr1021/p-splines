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

def build_model(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        var_names = ['beta_0', 'sigma_2', 'tau']
        if not penalised:
            tau = pm.Gamma("tau", alpha=a, beta=b)
            #tau = pm.Deterministic("tau", pt.as_tensor_variable(1.0))
        else:
            tau = pm.Gamma("tau", alpha=a, beta=b)
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']
        
        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='standard': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            w = pm.Normal("w",mu=0, tau=tau, shape=k, dims="w_dim")
            if penalised:
                Dw = pt.dot(difference_matrix(k, order=order), w)
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                penalty = pt.dot(Dw, Dw)
                Q = tau_p * K + tau * np.eye(k)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * k * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w)
            var_names += ['w']

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            if penalised:
                Dw = pt.dot(difference_matrix(k, order=order)[:, :-1], w_cd)
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                penalty = pt.dot(Dw, Dw)
                Q = tau_p * K + tau * np.eye(k-1)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * (k-1) * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z
            ZTZ = Z.T @ Z
            w_c = pm.MvNormal("w_c", mu=np.zeros(Z.shape[1]), tau=tau*ZTZ, shape=Z.shape[1], dims="w_c_dim")

            model.add_coord("pen_w_dim", range(k))
            pen_w = pm.Deterministic("pen_w", pt.dot(Z, w_c), dims="pen_w_dim")
            if penalised:
                Dw = pt.dot(difference_matrix(k, order=order)[:, :], pen_w)
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                penalty = pt.dot(Dw, Dw)
                Q = tau_p * K + tau * np.eye(k)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * (k-1) * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w_c)
            var_names += ['w_c']

        elif implementation=='spectral': # spectral = scheipl implementation, decompose improper prior
            X = B.copy()
            X_plot = B_plot.copy()
            #X_plot = X_plot - X.mean(axis=0)
            # X = X - X.mean(axis=0)  # centre the basis functions
            
            # if not penalised this is the standard implementation, needs improper prior for decomposition
            if penalised:
                # Penalised
                r = k - order
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                K_pinv = np.linalg.pinv(K)
                
                VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K_pinv@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            
            var_names += ['w_p', 'w_0']
            
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

            w_svd = pm.Normal("w_svd", mu=0, tau=tau, shape=k-1, dims="w_svd_dim")
            if penalised:
                DVpw = pt.dot(pt.dot(difference_matrix(k, order=order), Vp), w_svd)
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                penalty = pt.dot(DVpw, DVpw)
                Q = tau_p * K + tau * np.eye(k-1)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * (k-1) * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X, X_plot, var_names

def build_model_MvN(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        var_names = ['beta_0', 'sigma_2', 'tau']
        if not penalised:
            tau = pm.Gamma("tau", alpha=a, beta=b)
            #tau = pm.Deterministic("tau", pt.as_tensor_variable(1.0))
        else:
            tau = pm.Gamma("tau", alpha=a, beta=b)
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='standard': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            if not penalised:
                model.add_coord("sample_w_dim", range(k))
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                model.add_coord("sample_w_dim", range(k))
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                Q = tau_p * K + tau * pt.eye(k)
                sample_w = pm.MvNormal("sample_w", mu=np.zeros(k), tau=Q, dims="sample_w_dim")
                #Dw = pt.dot(difference_matrix(k, order=order), w)
                #penalty = pt.dot(Dw, Dw)
            w_post = pm.Deterministic("w_post", sample_w - sample_w.mean(axis=0), dims="sample_w_dim")
            var_names += ['w_post']
            f = pm.math.dot(X, w_post)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            if penalised:
                Dw = pt.dot(difference_matrix(k, order=order)[:, :-1], w_cd)
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                penalty = pt.dot(Dw, Dw)
                Q = tau_p * K + tau * np.eye(k-1)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * (k-1) * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z
            ZTZ = Z.T @ Z
            w_c = pm.MvNormal("w_c", mu=np.zeros(Z.shape[1]), tau=tau*ZTZ, shape=Z.shape[1], dims="w_c_dim")

            model.add_coord("pen_w_dim", range(k))
            pen_w = pm.Deterministic("pen_w", pt.dot(Z, w_c), dims="pen_w_dim")
            if penalised:
                Dw = pt.dot(difference_matrix(k, order=order)[:, :], pen_w)
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                penalty = pt.dot(Dw, Dw)
                Q = tau_p * K + tau * np.eye(k)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * (k-1) * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='spectral': # spectral = scheipl implementation, decompose improper prior
            X = B.copy()
            X_plot = B_plot.copy()
            #X_plot = X_plot - X.mean(axis=0)
            # X = X - X.mean(axis=0)  # centre the basis functions
            
            # if not penalised this is the standard implementation, needs improper prior for decomposition
            if penalised:
                # Penalised
                r = k - order
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                K_pinv = np.linalg.pinv(K)
                
                VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K_pinv@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']
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

            w_svd = pm.Normal("w_svd", mu=0, tau=tau, shape=k-1, dims="w_svd_dim")
            if penalised:
                DVpw = pt.dot(pt.dot(difference_matrix(k, order=order), Vp), w_svd)
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                penalty = pt.dot(DVpw, DVpw)
                Q = tau_p * K + tau * np.eye(k-1)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * (k-1) * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X, X_plot, var_names

def build_model_ortho_MvN(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        var_names = ['beta_0', 'sigma_2', 'tau']
        if not penalised:
            tau = pm.Gamma("tau", alpha=a, beta=b)
            #tau = pm.Deterministic("tau", pt.as_tensor_variable(1.0))
        else:
            tau = pm.Gamma("tau", alpha=a, beta=b)
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            #X_plot = X_plot - X.mean(axis=0)
            #X = X - X.mean(axis=0)  # centre the basis functions

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                Q = tau_p * K + tau * pt.eye(k)
                sample_w = pm.MvNormal("sample_w", mu=np.zeros(k), tau=Q, dims="sample_w_dim")
                #Dw = pt.dot(difference_matrix(k, order=order), w)
                #penalty = pt.dot(Dw, Dw)
            w_post = pm.Deterministic("w_post", sample_w - sample_w.mean(axis=0), dims="sample_w_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            #X_plot = X_plot - X.mean(axis=0)
            #X = X - X.mean(axis=0)  # centre the basis functions
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                Q = tau_p * K + tau * pt.eye(k)
                sample_w_ortho = pm.MvNormal("sample_w_ortho", mu=np.zeros(k), tau=Q, dims="sample_w_ortho_dim")
                #Dw = pt.dot(difference_matrix(k, order=order), w)
                #penalty = pt.dot(Dw, Dw)
            sample_w = Vt.T @ sample_w_ortho
            w_post = sample_w - sample_w.mean(axis=0)
            w_post_ortho = pm.Deterministic("w_post_ortho", Vt @ w_post, dims="sample_w_ortho_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                Q = tau_p * K + tau * pt.eye(k)
                sample_w = pm.MvNormal("sample_w", mu=np.zeros(k), tau=Q, dims="sample_w_dim")
                #Dw = pt.dot(difference_matrix(k, order=order), w)
                #penalty = pt.dot(Dw, Dw)
            w_post = pm.Deterministic("w_post", sample_w - sample_w.mean(axis=0), dims="sample_w_dim")
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                Q = tau_p * K + tau * pt.eye(k)
                sample_w_ortho = pm.MvNormal("sample_w_ortho", mu=np.zeros(k), tau=Q, dims="sample_w_ortho_dim")
                #Dw = pt.dot(difference_matrix(k, order=order), w)
                #penalty = pt.dot(Dw, Dw)
            sample_w = Vt.T @ sample_w_ortho
            w_post = sample_w - sample_w.mean(axis=0)
            w_post_ortho = pm.Deterministic("w_post_ortho", Vt @ w_post, dims="sample_w_ortho_dim")
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            model.add_coord("w_cd_dim", range(k-1))
            if not penalised:
                w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            else:
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                Q = tau_p * K + tau * np.eye(k-1)
                w_cd = pm.MvNormal("w_cd", mu=np.zeros(k-1), tau=Q, dims="w_cd_dim")
            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']
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

            model.add_coord("w_cd_ortho_dim", range(k-1))
            if not penalised:
                w_cd_ortho = pm.Normal("w_cd_ortho",mu=0, tau=tau, shape=k-1, dims="w_cd_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                Q = tau_p * K + tau * np.eye(k-1)
                w_cd_ortho = pm.MvNormal("w_cd_ortho", mu=np.zeros(k-1), tau=Q, dims="w_cd_ortho_dim")
            f = pm.math.dot(X, w_cd_ortho)
            var_names += ['w_cd_ortho']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            model.add_coord("w_c_dim", range(k-1))
            if not penalised:
                w_c = pm.Normal("w_c", mu=0, tau=tau, shape=k-1, dims="w_c_dim")
            else:
                K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                Q = tau_p * K + tau * pt.eye(k-1)
                w_c = pm.MvNormal("w_c", mu=np.zeros(k-1), tau=Q, dims="w_c_dim")
            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("w_c_ortho_dim", range(k-1))
            if not penalised:
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=tau, shape=k-1, dims="w_c_ortho_dim")
            else:
                K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                Q = tau_p * K + tau * pt.eye(k-1)
                w_c_ortho = pm.MvNormal("w_c_ortho", mu=np.zeros(k-1), tau=Q, dims="w_c_ortho_dim")
            f = pm.math.dot(X, w_c_ortho)
            var_names += ['w_c_ortho']

        elif implementation=='spectral': # spectral = scheipl implementation, decompose improper prior
            X = B.copy()
            X_plot = B_plot.copy()
            #X_plot = X_plot - X.mean(axis=0)
            # X = X - X.mean(axis=0)  # centre the basis functions
            
            # if not penalised this is the standard implementation, needs improper prior for decomposition
            if penalised:
                # Penalised
                r = k - order
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                K_pinv = np.linalg.pinv(K)
                
                VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K_pinv@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

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

            model.add_coord("w_svd_dim", range(k-1))
            if not penalised:
                w_svd = pm.Normal("w_svd", mu=0, tau=tau, shape=k-1, dims="w_svd_dim")
            else:
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                Q = tau_p * K + tau * np.eye(k-1)
                w_svd = pm.MvNormal("w_svd", mu=np.zeros(k-1), tau=Q, dims="w_svd_dim")
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        else:
            raise ValueError("Implementation not recognised")


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X, X_plot, var_names

def build_model_ortho_diag(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order, tau_val=1e-4):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        var_names = ['beta_0', 'sigma_2', 'tau']

        # tau = pm.Gamma("tau", alpha=a, beta=b)#+1e-2 #, initval=1.0)
        tau = pm.Deterministic("tau", pt.as_tensor_variable(tau_val))
        if penalised:
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)#+1e-2#, initval=1.0)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T
            pre = Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            var_names += ['w_post']
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

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            var_names += ['w_post_ortho']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            var_names += ['w_oc']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            var_names += ['w_oc']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            model.add_coord("w_cd_dim", range(k-1))
            if not penalised:
                w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            else:
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd = pm.Normal("w_cd", mu=0, tau=Q, shape=k-1, dims="w_cd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']
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

            model.add_coord("w_cd_ortho_dim", range(k-1))
            if not penalised:
                w_cd_ortho = pm.Normal("w_cd_ortho",mu=0, tau=tau, shape=k-1, dims="w_cd_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd_ortho = pm.Normal("w_cd_ortho", mu=0, tau=Q, shape=k-1, dims="w_cd_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd_ortho)
            var_names += ['w_cd_ortho']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            model.add_coord("w_c_dim", range(k-1))
            if not penalised:
                w_c = pm.Normal("w_c", mu=0, tau=tau, shape=k-1, dims="w_c_dim")
            else:
                K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c = pm.Normal("w_c", mu=0, tau=Q, shape=k-1, dims="w_c_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("w_c_ortho_dim", range(k-1))
            if not penalised:
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=tau, shape=k-1, dims="w_c_ortho_dim")
            else:
                K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=Q, shape=k-1, dims="w_c_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c_ortho)
            var_names += ['w_c_ortho']

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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']
            
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Orthogonal Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

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

            model.add_coord("w_svd_dim", range(k-1))
            if not penalised:
                w_svd = pm.Normal("w_svd", mu=0, tau=tau, shape=k-1, dims="w_svd_dim")
            else:
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        
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

            model.add_coord("w_svd_dim", range(k-1))
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
                
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        else:
            raise ValueError("Implementation not recognised")
        


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X, X_plot, var_names

def build_model_ortho_diag_tau(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        var_names = ['beta_0', 'sigma_2', 'tau']

        tau = pm.Gamma("tau", alpha=a, beta=b)#+1e-2 #, initval=1.0)
        # tau = pm.Deterministic("tau", pt.as_tensor_variable(1e-6))
        if penalised:
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)#+1e-2#, initval=1.0)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T
            pre = Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            var_names += ['w_post']
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

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            var_names += ['w_post_ortho']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            var_names += ['w_oc']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            var_names += ['w_oc']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            model.add_coord("w_cd_dim", range(k-1))
            if not penalised:
                w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            else:
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd = pm.Normal("w_cd", mu=0, tau=Q, shape=k-1, dims="w_cd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']
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

            model.add_coord("w_cd_ortho_dim", range(k-1))
            if not penalised:
                w_cd_ortho = pm.Normal("w_cd_ortho",mu=0, tau=tau, shape=k-1, dims="w_cd_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd_ortho = pm.Normal("w_cd_ortho", mu=0, tau=Q, shape=k-1, dims="w_cd_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd_ortho)
            var_names += ['w_cd_ortho']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            model.add_coord("w_c_dim", range(k-1))
            if not penalised:
                w_c = pm.Normal("w_c", mu=0, tau=tau, shape=k-1, dims="w_c_dim")
            else:
                K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c = pm.Normal("w_c", mu=0, tau=Q, shape=k-1, dims="w_c_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("w_c_ortho_dim", range(k-1))
            if not penalised:
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=tau, shape=k-1, dims="w_c_ortho_dim")
            else:
                K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=Q, shape=k-1, dims="w_c_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c_ortho)
            var_names += ['w_c_ortho']

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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']
            
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Orthogonal Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

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

            model.add_coord("w_svd_dim", range(k-1))
            if not penalised:
                w_svd = pm.Normal("w_svd", mu=0, tau=tau, shape=k-1, dims="w_svd_dim")
            else:
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        
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

            model.add_coord("w_svd_dim", range(k-1))
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
                
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        else:
            raise ValueError("Implementation not recognised")
        


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X, X_plot, var_names

def build_model_ortho_diag_inS(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        var_names = ['beta_0', 'sigma_2', 'tau']

        tau = pm.Gamma("tau", alpha=a, beta=b)+1e-2 #, initval=1.0)
        #tau = pm.Deterministic("tau", pt.as_tensor_variable(1.0))
        if penalised:
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)+1e-2#, initval=1.0)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")

                X = X @ K_U
                pre = pre @ K_U

            X_plot = X_plot @ pre

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U @ np.diag(S)
            pre = Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                X = U
                pre = pre @ np.diag(1/S)
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau/(S**2), shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")

                X = X @ K_U
                pre = pre @ K_U

            X_plot = X_plot @ pre

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")

                X = X @ K_U
                pre = pre @ K_U

            X_plot = X_plot @ pre

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U @ np.diag(S)
            pre = Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                X = U
                pre = pre @ np.diag(1/S)
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau/(S**2), shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")

                X = X @ K_U
                pre = pre @ K_U

            X_plot = X_plot @ pre

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            model.add_coord("w_cd_dim", range(k-1))
            if not penalised:
                w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            else:
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd = pm.Normal("w_cd", mu=0, tau=Q, shape=k-1, dims="w_cd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']
        elif implementation=='ortho_centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            pre = Vt.T

            model.add_coord("w_cd_ortho_dim", range(k-1))
            if not penalised:
                X = U
                pre = pre @ np.diag(1/S)
                w_cd_ortho = pm.Normal("w_cd_ortho",mu=0, tau=tau/(S**2), shape=k-1, dims="w_cd_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd_ortho = pm.Normal("w_cd_ortho", mu=0, tau=Q, shape=k-1, dims="w_cd_ortho_dim")

                X = X @ K_U
                pre = pre @ K_U
            
            X_plot = X_plot @ pre

            f = pm.math.dot(X, w_cd_ortho)
            var_names += ['w_cd_ortho']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            model.add_coord("w_c_dim", range(k-1))
            if not penalised:
                w_c = pm.Normal("w_c", mu=0, tau=tau, shape=k-1, dims="w_c_dim")
            else:
                K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c = pm.Normal("w_c", mu=0, tau=Q, shape=k-1, dims="w_c_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            pre = Vt.T

            model.add_coord("w_c_ortho_dim", range(k-1))
            if not penalised:
                X = U
                pre = pre @ np.diag(1/S)
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=tau/(S**2), shape=k-1, dims="w_c_ortho_dim")
            else:
                K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=Q, shape=k-1, dims="w_c_ortho_dim")
                X = X @ K_U
                pre = pre @ K_U

            X_plot = X_plot @ pre

            f = pm.math.dot(X, w_c_ortho)
            var_names += ['w_c_ortho']

        elif implementation=='spectral': # spectral = scheipl implementation, decompose improper prior
            X = B.copy()
            X_plot = B_plot.copy()
            
            # if not penalised this is the standard implementation, needs improper prior for decomposition
            if penalised:
                # Penalised
                r = k - order
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                K_pinv = np.linalg.pinv(K)
                
                VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K_pinv@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']
            
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
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K_pinv@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
                Xp = Up*np.sqrt(Vp)
                
                Phi_p = K_pinv @ X.T @ Up / np.sqrt(Vp)
                X_plot_p = X_plot @ Phi_p
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X0_plot])

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Orthogonal Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

        elif implementation=='svd': # svd = svd decomposition, decompose f_ construction
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            
            X_ = X - X.mean(axis=0)
            U, S, Vt = np.linalg.svd(X_, full_matrices=False)
            Up = U[:, :-1]
            Sp = S[:-1]
            Vp = Vt[:-1, :].T

            X = Up * Sp
            pre = Vp

            model.add_coord("w_svd_dim", range(k-1))
            if not penalised:
                X = Up
                pre = pre @ np.diag(1/Sp)
                w_svd = pm.Normal("w_svd", mu=0, tau=tau/(Sp**2), shape=k-1, dims="w_svd_dim")
            else:
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                pre = pre @ K_U

            X_plot = X_plot @ pre

            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        else:
            raise ValueError("Implementation not recognised")


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X, X_plot, var_names

def build_model_ortho_diag_sigma(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        #var_names = ['beta_0', 'sigma_2', 'tau']
        var_names = ['beta_0', 'sigma_2', 'sigma']

        # sigma = pm.HalfStudentT("sigma", nu=3, sigma=1)
        sigma = pm.InverseGamma('sigma', alpha=a, beta=b)
        tau = pm.Deterministic("tau", 1 / sigma**2)
        #tau = pm.Gamma("tau", alpha=a, beta=b)+1e-6#, initval=1.0)
        #tau = pm.Deterministic("tau", pt.as_tensor_variable(1.0))
        if penalised:
            # sigma_p = pm.HalfStudentT("sigma_p", nu=3, sigma=1)
            sigma_p = pm.InverseGamma('sigma_p', alpha=a, beta=b)
            tau_p = pm.Deterministic("tau_p", 1 / sigma_p**2)
            #tau_p = pm.Gamma("tau_p", alpha=a, beta=b)+1e-6#, initval=1.0)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            #var_names += ['tau_p']
            var_names += ['sigma_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, sigma=sigma, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T
            pre = Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, sigma=sigma, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, sigma=sigma, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            var_names += ['w_post']
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

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, sigma=sigma, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            model.add_coord("w_cd_dim", range(k-1))
            if not penalised:
                w_cd = pm.Normal("w_cd",mu=0, sigma=sigma, shape=k-1, dims="w_cd_dim")
            else:
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd = pm.Normal("w_cd", mu=0, tau=Q, shape=k-1, dims="w_cd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']
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

            model.add_coord("w_cd_ortho_dim", range(k-1))
            if not penalised:
                w_cd_ortho = pm.Normal("w_cd_ortho",mu=0, sigma=sigma, shape=k-1, dims="w_cd_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd_ortho = pm.Normal("w_cd_ortho", mu=0, tau=Q, shape=k-1, dims="w_cd_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd_ortho)
            var_names += ['w_cd_ortho']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            model.add_coord("w_c_dim", range(k-1))
            if not penalised:
                w_c = pm.Normal("w_c", mu=0, sigma=sigma, shape=k-1, dims="w_c_dim")
            else:
                K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c = pm.Normal("w_c", mu=0, tau=Q, shape=k-1, dims="w_c_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("w_c_ortho_dim", range(k-1))
            if not penalised:
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, sigma=sigma, shape=k-1, dims="w_c_ortho_dim")
            else:
                K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=Q, shape=k-1, dims="w_c_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c_ortho)
            var_names += ['w_c_ortho']

        elif implementation=='spectral': # spectral = scheipl implementation, decompose improper prior
            X = B.copy()
            X_plot = B_plot.copy()
            
            # if not penalised this is the standard implementation, needs improper prior for decomposition
            if penalised:
                # Penalised
                r = k - order
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                K_pinv = np.linalg.pinv(K)
                
                VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K_pinv@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
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

                w_p = pm.Normal("w_p",mu=0, sigma=sigma_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, sigma=sigma, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']
            
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
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K_pinv@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
                Xp = Up*np.sqrt(Vp)
                
                Phi_p = K_pinv @ X.T @ Up / np.sqrt(Vp)
                X_plot_p = X_plot @ Phi_p
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X0_plot])

                w_p = pm.Normal("w_p",mu=0, sigma=sigma_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, sigma=sigma, shape=(order-1), dims="w_0_dim")
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Orthogonal Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

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

            model.add_coord("w_svd_dim", range(k-1))
            if not penalised:
                w_svd = pm.Normal("w_svd", mu=0, sigma=sigma, shape=k-1, dims="w_svd_dim")
            else:
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        else:
            raise ValueError("Implementation not recognised")


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X, X_plot, var_names

def build_model_ortho_diag_c(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        var_names = ['beta_0', 'sigma_2', 'tau']

        tau = pm.Gamma("tau", alpha=a, beta=b)+1e-8 #, initval=1.0)
        #tau = pm.Deterministic("tau", pt.as_tensor_variable(1.0))
        if penalised:
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)+1e-8 #, initval=1.0)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 100)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)

                # Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                # sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")

                sqrt_Q_inv = 1/pt.sqrt(tau_p * K_V + tau)
                pre_sample_w = pm.Normal("pre_sample_w", mu=0, tau=1, shape=k, dims="sample_w_dim")
                sample_w = pm.Deterministic("sample_w", pre_sample_w * sqrt_Q_inv)

                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T
            pre = Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)

                # Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                # sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                
                sqrt_Q_inv = 1/pt.sqrt(tau_p * K_V + tau)
                pre_sample_w = pm.Normal("pre_sample_w", mu=0, tau=1, shape=k, dims="sample_w_ortho_dim")
                sample_w_ortho = pm.Deterministic("sample_w_ortho", pre_sample_w * sqrt_Q_inv)

                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)

                # Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                # sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")

                sqrt_Q_inv = 1/pt.sqrt(tau_p * K_V + tau)
                pre_sample_w = pm.Normal("pre_sample_w", mu=0, tau=1, shape=k, dims="sample_w_dim")
                sample_w = pm.Deterministic("sample_w", pre_sample_w * sqrt_Q_inv)

                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            var_names += ['w_post']
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

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)

                # Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                # sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                
                sqrt_Q_inv = 1/pt.sqrt(tau_p * K_V + tau)
                pre_sample_w = pm.Normal("pre_sample_w", mu=0, tau=1, shape=k, dims="sample_w_ortho_dim")
                sample_w_ortho = pm.Deterministic("sample_w_ortho", pre_sample_w * sqrt_Q_inv)
                
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            model.add_coord("w_cd_dim", range(k-1))
            if not penalised:
                w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            else:
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                K_V, K_U = np.linalg.eigh(K)

                # Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                # w_cd = pm.Normal("w_cd", mu=0, tau=Q, shape=k-1, dims="w_cd_dim")

                sqrt_Q_inv = 1/pt.sqrt(tau_p * K_V + tau)
                pre_sample_w = pm.Normal("pre_sample_w", mu=0, tau=1, shape=k-1, dims="w_cd_dim")
                w_cd = pm.Deterministic("w_cd", pre_sample_w * sqrt_Q_inv)

                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']
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

            model.add_coord("w_cd_ortho_dim", range(k-1))
            if not penalised:
                w_cd_ortho = pm.Normal("w_cd_ortho",mu=0, tau=tau, shape=k-1, dims="w_cd_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                K_V, K_U = np.linalg.eigh(K)

                # Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                # w_cd_ortho = pm.Normal("w_cd_ortho", mu=0, tau=Q, shape=k-1, dims="w_cd_ortho_dim")
                
                sqrt_Q_inv = 1/pt.sqrt(tau_p * K_V + tau)
                pre_sample_w = pm.Normal("pre_sample_w", mu=0, tau=1, shape=k-1, dims="w_cd_ortho_dim")
                w_cd_ortho = pm.Deterministic("w_cd_ortho", pre_sample_w * sqrt_Q_inv)
                
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd_ortho)
            var_names += ['w_cd_ortho']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            model.add_coord("w_c_dim", range(k-1))
            if not penalised:
                w_c = pm.Normal("w_c", mu=0, tau=tau, shape=k-1, dims="w_c_dim")
            else:
                K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                K_V, K_U = np.linalg.eigh(K)

                # Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                # w_c = pm.Normal("w_c", mu=0, tau=Q, shape=k-1, dims="w_c_dim")

                sqrt_Q_inv = 1/pt.sqrt(tau_p * K_V + tau)
                pre_sample_w = pm.Normal("pre_sample_w", mu=0, tau=1, shape=k-1, dims="w_c_dim")
                w_c = pm.Deterministic("w_c", pre_sample_w * sqrt_Q_inv)

                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("w_c_ortho_dim", range(k-1))
            if not penalised:
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=tau, shape=k-1, dims="w_c_ortho_dim")
            else:
                K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                K_V, K_U = np.linalg.eigh(K)

                # Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                # w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=Q, shape=k-1, dims="w_c_ortho_dim")

                sqrt_Q_inv = 1/pt.sqrt(tau_p * K_V + tau)
                pre_sample_w = pm.Normal("pre_sample_w", mu=0, tau=1, shape=k-1, dims="w_c_ortho_dim")
                w_c_ortho = pm.Deterministic("w_c_ortho", pre_sample_w * sqrt_Q_inv)

                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c_ortho)
            var_names += ['w_c_ortho']

        elif implementation=='spectral': # spectral = scheipl implementation, decompose improper prior
            X = B.copy()
            X_plot = B_plot.copy()
            
            # if not penalised this is the standard implementation, needs improper prior for decomposition
            if penalised:
                # Penalised
                r = k - order
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                K_pinv = np.linalg.pinv(K)
                
                VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K_pinv@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

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
                print(X.shape, np.linalg.matrix_rank(X))

                # Penalised
                r = k - order
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                K_pinv = np.linalg.pinv(K)

                VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K_pinv@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
                Xp = Up*np.sqrt(Vp)

                X = np.hstack([Xp, X0])

                Phi_p = K_pinv @ X.T @ Up / np.sqrt(Vp)
                X_plot_p = X_plot @ Phi_p
                X_plot = np.hstack([X_plot_p, X_plot_0])

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Orthogonal Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

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

            model.add_coord("w_svd_dim", range(k-1))
            if not penalised:
                w_svd = pm.Normal("w_svd", mu=0, tau=tau, shape=k-1, dims="w_svd_dim")
            else:
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                K_V, K_U = np.linalg.eigh(K)

                # Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                # w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")

                sqrt_Q_inv = 1/pt.sqrt(tau_p * K_V + tau)
                pre_sample_w = pm.Normal("pre_sample_w", mu=0, tau=1, shape=k-1, dims="w_svd_dim")
                w_svd = pm.Deterministic("w_svd", pre_sample_w * sqrt_Q_inv)

                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        else:
            raise ValueError("Implementation not recognised")


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X, X_plot, var_names


####


def build_nb_model_ortho_diag(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        # sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)
        alpha = pm.HalfNormal("alpha", sigma=5)

        var_names = ['beta_0', 'alpha', 'tau']

        # tau = pm.Gamma("tau", alpha=a, beta=b)#+1e-2 #, initval=1.0)
        tau = pm.Deterministic("tau", pt.as_tensor_variable(1e-2))
        if penalised:
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)#+1e-2#, initval=1.0)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T
            pre = Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            var_names += ['w_post']
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

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            var_names += ['w_post_ortho']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            var_names += ['w_oc']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            var_names += ['w_oc']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            model.add_coord("w_cd_dim", range(k-1))
            if not penalised:
                w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            else:
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd = pm.Normal("w_cd", mu=0, tau=Q, shape=k-1, dims="w_cd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']
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

            model.add_coord("w_cd_ortho_dim", range(k-1))
            if not penalised:
                w_cd_ortho = pm.Normal("w_cd_ortho",mu=0, tau=tau, shape=k-1, dims="w_cd_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd_ortho = pm.Normal("w_cd_ortho", mu=0, tau=Q, shape=k-1, dims="w_cd_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd_ortho)
            var_names += ['w_cd_ortho']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            model.add_coord("w_c_dim", range(k-1))
            if not penalised:
                w_c = pm.Normal("w_c", mu=0, tau=tau, shape=k-1, dims="w_c_dim")
            else:
                K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c = pm.Normal("w_c", mu=0, tau=Q, shape=k-1, dims="w_c_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("w_c_ortho_dim", range(k-1))
            if not penalised:
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=tau, shape=k-1, dims="w_c_ortho_dim")
            else:
                K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=Q, shape=k-1, dims="w_c_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c_ortho)
            var_names += ['w_c_ortho']

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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']
            
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Orthogonal Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

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

            model.add_coord("w_svd_dim", range(k-1))
            if not penalised:
                w_svd = pm.Normal("w_svd", mu=0, tau=tau, shape=k-1, dims="w_svd_dim")
            else:
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        
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

            model.add_coord("w_svd_dim", range(k-1))
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
                
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        else:
            raise ValueError("Implementation not recognised")
        


        eta = beta_0 + f
        # Likelihood
        # y_obs = pm.NegativeBinomial('y_obs', mu=pt.exp(eta), alpha=alpha, observed=y_data)
        y_obs = pm.NegativeBinomial('y_obs', mu=softplus(eta), alpha=alpha, observed=y_data)
    
    return model, X, X_plot, var_names

def build_popnb_model_ortho_diag(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    pop = y_data[1]
    y_data = y_data[0]
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        # sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)
        alpha = pm.HalfNormal("alpha", sigma=5)

        var_names = ['beta_0', 'alpha', 'tau']

        # tau = pm.Gamma("tau", alpha=a, beta=b)#+1e-2 #, initval=1.0)
        tau = pm.Deterministic("tau", pt.as_tensor_variable(1e1))
        if penalised:
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)#+1e-2#, initval=1.0)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T
            pre = Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            var_names += ['w_post']
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

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            var_names += ['w_post_ortho']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            var_names += ['w_oc']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            var_names += ['w_oc']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            model.add_coord("w_cd_dim", range(k-1))
            if not penalised:
                w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            else:
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd = pm.Normal("w_cd", mu=0, tau=Q, shape=k-1, dims="w_cd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']
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

            model.add_coord("w_cd_ortho_dim", range(k-1))
            if not penalised:
                w_cd_ortho = pm.Normal("w_cd_ortho",mu=0, tau=tau, shape=k-1, dims="w_cd_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd_ortho = pm.Normal("w_cd_ortho", mu=0, tau=Q, shape=k-1, dims="w_cd_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd_ortho)
            var_names += ['w_cd_ortho']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            model.add_coord("w_c_dim", range(k-1))
            if not penalised:
                w_c = pm.Normal("w_c", mu=0, tau=tau, shape=k-1, dims="w_c_dim")
            else:
                K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c = pm.Normal("w_c", mu=0, tau=Q, shape=k-1, dims="w_c_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("w_c_ortho_dim", range(k-1))
            if not penalised:
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=tau, shape=k-1, dims="w_c_ortho_dim")
            else:
                K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=Q, shape=k-1, dims="w_c_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c_ortho)
            var_names += ['w_c_ortho']

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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']
            
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Orthogonal Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

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

            model.add_coord("w_svd_dim", range(k-1))
            if not penalised:
                w_svd = pm.Normal("w_svd", mu=0, tau=tau, shape=k-1, dims="w_svd_dim")
            else:
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        
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

            model.add_coord("w_svd_dim", range(k-1))
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
                
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        else:
            raise ValueError("Implementation not recognised")
        


        eta = beta_0 + f + pt.log(pop)
        # Likelihood
        y_obs = pm.NegativeBinomial('y_obs', mu=pt.exp(eta), alpha=alpha, observed=y_data)
        # y_obs = pm.NegativeBinomial('y_obs', mu=softplus(eta), alpha=alpha, observed=y_data)
    
    return model, X, X_plot, var_names


def build_p_model_ortho_diag(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):

    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        # sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)
        # alpha = pm.HalfNormal("alpha", sigma=5)

        var_names = ['beta_0', 'tau']

        # tau = pm.Gamma("tau", alpha=a, beta=b)#+1e-2 #, initval=1.0)
        tau = pm.Deterministic("tau", pt.as_tensor_variable(1e-6))
        if penalised:
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)#+1e-2#, initval=1.0)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))
            var_names += ['tau_p']

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post']
            f = pm.math.dot(X, sample_w)
        elif implementation=='ortho_post_centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T
            pre = Vt.T

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            beta_0_post = pm.Deterministic("beta_0_post", beta_0 + true_sample_w.mean(axis=0))
            var_names[var_names.index('beta_0')] = 'beta_0_post'
            var_names += ['w_post_ortho']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            pre = pt.eye(k)

            model.add_coord("sample_w_dim", range(k))
            if not penalised:
                sample_w = pm.Normal("sample_w",mu=0, tau=tau, shape=k, dims="sample_w_dim")
            else:
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w = pm.Normal("sample_w", mu=0, tau=Q, shape=k, dims="sample_w_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w
            w_post = pm.Deterministic("w_post", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_dim")
            var_names += ['w_post']
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

            model.add_coord("sample_w_ortho_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("sample_w_ortho",mu=0, tau=tau, shape=k, dims="sample_w_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("sample_w_ortho", mu=0, tau=Q, shape=k, dims="sample_w_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
                pre = pre @ K_U

            true_sample_w = pre @ sample_w_ortho
            w_post_ortho = pm.Deterministic("w_post_ortho", pre.T @ (true_sample_w - true_sample_w.mean(axis=0)), dims="sample_w_ortho_dim")
            var_names += ['w_post_ortho']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            var_names += ['w_oc']
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

            model.add_coord("w_oc_dim", range(k))
            if not penalised:
                sample_w_ortho = pm.Normal("w_oc",mu=0, tau=tau, shape=k, dims="w_oc_dim")
            else:
                K = Vt @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                sample_w_ortho = pm.Normal("w_oc", mu=0, tau=Q, shape=k, dims="w_oc_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            var_names += ['w_oc']
            f = pm.math.dot(X, sample_w_ortho)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            model.add_coord("w_cd_dim", range(k-1))
            if not penalised:
                w_cd = pm.Normal("w_cd",mu=0, tau=tau, shape=k-1, dims="w_cd_dim")
            else:
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd = pm.Normal("w_cd", mu=0, tau=Q, shape=k-1, dims="w_cd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd)
            var_names += ['w_cd']
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

            model.add_coord("w_cd_ortho_dim", range(k-1))
            if not penalised:
                w_cd_ortho = pm.Normal("w_cd_ortho",mu=0, tau=tau, shape=k-1, dims="w_cd_ortho_dim")
            else:
                K = Vt @ difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1] @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_cd_ortho = pm.Normal("w_cd_ortho", mu=0, tau=Q, shape=k-1, dims="w_cd_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_cd_ortho)
            var_names += ['w_cd_ortho']

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            model.add_coord("w_c_dim", range(k-1))
            if not penalised:
                w_c = pm.Normal("w_c", mu=0, tau=tau, shape=k-1, dims="w_c_dim")
            else:
                K = Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c = pm.Normal("w_c", mu=0, tau=Q, shape=k-1, dims="w_c_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c)
            var_names += ['w_c']
        elif implementation=='ortho_conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            X = X @ Z
            X_plot = X_plot @ Z

            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U * S
            X_plot = X_plot @ Vt.T

            model.add_coord("w_c_ortho_dim", range(k-1))
            if not penalised:
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=tau, shape=k-1, dims="w_c_ortho_dim")
            else:
                K = Vt @ Z.T @difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :] @ Z @ Vt.T
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_c_ortho = pm.Normal("w_c_ortho", mu=0, tau=Q, shape=k-1, dims="w_c_ortho_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U

            f = pm.math.dot(X, w_c_ortho)
            var_names += ['w_c_ortho']

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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']
            
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

                w_p = pm.Normal("w_p",mu=0, tau=tau_p, shape=r, dims="w_p_dim")
                w_0 = pm.Normal("w_0", mu=0, tau=tau, shape=(order-1), dims="w_0_dim")
                f = pm.math.dot(Xp, w_p) + pm.math.dot(X0, w_0)
            else:
                raise ValueError("Unpenalised Orthogonal Spectral = standard implementation, not implemented separately")
            var_names += ['w_p', 'w_0']

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

            model.add_coord("w_svd_dim", range(k-1))
            if not penalised:
                w_svd = pm.Normal("w_svd", mu=0, tau=tau, shape=k-1, dims="w_svd_dim")
            else:
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                K_V, K_U = np.linalg.eigh(K)
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        
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

            model.add_coord("w_svd_dim", range(k-1))
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
                
                Q = tau_p * K_V + tau
                # Q = pt.maximum(Q, 1e-8)
                w_svd = pm.Normal("w_svd", mu=0, tau=Q, shape=k-1, dims="w_svd_dim")
                X = X @ K_U
                X_plot = X_plot @ K_U
            # return K, K_V, K_U, Q, X, X_plot
            f = pm.math.dot(X, w_svd)
            var_names += ['w_svd']
        else:
            raise ValueError("Implementation not recognised")
        


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Poisson("y_obs", mu=pt.exp(pt.log(pop) + eta), observed=y_data)
    return model, X, X_plot, var_names


####

builder_dict = {'0': build_model,
                'MvN': build_model_MvN,
                'ortho_MvN': build_model_ortho_MvN,
                'ortho_diag': build_model_ortho_diag,
                'ortho_diag_tau': build_model_ortho_diag_tau,
                'ortho_diag_inS': build_model_ortho_diag_inS,
                'ortho_diag_sigma': build_model_ortho_diag_sigma,
                'ortho_diag_c': build_model_ortho_diag_c,
                'nb_ortho_diag': build_nb_model_ortho_diag,
                'popnb_ortho_diag': build_popnb_model_ortho_diag,
                'p_ortho_diag': build_p_model_ortho_diag}








#####################
def build_model_0(x_data, y_data, a, b, spline_degree, n_internal_knots, implementation, penalised, order):
    model = pm.Model()
    with model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=100)
        sigma_2 = pm.InverseGamma('sigma_2', alpha=a, beta=b)

        if not penalised:
            tau = pm.Gamma("tau", alpha=a, beta=b)
            #tau = pm.Deterministic("tau", pt.as_tensor_variable(1.0))
        else:
            tau = pm.Gamma("tau", alpha=a, beta=b)
            tau_p = pm.Gamma("tau_p", alpha=a, beta=b)
            #tau_p = pm.Deterministic("tau_p", pt.as_tensor_variable(1.0))

        # Spline implementation
        eval_B = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_data)
        B = eval_B['B']
        k = B.shape[1]

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
        eval_B_plot = eval_spline_basis_equispaced_numeric(spline_degree, np.min(x_data), np.max(x_data), n_internal_knots, x_plot)
        B_plot = eval_B_plot['B'][:, :]

        if implementation=='standard': # standard = full B splines, unidentifiable, use for posterior centring evaluation
            X = B.copy()
            X_plot = B_plot.copy()
            w = pm.Normal("w",mu=0, tau=tau, shape=k, dims="w_dim")
            if penalised:
                Dw = pt.dot(difference_matrix(k, order=order), w)
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                penalty = pt.dot(Dw, Dw)
                Q = tau_p * K + tau * np.eye(k)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * k * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w)

        elif implementation=='centring+dropping': # centring+dropping = Gressani implementation, adjusted penalty
            X = B.copy()
            X_plot = B_plot.copy()
            X_plot = X_plot - X.mean(axis=0)
            X = X - X.mean(axis=0)  # centre the basis functions
            X = X[:, :-1]  # drop the last column to ensure identifiability
            X_plot = X_plot[:, :-1]

            w = pm.Normal("w",mu=0, tau=tau, shape=k-1, dims="w_dim")
            if penalised:
                Dw = pt.dot(difference_matrix(k, order=order)[:, :-1], w)
                K = difference_matrix(k, order=order)[:, :-1].T @ difference_matrix(k, order=order)[:, :-1]
                penalty = pt.dot(Dw, Dw)
                Q = tau_p * K + tau * np.eye(k-1)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * (k-1) * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w)

        elif implementation=='conditioning': # conditioning = Chen implementation, map to constrained conditioned space
            X = B.copy()
            X_plot = B_plot.copy()
            Z = null_space(np.ones((X.shape[0], 1)).T @ X)
            ZTZ = Z.T @ Z
            theta = pm.MvNormal("theta", mu=np.zeros(Z.shape[1]), tau=tau*ZTZ, shape=Z.shape[1], dims="theta_dim")

            model.add_coord("w_dim", range(k))
            w = pm.Deterministic("w", pt.dot(Z,theta), dims="w_dim")
            if penalised:
                Dw = pt.dot(difference_matrix(k, order=order)[:, :], w)
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                penalty = pt.dot(Dw, Dw)
                Q = tau_p * K + tau * np.eye(k)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * (k-1) * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w)
        
        elif implementation=='spectral': # spectral = scheipl implementation, decompose improper prior
            X = B.copy()
            X_plot = B_plot.copy()
            #X_plot = X_plot - X.mean(axis=0)
            #X = X - X.mean(axis=0)  # centre the basis functions
            
            # if not penalised this is the standard implementation, needs improper prior for decomposition
            if penalised:
                # Penalised
                r = k - order
                K = difference_matrix(k, order=order)[:, :].T @ difference_matrix(k, order=order)[:, :]
                K_pinv = np.linalg.pinv(K)
                
                VK_pinv, UK_pinv = np.linalg.eigh(K_pinv)
                #X_plot_p = X_plot @ (UK_pinv[:, -r:] * np.sqrt(VK_pinv[-r:]))
                V, U = np.linalg.eigh(X@K@X.T)
                Vp = V[-r:]
                Up = U[:, -r:]
                U0 = U[:, :-r]
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

                wp = pm.Normal("wp",mu=0, tau=tau_p, shape=r, dims="wp_dim")
                w0 = pm.Normal("w0", mu=0, tau=tau, shape=(order-1), dims="w0_dim")
                X = np.hstack([Xp, X0])
                X_plot = np.hstack([X_plot_p, X_plot_0])
                f = pm.math.dot(Xp, wp) + pm.math.dot(X0, w0)
            else:
                w = pm.Normal("w",mu=0, tau=tau, shape=k, dims="w_dim")
                print("Unpenalised Spectral = standard implementation")
                f = pm.math.dot(X, w)
            
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

            w = pm.Normal("w", mu=0, tau=tau, shape=k-1, dims="w_dim")
            if penalised:
                DVpw = pt.dot(pt.dot(difference_matrix(k, order=order), Vp), w)
                K = Vp.T @ difference_matrix(k, order=order).T @ difference_matrix(k, order=order) @ Vp
                penalty = pt.dot(DVpw, DVpw)
                Q = tau_p * K + tau * np.eye(k-1)
                sign, logdet = pt.linalg.slogdet(Q)
                pm.Potential("spline_penalty", - 0.5 * tau_p * penalty
                                            - 0.5 * (k-1) * pt.log(tau)
                                            + 0.5 * logdet)
            f = pm.math.dot(X, w)


        eta = beta_0 + f
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=eta, sigma=np.sqrt(sigma_2), observed=y_data)
    return model, X, X_plot
