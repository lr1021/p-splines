import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.linalg import null_space
from _utils_spline import eval_spline_basis_equispaced_numeric, difference_matrix

import warnings
warnings.filterwarnings('ignore')


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
            if not penalised:
                model.add_coord("w_dim", range(k))
                w = pm.Normal("w",mu=0, tau=tau, shape=k, dims="w_dim")
            else:
                model.add_coord("w_dim", range(k))
                K = difference_matrix(k, order=order).T @ difference_matrix(k, order=order)
                Q = tau_p * K + tau * pt.eye(k)
                w = pm.MvNormal("w", mu=np.zeros(k), tau=Q, dims="w_dim")
                #Dw = pt.dot(difference_matrix(k, order=order), w)
                #penalty = pt.dot(Dw, Dw)
            var_names += ['w']
                
            f = pm.math.dot(X, w)

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

builder_dict = {'0': build_model,
                'MvN': build_model_MvN}













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
