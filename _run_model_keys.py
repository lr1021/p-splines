import os
import numpy as np

run_name = '5 non penalised'
###

# f, sigma
f_sigma_list = [('f3', 0.33)]
# replication
replication_list = list(range(20))
# replication_list = [8]
# implementation
#implementation_list = ['conditioning', 'ortho_conditioning']#['post_centring', 'ortho_post_centring', 'centring', 'ortho_centring', 'centring+dropping', 'ortho_centring+dropping']#, 'centring+dropping', 'conditioning', 'spectral', 'svd']
implementation_list = ['post_centring', 'ortho_post_centring',
                       'centring', 'ortho_centring',
                       'centring+dropping', 'ortho_centring+dropping',
                       'conditioning', 'ortho_conditioning',
                       'svd']
# implementation_list = ['ortho_conditioning']
# 0, MvN, ortho_diag
builder_list = ['ortho_MvN']
# penalised
penalised_list = [False]

a = 1
b = 0.005
order = 2
spline_degree = 3
n_internal_knots = 20

def f1(x):
    return x/1.758
def f2(x):
    return x**2/2.75 - 1.5
def f3(x):
    return np.sin(x)/0.72

functions = {f.__name__: f for f in [f1, f2, f3]}

model_keys = []
for f, sigma in f_sigma_list:
    for replication in replication_list:
        for implementation in implementation_list:
            for builder in builder_list:
                for penalised in penalised_list:
                    if (implementation=='spectral')&(not penalised):
                        continue
                    model_keys.append((f, sigma, implementation, penalised, replication, builder))

# Paths
directory_path = "../p-splines/"
data_path = os.path.join(directory_path, "data.pkl")

reports_path = os.path.join(directory_path, f"run({run_name})/reports")
if not os.path.exists(reports_path):
    os.makedirs(reports_path)
idatas_path = os.path.join(directory_path, f"run({run_name})/idatas")
if not os.path.exists(idatas_path):
    os.makedirs(idatas_path)