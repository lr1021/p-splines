import os
import numpy as np

run_name = '1'
###

# 0, MvN
builder = 'MvN'

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
for f, sigma in [('f3', 0.33)]:
    for replication in [0]:
        for implementation in ['standard', 'centring+dropping', 'conditioning', 'spectral', 'svd']:
            if implementation=='spectral':
                model_keys.append((f, sigma, implementation, True, replication, builder))
            else:
                for penalised in [True, False]:
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