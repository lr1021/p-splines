import os
from _run_generate_data import functions

run_idatas = True
replace_idatas = False
idatas_workers = 128
run_reports = True
replace_reports = False
reports_workers = 50
run_replications_report = True
replications_report_workers = 100

run_name = '7 ortho diag (all, nop p)'
###

# f, sigma
f_sigma_list = [('f1', 0.33), ('f1', 0.5), ('f1', 1.0),
                ('f2', 0.33), ('f2', 0.5), ('f2', 1.0),
                ('f3', 0.33), ('f3', 0.5), ('f3', 1.0)]
# replication
replication_list = list(range(100))[:]
report_replication_list = [0]
# implementation
#implementation_list = ['conditioning', 'ortho_conditioning']#['post_centring', 'ortho_post_centring', 'centring', 'ortho_centring', 'centring+dropping', 'ortho_centring+dropping']#, 'centring+dropping', 'conditioning', 'spectral', 'svd']
implementation_list = ['post_centring', 'ortho_post_centring',
                       'centring', 'ortho_centring',
                       'centring+dropping', 'ortho_centring+dropping',
                       'conditioning', 'ortho_conditioning',
                       'svd',
                       'spectral'][:]
# implementation_list = ['conditioning']
# implementation_list = ['ortho_conditioning']
#implementation_list = ['post_centring', 'ortho_post_centring']
# 0, MvN, ortho_diag
builder_list = ['ortho_diag']
# penalised
penalised_list = [True, False]

a = 1
b = 0.005
order = 2
spline_degree = 3
n_internal_knots = 20

model_keys = []
for f, sigma in f_sigma_list:
    for replication in replication_list:
        for implementation in implementation_list:
            for builder in builder_list:
                for penalised in penalised_list:
                    if (implementation=='spectral')&(not penalised):
                        continue
                    model_keys.append((f, sigma, implementation, penalised, replication, builder))

report_model_keys = []
for f, sigma in f_sigma_list:
    for replication in report_replication_list:
        for implementation in implementation_list:
            for builder in builder_list:
                for penalised in penalised_list:
                    if (implementation=='spectral')&(not penalised):
                        continue
                    report_model_keys.append((f, sigma, implementation, penalised, replication, builder))

# Paths
directory_path = "../p-splines/"
data_path = os.path.join(directory_path, "data.pkl")

reports_path = os.path.join(directory_path, f"run({run_name})/reports")
if not os.path.exists(reports_path):
    os.makedirs(reports_path)
idatas_path = os.path.join(directory_path, f"run({run_name})/idatas")
if not os.path.exists(idatas_path):
    os.makedirs(idatas_path)