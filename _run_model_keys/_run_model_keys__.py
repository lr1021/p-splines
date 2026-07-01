import os
from _run._run_generate_data import data_name
import shutil


copy_keys = True

run_idatas = True
replace_idatas = False
idatas_workers = 512
run_reports = True
replace_reports = False
reports_workers = 200
run_replications_report = True
replications_report_workers = 100

run_name = 'ortho_diag_all_fxtau_long'
###

# f, sigma
f_sigma_list = [('f1', 0.33), ('f1', 0.5), ('f1', 1.0),
                ('f2', 0.33), ('f2', 0.5), ('f2', 1.0),
                ('f3', 0.33), ('f3', 0.5), ('f3', 1.0)]

f_sigma_list = [('f1', 1.0)]
f_sigma_list = [('f4', 0.33), ('f4', 0.5), ('f4', 1.0),
                ('f5', 0.33), ('f5', 0.5), ('f5', 1.0)]
#f_sigma_list = [('f4', 0.5),
                #('f5', 0.5)]
# replication
replication_list = [12]#list(range(100))[:]
replication_list = list(range(100))[:50]
report_replication_list = replication_list# list(range(100))[:]
report_replication_list = replication_list[:5]
# implementation
#implementation_list = ['conditioning', 'ortho_conditioning']#['post_centring', 'ortho_post_centring', 'centring', 'ortho_centring', 'centring+dropping', 'ortho_centring+dropping']#, 'centring+dropping', 'conditioning', 'spectral', 'svd']
implementation_list = ['post_centring', 'ortho_post_centring',
                       'centring', 'ortho_centring',
                       'centring+dropping', 'ortho_centring+dropping',
                       'conditioning', 'ortho_conditioning',
                       'svd',
                       'spectral',
                       'ortho_spectral'][:]
# implementation_list = ['centring+dropping', 'ortho_centring+dropping',
                       #'svd',
                       #'spectral', 'ortho_spectral'][:]
# implementation_list = ['ortho_centring+dropping']
# implementation_list = ['spectral', 'ortho_spectral'][:]
#implementation_list = ['post_centring']
# implementation_list = ['ortho_conditioning']
#implementation_list = ['post_centring', 'ortho_post_centring']
# 0, MvN, ortho_diag
builder_list = ['ortho_diag']
# penalised
penalised_list = [True, False]
# penalised_list = [False]


a = 1
b = 0.005
#b = 0.0005
order = 2
spline_degree = 3
n_internal_knots = 20
n_internal_knots = 5

n_tune = 2000
n_draws = 4000
n_chains = 4
n_cores = 4

target_accept = 0.8 #0.8
max_treedepth = 10 #10

model_keys = []
for f, sigma in f_sigma_list:
    for replication in replication_list:
        for implementation in implementation_list:
            for builder in builder_list:
                for penalised in penalised_list:
                    if ('spectral' in implementation)&(not penalised):
                        continue
                    model_keys.append((f, sigma, implementation, penalised, replication, builder))

report_model_keys = []
for f, sigma in f_sigma_list:
    for replication in report_replication_list:
        for implementation in implementation_list:
            for builder in builder_list:
                for penalised in penalised_list:
                    if ('spectral' in implementation)&(not penalised):
                        continue
                    report_model_keys.append((f, sigma, implementation, penalised, replication, builder))

# Paths
directory_path = "../p-splines/"
data_path = os.path.join(directory_path, data_name)

run_path = os.path.join(directory_path, f"run({run_name})")
reports_path = os.path.join(run_path, "reports")
idatas_path = os.path.join(run_path, "idatas")
os.makedirs(reports_path, exist_ok=True)
os.makedirs(idatas_path, exist_ok=True)

#reports_path = os.path.join(directory_path, f"run({run_name})/reports")
#if not os.path.exists(reports_path):
    #os.makedirs(reports_path)
#idatas_path = os.path.join(directory_path, f"run({run_name})/idatas")
#if not os.path.exists(idatas_path):
    #os.makedirs(idatas_path)
if copy_keys:
    current_script = os.path.abspath(__file__)
    destination = os.path.join(run_path, os.path.basename(current_script))
    shutil.copy2(current_script, destination)