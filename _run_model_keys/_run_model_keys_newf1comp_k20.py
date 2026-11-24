import os
import shutil

# newf1comp: rerun of run_tau1e_6_a1b1k{5,20}_data12345newf1 (tau = 1e-6, both penalisations)
# with svd, ortho_post_centring and opr_exact. Run with _run/_run_run_ms_tau_comp.py

copy_keys = True

run_idatas = True
replace_idatas = False
idatas_workers = 60

run_reports = False
replace_reports = False
reports_workers = 60

run_replications_report = True
replace_replications_report = True
replications_report_workers = 60

run_prefix = 'newf1comp'
tau_exp = 6        # tau = 10**(-tau_exp) = 1e-6, as in the original runs
data_name = 'data12345newf1'

generate_data = False
quiet = True
###
data_folder = 'data/'
os.makedirs(data_folder, exist_ok=True)
data_path = os.path.join(data_folder, f'{data_name}.pkl')
###

# f, sigma
f_sigma_list = [('f1', 0.5), ('f1', 1.0),
                ('f2', 0.5), ('f2', 1.0),
                ('f3', 0.5), ('f3', 1.0),
                ('f4', 0.5), ('f4', 1.0),]   # f5 = mirror image of f4, not run

###
run_comparison_report = False
comparison_list = [('svd', 'opr_exact')]
###

# replication
replication_list = list(range(100))
report_replication_list = replication_list[:2]
# implementation
implementation_list = ['svd',
                       'ortho_post_centring',
                       'opr_exact']

builder_list = ['ortho_diag']
# penalised
penalised_list = [True, False]

a = 1
b = 0.1
order = 2
spline_degree = 3
n_internal_knots = 20

n_tune = 2000
n_draws = 4000
n_chains = 4
n_cores = 4

target_accept = 0.8
max_treedepth = 10

model_keys = []
for f, sigma in f_sigma_list:
    for replication in replication_list:
        for implementation in implementation_list:
            for builder in builder_list:
                for penalised in penalised_list:
                    model_keys.append((f, sigma, implementation, penalised, replication, builder))

report_model_keys = []
for f, sigma in f_sigma_list:
    for replication in report_replication_list:
        for implementation in implementation_list:
            for builder in builder_list:
                for penalised in penalised_list:
                    report_model_keys.append((f, sigma, implementation, penalised, replication, builder))

# tau (fixed prior precision of the spline weights) and run name -- generated, do not edit here
# tau_val = 10 ** (-tau_exp);  run_name = f"{run_prefix}tau{tau_exp}_a{a}b{b}k{n_internal_knots}"
from _utils._utils_tau import tau_from_exp, make_run_name
tau_val = tau_from_exp(tau_exp)
run_name = make_run_name(run_prefix, tau_exp, a, b, n_internal_knots)

# Paths
directory_path = "../p-splines/"

run_folder_name = f"run_{run_name}_{data_name}"
run_path = os.path.join(directory_path, run_folder_name)
reports_path = os.path.join(run_path, "reports")
idatas_path = os.path.join(run_path, "idatas")
os.makedirs(reports_path, exist_ok=True)
os.makedirs(idatas_path, exist_ok=True)

if copy_keys:
    current_script = os.path.abspath(__file__)
    destination = os.path.join(run_path, '_run_model_keys.py')
    if not os.path.exists(destination):
        shutil.copy2(current_script, destination)
