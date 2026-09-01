import os
import shutil


copy_keys = True

run_idatas = True
replace_idatas = False
idatas_workers = 100

run_reports = False
replace_reports = False
reports_workers = 100

run_replications_report = False
replace_replications_report = False
replications_report_workers = 100

run_name = 'CDtau3_a1b1k5'
data_name = 'alt_data_4b_1_1000'

generate_data = False
quiet = True
###
data_folder = 'data/'
os.makedirs(data_folder, exist_ok=True)
data_path = os.path.join(data_folder, f'{data_name}.pkl')
###

# f, sigma
f_sigma_list = [('f4b', 0.5)]
###
run_comparison_report = True
top_n = 40
spread = True
spread_start = 0
spread_step = 1
comparison_list = [('ortho_centring+dropping', 'svd')]
replace_comparison_report = True
###

if generate_data:
    import _run._run_generate_data as _run_generate_data
    _run_generate_data.main(data_path, f_sigma_list, generate_data)

# replication
replication_list = list(range(360))[::]
report_replication_list = replication_list# list(range(100))[:]
report_replication_list = replication_list[:]
# implementation
implementation_list = ['svd',
                       'ortho_centring+dropping',]


# 0, MvN, ortho_diag
builder_list = ['ortho_diag']
# penalised
penalised_list = [False]


a = 1
b = 0.1
order = 2
spline_degree = 3
n_internal_knots = 5

n_tune = 1000
n_draws = 2000
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
                    if ('spectral' in implementation)&(not penalised) or ('svd_aligned' in implementation)&(not penalised):
                        continue
                    model_keys.append((f, sigma, implementation, penalised, replication, builder))

report_model_keys = []
for f, sigma in f_sigma_list:
    for replication in report_replication_list:
        for implementation in implementation_list:
            for builder in builder_list:
                for penalised in penalised_list:
                    if ('spectral' in implementation)&(not penalised) or ('svd_aligned' in implementation)&(not penalised):
                        continue
                    report_model_keys.append((f, sigma, implementation, penalised, replication, builder))

# Paths
directory_path = "../p-splines/"
# data_path = os.path.join(directory_path, data_name)

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
    
    