import os
import shutil


copy_keys = True

run_idatas = True
replace_idatas = False
idatas_workers = 512

run_reports = True
replace_reports = False
reports_workers = 100

run_replications_report = True
replace_replications_report = False
replications_report_workers = 100

run_name = 'test1'
data_name = 'data12345'

###
data_folder = 'data/'
os.makedirs(data_folder, exist_ok=True)
data_path = os.path.join(data_folder, f'{data_name}.pkl')
###

# f, sigma
f_sigma_list = [('f1', 0.33), ('f1', 0.5), ('f1', 1.0),
                ('f2', 0.33), ('f2', 0.5), ('f2', 1.0),
                ('f3', 0.33), ('f3', 0.5), ('f3', 1.0)]

f_sigma_list = [('f4', 0.33), ('f4', 0.5), ('f4', 1.0),
                ('f5', 0.33), ('f5', 0.5), ('f5', 1.0)]
f_sigma_list = [('f4', 0.33)]

# replication
replication_list = list(range(100))[:2]
report_replication_list = replication_list# list(range(100))[:]
report_replication_list = replication_list[:2]
# implementation
#implementation_list = ['conditioning', 'ortho_conditioning']#['post_centring', 'ortho_post_centring', 'centring', 'ortho_centring', 'centring+dropping', 'ortho_centring+dropping']#, 'centring+dropping', 'conditioning', 'spectral', 'svd']
#implementation_list = ['post_centring', 'ortho_post_centring',
                       #'centring', 'ortho_centring',
                       #'centring+dropping', 'ortho_centring+dropping',
                       #'conditioning', 'ortho_conditioning',
                       #'svd',
                       #'spectral',
                       #'ortho_spectral'][:]
implementation_list = ['ortho_centring_T', 'ortho_centring_T_exact', 'svd', 'svd_aligned'][0:2]
# implementation_list = ['centring+dropping', 'ortho_centring+dropping',
                       #'svd',
                       #'spectral', 'ortho_spectral'][:]

# 0, MvN, ortho_diag
builder_list = ['ortho_diag']
# penalised
penalised_list = [True]
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
    current_script = '_run_model_keys/_run_model_keys.py'
    destination = os.path.join(run_path, '_run_model_keys.py')
    if not os.path.exists(destination):
        shutil.copy2(current_script, destination)

    idatas_script = '_run_model_idatas.py'
    with open(idatas_script, 'r') as f:
        text = f.read()

    text = text.replace(
        'from _run_model_keys._run_model_keys import',
        f'from {run_folder_name}._run_model_keys import'
    )
    text = text.replace(
        'from _run_generate_data import',
        f'from {run_folder_name}._run_generate_data import'
    )
    
    destination = os.path.join(run_path, '_run_model_idatas.py')
    if not os.path.exists(destination):
        with open(destination, 'w') as f:
            f.write(text)

    generate_script = '_run_generate_data.py'
    with open(generate_script, 'r') as f:
        text = f.read()

    text = text.replace(
        'from _run_model_keys._run_model_keys import',
        f'from {run_folder_name}._run_model_keys import'
    )
    destination = os.path.join(run_path, '_run_generate_data.py')
    if not os.path.exists(destination):
        with open(destination, 'w') as f:
            f.write(text)
    
    reports_script = '_run_model_reports.py'
    with open(reports_script, 'r') as f:
        text = f.read()

    text = text.replace(
        'from _run_model_keys._run_model_keys import',
        f'from {run_folder_name}._run_model_keys import'
    )
    text = text.replace(
        'from _run_generate_data import',
        f'from {run_folder_name}._run_generate_data import'
    )
    destination = os.path.join(run_path, '_run_model_reports.py')
    if not os.path.exists(destination):
        with open(destination, 'w') as f:
            f.write(text)