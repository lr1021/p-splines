import os
import shutil


copy_keys = True

run_idatas = True
replace_idatas = False
idatas_workers = 20

run_reports = True
replace_reports = False
reports_workers = 20

run_replications_report = False
replace_replications_report = False
replications_report_workers = 20

run_name = 'DF3tau2_a1b1k20_sp_0'
data_name = 'df3_k20_0'

generate_data = False
quiet = False
###
data_folder = 'data/youtube/'
os.makedirs(data_folder, exist_ok=True)
data_path = os.path.join(data_folder, f'{data_name}.pkl')
###

# f, sigma
f_sigma_list = [('df3_k20_0', 0.5)]
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

#
keep_k5 =  {93:(55,74),
            187:(70, 83),
            209:(50, 65),
            222:(50, 69),
            403:(62, 96),
            409:(30, 65),
            420:(80, 99),
            436:(20, 32),
            439:(55, 70),
            458:(19, 35),
            463:(65, 84),
            544:(57, 73),
            554:(65, 81),
            597:(85, 105),
            599:(30, 50),
            620:(70, 95),
            747:(55, 86),
            755:(70, 90),
            757:(30, 42),
            766:(0, 62),
            801:(0, 50),
            805:(20, 56),
            825:(30, 41),
            854:(40, 75),
            907:(40, 74),
            936:(75, 98),
            955:(30, 44)}

keep_k20 =  {93:(0,74),
            187:(0, 83),
            209:(5, 65),
            222:(5, 69),
            409:(0, 65),
            420:(0, 99),
            436:(0, 32),
            439:(20, 70),
            458:(0, 35),
            463:(20, 84),
            544:(30, 73),
            554:(10, 81),
            597:(5, 105),
            620:(20, 95),
            747:(0, 86),
            755:(20, 90),
            757:(5, 42),            
            825:(0, 41),           
            854:(5, 75),           
            907:(20, 74),
            936:(0, 98),          
            955:(0, 44)}

#
# replication
replication_list = list(keep_k20.keys())[::]
report_replication_list = replication_list# list(range(100))[:]
report_replication_list = replication_list[:2]
# implementation
implementation_list = ['svd',
                       'ortho_centring+dropping',]


# 0, MvN, ortho_diag
builder_list = ['ortho_diag_softplus']
# penalised
penalised_list = [False]


a = 1
b = 0.1
order = 2
spline_degree = 3
n_internal_knots = 20

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
    
    