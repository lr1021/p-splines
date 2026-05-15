import os

model_keys = []
for f, sigma in [('f3', 0.33)]:
    for replication in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
        for implementation in ['standard', 'centring+dropping', 'conditioning', 'spectral', 'svd']:
            if implementation=='spectral':
                model_keys.append((f, sigma, implementation, True, replication))
            else:
                for penalised in [True, False]:
                    model_keys.append((f, sigma, implementation, penalised, replication))

# Paths
directory_path = "../p-splines/"
data_path = os.path.join(directory_path, "data.pkl")

run_name = '0'
reports_path = os.path.join(directory_path, f"run({run_name})/reports")
if not os.path.exists(reports_path):
    os.makedirs(reports_path)
idatas_path = os.path.join(directory_path, f"run({run_name})/idatas")