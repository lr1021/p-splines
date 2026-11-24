# Q2_01 recreated with the new tau runs + opr_exact (run with _run/_run_model_distortion_ms_mult_comp.py)
report_name = 'Q2_01comp'
distortion_report_path = f'_distortion_reports/_reports/{report_name}'

tau_vals = [-2, -1, 0, 1, 2]

comparison_list = [('svd', 'ortho_conditioning', 'opr_exact')]

f_sigma_distortion =  {
    ("('exponential', 'exponential')_5_10_1", 0.05): [(40, 'u')],
    ("('exponential', 'exponential')_5_10_1", 0.1): [(63, 'u')],
    ("('exponential', 'exponential')_5_10_1", 0.5): [(0, 'u')],
    ("('exponential', 'exponential')_5_10_1", 1.0): [],
    ("('exponential', 'exponential')_5_10_1", 5.0): [],

    ("('exponential', 'exponential')_5_50_1", 0.05): [(34, 'u'), (61, 'u')],
    ("('exponential', 'exponential')_5_50_1", 0.1): [(1, 'u'), (15, 'u'), (20, 'u'), (50, 'u'), (91, 'u')],
    ("('exponential', 'exponential')_5_50_1", 0.5): [],
    ("('exponential', 'exponential')_5_50_1", 1.0): [(15, 'u')],
    ("('exponential', 'exponential')_5_50_1", 5.0): [],

    ("('exponential', 'exponential')_5_100_1", 0.05): [],
    ("('exponential', 'exponential')_5_100_1", 0.1): [(10, 'u'), (27, 'u')],
    ("('exponential', 'exponential')_5_100_1", 0.5): [],
    ("('exponential', 'exponential')_5_100_1", 1.0): [(43, 'u'), (92, 'u')],
    ("('exponential', 'exponential')_5_100_1", 5.0): [],
}

# f_sigma_distortion =  {
#     ("('exponential', 'exponential')_5_10_1", 0.05): [(40, 'u'), (63, 'u'), (75, 'u'), (84, 'u'), (93, 'u')],
#     ("('exponential', 'exponential')_5_10_1", 0.1): [(0, 'u'), (63, 'u'), (75, 'u'), (84, 'u'), (93, 'u')],
#     ("('exponential', 'exponential')_5_10_1", 0.5): [(0, 'u'), (14, 'u'), (36, 'u'), (63, 'u'), (75, 'u'), (93, 'u')],
#     ("('exponential', 'exponential')_5_10_1", 1.0): [],
#     ("('exponential', 'exponential')_5_10_1", 5.0): [],

#     ("('exponential', 'exponential')_5_50_1", 0.05): [(1, 'u'), (2, 'u'), (7, 'u'), (15, 'u'), (20, 'u'), (22, 'u'), (23, 'u'), (27, 'u'), (34, 'u'), (50, 'u'), (61, 'u'), (65, 'u'), (86, 'u'), (90, 'u')],
#     ("('exponential', 'exponential')_5_50_1", 0.1): [(1, 'u'), (7, 'u'), (15, 'u'), (20, 'u'), (23, 'u'), (27, 'u'), (28, 'u'), (38, 'u'), (49, 'u'), (50, 'u'), (61, 'u'), (65, 'u'), (86, 'u'), (89, 'u'), (90, 'u'), (91, 'u')],
#     ("('exponential', 'exponential')_5_50_1", 0.5): [(7, 'u'), (15, 'u'), (31, 'u'), (32, 'u'), (50, 'u'), (56, 'u'), (61, 'u'), (65, 'u'), (86, 'u'), (89, 'u'), (90, 'u'), (91, 'u')],
#     ("('exponential', 'exponential')_5_50_1", 1.0): [(64, 'u'), (38, 'u'), (32, 'u'), (7, 'u'), (15, 'u')],
#     ("('exponential', 'exponential')_5_50_1", 5.0): [],

#     ("('exponential', 'exponential')_5_100_1", 0.05): [(1, 'u'), (3, 'u'), (6, 'u'), (10, 'u'), (27, 'u'), (38, 'u'), (50, 'u'), (78, 'u')],
#     ("('exponential', 'exponential')_5_100_1", 0.1): [(6, 'u'), (10, 'u'), (15, 'u'), (27, 'u'), (33, 'u'), (38, 'u'), (50, 'u'), (78, 'u')],
#     ("('exponential', 'exponential')_5_100_1", 0.5): [(38, 'u'), (43, 'u'), (49, 'u')],
#     ("('exponential', 'exponential')_5_100_1", 1.0): [(5, 'u'), (6, 'u'), (43, 'u'), (92, 'u')],
#     ("('exponential', 'exponential')_5_100_1", 5.0): [],
# }