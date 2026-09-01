report_name = 'Q_01_alt_bi'
distortion_report_path = f'_distortion_reports/_reports/{report_name}'

tau_vals = [-1, 0, 1, 2, 3]

f_sigma_distortion =    {
        ('bi_normal_5_10_1', 0.05):          [],
        ('bi_normal_5_10_1', 0.1):          [],
        ('bi_normal_5_10_1', 0.5):          [],
        ('bi_normal_5_10_1', 1.0):          [],
        ('bi_normal_5_10_1', 5.0):          [],

        ('bi_normal_5_50_1', 0.05):          [],
        ('bi_normal_5_50_1', 0.1):          [],
        ('bi_normal_5_50_1', 0.5):          [],
        ('bi_normal_5_50_1', 1.0):          [],
        ('bi_normal_5_50_1', 5.0):          [],

        ('bi_normal_5_100_1', 0.05):          [],
        ('bi_normal_5_100_1', 0.1):          [(47, 'u')],
        ('bi_normal_5_100_1', 0.5):          [],
        ('bi_normal_5_100_1', 1.0):          [],
        ('bi_normal_5_100_1', 5.0):          []
        }

# f_sigma_distortion =    {
#         ('bi_normal_5_10_1', 0.05):          [],
#         ('bi_normal_5_10_1', 0.1):          [],
#         ('bi_normal_5_10_1', 0.5):          [],
#         ('bi_normal_5_10_1', 1.0):          [],
#         ('bi_normal_5_10_1', 5.0):          [],

#         ('bi_normal_5_50_1', 0.05):          [(2 , 'u')],
#         ('bi_normal_5_50_1', 0.1):          [],
#         ('bi_normal_5_50_1', 0.5):          [],
#         ('bi_normal_5_50_1', 1.0):          [],
#         ('bi_normal_5_50_1', 5.0):          [],

#         ('bi_normal_5_100_1', 0.05):          [(80, 'u'), (98, 'u')],
#         ('bi_normal_5_100_1', 0.1):          [(47, 'u'), (74, 'u')],
#         ('bi_normal_5_100_1', 0.5):          [],
#         ('bi_normal_5_100_1', 1.0):          [],
#         ('bi_normal_5_100_1', 5.0):          []
#         }