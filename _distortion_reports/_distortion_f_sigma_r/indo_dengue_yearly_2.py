report_name = 'indo_dengue_yearly_2'
distortion_report_path = f'_distortion_reports/_reports/{report_name}'

tau_vals = [-1, 0, 1, 2, 3, 4]

f_sigma_distortion_cd = {('rh_mean_pop_weighted', 0.5):                 [(2, 'u'), (3, 'u'), (7, 'u'), (11, 'u'), (26, 'u')],
                         ('t2m_max_pop_weighted', 0.5):                 [(2, 'u'), (6, 'u')],
                         ('t2m_mean_pop_weighted', 0.5):                [],
                         ('t2m_min_pop_weighted', 0.5):                 [(1, 'u'), (19, 'u'), (23, 'u'), (27, 'u')],
                         ('tp_24hmax_pop_weighted_log', 0.5):           [(3, 'u'), (22, 'o')],
                         ('tp_24hmean_pop_weighted_log', 0.5):          [(1, 'u'), (2, 'o'), (5, 'o'), (9, 'o')]}

f_sigma_distortion_cd = {('rh_mean_pop_weighted', 0.5):                 [(7, 'u')],
                         ('t2m_max_pop_weighted', 0.5):                 [],
                         ('t2m_mean_pop_weighted', 0.5):                [],
                         ('t2m_min_pop_weighted', 0.5):                 [],
                         ('tp_24hmax_pop_weighted_log', 0.5):           [(3, 'u')],
                         ('tp_24hmean_pop_weighted_log', 0.5):          [(1, 'u'), (2, 'o'), (5, 'o'), (9, 'o')]}

f_sigma_distortion_q =  {('rh_mean_pop_weighted', 0.5):                 [],
                         ('t2m_max_pop_weighted', 0.5):                 [(11, 'u'), (13, 'u'), (18, 'u'), (27, 'u')],
                         ('t2m_mean_pop_weighted', 0.5):                [(10, 'u'), (11, 'u'), (17, 'u')],
                         ('t2m_min_pop_weighted', 0.5):                 [(6, 'u'), (13, 'u'), (26, 'u')],
                         ('tp_24hmax_pop_weighted_log', 0.5):           [(7, 'u'), (11, 'u')],
                         ('tp_24hmean_pop_weighted_log', 0.5):          [(10, 'u')]}

f_sigma_distortion_q =  {('rh_mean_pop_weighted', 0.5):                 [],
                         ('t2m_max_pop_weighted', 0.5):                 [(11, 'u')],
                         ('t2m_mean_pop_weighted', 0.5):                [(11, 'u'), (17, 'u')],
                         ('t2m_min_pop_weighted', 0.5):                 [(13, 'u')],
                         ('tp_24hmax_pop_weighted_log', 0.5):           [(7, 'u')],
                         ('tp_24hmean_pop_weighted_log', 0.5):          [(10, 'u')]}
f_sigma_distortion = f_sigma_distortion_cd