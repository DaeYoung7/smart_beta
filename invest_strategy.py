import matplotlib.pyplot as plt
import seaborn as sns

from factor import *
from utils import *

price_fn = ['price', 'share_number', 'price_mod']
quality_fn = ['gross_profit', 'asset', 'liability', 'cashflow', 'sales']
value_fn = ['per', 'pbr', 'psr', 'pcr']

price_df_i = read_data("kospi", price_fn)
quality_df_i = read_data("kospi", quality_fn)
value_df_i = read_data("kospi", value_fn)

price_df_q = read_data("kosdaq", price_fn)
quality_df_q = read_data("kosdaq", quality_fn)
value_df_q = read_data("kosdaq", value_fn)

m_p_i, m_pm_i, m_cap_i, m_ret_i, quality_i, value_i, value_r_i = modify_data(price_df_i, quality_df_i, value_df_i, 15)
quality_i_z = quality_factor(quality_i)
value_i_z = value_factor(value_i)
value_r_i_z = value_factor(value_r_i)

m_p_q, m_pm_q, m_cap_q, m_ret_q, quality_q, value_q, value_r_q = modify_data(price_df_q, quality_df_q, value_df_q, 15)
quality_q_z = quality_factor(quality_q)
value_q_z = value_factor(value_q)
value_r_q_z = value_factor(value_r_q)
vol_rank = m_pm_q.rolling(48).std().rank(axis=1, pct=1)

# kospi
s_mask_i = m_cap_i.rank(axis=1) <= 100
sq_mask_i = quality_i_z[s_mask_i].rank(axis=1, pct=True) < 0.2
sv_mask_i = value_i_z[sq_mask_i].rank(axis=1, pct=True) < 0.4

# kosdaq
s_mask_q = (m_cap_q.rank(axis=1) <= 200)
sq_mask_q = quality_q_z[s_mask_q].rank(axis=1, pct=True) < 0.2
sv_mask_r_q = value_r_q_z[sq_mask_q].rank(axis=1, pct=True) < 0.4
sv_mask_q = vol_rank[sv_mask_r_q] < 0.3

