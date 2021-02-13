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
quality_i_z = quality_factor(quality_i, 12)
value_i_z = value_factor(value_i)
value_r_i_z = value_factor(value_r_i)

m_p_q, m_pm_q, m_cap_q, m_ret_q, quality_q, value_q, value_r_q = modify_data(price_df_q, quality_df_q, value_df_q, 15)
quality_q_z = quality_factor(quality_q, 12)
value_q_z = value_factor(value_q)
value_r_q_z = value_factor(value_r_q)
momentum_q = momentum_factor(m_pm_q, 2, 1)
vol_rank = m_pm_q.rolling(48).std().rank(axis=1, pct=1)

# kospi
s_mask_i = m_cap_i.rank(axis=1, pct=True) < 0.1
sq_mask_i = quality_i_z[s_mask_i].rank(axis=1, pct=True) < 0.2
mask_i = value_r_i_z[sq_mask_i].rank(axis=1, pct=True) < 0.4

# kosdaq
s_mask_q = m_cap_q.rank(axis=1, pct=True) < 0.1
sq_mask_q = quality_q_z[s_mask_q].rank(axis=1, pct=True) < 0.4
sv_mask_q = value_q_z[sq_mask_q].rank(axis=1, pct=True) < 0.5
mask_q = momentum_q[sv_mask_q].rank(axis=1, pct=True, ascending=True) < 0.5

portfolio_ret_i = m_ret_i[mask_i.shift(1)].mean(axis=1).fillna(0)
portfolio_ret_q = m_ret_q[mask_q.shift(1)].mean(axis=1).fillna(0)
port_cumret_i = (portfolio_ret_i + 1).cumprod()
port_cumret_q = (portfolio_ret_q + 1).cumprod()

ratio_i = np.arange(0, 1.2, 0.2)
for r in ratio_i:
    port_cumret_iq = (portfolio_ret_i * r + portfolio_ret_q * (1 - r) + 1).cumprod()
    print(port_cumret_iq.round(2).values[-1], MDD(port_cumret_iq.values)[2])
print(portfolio_ret_q[26:].corr(portfolio_ret_i[26:]))

code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0][['회사명', '종목코드']]
sv_idx = list(mask_i.iloc[-1][mask_i.iloc[-1]==True].index) + list(mask_q.iloc[-1][mask_q.iloc[-1]==True].index)
for j in range(len(sv_idx)):
    print(code_df.query(f'종목코드=={int(sv_idx[j][1:])}').values[0][0], sv_idx[j][1:])