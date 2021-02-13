import numpy as np
import pandas as pd
import datetime

def read_data(kospi_kosdaq, fn):
    df_list = []
    for i in fn:
        df_list.append(pd.read_csv("../../jupyter/smart_beta/data/" + kospi_kosdaq + "/" + i + ".csv", index_col='date', parse_dates=True))
    return df_list

def modify_data(price, quality, value, shift_size):
    p = price[0]
    sn = price[1]
    pm = price[2]

    market_cap = p * sn
    market_cap.index = market_cap.index + pd.tseries.offsets.MonthEnd(0)
    m_cap = market_cap.groupby('date').last()

    pm.index = pm.index + pd.tseries.offsets.MonthEnd(0)
    m_pm = pm.groupby('date').last()
    m_ret = (m_pm.fillna(-1) / m_pm.shift(1)).clip(0, 10) - 1

    p.index = p.index + pd.tseries.offsets.MonthEnd(0)
    m_p = p.groupby('date').last()

    quality_list = []
    for i in quality:
        quality_list.append(pd.DataFrame(index=m_ret.index).join(i).shift(shift_size))

    value_list = []
    for i in value:
        value_list.append((m_pm / (m_p / i)).shift(shift_size))

    # 최근 주가로 계산 (eps를 shift)
    value_recent_list = []
    for i in value:
        value_recent_list.append((m_pm / (m_p / i).shift(shift_size)))

    return m_p, m_pm, m_cap, m_ret, quality_list, value_list, value_recent_list

def quality_factor(quality, compare_pivot):
    gp = quality[0]
    ast = quality[1]
    lb = quality[2]
    cf = quality[3]
    sl = quality[4]

    gpoa = gp / ast  # gross profit over asset
    cfoa = cf / ast  # cashflow profit over asset
    gmar = gp / sl  # gross margin
    turn = sl / ast

    # 성장성 팩터(수익성 팩터, 5년 전과 비교해서 변화 정도 / 총자산)
    gpoa_d = (gpoa - gpoa.shift(compare_pivot)) / ast.shift(compare_pivot)
    cfoa_d = (cfoa - cfoa.shift(compare_pivot)) / ast.shift(compare_pivot)
    gmar_d = (gmar - gmar.shift(compare_pivot)) / sl.shift(compare_pivot)
    turn_d = sl / ast - sl.shift(12) / ast.shift(12)

    # 안정성 팩터
    lev = lb / ast  # 총부채 / 총자산
    gpvol = gpoa.rolling(compare_pivot).std()  # gross profit volatility
    cfvol = cfoa.rolling(compare_pivot).std()  # cashflow volatility

    gpoa_rank = gpoa.rank(axis=1, ascending=False)
    cfoa_rank = cfoa.rank(axis=1, ascending=False)
    gmar_rank = gmar.rank(axis=1, ascending=False)
    turn_rank = turn.rank(axis=1, ascending=False)
    gpoa_d_rank = gpoa_d.rank(axis=1, ascending=False)
    cfoa_d_rank = cfoa_d.rank(axis=1, ascending=False)
    gmar_d_rank = gmar_d.rank(axis=1, ascending=False)
    turn_d_rank = turn_d.rank(axis=1, ascending=False)
    lev_rank = lev.rank(axis=1)
    gpvol_rank = gpvol.rank(axis=1)
    cfvol_rank = cfvol.rank(axis=1)

    gpoa_z = (gpoa_rank.T - gpoa_rank.mean(axis=1)) / gpoa_rank.std(axis=1)
    cfoa_z = (cfoa_rank.T - cfoa_rank.mean(axis=1)) / cfoa_rank.std(axis=1)
    gmar_z = (gmar_rank.T - gmar_rank.mean(axis=1)) / gmar_rank.std(axis=1)
    turn_z = (turn_rank.T - turn_rank.mean(axis=1)) / turn_rank.std(axis=1)
    gpoa_d_z = (gpoa_d_rank.T - gpoa_d_rank.mean(axis=1)) / gpoa_d_rank.std(axis=1)
    cfoa_d_z = (cfoa_d_rank.T - cfoa_d_rank.mean(axis=1)) / cfoa_d_rank.std(axis=1)
    gmar_d_z = (gmar_d_rank.T - gmar_d_rank.mean(axis=1)) / gmar_d_rank.std(axis=1)
    turn_d_z = (turn_d_rank.T - turn_d_rank.mean(axis=1)) / turn_d_rank.std(axis=1)
    lev_z = (lev_rank.T - lev_rank.mean(axis=1)) / lev_rank.std(axis=1)
    gpvol_z = (gpvol_rank.T - gpvol_rank.mean(axis=1)) / gpvol_rank.std(axis=1)
    cfvol_z = (cfvol_rank.T - cfvol_rank.mean(axis=1)) / cfvol_rank.std(axis=1)

    total_qz = gpoa_z.T + cfoa_z.T + gmar_z.T + turn_z.T + gpoa_d_z.T + cfoa_d_z.T + gmar_d_z.T + turn_d_z.T + lev_z.T + gpvol_z.T + cfvol_z.T
    total_qz_z = ((total_qz.T - total_qz.mean(axis=1)) / total_qz.std(axis=1)).T

    return total_qz_z

def value_factor(value):
    per = value[0]
    pbr = value[1]
    psr = value[2]
    pcr = value[3]

    per_rank = per.rank(axis=1, pct=True)
    pbr_rank = pbr.rank(axis=1, pct=True)
    psr_rank = psr.rank(axis=1, pct=True)
    pcr_rank = pcr.rank(axis=1, pct=True)

    per_z = ((per_rank.T - per_rank.mean(axis=1)) / per_rank.std(axis=1))
    pbr_z = ((pbr_rank.T - pbr_rank.mean(axis=1)) / pbr_rank.std(axis=1))
    psr_z = ((psr_rank.T - psr_rank.mean(axis=1)) / psr_rank.std(axis=1))
    pcr_z = ((pcr_rank.T - pcr_rank.mean(axis=1)) / pcr_rank.std(axis=1))

    total_vz = per_z.T + pbr_z.T + psr_z.T + pcr_z.T
    return total_vz

def momentum_factor(ret, back, forward):
    return ret / ret.shift(back) - ret / ret.shift(forward)