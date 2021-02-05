import numpy as np
import pandas as pd

def MDD(arr):
    low = np.argmax(np.maximum.accumulate(arr) - arr)
    high = np.argmax(arr[:low])
    return high, low, (arr[high] - arr[low]) / arr[high]

def hit_ratio(arr, r, recent_num_ratio=True):
    total = round(sum(arr > 0) / len(arr), 4)
    recent = round(sum(arr[-1*r:] > 0) / len(arr[-1*r:]), 4) if recent_num_ratio else sum(arr[-1*r:] > 0)
    return total, recent

def fast_rank(df: pd.DataFrame, axis=1, pct=True, ascending=True) -> pd.DataFrame:
    arr = df.values.copy()
    if not ascending:
        arr = -arr
    nan_mask = np.isnan(arr)
    # argsort 2번하면 rank를 값으로 갖게 됨
    # [70, 50, 60, nan] -> [1, 2, 0, 3] -> [2, 0, 1, 3]
    rank = arr.argsort(axis=axis).argsort(axis=axis)
    rank = rank.astype(float)

    # 1을 더해줌
    rank += np.ones(rank.shape)
    rank[nan_mask] = np.nan
    # np.nanmax : nan을 0으로 치환하고 max값 구함
    if pct:
        rank = rank / np.nanmax(rank, axis=axis)[:, np.newaxis]
    return pd.DataFrame(rank, index=df.index, columns=df.columns)