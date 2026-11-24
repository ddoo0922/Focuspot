"""KAMP 용접기 데이터 로딩, 복사 블록 표시, 일별 라벨."""
import numpy as np
import pandas as pd
from . import data

FEATS = data.FEATURE_SETS["kamp_full"]
BLOCK_FORCE = 3.0   # 가압력 3 bar 초과는 전부 4일 반복 복사 블록(1장 진단)


def load_kamp():
    raw = pd.read_excel(data.KAMP_XLSX, sheet_name="Raw data")
    res = pd.read_excel(data.KAMP_XLSX, sheet_name="result")
    K = pd.DataFrame({
        "idx": raw["idx"].values,
        "date": raw["working time"].dt.normalize().values,
        "force_mean": raw["weld force(bar)"].values, "cur_mean": raw["weld current(kA)"].values,
        "vol_mean": raw["weld Voltage(v)"].values, "weld_len": raw["weld time(ms)"].values,
    })
    K["heat"] = K.vol_mean * K.cur_mean * K.weld_len
    K["dynres"] = K.vol_mean / K.cur_mean
    K["i_over_f"] = K.cur_mean / K.force_mean
    K["block"] = K.force_mean > BLOCK_FORCE
    lab = res.groupby(res["working time"].dt.normalize())["defect"].sum()
    days = sorted(K.date.unique())
    cnt = pd.Series({d: lab.get(d, np.nan) for d in days})
    return K, cnt


def make_X(K, neutralize=True):
    """neutralize=True: 블록 행의 피처를 비블록 중앙값으로 대체(행 수·일별 개수 대응은 유지)."""
    X = K[FEATS].copy()
    if neutralize:
        X.loc[K.block, :] = X[~K.block].median().values
    return X.values.astype(float)
