"""Stellantis N.V. 파형 특징. 기본 16개(전류·전압 통계)와 도메인 특징(본 용접 구간의 모양·변화 지표).
1차 탐색에서 쓴 '기본 16특징 / 도메인 90특징'의 정의를 따른다."""
import numpy as np
import pandas as pd
from . import ROOT
from wl import data as wldata

CACHE = ROOT / "cache"
SIGS = ["I", "V", "R", "P"]                       # 전류, 전압, 상대 저항 V/I, 상대 전력 V·I (정규화 파형이라 실제 Ω·W 아님)
SIG_STATS = ["mean", "std", "min", "max", "range", "start", "end", "end_start",
             "seg1", "seg2", "seg3", "seg4", "seg5", "slope", "t_max", "t_min",
             "max_drop", "max_rise", "peak_end", "cv"]
EXTRA = ["pre_len", "gap_len", "main_len", "pre_I", "pre_V", "corr_IV", "P_late_early", "R_drop_rel", "E_total", "I2_total"]
DOMAIN = [f"{s}_{k}" for s in SIGS for k in SIG_STATS] + EXTRA
BASE = [f"{ch}_{k}" for ch in ("I", "V") for k in ("mean", "std", "max", "min", "median", "q10", "q90")] + ["on_len", "VI_sum"]


def _base(v, c):
    m = c > 0.05
    rows = []
    for vi, ci, mi in zip(v, c, m):
        r = []
        for x in (ci[mi], vi[mi]):
            r += [x.mean(), x.std(), x.max(), x.min(), np.median(x), np.quantile(x, .1), np.quantile(x, .9)]
        rows.append(r + [mi.sum(), (vi * ci).sum()])
    return pd.DataFrame(rows, columns=BASE)


def _smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) > k else x


def _sig_stats(x, n):
    a = max(n // 10, 1)
    xs = _smooth(x, max(n // 50, 3))
    d = np.diff(xs) * len(xs)                       # 길이로 정규화한 변화율
    seg = [s.mean() for s in np.array_split(x, 5)]
    t = np.linspace(0, 1, len(x))
    st, en = x[:a].mean(), x[-a:].mean()
    return [x.mean(), x.std(), x.min(), x.max(), x.max() - x.min(), st, en, en / max(abs(st), 1e-6),
            *seg, np.polyfit(t, x, 1)[0], xs.argmax() / len(xs), xs.argmin() / len(xs),
            d.min() if len(d) else 0.0, d.max() if len(d) else 0.0, xs.max() - xs[-1], x.std() / max(abs(x.mean()), 1e-6)]


def _domain(v, c):
    rows = []
    for vi, ci in zip(v, c):
        on = np.flatnonzero(ci > 0.05)
        gaps = np.flatnonzero(np.diff(on) > 1)
        has_pre = len(gaps) > 0 and on[gaps[0]] < 200     # 앞쪽 짧은 예열 통전 뒤 끊김
        start = on[gaps[0] + 1] if has_pre else on[0]
        seg = on[on >= start]
        pre = on[on < start]
        I, V = ci[seg], vi[seg]
        R, P = V / np.maximum(I, 1e-3), V * I
        n = len(seg)
        r = []
        for x in (I, V, R, P):
            r += _sig_stats(x, n)
        h = n // 2
        Rs = _smooth(R, max(n // 20, 3))
        r += [len(pre), (start - pre[-1] - 1) if has_pre else 0, n,
              ci[pre].mean() if len(pre) else 0.0, vi[pre].mean() if len(pre) else 0.0,
              np.corrcoef(I, V)[0, 1], P[h:].sum() / max(P[:h].sum(), 1e-6),
              (Rs.max() - Rs[-1]) / max(Rs.max(), 1e-6), P.sum(), (I ** 2).sum()]
        rows.append(r)
    return pd.DataFrame(rows, columns=DOMAIN)


def load(refresh=False):
    """meta(body, loc, date, y) + 기본 16개 + 도메인 특징. 캐시 사용."""
    p = CACHE / "polito_features.pkl"
    if p.exists() and not refresh:
        return pd.read_pickle(p)
    meta, v, c, _ = wldata.load_polito_waves()
    df = pd.concat([meta, _base(v, c), _domain(v, c)], axis=1)
    CACHE.mkdir(exist_ok=True)
    df.to_pickle(p)
    return df
