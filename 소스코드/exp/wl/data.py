"""Stellantis N.V. RSW 데이터 로딩과 KAMP형 타점 요약 피처."""
from pathlib import Path
import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
POLITO = ROOT.parent / "학습용 데이터" / "03_Stellantis N.V._RSW" / "data"
KAMP_XLSX = ROOT.parent / "학습용 데이터" / "01. KAMP_용접기 AI 데이터셋" / "2. 용접기 AI 데이터셋" / "Welding Data Set_01.xlsx"
CACHE = EXP / "cache"

BASE_FEATS = ["cur_mean", "vol_mean", "weld_len", "force_mean"]
DERIVED_FEATS = ["heat", "dynres", "i_over_f"]
SEQ_BASE = ["cur_mean", "vol_mean", "heat", "dynres"]
SEQ_FEATS = [f"dev_{c}" for c in SEQ_BASE] + [f"dl_{c}" for c in SEQ_BASE]
FEATURE_SETS = {
    "kamp_full": BASE_FEATS + DERIVED_FEATS,
    "kamp_noforce": ["cur_mean", "vol_mean", "weld_len", "heat", "dynres"],
    "seq": SEQ_FEATS,
    "kamp_seq": BASE_FEATS + DERIVED_FEATS + SEQ_FEATS,
}
SHAPE_FEATS = ["sh_dr_peak", "sh_dr_end", "sh_dr_drop", "sh_dr_drop_rel", "sh_cur_late_early", "sh_vol_late_early",
               "sh_cur_std", "sh_force_max", "sh_force_std", "sh_len_main"]
FEATURE_SETS["shape"] = SHAPE_FEATS                       # 파형 모양 피처(KAMP에는 없음, 양성 대조용)
FEATURE_SETS["shape_kamp"] = BASE_FEATS + DERIVED_FEATS + SHAPE_FEATS


def _shape(v, c, f):
    """본 용접 구간(예열 뒤) 파형 모양 요약. 예열은 앞쪽 짧은 통전 뒤 전류가 0으로 끊기는 구간으로 본다."""
    rows = []
    for vi, ci, fi in zip(v, c, f):
        on = np.flatnonzero(ci > 0.05)
        if len(on) < 20:
            rows.append([np.nan] * len(SHAPE_FEATS)); continue
        gaps = np.flatnonzero(np.diff(on) > 1)
        start = on[gaps[0] + 1] if len(gaps) and on[gaps[0]] < 200 else on[0]
        seg = on[on >= start]
        vv, cc = vi[seg], ci[seg]
        r = vv / np.maximum(cc, 1e-6)
        n = len(seg); a, b = max(n // 10, 1), max(n * 3 // 10, 1)
        k = max(n // 20, 3)
        rs = np.convolve(r, np.ones(k) / k, mode="valid")
        peak, end = rs.max(), rs[-1]
        ff = fi[fi > 0]
        rows.append([peak, end, peak - end, (peak - end) / max(peak, 1e-6),
                     cc[-b:].mean() / max(cc[a:b + a].mean(), 1e-6), vv[-b:].mean() / max(vv[a:b + a].mean(), 1e-6),
                     cc[a:].std(), ff.max() if len(ff) else np.nan, ff.std() if len(ff) else np.nan, float(n)])
    return pd.DataFrame(rows, columns=SHAPE_FEATS)


def seq_features(X, group, w=20):
    """순서 피처. X는 그룹 안에서 생산 순서대로 정렬된 DataFrame.
    dev_* = 직전 w개 타점 중앙값 대비 편차(느린 변화 제거), dl_* = 직전 타점 대비 변화량."""
    out = {}
    for c in SEQ_BASE:
        s = X[c]
        med = s.groupby(group).transform(lambda v: v.shift(1).rolling(w, min_periods=3).median())
        out[f"dev_{c}"] = s - med
        out[f"dl_{c}"] = s - s.groupby(group).shift(1)
    return pd.DataFrame(out, index=X.index).fillna(0.0)


def load_polito_waves():
    d = {k: pd.read_csv(POLITO / f"{k}.csv") for k in ["voltage", "current", "force", "labels"]}
    L = d["labels"]
    for k in ["voltage", "current", "force"]:
        assert (d[k][["Car Body", "Welding Spot"]].values == L[["Car Body", "Welding Spot"]].values).all()
    meta = pd.DataFrame({
        "body": L["Car Body"].values, "loc": L["Welding Spot"].values,
        "date": pd.to_datetime(L["Date"]).values, "y": L["Fault"].values.astype(int),
    })
    return meta, d["voltage"].iloc[:, 3:].values, d["current"].iloc[:, 3:].values, d["force"].iloc[:, 3:].values


def _summ(v, c, f):
    m = c > 0
    L = m.sum(1).astype(float)
    cur = np.where(m, c, 0).sum(1) / L
    vol = np.where(m, v, 0).sum(1) / L
    fm = f > 0
    fl = fm.sum(1)
    force = np.where(fl > 0, np.where(fm, f, 0).sum(1) / np.maximum(fl, 1), np.nan)
    heat = (v * c).sum(1)
    ratio = np.where(m & (c > 0.05), v / np.maximum(c, 1e-6), np.nan)
    dynres = np.nanmedian(ratio, axis=1)
    return pd.DataFrame({
        "cur_mean": cur, "vol_mean": vol, "weld_len": L, "force_mean": force,
        "heat": heat, "dynres": dynres, "i_over_f": cur / force,
    })


def load_polito(refresh=False):
    """meta(body, loc, date, y) + KAMP형 요약 피처 DataFrame. 가압력 전부 0인 타점은 NaN."""
    p = CACHE / "polito_summary.pkl"
    if p.exists() and not refresh:
        return pd.read_pickle(p)
    meta, v, c, f = load_polito_waves()
    X = _summ(v, c, f)
    out = pd.concat([meta, X], axis=1)
    # Stellantis N.V. 순서는 파일 순서(차체 → 타점 번호)를 가정한다. 위치별 기준이 달라 위치 평균(라벨 무관)을 먼저 뺀다.
    Xd = X[SEQ_BASE] - X[SEQ_BASE].groupby(meta["loc"]).transform("mean")
    out = pd.concat([out, seq_features(Xd, meta["body"]), _shape(v, c, f)], axis=1)
    CACHE.mkdir(exist_ok=True)
    out.to_pickle(p)
    return out
