"""KAMP와 Stellantis N.V.에 똑같이 적용하는 단일 모델 사양.
입력: 타점당 4개 값(전류·전압·통전시간·가압력) + 같은 식의 파생 3개. 정답: 날짜별 불량 개수만.
평가: 날짜 하나를 빼고 학습(나머지 날짜의 개수만 사용) → 뺀 날짜의 타점 점수. 데이터마다 따로 학습한다(단위가 달라 가중치는 옮기지 않음)."""
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from wl import data as wldata, kamp as wlkamp
from wl.models import WeakMLP, AutoEncoder
from .weak import AggLogistic, ConstRate, MonoCount, RulePrior, shift_to

FEATS = ["cur", "vol", "time", "force", "heat", "dynres", "i_over_f"]
PRIMARY = "PVC MLP"
MODELS = ["고정 불량률", "AE (정답 미사용)", "PVC MLP", "비율맞춤 MLP", "집계 로지스틱"]
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def features(cur, vol, time, force):
    """4개 값 → 7개 입력. 두 데이터에 같은 함수를 쓴다."""
    X = pd.DataFrame({"cur": cur, "vol": vol, "time": time, "force": force})
    X["heat"] = X.cur * X.vol * X.time
    X["dynres"] = X.vol / X.cur
    X["i_over_f"] = X.cur / X.force
    return X


PHYS = ["heat", "dynres", "i_over_f"]
DERIVED5 = PHYS + [f"dev_{c}" for c in PHYS] + [f"dl_{c}" for c in PHYS]


def derived5(X, day, skip=None, w=20):
    """1장의 파생 변수 5종만으로 입력을 만든다: 입열량(V·I·t), 동저항(V/I), 전류/가압력(I/F),
    이 3개 각각의 이동 중앙값 대비 편차(같은 날 직전 w타점 중앙값과의 차이)와 직전 타점 대비 변화량. 제공 변수 4개는 넣지 않는다.
    순서는 파일 순서(KAMP는 생산 순서, Stellantis N.V.는 차체→타점 번호)로 본다. skip 행(KAMP 복사 블록)은 순서 계산에서 빼고 비블록 중앙값으로 채운다."""
    skip = np.zeros(len(X), bool) if skip is None else skip
    out = pd.DataFrame(index=X.index, columns=DERIVED5, dtype=float)
    out[PHYS] = X[PHYS].values
    ok = ~skip
    sub = X.loc[ok, PHYS]; g = pd.Series(day[ok], index=sub.index)
    for c in PHYS:
        v = sub[c]
        med = v.groupby(g).transform(lambda s: s.shift(1).rolling(w, min_periods=3).median())
        out.loc[ok, f"dev_{c}"] = (v - med).fillna(0.0).values
        out.loc[ok, f"dl_{c}"] = (v - v.groupby(g).shift(1)).fillna(0.0).values
    if skip.any():
        out.loc[skip, :] = out.loc[ok].median().values
    return out


def day_transform(X, day, kind, skip=None):
    """날짜 안 변환(사양 후보 D1·D2). daystd = 날짜마다 (x − 평균)/표준편차. absz = 날짜마다 |x − 중앙값| / (1.4826·MAD).
    skip 행(KAMP 복사 블록)은 통계 계산에서 빼고 0으로 둔다."""
    skip = np.zeros(len(X), bool) if skip is None else skip
    out = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    ok = ~skip
    sub = X.loc[ok].copy(); sub = sub.fillna(sub.median()); g = pd.Series(day[ok], index=sub.index)
    if kind == "daystd":
        z = (sub - sub.groupby(g).transform("mean")) / (sub.groupby(g).transform("std").fillna(0) + 1e-9)
    else:
        med = sub.groupby(g).transform("median")
        mad = (sub - med).abs().groupby(g).transform("median") * 1.4826
        z = (sub - med).abs() / (mad + 1e-9 + 1e-6 * sub.abs().groupby(g).transform("median"))
        z = z.clip(upper=10)
    out.loc[ok] = z.values
    return out


def load_polito(featset="raw"):
    """Stellantis N.V.: 파형에서 KAMP와 같은 4개 요약값만 뽑는다. 가압력 파형이 전부 0인 284타점은 결측.
    featset="derived5"면 파생 변수 5종(9개 입력)만 쓴다."""
    P = wldata.load_polito()
    X = features(P.cur_mean.values, P.vol_mean.values, P.weld_len.values, P.force_mean.values)
    day = P.date.dt.strftime("%Y-%m-%d").values
    if featset.startswith("derived5"):
        X = derived5(X, day)
        if featset != "derived5":
            X = day_transform(X, day, featset.split("_")[1])
    cnt = pd.Series(P.y.values).groupby(day).sum()
    return X, day, cnt, P.y.values, np.zeros(len(P), bool)


def load_kamp(featset="raw"):
    """KAMP: 복사 블록(가압력>3bar, 291행×4일)은 1장 원칙대로 입력을 비블록 중앙값으로 중립화하고 순위에서 뺀다."""
    K, cnt_day = wlkamp.load_kamp()
    X = features(K.cur_mean.values, K.vol_mean.values, K.weld_len.values, K.force_mean.values)
    blk = K.block.values
    day = pd.to_datetime(K.date).dt.strftime("%Y-%m-%d").values
    if featset.startswith("derived5"):
        X = derived5(X, day, skip=blk)
        if featset != "derived5":
            X = day_transform(X, day, featset.split("_")[1], skip=blk)
    else:
        X.loc[blk, :] = X[~blk].median().values
    cnt = pd.Series({pd.Timestamp(d).strftime("%Y-%m-%d"): v for d, v in cnt_day.items()})
    return X, day, cnt, None, blk


def _prep(X, tr):
    X = X.values.astype(float)
    med = np.nanmedian(X[tr], axis=0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    return (X - mu) / sd


def fit_score(name, X, tr_bag, tr_cnt, tr, te, seed, calib=False):
    """학습 행 tr(날짜 번호 tr_bag, 날짜별 개수 tr_cnt)로 학습 → te 행의 (위험 점수, 불량 확률).
    calib=True: 학습 날짜의 확률 합이 학습 날짜의 실제 불량 합과 같아지도록 절편만 옮긴다(순위 불변, 사양 후보 2)."""
    Xs = _prep(X, tr)
    if name == "고정 불량률":
        m = ConstRate().fit(Xs[tr], tr_bag, tr_cnt, seed); z = m.logit(Xs[te]); return z, expit(z)
    if name == "AE (정답 미사용)":
        s = AutoEncoder(device=DEV).fit(Xs[tr], None, None, None, None, None, seed).predict(Xs[te], None)[0]
        return s, np.full(te.sum(), tr_cnt.sum() / tr.sum())
    if name == "집계 로지스틱":
        m = AggLogistic().fit(Xs[tr], tr_bag, tr_cnt, seed); z = m.logit(Xs[te]); return z, expit(z)
    if name == "규칙 사전지식 모델":
        m = RulePrior(device=DEV).fit(Xs[tr], tr_bag, tr_cnt, seed); z = m.logit(Xs[te]); return z, expit(z)
    if name == "단조 개수 모델":
        m = MonoCount(device=DEV).fit(Xs[tr], tr_bag, tr_cnt, seed); z = m.logit(Xs[te]); return z, expit(z)
    loss = {"PVC MLP": "count", "비율맞춤 MLP": "dllp", "PVC MLP 소형": "count"}[name]
    kw = dict(hidden=8, wd=1e-2) if name == "PVC MLP 소형" else {}      # 사양 후보 6: 은닉층 축소 + 가중치 감쇠 강화
    m = WeakMLP(loss=loss, device=DEV, **kw).fit(Xs[tr], tr_bag, tr_cnt, None, None, None, seed)
    z = m.predict(Xs[te], None)[1]
    if calib:
        z = z + shift_to(m.predict(Xs[tr], None)[1], tr_cnt.sum())
    return z, expit(z)


def lodo(name, X, day, cnt, seed, calib=False):
    """정답 있는 날짜마다 그 날을 빼고 학습 → 그 날 점수. 정답 없는 날(KAMP 3/27)은 점수를 매기지 않는다."""
    days = [d for d in cnt.index if not np.isnan(cnt[d])]
    lab = np.isin(day, days)
    score, prob = np.full(len(day), np.nan), np.full(len(day), np.nan)
    for d in days:
        te = day == d
        tr = lab & ~te
        ub, bag = np.unique(day[tr], return_inverse=True)
        c = cnt.loc[ub].values.astype(int)
        s, p = fit_score(name, X, bag, c, tr, te, seed, calib)
        score[te], prob[te] = s, p
    return score, prob
