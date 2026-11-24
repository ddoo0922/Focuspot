"""차체 단위 CV로 OOF 확률을 만들고 지표/부트스트랩을 계산한다."""
import pickle
import numpy as np
from scipy.special import expit
import pandas as pd
from pathlib import Path
from . import data, bags, metrics, models

RES = data.EXP / "results"


def _calibrate_shift(z, target):
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(z + mid).sum() < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def run_one(model_name, cond, seed, feature_set="kamp_full", k=7, gpu=0, df=None):
    """한 (모델, 조건, 시드)의 OOF 결과. 반환 dict: idx, y, loc, body, bag, p, score."""
    import torch
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    df = data.load_polito() if df is None else df
    locstd = feature_set.endswith("_locstd")
    feats = data.FEATURE_SETS[feature_set.replace("_locstd", "")]
    idx, bag = bags.build_condition(df, cond, seed)
    d = df.iloc[idx].reset_index(drop=True)
    y, loc, body = d["y"].values, d["loc"].values, d["body"].values
    Xraw = d[feats].values.astype(float)
    fold = bags.body_folds(body, k, seed)
    p_oof = np.zeros(len(d)); s_oof = np.zeros(len(d))
    for f in range(k):
        tr, te = fold != f, fold == f
        med = np.nanmedian(Xraw[tr], axis=0)
        Xi = np.where(np.isnan(Xraw), med, Xraw)
        if locstd:   # 위치별 평균을 뺀다(학습 폴드 타점으로만 계산, 처음 보는 위치는 전체 평균)
            gm = Xi[tr].mean(0)
            lm = {l: Xi[tr & (loc == l)].mean(0) for l in np.unique(loc[tr])}
            Xi = Xi - np.array([lm.get(l, gm) for l in loc])
        mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9
        X = (Xi - mu) / sd
        btr_raw = bag[tr]
        ub, btr = np.unique(btr_raw, return_inverse=True)
        cnt = np.bincount(btr, weights=y[tr]).astype(int)
        mdl = models.make_model(model_name, dev)
        ytr = y[tr] if mdl.needs_strong else None
        mdl.fit(X[tr], btr, cnt, ytr, loc[tr], feats, seed * 100 + f)
        s_te, z_te = mdl.predict(X[te], loc[te])
        s_oof[te] = s_te
        if z_te is None:
            p_oof[te] = cnt.sum() / tr.sum()   # 순위만 내는 모델: 개수는 학습 구간 평균 불량률로(묶음 개수만 사용)
        else:
            _, z_tr = mdl.predict(X[tr], loc[tr])
            sh = _calibrate_shift(z_tr, cnt.sum())
            p_oof[te] = expit(z_te + sh)
    return dict(model=model_name, cond=cond, seed=seed, feature_set=feature_set, idx=idx,
                y=y, loc=loc, body=body, bag=bag, p=p_oof, score=s_oof)


def eval_run(r, rows=None, **kw):
    sel = slice(None) if rows is None else rows
    _, bag = np.unique(r["bag"][sel], return_inverse=True)
    return metrics.evaluate(r["y"][sel], r["p"][sel], r["score"][sel], r["loc"][sel], bag, **kw)


def bootstrap(runs, n_boot=200, seed=0, keys=None):
    """같은 (모델, 조건)의 시드 반복 runs에 대해 차체 단위 부트스트랩. 반환: 점추정(시드평균), 95% 구간 DataFrame."""
    pts = pd.DataFrame([eval_run(r) for r in runs])
    rng = np.random.default_rng(seed)
    bodies = np.unique(np.concatenate([r["body"] for r in runs]))
    rows_by_body = [{b: np.flatnonzero(r["body"] == b) for b in bodies} for r in runs]
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(bodies, size=len(bodies), replace=True)
        vals = []
        for r, rb in zip(runs, rows_by_body):
            rows = np.concatenate([rb[b] for b in pick])
            vals.append(eval_run(r, rows))
        draws.append(pd.DataFrame(vals).mean())
    D = pd.DataFrame(draws)
    out = pd.DataFrame({"mean": pts.mean(), "lo": D.quantile(0.025), "hi": D.quantile(0.975)})
    return out


def save_runs(runs, name):
    RES.mkdir(exist_ok=True)
    with open(RES / f"{name}.pkl", "wb") as fh:
        pickle.dump(runs, fh)


def load_runs(name):
    with open(RES / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)
