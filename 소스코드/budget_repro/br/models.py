"""개별 정답으로 학습하는 검사 추천 모델(차체 단위 K-fold OOF). 위치 기준·AE는 개별 정답을 쓰지 않는다."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from wl.bags import body_folds
from wl.models import AutoEncoder


def _std(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    return (X - mu) / sd


def fit_predict(name, X, y, loc, tr, te, seed):
    """학습 구간 tr로 학습해 te의 위험 점수를 낸다. 점수는 클수록 위험."""
    if name == "loc_prior":                                  # 학습 구간 위치별 불량률(베타 평활), 처음 보는 위치는 전체 불량률
        g = y[tr].mean()
        s = pd.Series(y[tr]).groupby(loc[tr]).agg(["sum", "count"])
        rate = (s["sum"] + 2 * g) / (s["count"] + 2)
        return pd.Series(loc[te]).map(rate).fillna(g).values
    if name == "autoencoder":                                # 개별 정답 미사용, 복원오차
        Xs = _std(X, tr)
        return AutoEncoder(device="cpu").fit(Xs[tr], None, None, None, None, None, seed).predict(Xs[te], None)[0]
    if name == "logistic":
        Xs = _std(X, tr)
        m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(Xs[tr], y[tr])
        return m.predict_proba(Xs[te])[:, 1]
    if name == "catboost":
        m = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, random_seed=seed, verbose=0,
                               thread_count=8, allow_writing_files=False)
        return m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    if name == "extratrees":
        m = ExtraTreesClassifier(500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=8)
        return m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    if name == "tabicl":
        from tabicl import TabICLClassifier
        m = TabICLClassifier(n_estimators=8, random_state=seed, device="cuda")
        return m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    raise ValueError(name)


def run_oof(df, name, feats, seed, k=7):
    """차체 단위 k-fold OOF 점수. 반환 DataFrame: row, fold, score."""
    X = df[feats].values.astype(float) if feats else np.zeros((len(df), 1))
    y, loc = df["y"].values, df["loc"].values
    fold = body_folds(df["body"].values, k, seed)
    s = np.zeros(len(df))
    for f in range(k):
        tr, te = fold != f, fold == f
        s[te] = fit_predict(name, X, y, loc, tr, te, seed * 100 + f)
    return pd.DataFrame({"row": np.arange(len(df)), "fold": fold, "score": s})
