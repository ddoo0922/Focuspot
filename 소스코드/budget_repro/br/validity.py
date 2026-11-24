"""KAMP 타당성 검증(23~25)용 함수. 모델 사양(br/spec.py)은 그대로 쓰고, 평가·조건 변환만 여기에 둔다."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from . import spec as S


def top_sets(score, day, ok, frac=.1):
    """날짜마다 ok 행 중 점수 상위 ceil(frac·n)개 인덱스 집합."""
    out = {}
    for d in np.unique(day[ok]):
        m = ok & (day == d); k = int(np.ceil(frac * m.sum())); ix = np.flatnonzero(m)
        out[d] = set(ix[np.argsort(-score[m], kind="stable")[:k]])
    return out


def overlap_days(a, b, day, ok, frac=.1):
    """날짜별 상위 frac 겹침 비율(배열, 날짜 순)."""
    A, Bs = top_sets(a, day, ok, frac), top_sets(b, day, ok, frac)
    return np.array([len(A[d] & Bs[d]) / len(A[d]) for d in sorted(A)])


def low_current_frac(score, cur, day, ok, frac=.1, delta=0.1):
    """상위 frac 추천 중 '전류 < 기준 − delta' 비율. 기준 두 가지: 그날 중앙값, 전체 기간 중앙값(18번 노트북의 83.1% 정의)."""
    T = top_sets(score, day, ok, frac)
    idx = np.array(sorted(set().union(*T.values())))
    day_med = pd.Series(cur[ok]).groupby(day[ok]).median()
    per_day = (cur[idx] < day_med.loc[day[idx]].values - delta).mean()
    glob = (cur[idx] < np.median(cur[ok]) - delta).mean()
    return per_day, glob


def day_rank(score, day, ok):
    out = np.full(len(score), np.nan)
    out[ok] = pd.Series(score[ok]).groupby(day[ok]).rank(pct=True).values
    return out


def boot_ci(x, n=5000, seed=0):
    """날짜(또는 반복) 단위 복원추출 평균의 95% 구간."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return x.mean(), np.quantile(m, .025), np.quantile(m, .975)


def loc_standardize(X, loc):
    """위치마다 열별 (x − 위치 평균)/위치 표준편차. 정답을 쓰지 않는다. 표준편차가 0이거나 1행뿐인 위치는 0."""
    X = X.copy()
    X = X.fillna(X.median())
    g = pd.Series(loc, index=X.index)
    mu = X.groupby(g).transform("mean"); sd = X.groupby(g).transform("std")
    Z = (X - mu) / sd
    return Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class GuidebookAE:
    """KAMP 가이드북 기준 오토인코더: 4 → 3 → 2 → 3 → 4, RReLU(마지막 층 없음), MSE, Adam lr 0.01, 50 epoch, batch 64,
    입력은 학습 행 기준 MinMax 정규화. 이상 점수 = 행별 복원 오차(MSE). 정답·개수를 쓰지 않는다."""

    def fit(self, X, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        self.lo, self.hi = X.min(0), X.max(0)
        Xs = torch.tensor((X - self.lo) / (self.hi - self.lo + 1e-9), dtype=torch.float32)
        self.net = nn.Sequential(nn.Linear(4, 3), nn.RReLU(), nn.Linear(3, 2), nn.RReLU(), nn.Linear(2, 3), nn.RReLU(), nn.Linear(3, 4))
        opt = torch.optim.Adam(self.net.parameters(), lr=0.01)
        g = torch.Generator().manual_seed(seed)
        for _ in range(50):
            for idx in torch.randperm(len(Xs), generator=g).split(64):
                opt.zero_grad(); nn.functional.mse_loss(self.net(Xs[idx]), Xs[idx]).backward(); opt.step()
        return self

    def score(self, X):
        self.net.eval()
        with torch.no_grad():
            Xs = torch.tensor((X - self.lo) / (self.hi - self.lo + 1e-9), dtype=torch.float32)
            return ((self.net(Xs) - Xs) ** 2).mean(1).numpy()


def ae_scores(X4, tr, te, seeds):
    """가이드북 AE를 학습 행 tr로 학습(결측은 학습 행 중앙값)해 te 행 점수의 시드 평균."""
    X = X4.values.astype(float); med = np.nanmedian(X[tr], 0); X = np.where(np.isnan(X), med, X)
    return np.mean([GuidebookAE().fit(X[tr], s).score(X[te]) for s in seeds], 0)
