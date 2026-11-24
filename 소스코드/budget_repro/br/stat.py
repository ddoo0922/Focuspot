"""24~26 노트북 공통 통계 함수(statsmodels 없이 numpy로 구현). OLS(HC3 강건 표준오차), Firth 로지스틱, 비율 구간, 순열 검정."""
import itertools
import numpy as np
from scipy import stats
from scipy.special import expit

Z = stats.norm.ppf(0.975)


def ols(X, y):
    """X는 상수항 포함. 계수, HC3 표준오차(이분산에 강건)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.sum((X @ XtXi) * X, 1)
    meat = X.T @ (X * (e ** 2 / np.clip(1 - h, 1e-6, None) ** 2)[:, None])
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return b, se


def firth(X, y, iters=200):
    """Firth 벌점 로지스틱(사건이 드물거나 한 칸에 사건이 0일 때도 계수가 발산하지 않음). 계수, 표준오차."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(X @ b); W = p * (1 - p)
        Iinv = np.linalg.pinv(X.T @ (X * W[:, None]))
        h = W * np.sum((X @ Iinv) * X, 1)
        step = Iinv @ (X.T @ (y - p + h * (0.5 - p)))
        mx = np.max(np.abs(step))
        if mx > 2: step *= 2 / mx
        b += step
        if mx < 1e-8: break
    p = expit(X @ b); W = p * (1 - p)
    return b, np.sqrt(np.diag(np.linalg.pinv(X.T @ (X * W[:, None]))))


def wilson(k, n):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d; w = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return c - w, c + w


def newcombe(k1, n1, k2, n2):
    """비율 차이 p1−p2의 Newcombe 95% 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def spearman_boot(x, y, B=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed); n = len(x)
    bs = [stats.spearmanr(x[i], y[i])[0] for i in (rng.integers(0, n, n) for _ in range(B))]
    return r[0], np.nanquantile(bs, .025), np.nanquantile(bs, .975), r[1]


def perm_spearman(x, y):
    """정확 순열 검정(모든 순열, n≤9). 양측 p와 단측(양의 상관) p."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, np.nan, np.nan
    rx, ry = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    perms = np.array(list(itertools.permutations(range(len(y)))))
    rs = (ry[perms] * rx).mean(1)
    r0 = float((rx * ry).mean())
    return r0, float(np.mean(np.abs(rs) >= abs(r0) - 1e-12)), float(np.mean(rs >= r0 - 1e-12))
