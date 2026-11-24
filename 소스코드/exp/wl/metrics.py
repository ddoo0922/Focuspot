"""1단계(묶음 개수 오차)·2단계(검사 예산 포착률) 지표. 동점은 기대값으로 처리한다."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

BUDGETS = (0.05, 0.10, 0.20)
M_LIST = (5, 10)


def inclusion_prob(score, k, grp=None):
    """각 행이 '그룹 내 상위 k개'에 뽑힐 확률. k는 그룹별 실수(기대 개수), 동점은 균등 분할.
    grp=None이면 전체가 한 그룹(k는 스칼라)."""
    n = len(score)
    if grp is None:
        grp = np.zeros(n, int)
        k = np.array([k], float)
    k = np.asarray(k, float)
    order = np.lexsort((-score, grp))
    g, s = grp[order], score[order]
    new_grp = np.r_[True, g[1:] != g[:-1]]
    new_tie = new_grp | np.r_[True, s[1:] != s[:-1]]
    idx = np.arange(n)
    grp_start = np.maximum.accumulate(np.where(new_grp, idx, 0))
    tie_start = np.maximum.accumulate(np.where(new_tie, idx, 0))
    tie_id = np.cumsum(new_tie) - 1
    tie_size = np.bincount(tie_id)[tie_id]
    before = tie_start - grp_start
    p_sorted = np.clip((k[g] - before) / tie_size, 0.0, 1.0)
    out = np.empty(n)
    out[order] = p_sorted
    return out


def _prf(hits, inspected, total_pos):
    rec = hits / total_pos if total_pos > 0 else np.nan
    prec = hits / inspected if inspected > 0 else np.nan
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0.0
    return rec, prec, f1


def within_loc_auroc(y, score, loc):
    df = pd.DataFrame({"y": y, "s": score, "l": loc})
    df["r"] = df.groupby("l")["s"].rank(method="average")
    g = df.groupby("l")
    npos, n = g["y"].sum(), g["y"].count()
    nneg = n - npos
    rpos = df[df.y == 1].groupby("l")["r"].sum().reindex(npos.index).fillna(0)
    ok = (npos > 0) & (nneg > 0)
    if not ok.any():
        return np.nan
    auc = (rpos - npos * (npos + 1) / 2)[ok] / (npos * nneg)[ok]
    w = (npos * nneg)[ok]
    return float((auc * w).sum() / w.sum())


def stage2(y, p, score, loc, bag, budgets=BUDGETS, m_list=M_LIST):
    out = {}
    N, P = len(y), y.sum()
    if P == 0 or P == N:
        return out
    ranks = rankdata(score)
    out["auroc"] = float((ranks[y == 1].sum() - P * (P + 1) / 2) / (P * (N - P)))
    out["prauc"] = float(average_precision_score(y, score + 1e-12 * np.random.default_rng(0).random(N))) \
        if len(np.unique(score)) > 1 else float(P / N)
    out["wl_auroc"] = within_loc_auroc(y, score, loc)
    nloc = np.bincount(loc) * 1.0          # 그룹(위치 번호)별 타점 수 — k는 그룹 번호로 인덱싱된다
    for b in budgets:
        inc = inclusion_prob(score, b * N)
        r, pr, f = _prf((inc * y).sum(), inc.sum(), P)
        out[f"recall@{b}"], out[f"prec@{b}"], out[f"f1@{b}"], out[f"lift@{b}"] = r, pr, f, r / b
        inc = inclusion_prob(score, b * nloc, grp=loc)
        r, pr, f = _prf((inc * y).sum(), inc.sum(), P)
        out[f"wl_recall@{b}"] = r
    nb = np.bincount(bag)
    psum = np.bincount(bag, weights=p)
    for m in m_list:
        k = np.minimum(m * psum, nb)
        inc = inclusion_prob(score, k, grp=bag)
        r, pr, f = _prf((inc * y).sum(), inc.sum(), P)
        out[f"recall@{m}x"], out[f"prec@{m}x"], out[f"f1@{m}x"], out[f"budget@{m}x"] = r, pr, f, inc.sum() / N
    return out


def stage1(y, p, bag):
    cnt = np.bincount(bag, weights=y)
    pred = np.bincount(bag, weights=p)
    return {"mae": float(np.abs(pred - cnt).mean()), "bias": float((pred - cnt).mean())}


def evaluate(y, p, score, loc, bag, **kw):
    return {**stage1(y, p, bag), **stage2(y, p, score, loc, bag, **kw)}
