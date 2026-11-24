"""검사 예산 평가. 예산 b = 검사할 기록의 비율. 평가 그룹(fold 또는 날짜)마다 ceil(b·n)개를 위험 점수 순으로 고른다.
동점은 정답을 쓰지 않고 기대값(균등 분할)으로 처리한다."""
import numpy as np
import pandas as pd
from wl.metrics import inclusion_prob

BUDGETS = np.round(np.arange(0.01, 1.0001, 0.01), 2)


def pick_counts(grp, b):
    """그룹별 검사 수 ceil(b·n_g)."""
    ug, n = np.unique(grp, return_counts=True)
    return ug, np.ceil(b * n - 1e-9).astype(int)


def hits_at(y, score, grp, b):
    """예산 b에서 (검사 수, 기대 포착 불량 수)."""
    ug, k = pick_counts(grp, b)
    gi = np.searchsorted(ug, grp)
    p = inclusion_prob(np.asarray(score, float), k.astype(float), gi)
    return int(k.sum()), float((p * y).sum())


def curve(y, score, grp, budgets=BUDGETS):
    """예산별 검사 수·포착·미포착·포착률·정밀도."""
    P = y.sum()
    rows = []
    for b in budgets:
        n, h = hits_at(y, score, grp, b)
        rows.append(dict(budget=b, inspected=n, caught=h, missed=P - h, recall=h / P, precision=h / n))
    return pd.DataFrame(rows)


def random_curve(y, grp, budgets=BUDGETS):
    """무작위 선택 기대값: 그룹마다 k_g/n_g · (그룹 불량 수)."""
    ug, n = np.unique(grp, return_counts=True)
    pos = np.array([y[grp == g].sum() for g in ug])
    P = y.sum()
    rows = []
    for b in budgets:
        k = np.ceil(b * n - 1e-9)
        h = float((k / n * pos).sum())
        rows.append(dict(budget=b, inspected=int(k.sum()), caught=h, missed=P - h, recall=h / P, precision=h / k.sum()))
    return pd.DataFrame(rows)


def budget_for_recall(cv, targets=(0.8, 0.9, 0.95, 1.0)):
    """목표 포착률을 처음 만족하는 예산(%)과 검사 수."""
    out = {}
    for t in targets:
        ok = cv[cv["recall"] >= t - 1e-9]
        out[t] = (float(ok["budget"].iloc[0]), int(ok["inspected"].iloc[0])) if len(ok) else (np.nan, np.nan)
    return out
