# %% [markdown]
# # 10. 사양 후보 4·5 — 06·08 캐시만 사용
# - **후보 4 (순위 앙상블):** 날짜 안 순위 백분위로 PVC와 비율 맞춤 MLP를 평균한다. PVC 단독과 차체 복원추출로 대응 비교한다.
# - **후보 5 (1단계 = 고정 불량률):** 개수 연동 예산의 추정 개수를 학습 날짜 평균 불량률 × 그날 타점 수로 바꾼다. 집계 로지스틱 1단계와 비교한다.
# 두 후보 모두 두 데이터에 똑같이 적용한다.

# %%
import numpy as np, pandas as pd
import br
from br import spec as S
from wl.metrics import inclusion_prob
from wl import data as wldata
pd.set_option("display.width", 220)
DATA = {"polito": S.load_polito(), "kamp": S.load_kamp()}
sc = pd.read_pickle(br.RES / "same_model_scores.pkl")
def per_seed(ds, model, col):
    return [g.sort_values("row")[col].values for _, g in sc[(sc.data == ds) & (sc.model == model)].groupby("seed")]
def pct_in_day(s, day):
    return pd.Series(s).groupby(day).rank(pct=True).values

X, day, cnt, y, _ = DATA["polito"]
body = wldata.load_polito().body.values
rng = np.random.default_rng(2); ub = np.unique(body)
Wb = np.stack([np.bincount(d_, minlength=len(ub)) for d_ in rng.integers(0, len(ub), (5000, len(ub)))])[:, np.searchsorted(ub, body)]
ud, gi = np.unique(day, return_inverse=True); n = np.bincount(gi); pos = np.bincount(gi, weights=y)
PVC = per_seed("polito", "PVC MLP", "score"); RAT = per_seed("polito", "비율맞춤 MLP", "score")
ENS = [(pct_in_day(a, day) + pct_in_day(b, day)) / 2 for a, b in zip(PVC, RAT)]
def recall_boot(scores, k):
    ps = np.mean([inclusion_prob(s, k.astype(float), gi) for s in scores], 0)
    return (ps * y).sum() / y.sum(), (Wb @ (ps * y)) / (Wb @ y)
def ci(a): return f"{np.mean(a):+.3f} [{np.quantile(a,.025):+.3f}, {np.quantile(a,.975):+.3f}]"

# %% [markdown]
# ## 1. 후보 4 — 순위 앙상블 vs PVC 단독 (Stellantis N.V.)

# %%
rows = []
for b in (.05, .1, .2, .3):
    k = np.ceil(b * n - 1e-9)
    r_p, bp = recall_boot(PVC, k); r_r, br_ = recall_boot(RAT, k); r_e, be = recall_boot(ENS, k)
    rows.append({"예산": f"{int(b*100)}%", "PVC": round(r_p, 3), "비율맞춤": round(r_r, 3), "앙상블": round(r_e, 3),
                 "앙상블−PVC [95%]": ci(be - bp), "차이≤0 비율": round(float((be - bp <= 0).mean()), 3)})
t4 = pd.DataFrame(rows); t4.to_csv(br.RES / "cand4_ensemble.csv", index=False, encoding="utf-8-sig"); t4

# %% [markdown]
# ## 2. 후보 4 — KAMP에서 두 순위가 얼마나 겹치나 (정답 없이 볼 수 있는 일관성)
# 날짜별 상위 10% 집합의 겹침 비율. 낮으면 두 모델이 서로 다른 타점을 추천한다는 뜻이다.

# %%
Xk, kday, kcnt, _, kblk = DATA["kamp"]
kp = np.mean(per_seed("kamp", "PVC MLP", "score"), 0); kr = np.mean(per_seed("kamp", "비율맞춤 MLP", "score"), 0)
ok = ~np.isnan(kp) & ~kblk
rows = []
for d in np.unique(kday[ok]):
    m = ok & (kday == d); kk = int(np.ceil(.1 * m.sum()))
    a = set(np.flatnonzero(m)[np.argsort(-kp[m])[:kk]]); b_ = set(np.flatnonzero(m)[np.argsort(-kr[m])[:kk]])
    rows.append(dict(date=d, 상위10퍼센트수=kk, 겹침비율=len(a & b_) / kk))
pd.DataFrame(rows).round(3)

# %% [markdown]
# ## 3. 후보 5 — 개수 연동 예산의 1단계: 집계 로지스틱 vs 고정 불량률 (Stellantis N.V., 순위는 PVC)

# %%
C_AGG = [np.bincount(gi, weights=p) for p in per_seed("polito", "집계 로지스틱", "prob")]
C_CON = [np.bincount(gi, weights=p) for p in per_seed("polito", "고정 불량률", "prob")]
rows = []
for m in (2.5, 5, 10):
    for nm, C in [("집계 로지스틱", C_AGG), ("고정 불량률", C_CON)]:
        k = np.minimum(np.ceil(m * np.mean(C, 0) - 1e-9), n)
        r, _ = recall_boot(PVC, k)
        rows.append({"배수": m, "1단계": nm, "검사수": int(k.sum()), "검사비율": round(k.sum() / len(y), 3),
                     "포착": round(r * y.sum(), 1), "같은 수 무작위": round((k / n * pos).sum(), 1), "포착/검사": round(r * y.sum() / k.sum(), 3)})
t5 = pd.DataFrame(rows); t5.to_csv(br.RES / "cand5_count_stage.csv", index=False, encoding="utf-8-sig"); t5

# %% [markdown]
# ## 4. 추천의 안정성 — PVC 시드끼리 날짜별 상위 10% 겹침 비율 (정답 불필요)
# 같은 사양인데 시드만 바꿨을 때 추천 타점이 얼마나 같은가. Stellantis N.V.와 KAMP를 같은 방식으로 본다.

# %%
from itertools import combinations
rows = []
for ds in DATA:
    Xd, dd, cc, yy, bb = DATA[ds]
    S_ = per_seed(ds, "PVC MLP", "score")
    ok = ~np.isnan(S_[0]) & ~bb
    for d in np.unique(dd[ok]):
        m = ok & (dd == d); kk = int(np.ceil(.1 * m.sum())); ix = np.flatnonzero(m)
        tops = [set(ix[np.argsort(-s[m])[:kk]]) for s in S_]
        ov = [len(a & b_) / kk for a, b_ in combinations(tops, 2)]
        rows.append(dict(데이터=ds, date=d, 타점수=int(m.sum()), 상위10퍼센트=kk, 시드간겹침_평균=np.mean(ov), 최소=np.min(ov)))
stab = pd.DataFrame(rows).round(3); stab.to_csv(br.RES / "seed_stability_top10.csv", index=False, encoding="utf-8-sig")
display(stab.groupby("데이터")[["시드간겹침_평균", "최소"]].mean().round(3)); stab[stab.데이터 == "kamp"]
