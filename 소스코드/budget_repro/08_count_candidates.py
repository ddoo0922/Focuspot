# %% [markdown]
# # 08. 사양 후보 2·3 — 추천 순위는 그대로 두고, 예산 크기(추정 불량 수)를 고친다
# 06에서 PVC는 Stellantis N.V. 순위는 좋았지만 KAMP 개수를 과대추정했다(MAE 2.84, 고정 불량률 2.19).
# - **후보 2 (PVC+보정)**: 학습 날짜의 확률 합을 실제 합에 맞추도록 절편만 옮긴다. 순위는 PVC와 같다.
# - **후보 3 (2단계)**: 순위는 PVC, 추정 불량 수는 집계 로지스틱(06에서 KAMP MAE 1.69). 06 캐시로 계산하므로 새 학습이 없다.
# 두 후보 모두 두 데이터에 똑같이 적용한다. 판단 기준: KAMP 개수 MAE, Stellantis N.V. 개수 MAE, Stellantis N.V. 개수 연동 예산 포착.

# %%
import time, warnings
import numpy as np, pandas as pd
import br
from br import spec as S
from wl.metrics import inclusion_prob
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
DATA = {"polito": S.load_polito(), "kamp": S.load_kamp()}
cache = br.RES / "same_model_scores.pkl"
sc = pd.read_pickle(cache)
NEW = "PVC MLP+보정"
done = set(map(tuple, sc[["data", "model", "seed"]].drop_duplicates().values))
for ds, (X, day, cnt, y, blk) in DATA.items():
    for s in range(5):
        if (ds, NEW, s) in done:
            continue
        t = time.time()
        score, prob = S.lodo("PVC MLP", X, day, cnt, s, calib=True)
        sc = pd.concat([sc, pd.DataFrame({"data": ds, "model": NEW, "seed": s, "row": np.arange(len(day)),
                                          "score": score, "prob": prob})], ignore_index=True)
        sc.to_pickle(cache)
        print(f"{ds:6s} {NEW} seed={s}  {time.time()-t:5.1f}초", flush=True)

# %% [markdown]
# ## 1. 일별 개수 MAE — 후보별
# 2단계의 개수는 집계 로지스틱과 같다.

# %%
def day_pred(ds, model):
    X, day, cnt, y, blk = DATA[ds]
    g = sc[(sc.data == ds) & (sc.model == model)]
    out = []
    for s, gg in g.groupby("seed"):
        gg = gg.sort_values("row"); ok = ~np.isnan(gg.prob.values)
        out.append(pd.Series(gg.prob.values[ok]).groupby(day[ok]).sum())
    return pd.concat(out, axis=1)
rows = []
for ds in DATA:
    cnt = DATA[ds][2]
    for lab, model in [("고정 불량률", "고정 불량률"), ("PVC MLP (06)", "PVC MLP"), ("후보 2: PVC+보정", NEW), ("후보 3: 2단계 (개수=집계 로지스틱)", "집계 로지스틱")]:
        P_ = day_pred(ds, model)
        e = P_.sub(cnt.loc[P_.index], axis=0).abs().mean()
        rows.append(dict(데이터=ds, 후보=lab, MAE=e.mean(), 최소=e.min(), 최대=e.max(), 추정합계=P_.sum().mean(), 실제합계=cnt.loc[P_.index].sum()))
mae = pd.DataFrame(rows).round(2); mae.to_csv(br.RES / "count_candidates_mae.csv", index=False, encoding="utf-8-sig")
mae

# %% [markdown]
# ## 2. Stellantis N.V. 개수 연동 예산 — 날짜마다 (추정 불량 수 × m)개, 순위는 모두 PVC

# %%
X, day, cnt, y, _ = DATA["polito"]
ud, gi = np.unique(day, return_inverse=True); n = np.bincount(gi); pos = np.bincount(gi, weights=y)
def rank_scores(model):
    return [g.sort_values("row").score.values for _, g in sc[(sc.data == "polito") & (sc.model == model)].groupby("seed")]
def counts(model):
    return [np.bincount(gi, weights=g.sort_values("row").prob.values) for _, g in sc[(sc.data == "polito") & (sc.model == model)].groupby("seed")]
RANK = rank_scores("PVC MLP")
rows = []
for lab, cm in [("PVC MLP (06)", "PVC MLP"), ("후보 2: PVC+보정", NEW), ("후보 3: 2단계", "집계 로지스틱")]:
    C = counts(cm)
    for mult in (2.5, 5, 10):
        res = []
        for i, s_ in enumerate(RANK):
            c = C[i % len(C)]
            k = np.minimum(np.ceil(mult * c - 1e-9), n)
            p = inclusion_prob(s_, k.astype(float), gi)
            res.append((k.sum(), (p * y).sum(), (k / n * pos).sum()))
        r = np.mean(res, 0)
        rows.append({"후보": lab, "배수": mult, "검사수": r[0], "검사비율": r[0] / len(y), "포착": r[1], "같은 수 무작위": r[2], "포착/무작위": r[1] / r[2]})
cb = pd.DataFrame(rows).round(2); cb.to_csv(br.RES / "count_candidates_budget.csv", index=False, encoding="utf-8-sig")
cb
