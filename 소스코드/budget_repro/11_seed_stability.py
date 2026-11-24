# %% [markdown]
# # 11. 시드 평균 추천은 안정적인가
# PVC 시드 5~9를 추가로 학습한다(두 데이터, 같은 사양). 비교 항목:
# - "시드 0~4 평균 점수"와 "시드 5~9 평균 점수"의 날짜별 상위 5·10·20% 겹침 비율
# - 10시드 평균의 Stellantis N.V. 검사 예산 성능이 5시드와 같은가
# 정답 없이 계산하는 안정성이라 KAMP에서도 측정할 수 있다.

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
done = set(map(tuple, sc[["data", "model", "seed"]].drop_duplicates().values))
for ds, (X, day, cnt, y, blk) in DATA.items():
    for s in range(5, 10):
        if (ds, "PVC MLP", s) in done:
            continue
        t = time.time()
        score, prob = S.lodo("PVC MLP", X, day, cnt, s)
        sc = pd.concat([sc, pd.DataFrame({"data": ds, "model": "PVC MLP", "seed": s, "row": np.arange(len(day)),
                                          "score": score, "prob": prob})], ignore_index=True)
        sc.to_pickle(cache)
        print(f"{ds:6s} PVC seed={s}  {time.time()-t:5.1f}초", flush=True)

# %%
def seed_scores(ds, seeds):
    g = sc[(sc.data == ds) & (sc.model == "PVC MLP") & sc.seed.isin(seeds)]
    return np.mean([gg.sort_values("row").score.values for _, gg in g.groupby("seed")], 0)

def overlap(a, b, day, ok, frac):
    out = []
    for d in np.unique(day[ok]):
        m = ok & (day == d); k = int(np.ceil(frac * m.sum())); ix = np.flatnonzero(m)
        A = set(ix[np.argsort(-a[m], kind="stable")[:k]]); B_ = set(ix[np.argsort(-b[m], kind="stable")[:k]])
        out.append(len(A & B_) / k)
    return np.array(out)

rows = []
for ds, (X, day, cnt, y, blk) in DATA.items():
    single = [seed_scores(ds, [s]) for s in range(10)]
    a5, b5 = seed_scores(ds, range(5)), seed_scores(ds, range(5, 10))
    ok = ~np.isnan(a5) & ~blk
    for frac in (.05, .1, .2):
        sgl = np.mean([overlap(single[i], single[j], day, ok, frac).mean() for i in range(10) for j in range(i + 1, 10)])
        o5 = overlap(a5, b5, day, ok, frac)
        rows.append({"데이터": ds, "예산": f"{int(frac*100)}%", "단일 시드끼리": round(sgl, 3),
                     "5시드 평균끼리": round(o5.mean(), 3), "5시드 평균끼리 최저 날짜": round(o5.min(), 3)})
st = pd.DataFrame(rows); st.to_csv(br.RES / "seed_ensemble_stability.csv", index=False, encoding="utf-8-sig"); st

# %% [markdown]
# ## Stellantis N.V. 검사 예산 — 5시드 평균 vs 10시드 평균 순위

# %%
X, day, cnt, y, _ = DATA["polito"]
ud, gi = np.unique(day, return_inverse=True); n = np.bincount(gi)
rows = []
for nm, seeds in [("시드 0~4", range(5)), ("시드 5~9", range(5, 10)), ("시드 0~9", range(10))]:
    s_ = seed_scores("polito", seeds)
    r = {"순위 점수": nm}
    for b in (.05, .1, .2, .3):
        p = inclusion_prob(s_, np.ceil(b * n - 1e-9), gi); r[f"{int(b*100)}%"] = round((p * y).sum() / y.sum(), 3)
    rows.append(r)
pd.DataFrame(rows)
