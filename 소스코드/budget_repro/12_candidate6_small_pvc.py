# %% [markdown]
# # 12. 사양 후보 6 — PVC 소형(은닉층 32 → 8, 가중치 감쇠 1e-3 → 1e-2)
# 반복 6에서 사전 지정한 마지막 순위 후보다. 두 데이터에 10시드로 적용한다. 한 번 실행에 약 4.5분만 학습하고 캐시에 쌓는다(두 번 실행).
# 판단: Stellantis N.V. 10% 예산 포착률(10시드 평균 점수 순위)을 현재 PVC와 차체 복원추출로 대응 비교하고, KAMP 5시드 평균끼리의 추천 겹침을 본다.
# 개선이 구간상 확실하지 않으면 현재 사양(PVC 32)을 확정한다.

# %%
import time, warnings
import numpy as np, pandas as pd
import br
from br import spec as S
from wl.metrics import inclusion_prob
from wl import data as wldata
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
NEW = "PVC MLP 소형"
DATA = {"polito": S.load_polito(), "kamp": S.load_kamp()}
cache = br.RES / "same_model_scores.pkl"
sc = pd.read_pickle(cache)
done = set(map(tuple, sc[["data", "model", "seed"]].drop_duplicates().values))
t0 = time.time()
for s in range(10):
    for ds, (X, day, cnt, y, blk) in DATA.items():
        if (ds, NEW, s) in done or time.time() - t0 > 270:
            continue
        t = time.time()
        score, prob = S.lodo(NEW, X, day, cnt, s)
        sc = pd.concat([sc, pd.DataFrame({"data": ds, "model": NEW, "seed": s, "row": np.arange(len(day)),
                                          "score": score, "prob": prob})], ignore_index=True)
        sc.to_pickle(cache)
        print(f"{ds:6s} {NEW} seed={s}  {time.time()-t:5.1f}초", flush=True)
have = sc[sc.model == NEW].groupby("data").seed.nunique().to_dict()
print("학습된 시드 수:", have)
COMPLETE = all(have.get(ds, 0) == 10 for ds in DATA)

# %%
def mean_score(ds, model, seeds=range(10)):
    g = sc[(sc.data == ds) & (sc.model == model) & sc.seed.isin(seeds)]
    return np.mean([gg.sort_values("row").score.values for _, gg in g.groupby("seed")], 0)
if COMPLETE:
    X, day, cnt, y, _ = DATA["polito"]
    body = wldata.load_polito().body.values
    rng = np.random.default_rng(3); ub = np.unique(body)
    Wb = np.stack([np.bincount(d_, minlength=len(ub)) for d_ in rng.integers(0, len(ub), (5000, len(ub)))])[:, np.searchsorted(ub, body)]
    ud, gi = np.unique(day, return_inverse=True); n = np.bincount(gi)
    A, Bm = mean_score("polito", "PVC MLP"), mean_score("polito", NEW)
    rows = []
    for b in (.05, .1, .2, .3):
        k = np.ceil(b * n - 1e-9)
        pa, pb = inclusion_prob(A, k, gi), inclusion_prob(Bm, k, gi)
        d_ = (Wb @ ((pb - pa) * y)) / (Wb @ y)
        rows.append({"예산": f"{int(b*100)}%", "PVC(현재)": round((pa * y).sum() / y.sum(), 3), "PVC 소형": round((pb * y).sum() / y.sum(), 3),
                     "소형−현재 [95%]": f"{d_.mean():+.3f} [{np.quantile(d_,.025):+.3f}, {np.quantile(d_,.975):+.3f}]"})
    display(pd.DataFrame(rows))
    # KAMP 안정성: 5시드 평균끼리 날짜별 상위 10% 겹침
    X, kday, kcnt, _, kblk = DATA["kamp"]
    def ov(model):
        a, b_ = mean_score("kamp", model, range(5)), mean_score("kamp", model, range(5, 10))
        ok = ~np.isnan(a) & ~kblk; out = []
        for d in np.unique(kday[ok]):
            m = ok & (kday == d); k = int(np.ceil(.1 * m.sum())); ix = np.flatnonzero(m)
            out.append(len(set(ix[np.argsort(-a[m])[:k]]) & set(ix[np.argsort(-b_[m])[:k]])) / k)
        return np.mean(out), np.min(out)
    print("KAMP 5시드 평균끼리 상위 10% 겹침 (평균, 최저):", {m: tuple(np.round(ov(m), 3)) for m in ["PVC MLP", NEW]})
    ka, kb = mean_score("kamp", "PVC MLP"), mean_score("kamp", NEW); ok = ~np.isnan(ka) & ~kblk
    o = []
    for d in np.unique(kday[ok]):
        m = ok & (kday == d); k = int(np.ceil(.1 * m.sum())); ix = np.flatnonzero(m)
        o.append(len(set(ix[np.argsort(-ka[m])[:k]]) & set(ix[np.argsort(-kb[m])[:k]])) / k)
    print("KAMP 현재 PVC vs 소형 PVC 상위 10% 겹침:", round(np.mean(o), 3))
