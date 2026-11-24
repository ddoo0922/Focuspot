# %% [markdown]
# # 22. 재현 확인 — 새 검증 실험(23~25) 전에 기존 핵심 수치 두 개를 다시 계산한다
# 1. Stellantis N.V. 시간순(2022~23 학습 → 2024 추천), 10% 예산, 결합 모델: 기존 51개 중 17개
# 2. KAMP raw PVC 10시드: 시드 0~4 평균과 5~9 평균의 날짜별 상위 10% 겹침: 기존 0.849
# 캐시를 쓰지 않고 새로 학습한다. 기존 캐시와 점수가 같은지(결정성)도 본다. 모델 1회 학습 시간을 재서 이후 반복 횟수를 정한다.

# %%
import time, warnings, numpy as np, pandas as pd
import br
from br import spec as S
from wl.metrics import inclusion_prob
warnings.filterwarnings("ignore"); pd.set_option("display.width", 200)
SEEDS = list(range(10))

# 1) Stellantis N.V. 시간순 결합
X, day, cnt, y, _ = S.load_polito(); Xd, *_ = S.load_polito("derived5_absz")
TR = np.array([d_ < "2024" for d_ in day]); TE = np.array([d_ >= "2024-02-01" for d_ in day])
ub, bag = np.unique(day[TR], return_inverse=True)
t = time.time()
raw_t = np.mean([S.fit_score("PVC MLP", X, bag, cnt.loc[ub].values.astype(int), TR, TE, s)[0] for s in SEEDS], 0)
t_fit_polito = (time.time() - t) / len(SEEDS)
dt, yt = day[TE], y[TE]
rk = lambda s: pd.Series(s).groupby(dt).rank(pct=True).values
comb = (rk(raw_t) + rk(Xd.sum(axis=1).values[TE])) / 2
ud, gi = np.unique(dt, return_inverse=True); n = np.bincount(gi)
hit = (inclusion_prob(comb, np.ceil(.1 * n - 1e-9), gi) * yt).sum()
print(f"[1] Stellantis N.V. 시간순 10% 결합: {hit:.1f} / {int(yt.sum())} (기존 17/51) | Stellantis N.V. 시간순 1회 학습 {t_fit_polito:.2f}초")

# %%
# 2) KAMP raw PVC 10시드 LODO 새로 학습
Xk, kday, kcnt, _, kblk = S.load_kamp()
t = time.time(); new = []
for s in SEEDS:
    sc_, _ = S.lodo("PVC MLP", Xk, kday, kcnt, s); new.append(sc_)
    print(f"  KAMP seed {s} 완료 {time.time()-t:.0f}초", flush=True)
t_lodo_kamp = (time.time() - t) / len(SEEDS)
new = np.array(new)
old = pd.read_pickle(br.RES / "same_model_scores.pkl")
old = np.array([g.sort_values("row").score.values for _, g in old[(old.data == "kamp") & (old.model == "PVC MLP")].groupby("seed")])
ok = ~np.isnan(new[0])
print("기존 캐시와 점수 최대 차이:", float(np.nanmax(np.abs(new[:, ok] - old[:, ok]))))
def overlap(a, b, frac=.1):
    o = []
    for d in np.unique(kday[ok & ~kblk]):
        m = ok & ~kblk & (kday == d); k = int(np.ceil(frac * m.sum())); ix = np.flatnonzero(m)
        o.append(len(set(ix[np.argsort(-a[m])[:k]]) & set(ix[np.argsort(-b[m])[:k]])) / k)
    return np.mean(o)
print(f"[2] KAMP raw PVC 5시드 평균끼리 상위 10% 겹침: {overlap(new[:5].mean(0), new[5:].mean(0)):.3f} (기존 0.849) | KAMP LODO 1회(8번 학습) {t_lodo_kamp:.1f}초")

# %%
# 3) 이후 계획용: KAMP 8일 전체 1회 학습 시간, Stellantis N.V. 2022년 가압력 결측 확인
lab = np.isin(kday, kcnt.dropna().index); u, bg = np.unique(kday[lab], return_inverse=True)
t = time.time(); S.fit_score("PVC MLP", Xk, bg, kcnt.loc[u].values.astype(int), lab, lab, 0); print(f"KAMP 8일 전체 1회 학습 {time.time()-t:.2f}초")
y22 = np.array([d_ < "2023" for d_ in day])
print("Stellantis N.V. 2022년 타점", y22.sum(), "| 그중 가압력 결측", int(X.force[y22].isna().sum()), "| 전체 가압력 결측", int(X.force.isna().sum()),
      "| 2022년 불량", int(y[y22].sum()))
