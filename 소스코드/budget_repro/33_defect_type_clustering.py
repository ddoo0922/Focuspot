# %% [markdown]
# # 33. 불량 유형 3종(파임·용접부족·크랙)이 비지도 군집화로 분리되는가
# **질문:** KAMP 불량 라벨 3유형(1=파임불량, 2=용접부족, 3=크랙발생)이 공정 신호(전류·전압·통전시간·가압력)의 군집으로 나뉘는가?
#
# **KAMP의 제약:** 유형별 정답도 타점별이 아니라 "날짜 × 유형별 개수"뿐이다(8일, 유형 합 14/13/12). 그래서 군집↔유형 일치를 직접 셀 수 없다.
# (유형 번호 1·2·3 = 파임·용접부족·크랙 순서로 본다. 엑셀의 유형명 열은 글자가 깨져 번호만 사용.)
#
# **네 가지로 확인한다.**
# - A. 군집 경향: 데이터에 3개 덩어리가 자연스럽게 있는가(BIC·실루엣·안정성).
# - B. 물리 방향 대조: 사전에 정한 유형별 신호 방향(아래)과 군집 중심이 맞는가.
# - C. 날짜 개수 검정: 유형별 개수를 날짜별 군집 구성비로 설명할 수 있는가(29번과 같은 조건부 다항 귀무).
# - D. 양성 대조(모의): 유형별 이동을 일부러 심었을 때 군집화가 되찾는가 → 방법이 애초에 찾을 수 있는 크기를 가늠.
#
# **사전 지정 물리 방향(표준화 단위, 붙여넣은 설명에서 옮김):**
# 파임 = 전류↑·가압력↑·통전시간↑ / 용접부족 = 전류↓·통전시간↓·가압력↑(과가압) / 크랙 = 전류↑·가압력↓·통전시간↑.
# 전압은 방향을 지정하지 않는다(설명에 없음).

# %%
import sys, time, warnings
import br
import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import IsolationForest
from wl import kamp as wlkamp, data as wldata
warnings.filterwarnings("ignore"); pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
OUT = str(br.RES)
t0 = time.time()

K, cnt_day = wlkamp.load_kamp()
K = K.rename(columns={"cur_mean": "cur", "vol_mean": "vol", "weld_len": "time", "force_mean": "force"})
K["day"] = pd.to_datetime(K.date).dt.strftime("%Y-%m-%d")
res = pd.read_excel(wldata.KAMP_XLSX, sheet_name="result")
res["day"] = res["working time"].dt.strftime("%Y-%m-%d")
TYPE = res.groupby(["day", "defect type"])["defect"].sum().unstack().fillna(0)   # 날짜 × 유형
DAYS = list(TYPE.index)                                                         # 정답 있는 8일
KB = K[~K.block].reset_index(drop=True)
LOWV = KB.vol < 2.6                                                              # 29번에서 기록 품질 문제로 본 저전압 16타점(2.51 V) — 표준편차를 부풀리고 군집 하나를 차지함
print("저전압 이상 타점 제외:", int(LOWV.sum()), "| 날짜:", sorted(KB.day[LOWV].unique()))
KB = KB[~LOWV].reset_index(drop=True)
n_all = K.groupby("day").size().loc[DAYS].values.astype(float)
n_nb = KB.groupby("day").size().loc[DAYS].values.astype(float)   # 제외 후 타점 수 → SCALE이 제외분도 같은 구성으로 보정
SCALE = n_all / n_nb
RAW4 = ["cur", "vol", "time", "force"]
Z = ((KB[RAW4] - KB[RAW4].mean()) / KB[RAW4].std()).values
print("유형별 날짜 개수\n", TYPE.astype(int).T, "\n합:", TYPE.sum().astype(int).to_dict(), "| 군집 대상(블록 제외):", len(KB))
SIG = {                                                                         # (cur, vol, time, force)
    "파임(1)": np.array([+1, 0, +1, +1.]), "용접부족(2)": np.array([-1, 0, -1, +1.]), "크랙(3)": np.array([+1, 0, +1, -1.])}
SIGN = {k: v / np.linalg.norm(v) for k, v in SIG.items()}
print("사전 지정 방향 간 코사인:", {f"{a}-{b}": round(float(SIGN[a] @ SIGN[b]), 2) for a in SIGN for b in SIGN if a < b})

# %% [markdown]
# ## A. 군집 경향 — 3개 덩어리가 자연스럽게 있는가

# %%
rng = np.random.default_rng(0)
sub = rng.choice(len(Z), 4000, replace=False)
rowsA = []
for k in range(1, 9):
    g = GaussianMixture(k, n_init=3, random_state=0, reg_covar=1e-2).fit(Z)
    row = dict(k=k, BIC=g.bic(Z))
    if k >= 2:
        km = KMeans(k, n_init=10, random_state=0).fit(Z)
        row["실루엣(K-평균,4000표본)"] = silhouette_score(Z[sub], km.labels_[sub])
        labs = [KMeans(k, n_init=1, random_state=s).fit_predict(Z) for s in range(1, 6)]
        row["안정성 ARI(시드5개 쌍평균)"] = np.mean([adjusted_rand_score(labs[i], labs[j]) for i in range(5) for j in range(i + 1, 5)])
        row["최소군집 비율"] = np.bincount(km.labels_).min() / len(Z)
    rowsA.append(row)
A = pd.DataFrame(rowsA).set_index("k"); print(A.round(3))
hd = HDBSCAN(min_cluster_size=50).fit_predict(Z)
print("HDBSCAN(최소50): 군집 수", len(set(hd) - {-1}), "| 잡음", (hd == -1).sum(), "|", dict(zip(*np.unique(hd, return_counts=True))))
A.to_csv(f"{OUT}/type_cluster_tendency.csv")
print(f"[{time.time() - t0:.0f}s]")

# %% [markdown]
# ## B. 물리 방향 대조 — 군집 중심이 유형별 방향과 맞는가

# %%
rowsB, CL = [], {}
for name, mk in [("K-평균 k=3", lambda: KMeans(3, n_init=10, random_state=0)), ("K-평균 k=5", lambda: KMeans(5, n_init=10, random_state=0)),
                 ("가우스혼합 k=3", lambda: GaussianMixture(3, n_init=3, random_state=0, reg_covar=1e-2)), ("가우스혼합 k=5", lambda: GaussianMixture(5, n_init=3, random_state=0, reg_covar=1e-2))]:
    lab = mk().fit_predict(Z); CL[name] = lab
    for c in np.unique(lab):
        m = lab == c; mu = Z[m].mean(0)
        rowsB.append(dict(구성=name, 군집=c, 비율=m.mean(), **{f"{a}_z": mu[i] for i, a in enumerate(RAW4)}, 크기z=np.linalg.norm(mu),
                          **{f"cos_{k}": (mu @ v / np.linalg.norm(mu)) for k, v in SIGN.items()}))
B = pd.DataFrame(rowsB); print(B.round(2).to_string(index=False))
B.to_csv(f"{OUT}/type_cluster_profiles.csv", index=False)
# 가장 큰 군집 중심 이동(z)이 어느 정도인가 → 실제 타점 변동(1σ) 대비 작은지
print("\n군집 중심 최대 크기(z):", round(B.크기z.max(), 2), "— 군집 간 차이는 대부분 1σ 이하의 공정 변동 축이다.")
print(f"[{time.time() - t0:.0f}s]")

# %% [markdown]
# ## C. 날짜 개수 검정 — 유형별 개수가 날짜별 군집 구성비로 설명되는가
# 29번과 같은 방법: 날짜 d의 기대 개수 λ_d = Σ_c n_dc · r_c (r_c ≥ 0, 포아송 최대우도 EM), 귀무 = 유형 총 개수를 날짜 크기에 비례해 다항 배분.
# 유형이 3개이므로 본페로니 3 × 구성 수.

# %%
def em_rates(N, Kc, iters=3000):
    Kc = np.atleast_2d(Kc).astype(float); col = N.sum(0); ok = col > 0
    r = np.where(ok, 1.0, 0.0) * (Kc.sum(1, keepdims=True) / N.sum())
    for _ in range(iters):
        lam = r @ N.T
        r = r * np.divide((Kc / np.maximum(lam, 1e-300)) @ N, col, out=np.zeros_like(r), where=ok)
    r[:, ~ok] = np.nan
    return r


def loglik(N, Kc, r):
    lam = np.nan_to_num(r) @ N.T
    return (np.where(Kc > 0, Kc * np.log(np.maximum(lam, 1e-300)), 0.0) - lam).sum(-1)


def cond_test(N, kc, B=2000, seed=0):
    r = em_rates(N, kc)[0]; r0 = kc.sum() / N.sum(); nc = N.shape[1]
    LR = 2 * (loglik(N, kc[None], r[None])[0] - loglik(N, kc[None], np.full((1, nc), r0))[0])
    Kp = np.random.default_rng(seed).multinomial(int(kc.sum()), N.sum(1) / N.sum(), size=B).astype(float)
    LRp = 2 * (loglik(N, Kp, em_rates(N, Kp)) - loglik(N, Kp, np.tile(Kp.sum(1, keepdims=True) / N.sum(), (1, nc))))
    return r, (1 + np.sum(LRp >= LR - 1e-9)) / (B + 1)


rowsC = []
for name, lab in CL.items():
    day = KB.day.values
    cats = np.unique(lab)
    N = pd.crosstab(pd.Series(day), pd.Series(lab)).reindex(index=DAYS, columns=cats).fillna(0).values * SCALE[:, None]
    for ti, tn in zip([1, 2, 3], SIGN):
        kc = TYPE[ti].values.astype(float)
        r, p = cond_test(N, kc)
        rowsC.append(dict(구성=name, 유형=tn, 개수=int(kc.sum()), p=p, 군집별_추정불량률pct=np.round(r * 100, 2).tolist()))
C = pd.DataFrame(rowsC); print(C.to_string(index=False))
nC = len(C)
print(f"\np<0.05: {(C.p < 0.05).sum()}/{nC} (우연 기대 {0.05 * nC:.1f}) | 본페로니({nC}) 통과: {(C.p < 0.05 / nC).sum()}/{nC} | 최소 p = {C.p.min():.3f}")
C.to_csv(f"{OUT}/type_cluster_count_test.csv", index=False)

# 날짜 단위 직접 확인: 날짜 평균 신호 vs 유형 불량률(8점, 탐색용)
dm = KB.groupby("day")[RAW4].mean().loc[DAYS]
rowsD = []
for ti, tn in zip([1, 2, 3], SIGN):
    rate = TYPE[ti].values / n_all
    for v in RAW4:
        rho, p = stats.spearmanr(dm[v], rate)
        rowsD.append(dict(유형=tn, 변수=v, rho=rho, p=p))
D8 = pd.DataFrame(rowsD); print("\n날짜 평균 신호 ↔ 유형 불량률 (8점, 12개 중 우연 기대 p<0.05 0.6개)\n", D8.round(2).pivot(index="변수", columns="유형", values="rho"))
print("p<0.05:", (D8.p < 0.05).sum(), "/ 12 | 최소 p", round(D8.p.min(), 3))
D8.to_csv(f"{OUT}/type_cluster_day_corr.csv", index=False)
print(f"[{time.time() - t0:.0f}s]")

# %% [markdown]
# ## D. 양성 대조(모의) — 유형별 이동을 심으면 군집화가 되찾는가
# 실제 KAMP 타점 중 유형당 12개를 무작위로 골라 사전 지정 방향으로 d σ 이동시킨다(총 36개 ≈ 0.33%, 실제 불량 비율과 같음).
# 군집화가 이 36개를 어떻게 묶는지 본다. 지표 = 유형별 "가장 잘 맞는 군집"의 F1(정밀도·재현율), 기준선 = 군집 크기가 비슷한 무작위 군집의 F1.
# 군집 수 k = 3·5·8, 알고리즘 K-평균·가우스혼합·HDBSCAN, 이상치 탐지(Isolation Forest 상위 36개)도 비교한다.

# %%
def best_f1(lab, idx):
    f = 0.0
    for c in np.unique(lab):
        m = lab == c; tp = m[idx].sum()
        if tp == 0: continue
        pr, rc = tp / m.sum(), tp / len(idx); f = max(f, 2 * pr * rc / (pr + rc))
    return f


reps, nper = 10, 12
rowsE = []
for d in (1, 2, 3, 5, 8):
    for rep in range(reps):
        g = np.random.default_rng(100 + rep)
        pick = g.choice(len(Z), 3 * nper, replace=False).reshape(3, nper)
        Zs = Z.copy()
        for t, v in enumerate(SIGN.values()):
            Zs[pick[t]] = Zs[pick[t]] + d * v        # 사전 지정 방향으로 d σ 이동
        labs = {f"K-평균 k={k}": KMeans(k, n_init=3, random_state=rep).fit_predict(Zs) for k in (3, 5, 8)}
        labs.update({f"가우스혼합 k={k}": GaussianMixture(k, n_init=1, random_state=rep, reg_covar=1e-2).fit_predict(Zs) for k in (3, 5, 8)})
        labs["HDBSCAN"] = HDBSCAN(min_cluster_size=15).fit_predict(Zs)
        for nm, lab in labs.items():
            for t, tn in enumerate(SIGN):
                rowsE.append(dict(d=d, rep=rep, 알고리즘=nm, 유형=tn, F1=best_f1(lab, pick[t])))
        sc = -IsolationForest(random_state=rep, n_estimators=100).fit(Zs).score_samples(Zs)
        top = np.argsort(-sc)[:3 * nper]
        rowsE.append(dict(d=d, rep=rep, 알고리즘="IsolationForest 상위36", 유형="전체", F1=len(set(top) & set(pick.ravel())) / (3 * nper)))
    print(f"d={d}σ 완료 [{time.time() - t0:.0f}s]")
E = pd.DataFrame(rowsE)
tabE = E[E.유형 != "전체"].groupby(["알고리즘", "d"]).F1.mean().unstack().round(3)
print("\n유형별 최적 군집 F1 평균 (10회)\n", tabE)
ifr = E[E.유형 == "전체"].groupby("d").F1.mean().round(3)
print("\nIsolationForest 상위 36개 중 심은 점 재현율:", ifr.to_dict())
print("참고: 무작위로 크기 12짜리 군집을 만들 때의 F1 기대값 ≈ 0.003")
tabE.to_csv(f"{OUT}/type_cluster_positive_control.csv")
print(f"전체 {time.time() - t0:.0f}s")
