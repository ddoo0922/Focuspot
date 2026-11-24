# %% [markdown]
# # 32. 군집화(비지도 영역 분리)와 불량률 비교
# **질문:** 공정 조건을 비지도로 몇 개 영역(군집)으로 나누면, 불량률이 낮은 영역 = "안정 공정조건"을 정의할 수 있는가? 그 영역은 왜 불량률이 낮은가?
#
# **KAMP의 제약:** 타점별 정답이 없고 날짜별 불량 개수(8일)만 있다. 그래서 군집별 불량률을 직접 셀 수 없다.
# 대신 **날짜마다 군집 구성비가 다르다는 점**을 이용해 추정한다. 날짜 d의 기대 불량 수를 λ_d = Σ_c n_dc · r_c로 두고(n_dc = 그날 군집 c의 타점 수, r_c = 군집 c의 불량률), 포아송 최대우도(EM, r_c ≥ 0)로 r_c를 구한다.
# 날짜가 8개뿐이라 군집 수가 늘면 과적합이 된다. 그래서 아래 세 가지로 판단한다.
# 1. **조건부 검정('순열 p'):** 귀무가설(군집 간 불량률 같음)에서는 총 불량 수가 주어지면 날짜별 개수가 날짜 크기에 비례한 다항분포를 따른다. 이 분포에서 개수를 2,000번 뽑아 다시 맞추고, 실제 우도비가 그보다 큰지 본다.
#    (처음에는 개수를 날짜 사이에 섞는 순열을 썼으나, 날짜 크기가 509~2,000으로 달라 귀무분포가 부풀려졌다. 8-2 검정력 확인에서 발견해 바꿨다.)
# 2. **날짜 하나 빼기(LODO):** 7일로 r_c를 구해 남은 날의 개수를 예측한다. 고정 불량률보다 오차(MAE)가 작은지 본다.
# 3. **Stellantis N.V. 대조:** 타점별 정답이 있는 Stellantis N.V.에 같은 방법을 적용한다. 날짜별 개수만으로 추정한 군집 불량률이 실제 군집 불량률을 되찾는지 확인한다. 위치 효과를 통제한 경우도 따로 본다.
#
# **결과를 보기 전에 정한 판정 기준.** 어떤 군집을 "안정 공정조건"이라 부르려면 셋 다 만족해야 한다.
# (가) KAMP 순열 p < 0.05 / 구성 수(본페로니), (나) KAMP LODO MAE < 고정 불량률 MAE, (다) 같은 구성을 Stellantis N.V.에 적용했을 때 위치 통제 후에도 실제 군집 간 불량률 차이가 있고(같은 위치 안 순열 p < 0.05), 날짜별 개수 추정이 그 차이를 되찾음(추정·실제 순위 상관 > 0).
#
# - 입력: KAMP 4개 값(전류·전압·통전시간·가압력)과 파생 3개(입열량·동저항·I/F). 복사 블록 291행 × 4일은 군집화에서 뺀다. 블록 행은 그날 나머지 행과 같은 군집 구성이라고 가정하고 노출을 비례 확대한다.
# - 군집 구성: 입력 3종(원값4, 원값4+파생3, 공정 상태 = 같은 날 생산 순서 이동평균 50타점) × 알고리즘 9종(K-평균 k=2~5, 가우스 혼합 k=2~5, HDBSCAN) + 영역 분리 2종(전류 3분위, 전류×가압력 중앙값 4칸).
# - 모델을 학습하거나 기존 추천을 바꾸지 않는다.

# %%
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings, time
from scipy import stats
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import br
from br import spec as S, plot as P, stat as T
from br.validity import loc_standardize
from wl import kamp as wlkamp, data as wldata
warnings.filterwarnings("ignore"); P.setup(); pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
t0 = time.time()

K, cnt_day = wlkamp.load_kamp()
K = K.rename(columns={"cur_mean": "cur", "vol_mean": "vol", "weld_len": "time", "force_mean": "force"})
K["day"] = pd.to_datetime(K.date).dt.strftime("%Y-%m-%d")
cnt = pd.Series({pd.Timestamp(d).strftime("%Y-%m-%d"): v for d, v in cnt_day.items()})
LAB = [d for d in cnt.index if not np.isnan(cnt[d])]
KB = K[~K.block].reset_index(drop=True)
n_all, n_nb = K.groupby("day").size(), KB.groupby("day").size()
SCALE = (n_all / n_nb).loc[LAB].values                       # 블록 행 노출 보정 배수
KCNT = cnt.loc[LAB].values.astype(float)
RATE_D = KCNT / n_all.loc[LAB].values
RAW4, RAW7 = ["cur", "vol", "time", "force"], ["cur", "vol", "time", "force", "heat", "dynres", "i_over_f"]
NAME = {"cur": "전류(kA)", "vol": "전압(V)", "time": "통전시간(ms)", "force": "가압력(bar)", "heat": "입열량 V·I·t", "dynres": "동저항 V/I", "i_over_f": "전류/가압력"}
print("군집 대상(블록 제외):", len(KB), "| 정답 있는 날:", len(LAB), "| 불량 합:", int(KCNT.sum()))
print("고정 불량률:", f"{KCNT.sum() / n_all.loc[LAB].sum() * 100:.3f}%")

# %% [markdown]
# ## 0. 공통 함수 — 군집 만들기, 날짜별 개수로 군집 불량률 추정

# %%
def make_Z(D, kind, order):
    if kind == "원값4":
        Z = D[RAW4]
    elif kind == "원값4+파생3":
        Z = D[RAW7]
    else:                                                     # 공정 상태: 같은 날 생산 순서 이동평균(앞뒤 25타점)
        Z = D[RAW4].groupby(order).transform(lambda s: s.rolling(50, center=True, min_periods=10).mean())
    Z = Z.fillna(Z.median())
    return ((Z - Z.mean()) / Z.std()).values


def clusterings(Z, seed=0):
    out = {}
    for k in (2, 3, 4, 5):
        out[f"K-평균 k={k}"] = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    for k in (2, 3, 4, 5):
        out[f"가우스혼합 k={k}"] = GaussianMixture(k, n_init=3, random_state=seed).fit(Z).predict(Z)
    out["HDBSCAN"] = HDBSCAN(min_cluster_size=max(50, len(Z) // 50)).fit_predict(Z)    # -1 = 잡음(별도 군집으로 둠)
    return out


def regions(D):
    c3 = pd.qcut(D.cur.rank(method="first"), 3, labels=False).values
    cf = (D.cur > D.cur.median()).astype(int).values * 2 + (D.force > D.force.median()).astype(int).values
    return {"영역: 전류 3분위": c3, "영역: 전류×가압력 중앙값": cf}


def em_rates(N, Kc, iters=3000):
    """포아송 λ_d = Σ_c N_dc r_c, r_c ≥ 0 의 최대우도(EM). N: 날짜×군집 노출, Kc: 반복×날짜 개수. 노출 0인 군집은 NaN."""
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


def assess(lab, day, days, kc, scale, B=2000, seed=0):
    """날짜별 개수만으로 군집 불량률을 추정하고, 순열 p·LODO MAE·날짜 구성비 상관을 낸다."""
    cats = np.unique(lab)
    N = pd.crosstab(pd.Series(day), pd.Series(lab)).reindex(index=days, columns=cats).fillna(0).values * scale[:, None]
    r = em_rates(N, kc)[0]
    r0 = kc.sum() / N.sum()
    ll0 = loglik(N, kc[None], np.full((1, len(cats)), r0))[0]
    LR = 2 * (loglik(N, kc[None], r[None])[0] - ll0)
    rng = np.random.default_rng(seed)
    Kp = rng.multinomial(int(kc.sum()), N.sum(1) / N.sum(), size=B).astype(float)   # 귀무: 총 불량 수 고정, 날짜 크기 비례 배분
    LRp = 2 * (loglik(N, Kp, em_rates(N, Kp)) - loglik(N, Kp, np.tile(Kp.sum(1, keepdims=True) / N.sum(), (1, len(cats)))))
    p = (1 + np.sum(LRp >= LR - 1e-9)) / (B + 1)
    e1, e0 = [], []
    for i in range(len(days)):
        tr = np.arange(len(days)) != i
        ri = em_rates(N[tr], kc[tr])[0]
        e1.append(abs(np.nan_to_num(ri) @ N[i] - kc[i])); e0.append(abs(kc[tr].sum() / N[tr].sum() * N[i].sum() - kc[i]))
    share = N / N.sum(1, keepdims=True); rate_d = kc / N.sum(1)
    rho = [stats.spearmanr(share[:, j], rate_d)[0] if share[:, j].std() > 0 else np.nan for j in range(len(cats))]
    return dict(cats=cats, rates=r, n=N.sum(0), LR=LR, p=p, mae=np.mean(e1), mae0=np.mean(e0), rho=np.array(rho), N=N)

# %% [markdown]
# ## 1. KAMP — 군집 구성 31개(입력 3종 × 알고리즘 9종 + 영역 분리 2종)

# %%
kinds = ["원값4", "원값4+파생3", "공정상태(이동평균50)"]
LABS = {}
for kind in kinds:
    Z = make_Z(KB, kind, KB.day)
    for nm, lab in clusterings(Z).items():
        LABS[(kind, nm)] = lab
    print(f"  {kind} 완료 ({time.time() - t0:.0f}s)")
for nm, lab in regions(KB).items():
    LABS[("영역 분리", nm)] = lab
ok = KB.day.isin(LAB).values
rows, RESK = [], {}
for key, lab in LABS.items():
    a = assess(lab[ok], KB.day.values[ok], LAB, KCNT, SCALE)
    RESK[key] = a
    rr = a["rates"] * 100
    rows.append({"입력": key[0], "군집": key[1], "군집 수": len(a["cats"]), "최소 군집 크기": int(np.bincount(pd.factorize(lab)[0]).min()),
                 "추정 불량률 최저%": np.nanmin(rr), "추정 불량률 최고%": np.nanmax(rr), "우도비": a["LR"], "순열 p": a["p"],
                 "LODO MAE": a["mae"], "고정률 MAE": a["mae0"]})
SK = pd.DataFrame(rows)
NCONF = len(SK); ALPHA = 0.05 / NCONF
SK["(가) 순열 통과"] = SK["순열 p"] < ALPHA; SK["(나) LODO 개선"] = SK["LODO MAE"] < SK["고정률 MAE"]
SK.round(3).to_csv(br.RES / "cluster_kamp_summary.csv", index=False, encoding="utf-8-sig")
print(f"구성 수 {NCONF}, 본페로니 기준 p < {ALPHA:.4f} | 소요 {time.time() - t0:.0f}s")
display(SK.round(3))
print("조건부 p < 0.05(보정 전):", int((SK['순열 p'] < .05).sum()), "/", NCONF, "| (가) 통과:", int(SK['(가) 순열 통과'].sum()),
      "| (나) LODO 개선:", int(SK['(나) LODO 개선'].sum()), "| 둘 다:", int((SK['(가) 순열 통과'] & SK['(나) LODO 개선']).sum()))
print("보정 전 p 분포가 균등(무신호)이면 기대되는 p<0.05 개수:", f"{NCONF * .05:.1f}")

# %% [markdown]
# ## 2. 군집이 안정적으로 다시 나오는가 (시드 0~4, 원값4)
# 같은 데이터를 시드만 바꿔 다시 나눴을 때 군집 배정이 얼마나 같은지(조정 랜드 지수, 1 = 완전히 같음).

# %%
Z4 = make_Z(KB, "원값4", KB.day)
stab = []
for k in (2, 3, 4, 5):
    for alg in ("K-평균", "가우스혼합"):
        labs = [KMeans(k, n_init=10, random_state=s).fit_predict(Z4) if alg == "K-평균" else GaussianMixture(k, n_init=3, random_state=s).fit(Z4).predict(Z4) for s in range(5)]
        aris = [adjusted_rand_score(labs[0], l) for l in labs[1:]]
        stab.append({"알고리즘": alg, "k": k, "시드 간 ARI 평균": np.mean(aris), "최소": np.min(aris)})
display(pd.DataFrame(stab).round(3))
print("원값4 군집 간 ARI(K-평균 k=3 vs 가우스혼합 k=3):", round(adjusted_rand_score(LABS[("원값4", "K-평균 k=3")], LABS[("원값4", "가우스혼합 k=3")]), 3))
print("원값4 K-평균 k=3 vs 공정상태 K-평균 k=3:", round(adjusted_rand_score(LABS[("원값4", "K-평균 k=3")], LABS[("공정상태(이동평균50)", "K-평균 k=3")]), 3))

# %% [markdown]
# ## 3. 군집 모습과 날짜별 구성비 (대표 구성: 원값4 K-평균 k=3, 결과를 보기 전에 정함)
# 각 군집의 조건 범위(10~90 백분위)와 추정 불량률. 추정 불량률 구간은 8일을 복원추출한 부트스트랩(500회)이라 대략적이다.

# %%
def profile(D, lab, a, boot=True, kc=None, scale=None, days=None, okm=None):
    rows = []
    rb = None
    if boot:
        rng = np.random.default_rng(1); N = a["N"]; rs = []
        for _ in range(500):
            i = rng.integers(0, len(kc), len(kc)); rs.append(em_rates(N[i], kc[i])[0])
        rb = np.array(rs) * 100
    for j, c in enumerate(a["cats"]):
        m = lab == c; r = {"군집": int(c), "타점 수": int(m.sum()), "비율%": m.mean() * 100}
        for v in RAW4 + ["heat", "i_over_f"]:
            q = D.loc[m, v].quantile([.1, .5, .9]).values
            r[NAME[v]] = f"{q[1]:.3g} [{q[0]:.3g}~{q[2]:.3g}]"
        r["추정 불량률%"] = a["rates"][j] * 100
        if rb is not None:
            r["부트 95%"] = f"[{np.nanquantile(rb[:, j], .025):.2f}, {np.nanquantile(rb[:, j], .975):.2f}]"
        r["날짜 구성비↔일별 불량률 ρ"] = a["rho"][j]
        rows.append(r)
    return pd.DataFrame(rows)

REP = ("원값4", "K-평균 k=3")
lab_rep = LABS[REP]
PR = profile(KB, lab_rep, RESK[REP], kc=KCNT)
display(PR.round(3))
PR.round(4).to_csv(br.RES / "cluster_kamp_profiles_rep.csv", index=False, encoding="utf-8-sig")
SH = pd.crosstab(KB.day, lab_rep, normalize="index") * 100
SH["불량 수"] = cnt; SH["일별 불량률%"] = cnt / n_all * 100
print("날짜별 군집 구성비(%)와 일별 불량률:"); display(SH.round(2))
a = RESK[REP]
for j, c in enumerate(a["cats"]):
    share = a["N"][:, j] / a["N"].sum(1)
    r0, p2, _ = T.perm_spearman(share, RATE_D)
    print(f"  군집 {c}: 구성비↔일별 불량률 ρ={r0:+.2f}, 정확 순열 p(양측)={p2:.3f}")
print("3/27(정답 없음) 예상 불량 수(대표 구성):",
      f"{np.nansum(a['rates'] * pd.Series(lab_rep[KB.day.values == '2020-03-27']).value_counts().reindex(a['cats']).fillna(0).values) * n_all['2020-03-27'] / n_nb['2020-03-27']:.2f}")

# %% [markdown]
# ## 4. 우도비가 가장 큰 구성 (결과를 보고 고른 것 — 탐색용)

# %%
best = SK.sort_values("순열 p").head(3)
for _, b in best.iterrows():
    key = (b["입력"], b["군집"])
    print(f"\n▶ {key}  순열 p={b['순열 p']:.4f}  LODO MAE {b['LODO MAE']:.2f} vs 고정률 {b['고정률 MAE']:.2f}")
    display(profile(KB, LABS[key], RESK[key], kc=KCNT).round(3))

# %% [markdown]
# ## 5. Stellantis N.V. 대조 — 같은 방법이 실제 군집 불량률을 되찾는가
# Stellantis N.V.는 타점별 정답이 있으므로 (1) 군집별 **실제** 불량률과 그 차이의 검정, (2) 날짜별 개수만으로 한 **추정**을 비교할 수 있다.
# - 가압력이 0인 284타점(불량 8)은 4개 값 군집에서 뺀다 → 1,692타점, 불량 71.
# - 위치 교란: 군집이 특정 위치와 겹치면 군집 불량률 차이는 위치 차이일 뿐이다. 그래서 (a) 군집과 위치의 겹침(크라메르 V), (b) **같은 위치 안에서 정답을 섞은 순열 검정**을 함께 낸다.
# - 위치 통제 입력: 위치마다 표준화한 값(KAMP는 설비·제품 1개라 위치 차이가 없는 조건에 가깝다).

# %%
P0 = wldata.load_polito()
X7, pday, _, py, _ = S.load_polito("raw")
okf = X7.force.notna().values
PD = X7[okf].reset_index(drop=True); PD["day"] = pday[okf]; PD["loc"] = P0["loc"].values[okf]; PD["y"] = py[okf].astype(int)
PDAYS = sorted(PD.day.unique()); PK = PD.groupby("day").y.sum().reindex(PDAYS).values.astype(float)
print("Stellantis N.V. 대상:", len(PD), "불량", PD.y.sum(), "날짜", len(PDAYS))


def G_stat(lab, y):
    t = pd.crosstab(lab, y).values
    if t.shape[1] < 2: return 0.0
    return stats.chi2_contingency(t, lambda_="log-likelihood", correction=False)[0]


def within_loc_perm(lab, y, loc, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y)
    base = np.lexsort((np.arange(n), loc)); g0 = G_stat(lab, y); cnt_ = 0
    for _ in range(B):
        idx = np.lexsort((rng.random(n), loc)); yp = np.empty(n, int); yp[base] = y[idx]
        cnt_ += G_stat(lab, yp) >= g0 - 1e-9
    return (1 + cnt_) / (B + 1)


def cramers_v(a, b):
    t = pd.crosstab(a, b).values; chi2 = stats.chi2_contingency(t, correction=False)[0]
    return np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))


prow = []
for ctrl in ("원 입력", "위치 표준화"):
    for kind in kinds:
        if ctrl == "원 입력":
            Z = make_Z(PD, kind, PD.day)
        else:
            base = loc_standardize(PD[RAW7], PD["loc"].values)
            tmp = base.copy(); tmp["day"] = PD.day.values
            Z = make_Z(tmp, kind, tmp.day)
        CL = clusterings(Z)
        if kind == "원값4":
            src = PD if ctrl == "원 입력" else pd.concat([loc_standardize(PD[RAW7], PD["loc"].values)], axis=1)
            CL.update(regions(src))
        for nm, lab in CL.items():
            y = PD.y.values
            tr = pd.Series(y).groupby(lab).mean().reindex(np.unique(lab)).values
            a = assess(lab, PD.day.values, PDAYS, PK, np.ones(len(PDAYS)), B=1000)
            rho_est = stats.spearmanr(a["rates"], tr)[0] if len(tr) > 2 else np.sign(np.corrcoef(a["rates"], tr)[0, 1])
            prow.append({"입력 통제": ctrl, "입력": kind if not nm.startswith("영역") else "영역 분리", "군집": nm, "군집 수": len(tr),
                         "실제 불량률 최저%": tr.min() * 100, "실제 불량률 최고%": tr.max() * 100,
                         "실제 차이 p(정답 섞기)": stats.chi2_contingency(pd.crosstab(lab, y).values, correction=False)[1],
                         "실제 차이 p(같은 위치 안 섞기)": within_loc_perm(lab, y, PD["loc"].values, B=1000),
                         "군집↔위치 V": cramers_v(lab, PD["loc"].values),
                         "추정 순열 p(날짜 개수)": a["p"], "추정↔실제 순위상관": rho_est,
                         "LODO MAE": a["mae"], "고정률 MAE": a["mae0"]})
        print(f"  Stellantis N.V. {ctrl} · {kind} 완료 ({time.time() - t0:.0f}s)")
SP = pd.DataFrame(prow)
SP.round(4).to_csv(br.RES / "cluster_polito_summary.csv", index=False, encoding="utf-8-sig")
display(SP.round(3))
for ctrl in ("원 입력", "위치 표준화"):
    s = SP[SP["입력 통제"] == ctrl]
    print(f"{ctrl}: 실제 차이 p<0.05 {int((s['실제 차이 p(정답 섞기)'] < .05).sum())}/{len(s)}, 같은 위치 안 p<0.05 {int((s['실제 차이 p(같은 위치 안 섞기)'] < .05).sum())}/{len(s)}, "
          f"날짜 개수 추정 순열 p<0.05 {int((s['추정 순열 p(날짜 개수)'] < .05).sum())}/{len(s)}, 추정↔실제 순위상관 중앙값 {s['추정↔실제 순위상관'].median():+.2f}, "
          f"LODO 개선 {int((s['LODO MAE'] < s['고정률 MAE']).sum())}/{len(s)}")

# %% [markdown]
# ## 6. 판정 — 사전 기준 (가)·(나)·(다)

# %%
J = SK.merge(SP[SP["입력 통제"] == "위치 표준화"][["입력", "군집", "실제 차이 p(같은 위치 안 섞기)", "추정↔실제 순위상관"]], on=["입력", "군집"], how="left")
J["(다) Stellantis N.V. 위치 통제 확인"] = (J["실제 차이 p(같은 위치 안 섞기)"] < .05) & (J["추정↔실제 순위상관"] > 0)
J["안정 공정조건 판정"] = J["(가) 순열 통과"] & J["(나) LODO 개선"] & J["(다) Stellantis N.V. 위치 통제 확인"]
display(J[["입력", "군집", "순열 p", "(가) 순열 통과", "(나) LODO 개선", "실제 차이 p(같은 위치 안 섞기)", "추정↔실제 순위상관", "(다) Stellantis N.V. 위치 통제 확인", "안정 공정조건 판정"]].round(3))
print("기준 통과 구성:", int(J["안정 공정조건 판정"].sum()), "/", len(J))
J.round(4).to_csv(br.RES / "cluster_judgement.csv", index=False, encoding="utf-8-sig")

# %% [markdown]
# ## 7. 그림

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
rng = np.random.default_rng(0); sub = rng.choice(len(KB), 4000, replace=False)
for j, c in enumerate(np.unique(lab_rep)):
    m = sub[lab_rep[sub] == c]
    ax[0].scatter(KB.cur.values[m] + rng.normal(0, .004, len(m)), KB.force.values[m] + rng.normal(0, .003, len(m)), s=4, alpha=.35, color=P.PAL[j], label=f"군집 {c}")
ax[0].set(xlabel="전류(kA)", ylabel="가압력(bar)", title="(a) KAMP 원값4 K-평균 k=3 (4,000타점 표본)"); ax[0].legend(markerscale=4)
shr = SH.loc[LAB, np.unique(lab_rep)]
bot = np.zeros(len(LAB))
for j, c in enumerate(shr.columns):
    ax[1].bar(range(len(LAB)), shr[c], bottom=bot, color=P.PAL[j], alpha=.8, label=f"군집 {c}"); bot += shr[c].values
ax2 = ax[1].twinx(); ax2.plot(range(len(LAB)), RATE_D * 100, "k-o", label="일별 불량률"); ax2.set_ylabel("일별 불량률(%)"); ax2.grid(False)
ax[1].set_xticks(range(len(LAB))); ax[1].set_xticklabels([d[5:] for d in LAB]); ax[1].set(ylabel="군집 구성비(%)", title="(b) 날짜별 구성비와 불량률"); ax[1].legend(loc="lower left", fontsize=8)
o = SK.sort_values("순열 p").reset_index(drop=True)
ax[2].scatter(range(len(o)), o["순열 p"], color=[P.PAL[kinds.index(k)] if k in kinds else P.PAL[3] for k in o["입력"]], s=22)
ax[2].plot(range(len(o)), (np.arange(len(o)) + 1) / len(o), color=P.GRAY, ls="--", label="무신호 기대(균등)")
ax[2].axhline(ALPHA, color=P.PAL[7], ls=":", label=f"본페로니 {ALPHA:.4f}")
ax[2].set_yscale("log"); ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax[2].set(xlabel="구성(p 순)", ylabel="조건부 p", title=f"(c) KAMP {NCONF}개 구성의 조건부 p"); ax[2].legend(fontsize=8)
fig.tight_layout(); P.save(fig, "25_kamp_clusters"); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, ctrl in enumerate(("원 입력", "위치 표준화")):
    s = SP[SP["입력 통제"] == ctrl]
    ax[i].scatter(s["실제 차이 p(같은 위치 안 섞기)"], s["추정 순열 p(날짜 개수)"], c=[P.PAL[0] if v < .05 else P.GRAY for v in s["실제 차이 p(정답 섞기)"]], s=26)
    ax[i].axvline(.05, color=P.GRAY, ls="--"); ax[i].axhline(.05, color=P.GRAY, ls="--")
    ax[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); ax[i].set(xscale="log", yscale="log", xlabel="실제 군집 차이 p (같은 위치 안 정답 섞기)", ylabel="날짜 개수 추정 조건부 p", title=f"Stellantis N.V. {ctrl} (파랑 = 위치 무시 시 실제 차이 p<0.05)")
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); ax[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
fig.tight_layout(); P.save(fig, "26_polito_clusters"); plt.show()
print(f"총 소요 {time.time() - t0:.0f}s")

# %% [markdown]
# ## 8. 추가 확인 (결과를 본 뒤 추가)
# **8-1. LODO 개선이 진짜인가.** 군집 모델은 매개변수가 많아 신호가 없어도 LODO가 좋아질 수 있다. 귀무가설(군집 간 차이 없음)에서 뽑은 개수 300벌로 LODO MAE를 다시 계산해, 실제 MAE가 그보다 작은 비율을 본다('LODO 순열 p'라 부르지만 조건부 다항 귀무).

# %%
def lodo_null(N, kc, B=300, seed=0):
    rng = np.random.default_rng(seed)
    Kp = np.vstack([kc[None], rng.multinomial(int(kc.sum()), N.sum(1) / N.sum(), size=B).astype(float)])
    err = np.zeros(len(Kp))
    for i in range(len(kc)):
        tr = np.arange(len(kc)) != i
        r = em_rates(N[tr], Kp[:, tr])
        err += np.abs(np.nan_to_num(r) @ N[i] - Kp[:, i])
    err /= len(kc)
    return err[0], (1 + np.sum(err[1:] <= err[0] + 1e-9)) / (B + 1), np.median(err[1:])

ln = []
for key, a in RESK.items():
    m, p, med = lodo_null(a["N"], KCNT)
    ln.append({"입력": key[0], "군집": key[1], "LODO MAE": m, "귀무 LODO MAE 중앙값": med, "LODO 순열 p": p})
LN = pd.DataFrame(ln)
display(LN.round(3))
print("LODO 순열 p < 0.05:", int((LN['LODO 순열 p'] < .05).sum()), "/", len(LN), "| 귀무에서도 고정률(2.19)보다 나은 구성:", int((LN['귀무 LODO MAE 중앙값'] < 2.19).sum()))
LN.round(4).to_csv(br.RES / "cluster_kamp_lodo_null.csv", index=False, encoding="utf-8-sig")

# %% [markdown]
# **8-2. 검정력: 8일로 군집 차이를 찾을 수 있는가.** 대표 구성 중 시드에 흔들리지 않는 원값4 K-평균 k=2(ARI 1.0)의 실제 날짜별 노출을 그대로 둔다.
# 두 군집의 참 불량률 비가 R배라고 가정하고(전체 불량률 0.379% 유지) 날짜별 개수를 포아송으로 100번 만든 뒤, 같은 조건부 검정(500회)이 p < 0.05를 내는 비율을 센다.

# %%
a2 = RESK[("원값4", "K-평균 k=2")]; N2 = a2["N"]
print("K-평균 k=2 군집 크기:", a2["n"].round(0), "| 날짜별 군집1 구성비:", (N2[:, 1] / N2.sum(1)).round(2))
pw = []
rng = np.random.default_rng(7)
for R in (1.5, 2, 3, 5, 10):
    for hi in (0, 1):
        r = np.ones(2); r[hi] = R
        r = r * KCNT.sum() / (N2 @ r).sum()
        lam = N2 @ r
        hits = 0
        for s in range(100):
            kc = rng.poisson(lam).astype(float)
            Kp = np.vstack([kc[None], rng.multinomial(int(kc.sum()), N2.sum(1) / N2.sum(), size=500).astype(float)])
            r0 = Kp.sum(1, keepdims=True) / N2.sum()
            LR = 2 * (loglik(N2, Kp, em_rates(N2, Kp, 1500)) - loglik(N2, Kp, np.tile(r0, (1, 2))))
            hits += (1 + np.sum(LR[1:] >= LR[0] - 1e-9)) / 501 < .05
        pw.append({"참 불량률 비 R": R, "높은 쪽 군집": hi, "검정력(p<0.05)": hits / 100})
    print(f"  R={R} 완료 ({time.time() - t0:.0f}s)")
PW = pd.DataFrame(pw); display(PW)
PW.to_csv(br.RES / "cluster_kamp_power.csv", index=False, encoding="utf-8-sig")
