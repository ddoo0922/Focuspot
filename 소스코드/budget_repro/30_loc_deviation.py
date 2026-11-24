# %% [markdown]
# # 30. 위치별 상대 편차를 입력으로 주면 위치 의존도가 줄고 성능이 남는가
# 제안: 값 자체(절대 수치)가 아니라 [현재 타점 값 − 그 위치의 기준값]을 모델 입력으로 준다.
# 평가는 24번과 같다: 시간순(2022~23년 14일 학습 → 2024년 5일 931타점, 불량 51건), 날짜별 5·10·20% 예산, 95% 구간은 시험 차체 9대 복원추출.
#
# 입력 정의(실행 전 고정):
# - raw      : 원래 입력 7개(전류·전압·통전시간·가압력·입열량·동저항·전류/가압력). 24번 base 재현.
# - c1       : 위치별 표준화(전 기간 평균·표준편차). 24번 (c1) 재현.
# - dev      : **제안안.** x − (그 위치의 학습 기간 중앙값). 기준값은 2022~23년 타점만으로 계산한다(정답 미사용).
#              학습 행은 자기 날짜를 뺀 나머지 날짜로 기준값을 구한다(시험 행과 같은 조건). 이력이 없는 위치는 편차 0.
# - devz     : dev를 그 위치의 학습 기간 흩어짐(1.4826·MAD, 하한 = 전체 중앙 MAD의 25%)으로 나눈 것(민감도).
# - raw+dev  : 원래 입력 7개 + dev 7개(14개).
# 모델: PVC MLP 10시드 평균(사양 그대로), 집계 로지스틱(선형), D4 규칙(파생 9개의 날짜 안 |강건 z| 합).
#   D4-dev = 수준 3개를 dev로 바꾼 뒤 같은 순서 변수·같은 규칙. 결합 = 두 순위의 날짜 안 평균.
# 진단: 타점 정답으로 학습한 로지스틱(상한선) — 4개 요약값의 위치 안 편차에 신호가 있기는 한가.
# 지표: 포착 수/무작위, 포착률−무작위 [95%], AUROC, 같은 위치 안 AUROC, 위치마다 10%씩 검사했을 때 포착,
#        위치 설명 비율(점수의 날짜 안 순위 분산 중 위치 평균이 설명하는 비율. 1에 가까울수록 '위치만 보고 고른 것').

# %%
import sys, time, pickle, warnings, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))
import br
from br import spec as S, validity as V
from wl.metrics import inclusion_prob, within_loc_auroc
from wl import data as wldata
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
torch.set_num_threads(2)
RES = br.RES; RES.mkdir(exist_ok=True)

X, day, cnt, y, _ = S.load_polito()
P0 = wldata.load_polito(); loc, body = P0["loc"].values, P0.body.values
TR = np.array([d_ < "2024" for d_ in day]); TE = np.array([d_ >= "2024-02-01" for d_ in day])
SEEDS = range(10)
print(f"학습 {TR.sum()}타점(불량 {y[TR].sum()}건, {len(np.unique(day[TR]))}일) → 시험 {TE.sum()}타점(불량 {y[TE].sum()}건, {len(np.unique(day[TE]))}일)")


# %% [markdown]
# ## 1. 위치별 상대 편차 만들기

# %%
def loc_deviation(Xin, fit_mask, scale=False):
    """x − 위치 기준값. 기준값 = fit_mask 행(학습 기간)의 위치별 중앙값. fit_mask 안의 행은 자기 날짜를 빼고 구한다.
    scale=True면 위치별 1.4826·MAD로 나눈다. 정답을 쓰지 않는다."""
    out = pd.DataFrame(0.0, index=Xin.index, columns=Xin.columns)
    n_hist = np.zeros(len(Xin), int)
    for c in Xin.columns:
        v = Xin[c].values.astype(float)
        dev = np.zeros(len(v));
        glob_mad = np.nanmedian(np.abs(v[fit_mask] - np.nanmedian(v[fit_mask]))) * 1.4826
        for l in np.unique(loc):
            m = loc == l
            hist = m & fit_mask & ~np.isnan(v)
            for d_ in np.unique(day[m]):
                rows = m & (day == d_)
                ref = v[hist & (day != d_)]
                if c == Xin.columns[0]: n_hist[rows] = len(ref)
                if len(ref) == 0: continue
                base = np.median(ref)
                dv = v[rows] - base
                if scale:
                    mad = np.median(np.abs(ref - base)) * 1.4826 if len(ref) >= 3 else np.nan
                    dv = dv / max(mad if mad == mad else 0.0, 0.25 * glob_mad, 1e-12)
                dev[rows] = np.where(np.isnan(dv), 0.0, dv)
        out[c] = dev
    return out, n_hist

X_dev, n_hist = loc_deviation(X, TR)
X_devz, _ = loc_deviation(X, TR, scale=True)
X_c1 = V.loc_standardize(X, loc)
X_both = pd.concat([X, X_dev.add_prefix("dev_")], axis=1)
print("시험 타점의 위치 이력(학습 기간 같은 위치 타점 수):", pd.Series(n_hist[TE]).describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())
INPUTS = {"raw": X, "c1": X_c1, "dev": X_dev, "devz": X_devz, "raw+dev": X_both}

# D4 규칙: 파생 9개(수준 3 + 이동 중앙값 편차 3 + 직전 변화량 3)의 날짜 안 |강건 z| 합
Xd9, *_ = S.load_polito("derived5")
def d9_from(Xlvl):
    """수준 3개가 주어지면 같은 식으로 순서 변수 6개를 다시 만든다."""
    return S.derived5(Xlvl[S.PHYS].assign(**{}), day)
absz = lambda Xd: S.day_transform(Xd, day, "absz")
D4 = {"raw": absz(Xd9).sum(axis=1).values,
      "c1": absz(V.loc_standardize(Xd9, loc)).sum(axis=1).values,
      "dev": absz(d9_from(X_dev)).sum(axis=1).values,
      "devz": absz(d9_from(X_devz)).sum(axis=1).values}
D4["raw+dev"] = D4["dev"]
D4_LEVEL_DEV = absz(X_dev[S.PHYS]).sum(axis=1).values          # 수준 3개의 위치 편차 크기만


# %% [markdown]
# ## 2. 학습 (정답은 날짜별 개수만)

# %%
ub, bag = np.unique(day[TR], return_inverse=True)
c_tr = pd.Series(y[TR]).groupby(day[TR]).sum().loc[ub].values.astype(int)
cache = RES / "exp4_locdev_scores.pkl"
SC = pickle.load(open(cache, "rb")) if cache.exists() else {}
t0 = time.time()
for nm, Xin in INPUTS.items():
    for model in ("PVC MLP", "집계 로지스틱"):
        if (nm, model) in SC: continue
        SC[(nm, model)] = np.mean([S.fit_score(model, Xin, bag, c_tr, TR, TE, s)[0] for s in SEEDS], 0)
        pickle.dump(SC, open(cache, "wb")); print(f"  {nm:8s} {model:8s} 완료 ({time.time()-t0:.0f}초)", flush=True)

# 상한선 진단: 타점 정답으로 학습한 로지스틱(실제 사양에서는 쓸 수 없음)
UB = {}
for nm in ("raw", "dev", "devz"):
    Xs = S._prep(INPUTS[nm], TR)
    UB[nm] = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(Xs[TR], y[TR]).decision_function(Xs[TE])


# %% [markdown]
# ## 3. 채점 (숨긴 타점 정답)

# %%
dt, lt, bt, yt = day[TE], loc[TE], body[TE], y[TE]
ud, gi = np.unique(dt, return_inverse=True); n = np.bincount(gi)
ul, li = np.unique(lt, return_inverse=True); nl = np.bincount(li)
rk = lambda s: pd.Series(s).groupby(dt).rank(pct=True).values
ubt = np.unique(bt); rngb = np.random.default_rng(11)
Wb = np.stack([np.bincount(d_, minlength=len(ubt)) for d_ in rngb.integers(0, len(ubt), (3000, len(ubt)))])[:, np.searchsorted(ubt, bt)]
BOOT = [np.concatenate([np.flatnonzero(bt == b_) for b_ in np.random.default_rng(100 + i).choice(ubt, len(ubt))]) for i in range(1000)]

def loc_share(s):
    """점수(날짜 안 순위)의 분산 중 위치 평균이 설명하는 비율."""
    r = pd.Series(rk(s)); m = r.groupby(lt).transform("mean")
    return float(((m - r.mean()) ** 2).sum() / ((r - r.mean()) ** 2).sum())

def score_row(label, s):
    r = {"모델": label}
    for b in (5, 10, 20):
        k = np.ceil(b / 100 * n - 1e-9); p_ = inclusion_prob(s, k, gi); pr = (k / n)[gi]
        hit, rnd = (p_ * yt).sum(), (pr * yt).sum()
        d_ = (Wb @ ((p_ - pr) * yt)) / (Wb @ yt); lo, hi = np.quantile(d_, [.025, .975])
        r[f"{b}% 포착/무작위"] = f"{hit:.1f}/{rnd:.1f}"
        r[f"{b}% 포착률−무작위 [95%]"] = f"{(hit - rnd) / yt.sum() * 100:+.1f}%p [{lo*100:+.1f}, {hi*100:+.1f}]"
        if b == 10: r["_hit10"], r["_lo10"], r["_hi10"], r["_rnd10"] = hit, lo, hi, rnd
    r["AUROC"] = round(roc_auc_score(yt, s), 3)
    wa = within_loc_auroc(yt, s, lt)
    bs = [within_loc_auroc(yt[ii], s[ii], lt[ii]) for ii in BOOT]
    r["같은 위치 안 AUROC [95%]"] = f"{wa:.3f} [{np.nanquantile(bs,.025):.3f}, {np.nanquantile(bs,.975):.3f}]"; r["_wa"] = wa
    pin = inclusion_prob(s, 0.10 * nl, li)                 # 위치마다 10%씩 검사(위치 간 차이를 쓰지 않는 정책)
    r["위치마다 10% 검사 시 포착(무작위 5.1)"] = round(float((pin * yt).sum()), 1)
    r["위치 설명 비율"] = round(loc_share(s), 2)
    return r

rows = []
for nm in INPUTS:
    pvc, agg, d4 = SC[(nm, "PVC MLP")][...], SC[(nm, "집계 로지스틱")], D4[nm][TE]
    rows += [score_row(f"{nm} | PVC", pvc), score_row(f"{nm} | 집계 로지스틱", agg)]
    if nm != "raw+dev": rows.append(score_row(f"{nm} | D4", d4))
    rows.append(score_row(f"{nm} | 결합(PVC+D4)", (rk(pvc) + rk(d4)) / 2))
rows.append(score_row("dev | D4 수준 3개만", D4_LEVEL_DEV[TE]))
rows.append(score_row("혼합 | raw PVC + D4-dev", (rk(SC[("raw", "PVC MLP")]) + rk(D4["dev"][TE])) / 2))
rows.append(score_row("혼합 | dev PVC + D4(raw)", (rk(SC[("dev", "PVC MLP")]) + rk(D4["raw"][TE])) / 2))
rows.append(score_row("혼합 | 결합(raw) + dev PVC", (rk(SC[("raw", "PVC MLP")]) + rk(D4["raw"][TE]) + rk(SC[("dev", "PVC MLP")])) / 3))
for nm in UB: rows.append(score_row(f"[상한선·타점 정답] {nm} 로지스틱", UB[nm]))
res = pd.DataFrame(rows)
show = [c for c in res.columns if not c.startswith("_")]
res[show].to_csv(RES / "exp4_locdev.csv", index=False, encoding="utf-8-sig")
print(res[show].to_string(index=False))

# 24번 저장 점수와의 일치 확인(재현 검증)
old = RES / "exp2_runs.pkl"
if old.exists():
    R = pickle.load(open(old, "rb"))
    for k_, nm in ((("base", 0), "raw"), (("c1", 0), "c1")):
        a, b_ = R[k_]["raw"], SC[(nm, "PVC MLP")]
        print(f"재현 확인 {nm}: 저장 점수와 순위상관 {pd.Series(a).corr(pd.Series(b_), method='spearman'):.4f}, 최대 차이 {np.abs(a-b_).max():.2e}")


# %% [markdown]
# ## 4. 확인 실험 — 날짜 하나 빼기(20일, 불량 79건)
# 시험 불량이 51건뿐이라 같은 질문을 전체 20일로 다시 본다. 날짜 d를 뺄 때 위치 기준값도 d를 뺀 나머지 날짜로만 구한다.
# 비교 기준(기존 결과, README): raw PVC 22 / D4 17 / 결합 24 / 무작위 8.6 (10% 예산).
# 단일 규칙 6개는 28번(전체 데이터로 방향을 본 분석) 뒤에 방향을 정한 것이라 탐색 결과다.

# %%
days_all = np.unique(day)
ALL = np.ones(len(y), bool)
udA, giA = np.unique(day, return_inverse=True); nA = np.bincount(giA)
ubA = np.unique(body); rngA = np.random.default_rng(8)
WbA = np.stack([np.bincount(d_, minlength=len(ubA)) for d_ in rngA.integers(0, len(ubA), (3000, len(ubA)))])[:, np.searchsorted(ubA, body)]
BOOTA = [np.concatenate([np.flatnonzero(body == b_) for b_ in np.random.default_rng(500 + i).choice(ubA, len(ubA))]) for i in range(500)]
rkA = lambda s: pd.Series(s).groupby(day).rank(pct=True).values

cacheL = RES / "exp4_locdev_lodo.pkl"
L = pickle.load(open(cacheL, "rb")) if cacheL.exists() else {}
if "Xdev" not in L:
    Xdev_l = pd.DataFrame(0.0, index=X.index, columns=X.columns); folds = {}
    for d in days_all:
        Xd_, _ = loc_deviation(X, day != d)
        folds[d] = Xd_; Xdev_l.loc[day == d] = Xd_.loc[day == d].values
    L["Xdev"], L["folds"] = Xdev_l, folds; pickle.dump(L, open(cacheL, "wb"))
Xdev_l, folds = L["Xdev"], L["folds"]

def lodo_fit(fn, key):
    """날짜 d를 빼고 학습 → d 점수. fn(Xfold, tr, te, d) → te 점수."""
    if key in L: return L[key]
    s = np.full(len(y), np.nan)
    for d in days_all:
        te = day == d; s[te] = fn(folds[d], ~te, te, d)
    L[key] = s; pickle.dump(L, open(cacheL, "wb")); return s

def weak(model):
    def f(Xf, tr, te, d):
        ub_, bag_ = np.unique(day[tr], return_inverse=True)
        c_ = pd.Series(y[tr]).groupby(day[tr]).sum().loc[ub_].values.astype(int)
        return np.mean([S.fit_score(model, Xf, bag_, c_, tr, te, s)[0] for s in SEEDS], 0)
    return f
def sup(Xsrc):
    def f(Xf, tr, te, d):
        Xs = S._prep(Xf if Xsrc == "dev" else X, tr)
        return LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(Xs[tr], y[tr]).decision_function(Xs[te])
    return f

def score_all(label, s):
    r = {"모델": label}
    for b in (5, 10, 20):
        k = np.ceil(b / 100 * nA - 1e-9); p_ = inclusion_prob(s, k, giA); pr = (k / nA)[giA]
        hit, rnd = (p_ * y).sum(), (pr * y).sum()
        d_ = (WbA @ ((p_ - pr) * y)) / (WbA @ y); lo, hi = np.quantile(d_, [.025, .975])
        r[f"{b}% 포착/무작위"] = f"{hit:.1f}/{rnd:.1f}"
        r[f"{b}% 포착률−무작위 [95%]"] = f"{(hit - rnd) / y.sum() * 100:+.1f}%p [{lo*100:+.1f}, {hi*100:+.1f}]"
    r["AUROC"] = round(roc_auc_score(y, s), 3)
    bs = [within_loc_auroc(y[ii], s[ii], loc[ii]) for ii in BOOTA]
    r["같은 위치 안 AUROC [95%]"] = f"{within_loc_auroc(y, s, loc):.3f} [{np.nanquantile(bs,.025):.3f}, {np.nanquantile(bs,.975):.3f}]"
    r_ = pd.Series(rkA(s)); m_ = r_.groupby(loc).transform("mean")
    r["위치 설명 비율"] = round(float(((m_ - r_.mean()) ** 2).sum() / ((r_ - r_.mean()) ** 2).sum()), 2)
    return r

d4_raw_l = absz(Xd9).sum(axis=1).values
d4_dev_l = S.day_transform(S.derived5(Xdev_l[S.PHYS], day), day, "absz").sum(axis=1).values
rowsL = [score_all("raw | D4 (기존 17 재현 확인)", d4_raw_l), score_all("dev | D4", d4_dev_l)]
for c, nm in (("heat", "입열량"), ("cur", "전류"), ("time", "통전시간"), ("force", "가압력"), ("dynres", "동저항"), ("i_over_f", "전류/가압력")):
    rowsL.append(score_all(f"dev | 단일 규칙: {nm} 편차가 낮을수록 위험", -Xdev_l[c].values))
rowsL.append(score_all("[상한선·타점 정답] raw 로지스틱", lodo_fit(sup("raw"), "sup_raw")))
rowsL.append(score_all("[상한선·타점 정답] dev 로지스틱", lodo_fit(sup("dev"), "sup_dev")))
agg_l = lodo_fit(weak("집계 로지스틱"), "agg_dev"); rowsL.append(score_all("dev | 집계 로지스틱", agg_l))
RUN_PVC_LODO = False   # PVC 10시드 × 20일은 CPU에서 1시간 넘게 걸려 기본은 끈다(시간순 평가에는 PVC가 들어 있다). GPU에서는 True로 바꿔 실행.
if RUN_PVC_LODO:
    pvc_l = lodo_fit(weak("PVC MLP"), "pvc_dev")
    rowsL += [score_all("dev | PVC", pvc_l), score_all("dev | 결합(PVC+D4)", (rkA(pvc_l) + rkA(d4_dev_l)) / 2)]
resL = pd.DataFrame(rowsL); resL.to_csv(RES / "exp4_locdev_lodo.csv", index=False, encoding="utf-8-sig")
print("\n[날짜 하나 빼기]"); print(resL.to_string(index=False))
