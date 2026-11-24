# %% [markdown]
# # 25. 실험 3 — 보고서 1장과의 정합성 보강
# A. Stellantis N.V.에서 raw PVC · D4 · 결합의 F1, PR-AUC, AUROC를 "찾은 불량 수"와 함께 낸다(날짜 하나 빼기, 시간순).
#    F1 임계값: (i) 날짜별 상위 c개(c = 그날 실제 불량 수, KAMP에서도 일별 개수는 알려져 있다) (ii) 날짜별 상위 10%.
#    PR-AUC(정밀도-재현율 곡선 아래 면적)와 AUROC는 날짜마다 점수를 순위 백분위로 바꿔 모은 값으로 계산한다(날짜 하나 빼기에서는 날짜마다 모델이 달라 원점수를 섞을 수 없다).
# B. KAMP 가이드북 방식 오토인코더(AE: 4→3→2→3→4, RReLU, MSE, Adam 0.01, 50 epoch, batch 64, MinMax)를 KAMP와 같은 입력 4개로 Stellantis N.V.에 적용하고, 같은 예산에서 결합 모델과 비교한다. 10시드 평균.
# C. Stellantis N.V. 2022년 136타점(가압력 전부 0 → 결측)의 처리 확인과 민감도: (i) 2022년 제외 (ii) 가압력 관련 변수(가압력, I/F, D4의 I/F 3개) 중립화.
# 모델 사양은 바꾸지 않는다. 95% 구간은 차체 복원추출.

# %%
import time, pickle, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import br
from br import spec as S, validity as V, plot as P
from wl.metrics import inclusion_prob
from wl import data as wldata
warnings.filterwarnings("ignore"); P.setup(); pd.set_option("display.width", 230)
X, day, cnt, y, _ = S.load_polito(); Xd9, *_ = S.load_polito("derived5")
body = wldata.load_polito().body.values
TR = np.array([d_ < "2024" for d_ in day]); TE = np.array([d_ >= "2024-02-01" for d_ in day]); Y22 = np.array([d_ < "2023" for d_ in day])
ALL = np.ones(len(y), bool); SEEDS = range(10)
RAW = pd.read_pickle(br.RES / "same_model_scores.pkl")
raw_lodo = np.mean([g.sort_values("row").score.values for _, g in RAW[(RAW.data == "polito") & (RAW.model == "PVC MLP")].groupby("seed")], 0)
def d4(Xd): return S.day_transform(Xd, day, "absz").sum(axis=1).values
def temporal_raw(Xin, trm):
    ub, bag = np.unique(day[trm], return_inverse=True)
    return np.mean([S.fit_score("PVC MLP", Xin, bag, cnt.loc[ub].values.astype(int), trm, TE, s)[0] for s in SEEDS], 0)
def lodo_raw(Xin, mask):
    """mask 행만 쓰는 날짜 하나 빼기(10시드)."""
    c_ = pd.Series(y[mask]).groupby(day[mask]).sum()
    out = []
    for s in SEEDS:
        sc_ = np.full(len(y), np.nan); dd = day.copy(); dd[~mask] = "제외"
        for d in c_.index:
            te = mask & (day == d); tr = mask & ~te
            ub, bag = np.unique(day[tr], return_inverse=True)
            sc_[te] = S.fit_score("PVC MLP", Xin, bag, c_.loc[ub].values.astype(int), tr, te, s)[0]
        out.append(sc_)
    return np.mean(out, 0)

# 학습이 필요한 것: 캐시
cache = br.RES / "exp3_scores.pkl"
C = pickle.load(open(cache, "rb")) if cache.exists() else {}
X_nf = X.copy(); X_nf["force"] = 0.0; X_nf["i_over_f"] = 0.0          # 가압력 관련 중립화(상수 → 표준화 후 0)
Xd9_nf = Xd9.copy(); Xd9_nf[["i_over_f", "dev_i_over_f", "dl_i_over_f"]] = 0.0
X4 = X[["cur", "vol", "time", "force"]]
JOBS = {
    "raw_temporal": lambda: temporal_raw(X, TR),
    "ae_temporal": lambda: V.ae_scores(X4, TR, TE, SEEDS),
    "ae_lodo": lambda: _ae_lodo(),
    "raw_temporal_no2022": lambda: temporal_raw(X, TR & ~Y22),
    "raw_temporal_noforce": lambda: temporal_raw(X_nf, TR),
    "raw_lodo_no2022": lambda: lodo_raw(X, ~Y22),
    "raw_lodo_noforce": lambda: lodo_raw(X_nf, ALL),
}
def _ae_lodo():
    out = np.full(len(y), np.nan)
    for d in np.unique(day):
        te = day == d; out[te] = V.ae_scores(X4, ~te, te, SEEDS)
    return out
t0 = time.time()
for k, f in JOBS.items():
    if k in C or time.time() - t0 > 480: continue
    C[k] = f(); pickle.dump(C, open(cache, "wb")); print(f"{k} 완료 {time.time()-t0:.0f}초", flush=True)
DONE = all(k in C for k in JOBS)
print("학습:", sum(k in C for k in JOBS), "/", len(JOBS))

# %% [markdown]
# ## A·B. 지표 표 (Stellantis N.V.)

# %%
def rk(s, dd): return pd.Series(s).groupby(dd).rank(pct=True).values
def metrics_block(scores, mask, label):
    dd, yy, bb = day[mask], y[mask], body[mask]
    ud, gi = np.unique(dd, return_inverse=True); n = np.bincount(gi); cd = np.bincount(gi, weights=yy)
    ubt = np.unique(bb); rng = np.random.default_rng(5)
    W = np.stack([np.bincount(d_, minlength=len(ubt)) for d_ in rng.integers(0, len(ubt), (2000, len(ubt)))])[:, np.searchsorted(ubt, bb)]
    rows = []
    for nm, s in scores.items():
        r_ = rk(s[mask] if len(s) == len(y) else s, dd)
        pc = inclusion_prob(r_, np.minimum(cd, n).astype(float), gi)          # 날짜별 상위 c개
        k10 = np.ceil(.1 * n - 1e-9); p10 = inclusion_prob(r_, k10, gi)
        tp_c, sel_c = (pc * yy).sum(), cd.sum()
        tp10, sel10 = (p10 * yy).sum(), k10.sum()
        prec10, rec10 = tp10 / sel10, tp10 / yy.sum()
        f1_10 = 2 * prec10 * rec10 / (prec10 + rec10)
        f1c_b = (W @ (pc * yy)) / (W @ yy)                                      # 상위 c개에서는 정밀도 = 재현율 = F1(선택 수 = 불량 수)
        f110_b = 2 * ((W @ (p10 * yy)) / (W @ np.ones(len(yy)) * sel10 / len(yy))) * ((W @ (p10 * yy)) / (W @ yy))
        f110_b = f110_b / ((W @ (p10 * yy)) / (W @ np.ones(len(yy)) * sel10 / len(yy)) + (W @ (p10 * yy)) / (W @ yy))
        ci = lambda a: f"[{np.nanquantile(a,.025):.3f}, {np.nanquantile(a,.975):.3f}]"
        auc_b, ap_b = [], []
        for _ in range(500):
            ii = np.concatenate([np.flatnonzero(bb == b_) for b_ in rng.choice(ubt, len(ubt))])
            if 0 < yy[ii].sum() < len(ii): auc_b.append(roc_auc_score(yy[ii], r_[ii])); ap_b.append(average_precision_score(yy[ii], r_[ii]))
        rows.append({"평가": label, "모델": nm, "10% 찾은 불량": f"{tp10:.1f}/{int(yy.sum())}",
                     "F1 (날짜별 상위 c개)": f"{tp_c / sel_c:.3f} {ci(f1c_b)}", "F1 (날짜별 10%)": f"{f1_10:.3f} {ci(f110_b)}",
                     "정밀도/재현율 (10%)": f"{prec10:.3f} / {rec10:.3f}",
                     "PR-AUC": f"{average_precision_score(yy, r_):.3f} {ci(ap_b)}", "AUROC": f"{roc_auc_score(yy, r_):.3f} {ci(auc_b)}",
                     "PR-AUC 무작위 기준": f"{yy.mean():.3f}"})
    return rows

if DONE:
    D4_all = d4(Xd9)
    comb_l = (rk(raw_lodo, day) + rk(D4_all, day)) / 2
    S_lodo = {"raw PVC": raw_lodo, "D4": D4_all, "결합": comb_l, "가이드북 AE": C["ae_lodo"]}
    dt = day[TE]
    comb_t = (rk(C["raw_temporal"], dt) + rk(D4_all[TE], dt)) / 2
    S_temp = {"raw PVC": C["raw_temporal"], "D4": D4_all[TE], "결합": comb_t, "가이드북 AE": C["ae_temporal"]}
    tabA = pd.DataFrame(metrics_block(S_lodo, ALL, "날짜 하나 빼기") + metrics_block(S_temp, TE, "시간순 2024"))
    tabA.to_csv(br.RES / "exp3_metrics.csv", index=False, encoding="utf-8-sig"); display(tabA)

# %% [markdown]
# ## B. 가이드북 AE vs 결합 — 같은 예산에서 찾은 불량 수 차이 (차체 복원추출 95%)

# %%
def diff_table(A_, B_, mask, label):
    dd, yy, bb = day[mask], y[mask], body[mask]
    ud, gi = np.unique(dd, return_inverse=True); n = np.bincount(gi)
    ubt = np.unique(bb); rng = np.random.default_rng(6)
    W = np.stack([np.bincount(d_, minlength=len(ubt)) for d_ in rng.integers(0, len(ubt), (5000, len(ubt)))])[:, np.searchsorted(ubt, bb)]
    rows = []
    for b in (.05, .1, .2):
        k = np.ceil(b * n - 1e-9)
        pa, pb = inclusion_prob(rk(A_, dd), k, gi), inclusion_prob(rk(B_, dd), k, gi); pr = (k / n)[gi]
        d_ = (W @ ((pa - pb) * yy)) / (W @ yy)
        rows.append({"평가": label, "예산": f"{int(b*100)}%", "결합": round((pa * yy).sum(), 1), "가이드북 AE": round((pb * yy).sum(), 1),
                     "무작위": round((pr * yy).sum(), 1), "결합−AE 포착률 [95%]": f"{d_.mean()*100:+.1f}%p [{np.quantile(d_,.025)*100:+.1f}, {np.quantile(d_,.975)*100:+.1f}]"})
    return rows
if DONE:
    tabB = pd.DataFrame(diff_table(comb_l, C["ae_lodo"], ALL, "날짜 하나 빼기") + diff_table(comb_t, C["ae_temporal"], TE, "시간순 2024"))
    tabB.to_csv(br.RES / "exp3_ae_vs_combined.csv", index=False, encoding="utf-8-sig"); display(tabB)

# %% [markdown]
# ## C. 2022년 가압력 0 타점 — 처리 확인과 민감도

# %%
print("가압력 결측 타점: 2022년", int(X.force[Y22].isna().sum()), "/", int(Y22.sum()), "| 2023~24년", int(X.force[~Y22].isna().sum()), "/", int((~Y22).sum()))
print("raw PVC 처리: spec._prep이 가압력·I/F 결측을 '학습 행 중앙값'으로 채운 뒤 표준화한다(시간순이면 2023년 타점의 중앙값).")
Xd_abs = S.day_transform(Xd9, day, "absz")
print("D4 처리: I/F 수준값 결측 → day_transform에서 전체 중앙값으로 채움 → 같은 날 안에서 모두 같아 |z| = 0. 편차·변화량은 0으로 채움 → |z| = 0.")
print("   확인: 2022년 타점의 D4 I/F 성분 합 최댓값 =", float(Xd_abs.loc[Y22, ["i_over_f", "dev_i_over_f", "dl_i_over_f"]].values.max()))
def cap10(score, mask):
    dd, yy = day[mask], y[mask]; ud, gi = np.unique(dd, return_inverse=True); n = np.bincount(gi)
    return round((inclusion_prob(rk(score, dd), np.ceil(.1 * n - 1e-9), gi) * yy).sum(), 1), int(yy.sum())
if DONE:
    D4_nf = d4(Xd9_nf); dt = day[TE]
    rows = []
    # 시간순(시험 2024는 같고, 학습만 바뀐다)
    for nm, rawt, d4t in [("원래", C["raw_temporal"], D4_all[TE]), ("(i) 2022년 학습 제외", C["raw_temporal_no2022"], D4_all[TE]),
                          ("(ii) 가압력 관련 중립화", C["raw_temporal_noforce"], D4_nf[TE])]:
        cb = (rk(rawt, dt) + rk(d4t, dt)) / 2
        rows.append({"평가": "시간순 2024 (10%)", "설정": nm, "raw PVC": cap10(rawt, TE)[0], "D4": cap10(d4t, TE)[0], "결합": cap10(cb, TE)[0], "불량": cap10(cb, TE)[1]})
    # 날짜 하나 빼기: (i)은 2022년 날짜를 학습·평가 모두에서 뺀다 → 비교를 위해 원래 모델도 2022년 제외 날짜로 채점
    M = ~Y22
    for nm, rawl, d4l, mask in [("원래 (2022년 제외 날짜로 채점)", raw_lodo, D4_all, M), ("(i) 2022년 제외", C["raw_lodo_no2022"], D4_all, M),
                                ("원래 (전체 날짜)", raw_lodo, D4_all, ALL), ("(ii) 가압력 관련 중립화", C["raw_lodo_noforce"], D4_nf, ALL)]:
        cb = (rk(rawl[mask], day[mask]) + rk(d4l[mask], day[mask])) / 2
        rows.append({"평가": "날짜 하나 빼기 (10%)", "설정": nm, "raw PVC": cap10(rawl[mask], mask)[0], "D4": cap10(d4l[mask], mask)[0], "결합": cap10(cb, mask)[0], "불량": cap10(cb, mask)[1]})
    tabC = pd.DataFrame(rows); tabC.to_csv(br.RES / "exp3_force_sensitivity.csv", index=False, encoding="utf-8-sig"); display(tabC)

# %% [markdown]
# ## C 보완. 민감도 차이의 95% 구간 (차체 복원추출, 같은 표본에서 대응 비교)

# %%
def diff_ci(sa, sb, mask, b=.1, nb=5000, seed=12):
    dd, yy, bb = day[mask], y[mask], body[mask]
    ud, gi = np.unique(dd, return_inverse=True); n = np.bincount(gi); k = np.ceil(b * n - 1e-9)
    pa, pb = inclusion_prob(rk(sa, dd), k, gi), inclusion_prob(rk(sb, dd), k, gi)
    ubt = np.unique(bb); rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(d_, minlength=len(ubt)) for d_ in rng.integers(0, len(ubt), (nb, len(ubt)))])[:, np.searchsorted(ubt, bb)]
    d_ = (W @ ((pb - pa) * yy)) / (W @ yy)
    return f"{(pb - pa) @ yy:+.1f}개, {d_.mean()*100:+.1f}%p [{np.quantile(d_,.025)*100:+.1f}, {np.quantile(d_,.975)*100:+.1f}]"
if DONE:
    dt = day[TE]; cb = lambda r_, d_, dd: (rk(r_, dd) + rk(d_, dd)) / 2
    rows = []
    for nm, rawt, d4t in [("(i) 2022년 학습 제외", C["raw_temporal_no2022"], D4_all[TE]), ("(ii) 가압력 관련 중립화", C["raw_temporal_noforce"], D4_nf[TE])]:
        rows.append({"평가": "시간순 2024 (10%)", "설정 − 원래": nm, "raw PVC": diff_ci(C["raw_temporal"], rawt, TE), "D4": diff_ci(D4_all[TE], d4t, TE),
                     "결합": diff_ci(cb(C["raw_temporal"], D4_all[TE], dt), cb(rawt, d4t, dt), TE)})
    M = ~Y22
    rows.append({"평가": "날짜 하나 빼기 (10%, 2022년 제외 날짜)", "설정 − 원래": "(i) 2022년 제외", "raw PVC": diff_ci(raw_lodo[M], C["raw_lodo_no2022"][M], M), "D4": "변화 없음",
                 "결합": diff_ci(cb(raw_lodo[M], D4_all[M], day[M]), cb(C["raw_lodo_no2022"][M], D4_all[M], day[M]), M)})
    rows.append({"평가": "날짜 하나 빼기 (10%)", "설정 − 원래": "(ii) 가압력 관련 중립화", "raw PVC": diff_ci(raw_lodo, C["raw_lodo_noforce"], ALL), "D4": diff_ci(D4_all, D4_nf, ALL),
                 "결합": diff_ci(cb(raw_lodo, D4_all, day), cb(C["raw_lodo_noforce"], D4_nf, day), ALL)})
    tabC2 = pd.DataFrame(rows); tabC2.to_csv(br.RES / "exp3_force_sensitivity_ci.csv", index=False, encoding="utf-8-sig"); display(tabC2)
