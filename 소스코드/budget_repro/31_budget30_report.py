# %% [markdown]
# # 31. 검사예산 30% 기준 재계산과 안정 공정범위 (보고서 2~5장)
# 운영 권장 예산을 30%로 두고 보고서의 표·그림 수치를 다시 낸다. 새 학습은 하지 않고 저장된 점수만 쓴다(29·24·30번과 같은 식).
# 1. 모델별 포착(일자 단위 교차검증·시간순)  2. 포착·미탐지·오경보와 위치 그룹  3. 구성 요소  4. KAMP 조건 재현
# 5. KAMP 날짜별 검사 수량·이탈 방향·전류×가압력  6. 안정 공정범위(타점의 가운데 95% 구간)  7. 검사 시간·비용 환산  8. 그림
# 예산을 바꾸려면 B만 고치면 된다(B=.10이면 29번 노트북의 값과 같다).

# %%
import pickle, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, warnings
from scipy.stats import spearmanr, fisher_exact
import br
from br import spec as S, validity as V, plot as P
from wl.metrics import inclusion_prob, within_loc_auroc
from wl import data as wldata
warnings.filterwarnings("ignore"); P.setup(); pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
if "Malgun Gothic" not in {f.name for f in mpl.font_manager.fontManager.ttflist}: mpl.rcParams["font.family"] = "Noto Sans CJK KR"
PAL, GRAY, INK, INK2 = P.PAL, P.GRAY, P.INK, P.INK2
B = .30
X, day, cnt, y, _ = S.load_polito(); Xd, *_ = S.load_polito("derived5_absz"); Xd9, *_ = S.load_polito("derived5")
P0 = wldata.load_polito(); loc, body = P0["loc"].values, P0.body.values
RAW = pd.read_pickle(br.RES / "same_model_scores.pkl"); E3 = pd.read_pickle(br.RES / "exp3_scores.pkl")
raw = np.mean([g.sort_values("row").score.values for _, g in RAW[(RAW.data == "polito") & (RAW.model == "PVC MLP")].groupby("seed")], 0)
d4 = Xd.sum(axis=1).values
rk = lambda s, g: pd.Series(s).groupby(g).rank(pct=True).values
comb = (rk(raw, day) + rk(d4, day)) / 2
ud, gi = np.unique(day, return_inverse=True); n = np.bincount(gi); pos = np.bincount(gi, weights=y)
TE = np.array([d_ >= "2024-02-01" for d_ in day]); yt, dt = y[TE], day[TE]
comb_t = (rk(E3["raw_temporal"], dt) + rk(d4[TE], dt)) / 2
_, git = np.unique(dt, return_inverse=True); nt = np.bincount(git); post = np.bincount(git, weights=yt)
caught = lambda s, b: (inclusion_prob(s, np.ceil(b * n - 1e-9), gi) * y).sum()
ct = lambda s, b: (inclusion_prob(s, np.ceil(b * nt - 1e-9), git) * yt).sum()
rnd = lambda b: (np.ceil(b * n - 1e-9) / n * pos).sum(); rndt = lambda b: (np.ceil(b * nt - 1e-9) / nt * post).sum()
def pick(s, b):
    out = np.zeros(len(s), bool)
    for d in ud:
        ix = np.flatnonzero(day == d); k = int(np.ceil(b * len(ix) - 1e-9)); out[ix[np.argsort(-s[ix], kind="stable")[:k]]] = True
    return out
def rz(s, g):
    med = s.groupby(g).transform("median"); mad = (s - med).abs().groupby(g).transform("median") * 1.4826
    return ((s - med) / (mad + 1e-9 + 1e-6 * s.abs().groupby(g).transform("median"))).clip(-10, 10)
save = lambda t, name, **k: t.to_csv(br.RES / f"b30_{name}.csv", encoding="utf-8-sig", **k)

# %% [markdown]
# ## 1. 모델별 포착 — [표 2-5], [표 2-6], [그림 2-1], [그림 2-2]

# %%
MS = {"가이드북 AE": (E3["ae_lodo"], E3["ae_temporal"]), "raw PVC": (raw, E3["raw_temporal"]), "D4": (d4, d4[TE]), "raw PVC + D4": (comb, comb_t)}
rows = [{"예산%": int(b * 100), "모델": m, "교차검증 포착(79)": round(caught(a, b), 1), "시간순 포착(51)": round(ct(c, b), 1)} for b in (.05, .1, .2, B, .4) for m, (a, c) in MS.items()]
rows += [{"예산%": int(b * 100), "모델": "무작위", "교차검증 포착(79)": round(rnd(b), 1), "시간순 포착(51)": round(rndt(b), 1)} for b in (.05, .1, .2, B, .4)]
t1 = pd.DataFrame(rows).sort_values(["예산%"]); save(t1, "model_capture", index=False); print(t1.to_string(index=False))

# %% [markdown]
# ## 2. 포착·미탐지·오경보와 위치 그룹 — [표 3-1], [표 3-2], [표 3-4], 3-4·3-5절

# %%
sel = pick(comb, B); df = pd.DataFrame(dict(day=day, loc=loc, y=y, sel=sel))
df["grp"] = np.select([(df.y == 1) & df.sel, (df.y == 1) & ~df.sel, (df.y == 0) & df.sel], ["포착", "미탐지", "오경보"], "예산 밖 정상")
NM = {"heat": "입열량", "cur": "전류", "time": "통전시간", "force": "가압력"}
for c in X.columns: df["z_" + c] = rz(X[c], pd.Series(day)).values
t2 = df.groupby("grp")[["z_" + c for c in NM]].median().round(2); t2.columns = list(NM.values()); t2.insert(0, "타점 수", df.grp.value_counts())
plain = (df[["z_" + c for c in X.columns]].abs().fillna(0) < 2).all(axis=1); t2["조건값이 모두 평범한 타점 수"] = plain.groupby(df.grp).sum()
t2 = t2.loc[["포착", "미탐지", "오경보", "예산 밖 정상"]]; save(t2, "group_profile"); print(t2)
print(f"검사 {int(sel.sum())}타점, 추천 중 불량 {df[df.sel].y.mean():.3f}, 전체 불량률 {y.mean():.3f}")
L = df.groupby("loc").agg(n=("y", "size"), pos=("y", "sum"), sel=("sel", "sum")).assign(selrate=lambda d: d.sel / d.n)
lg = lambda l: "자주 추천(추천율>50%)" if L.selrate[l] > .5 else ("추천 안 됨(추천율≤5%)" if L.selrate[l] <= .05 else "가끔 추천")
df["lgrp"] = df["loc"].map(lg)
t3 = df.groupby("lgrp").agg(타점=("y", "size"), 불량=("y", "sum"), 포착=("grp", lambda s: (s == "포착").sum()), 미탐지=("grp", lambda s: (s == "미탐지").sum()), 오경보=("grp", lambda s: (s == "오경보").sum()), 입열이탈_중앙값=("z_heat", "median"))
t3.insert(0, "위치 수", df.drop_duplicates("loc").lgrp.value_counts()); t3["불량률%"] = (t3.불량 / t3.타점 * 100).round(1); save(t3.round(2), "location_groups"); print(t3.round(2))
hot = df[df.lgrp.str.startswith("자주")]
print(f"자주 추천 위치: 추천 타점 불량 비율 {hot[hot.sel].y.mean():.3f}, 위치 전체 {hot.y.mean():.3f}, 불량 점유 {hot.y.sum()/y.sum():.3f}, 평균 대비 {hot.y.mean()/y.mean():.2f}배")
print("84번 위치:", L.loc[84].to_dict(), "| 불량 0건인데 가장 많이 추천된 위치:", L[L.pos == 0].sort_values("sel", ascending=False).head(3)[["n", "sel"]].to_dict("index"))
small = df[df.day < "2023"]; print(f"2022년: {small.day.nunique()}일 {len(small)}타점, 하루 예산 {small.groupby('day').sel.sum().min()}~{small.groupby('day').sel.sum().max()}타점, 불량 {int(small.y.sum())}, 포착 {(small.grp=='포착').sum()}")
print("같은 위치 안 AUROC:", round(within_loc_auroc(y, comb, loc), 3))
bands = [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.0)]; bv = [caught(comb, hi) - (caught(comb, lo) if lo > 0 else 0) for lo, hi in bands]

# %% [markdown]
# ## 3. 구성 요소 — 3-2절 (예산 10%와 30% 비교)

# %%
PHYS = ["heat", "dynres", "i_over_f"]; groups = {"수준 이탈": PHYS, "이동 중앙값 편차": [c for c in Xd if c.startswith("dev_")], "직전 변화량": [c for c in Xd if c.startswith("dl_")], "D4 전체 9개": list(Xd.columns)}
rows = []
for ev, mask, rw in [("교차검증", np.ones(len(y), bool), raw), ("시간순", TE, np.asarray(E3["raw_temporal"]))]:
    yv, dv = y[mask], day[mask]; _, g2 = np.unique(dv, return_inverse=True); n2 = np.bincount(g2); r2 = lambda s: pd.Series(s).groupby(dv).rank(pct=True).values
    for b in (.1, B):
        k = np.ceil(b * n2 - 1e-9); r = {"평가": ev, "예산%": int(b * 100), "raw PVC 단독": round((inclusion_prob(rw, k, g2) * yv).sum(), 1)}
        for g, cols in groups.items(): r["raw PVC + " + g] = round((inclusion_prob((r2(rw) + r2(Xd.loc[mask, cols].sum(axis=1).values)) / 2, k, g2) * yv).sum(), 1)
        rows.append(r)
t4 = pd.DataFrame(rows); save(t4, "components", index=False); print(t4.to_string(index=False))

# %% [markdown]
# ## 4. KAMP 조건 재현 — [표 3-5], [그림 3-5] (24번 노트북의 저장된 실행 결과를 예산 B로 다시 집계)

# %%
runs = pickle.load(open(br.RES / "exp2_runs.pkl", "rb"))
D4c = {"orig": S.day_transform(Xd9, day, "absz").sum(axis=1).values[TE], "loc": S.day_transform(V.loc_standardize(Xd9, loc), day, "absz").sum(axis=1).values[TE]}
rkt = lambda s: pd.Series(s).groupby(dt).rank(pct=True).values; bt = body[TE]; ubt = np.unique(bt); rngb = np.random.default_rng(11)
Wb = np.stack([np.bincount(d_, minlength=len(ubt)) for d_ in rngb.integers(0, len(ubt), (3000, len(ubt)))])[:, np.searchsorted(ubt, bt)]
sc_of = lambda R, m: {"raw PVC": R["raw"], "D4": D4c[R["d4"]], "결합": (rkt(R["raw"]) + rkt(D4c[R["d4"]])) / 2}[m]
rows = []; kB = np.ceil(B * nt - 1e-9)
for cond in ["base", "a", "b", "c1", "abc"]:
    keys = [k for k in runs if k[0] == cond]
    for m in ["raw PVC", "D4", "결합"]:
        per = np.array([[(inclusion_prob(sc_of(runs[k], m), kB, git) * runs[k]["y"]).sum(), (kB / nt * np.bincount(git, weights=runs[k]["y"])).sum(), runs[k]["y"].sum()] for k in keys], float); hit, rd, tot = per.sum(0)
        if len(keys) > 1:
            rng = np.random.default_rng(3); bs = []
            for _ in range(3000):
                ii = rng.integers(0, len(keys), len(keys)); den = per[ii, 2].sum(); bs.append((per[ii, 0].sum() - per[ii, 1].sum()) / den if den > 0 else np.nan)
            lo, hi = np.nanquantile(bs, [.025, .975])
        else:
            yv = runs[keys[0]]["y"]; p_ = inclusion_prob(sc_of(runs[keys[0]], m), kB, git); d_ = (Wb @ ((p_ - (kB / nt)[git]) * yv)) / (Wb @ yv); lo, hi = np.quantile(d_, [.025, .975])
        rows.append({"조건": cond, "모델": m, "반복": len(keys), "포착": round(hit, 1), "무작위": round(rd, 1), "배수": round(hit / rd, 2), "포착률−무작위 [95%]": f"{(hit-rd)/tot*100:+.1f}%p [{lo*100:+.1f}, {hi*100:+.1f}]"})
t5 = pd.DataFrame(rows); save(t5, "conditions", index=False); print(t5.to_string(index=False))

# %% [markdown]
# ## 5. KAMP 적용 — [표 2-7], [표 3-6], [그림 3-4], 3-7절

# %%
K = pd.read_csv(br.RES / "final_kamp_recommendations_combined.csv"); K = K[K["상태"] == "순위 부여"].copy()
K = K.rename(columns={"weld current(kA)": "cur", "weld Voltage(v)": "vol", "weld force(bar)": "force", "weld time(ms)": "time"}); K["heat"] = K.cur * K.vol * K.time
m_ = K.groupby("date")["idx"].transform("size"); K["rB"] = K["일내순위(결합)"] <= np.ceil(B * m_ - 1e-9)      # 예측결과 파일의 '일내순위'에서 상위 B 추천을 얻는다
t6 = K.groupby("date").agg(순위대상=("idx", "size"), 검사수=("rB", "sum")); t6.loc["합계"] = t6.sum(); save(t6, "kamp_daily_plan"); print(t6.T)
for c in ["cur", "force", "time", "heat"]: K["d_" + c] = K[c] - K.groupby("date")[c].transform("median"); K["m_" + c] = K.groupby("date")[c].transform("median")
t7 = pd.DataFrame([dict(변수=nm, 집단=g_, 낮음=(K[m][c] < -1e-12).mean() * 100, 높음=(K[m][c] > 1e-12).mean() * 100) for c, nm in [("d_cur", "전류"), ("d_heat", "입열량"), ("d_force", "가압력"), ("d_time", "통전시간")] for g_, m in [("추천", K.rB), ("나머지", ~K.rB)]]).round(1)
save(t7, "kamp_direction", index=False); print(t7.to_string(index=False))
g = K[K.rB]; print(f"추천 타점의 이탈 중앙값: 전류 {(g.d_cur/g.m_cur).median()*100:.2f}%, 가압력 {g.d_force.median():+.3f}bar")
K["cb"] = pd.cut(K.d_cur, [-9, -.10, -.05, .05, 9], labels=["0.10 이상 낮음", "0.05~0.10 낮음", "±0.05 이내", "0.05 넘게 높음"]); K["fb"] = pd.cut(K.d_force, [-9, -.03, .03, 9], labels=["0.03 넘게 낮음", "±0.03 이내", "0.03 넘게 높음"])
i2 = K.groupby(["cb", "fb"], observed=False).agg(타점=("rB", "size"), 추천율=("rB", "mean")); i2["추천율%"] = (i2.추천율 * 100).round(1); save(i2, "kamp_interaction"); print(i2[["타점", "추천율%"]].unstack())
res = pd.read_excel(wldata.KAMP_XLSX, sheet_name="result").iloc[:, :6]; res.columns = ["idx", "machine", "item", "date", "defect", "dtype"]; res["date"] = pd.to_datetime(res.date).dt.strftime("%Y-%m-%d")
pv = res.pivot_table(index="date", columns="dtype", values="defect", aggfunc="sum"); low = g.groupby("date").apply(lambda q: (q.d_heat < 0).mean()).reindex(pv.index)
print("저입열 추천 비율과 용접부족 비율의 날짜별 순위상관: %.2f (p=%.2f)" % spearmanr(low, pv[2] / pv.sum(axis=1)))

# %% [markdown]
# ## 6. 안정 공정범위 — [표 4-3], [표 4-4], [그림 4-3]
# 타점의 가운데 95% 구간(2.5~97.5 백분위). 강건 z ±2는 값이 소수 둘째 자리로 기록되는 KAMP에서 이탈이 너무 많아 쓰지 않는다.

# %%
V5 = ["cur", "vol", "time", "force", "heat"]; VN = {"cur": "전류 (kA)", "vol": "전압 (V)", "time": "통전시간 (ms)", "force": "가압력 (bar)", "heat": "입열량"}
gk = K.groupby("date"); lo = {c: gk[c].quantile(.025) for c in V5}; hi = {c: gk[c].quantile(.975) for c in V5}; med = {c: gk[c].median() for c in V5}; allr = {c: (K[c].quantile(.025), K[c].quantile(.975)) for c in V5}
t8 = pd.DataFrame([{"변수": VN[c], "제안 하한(9일 전체)": allr[c][0], "제안 상한": allr[c][1], "일별 하한 최소": lo[c].min(), "일별 하한 최대": lo[c].max(), "일별 상한 최소": hi[c].min(), "일별 상한 최대": hi[c].max(),
                    "제안 범위 밖 %": ((K[c] < allr[c][0]) | (K[c] > allr[c][1])).mean() * 100} for c in V5]).round(3); save(t8, "stable_range_kamp", index=False); print(t8.to_string(index=False))
def n_out(df, gcol):
    o = np.zeros(len(df), int)
    for c in V5: o += ((df[c] < df.groupby(gcol)[c].transform(lambda s: s.quantile(.025))) | (df[c] > df.groupby(gcol)[c].transform(lambda s: s.quantile(.975)))).values
    return o
K["nout"] = n_out(K, "date"); inall = np.all([(K[c] >= allr[c][0]) & (K[c] <= allr[c][1]) for c in V5], axis=0)
def zout(q):
    o = np.zeros(len(q), bool)
    for c in V5: m = q[c].median(); mad = (q[c] - m).abs().median() * 1.4826; o |= ((q[c] - m).abs() / (mad if mad > 0 else q[c].std()) > 2).values
    return o.mean() * 100
zr = gk.apply(zout); dr = (K.nout > 0).groupby(K.date).mean() * 100
print(f"그날 범위 밖 타점 {dr.min():.1f}~{dr.max():.1f}% (전체 {(K.nout>0).mean()*100:.1f}%), 제안 범위 5개 모두 안 {inall.mean()*100:.1f}%, 강건 z ±2 기준 이탈 {zr.min():.1f}~{zr.max():.1f}%")
print(f"KAMP 추천율: 입열 하한 미만 {K[K.heat < K.date.map(lo['heat'])].rB.mean()*100:.1f}%, 가압력 상한 초과 {K[K.force > K.date.map(hi['force'])].rB.mean()*100:.1f}%")
Dp = pd.DataFrame({c: X[c].values for c in V5}); Dp["day"] = day; Dp["y"] = y; no = n_out(Dp, "day")
t9 = pd.DataFrame([{"이탈 변수 수": k, "타점": int(m.sum()), "불량": int(Dp[m].y.sum()), "불량률%": round(Dp[m].y.mean() * 100, 1)} for k, m in [("0개", no == 0), ("1개", no == 1), ("2개 이상", no >= 2)]]); save(t9, "stable_range_stellantis", index=False); print(t9.to_string(index=False))
print("2개 이상 대 나머지 Fisher p = %.4f" % fisher_exact([[int(Dp[no >= 2].y.sum()), int((no >= 2).sum() - Dp[no >= 2].y.sum())], [int(Dp[no < 2].y.sum()), int((no < 2).sum() - Dp[no < 2].y.sum())]])[1])

# %% [markdown]
# ## 7. 검사 시간·비용 환산 — [표 4-2]
# 단가는 차체 수동 초음파 검사의 외부 참고값(타점당 25초, 0.5유로). 포착률은 Stellantis N.V. 일자 단위 교차검증 값이다.

# %%
SEC, EUR = 25, 0.5; Nd = len(K) / K.date.nunique(); rec = caught(comb, B) / y.sum()
t10 = pd.DataFrame([{"검사 방식": nm, "하루 검사 타점": round(Nd * f), "검사 시간(h)": round(Nd * f * SEC / 3600, 1), "검사 비용(유로)": round(Nd * f * EUR), "찾는 불량 비율%": round(r * 100)}
                    for nm, f, r in [("전수 검사", 1, 1), (f"무작위 {rec*100:.0f}% 검사", rec, rec), (f"raw PVC + D4 {B*100:.0f}% 검사", B, rec), (f"무작위 {B*100:.0f}% 검사", B, rnd(B) / y.sum())]]); save(t10, "inspection_cost", index=False); print(t10.to_string(index=False))
print(f"같은 불량 비율을 찾는 검사량 절감: {(1 - B / rec) * 100:.1f}%")

# %% [markdown]
# ## 8. 그림

# %%
Wd = 6.4; mpl.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "savefig.dpi": 300})
# [그림 3-1] 위험 순위 구간별 불량
hit = caught(comb, B); fig, ax = plt.subplots(figsize=(Wd, 3.1)); x = np.arange(6)
ax.bar(x, bv, color=[PAL[0]] * 4 + [GRAY] * 2, width=.62); ax.plot(x, [(h_ - l_) * 79 for l_, h_ in bands], "o--", color=INK2, lw=1.1, ms=4, label="무작위로 섞였을 때의 기대 불량 수")
for xi, v in zip(x, bv): ax.text(xi, v + .7, f"{v:.1f}".rstrip("0").rstrip("."), ha="center", fontsize=10.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["상위 5%", "5~10%", "10~20%", "20~30%", "30~50%", "하위 50%"]); ax.set_ylabel("그 구간에 든 불량 수 (전체 79건)"); ax.set_xlabel("raw PVC + D4의 하루 안 위험 순위 구간"); ax.set_ylim(0, 44)
ax.axvspan(-.5, 3.5, color=PAL[0], alpha=.07); ax.text(1.5, 40.5, f"예산 30% 안 : 포착 {hit:.0f}건", ha="center", color=PAL[0], fontsize=9.5, fontweight="bold")
ax.text(4.5, 40.5, f"예산 30% 밖 : 미탐지 {79-hit:.0f}건", ha="center", color=INK2, fontsize=9.5, fontweight="bold"); ax.legend(loc="upper center", bbox_to_anchor=(.36, .86)); fig.tight_layout(); P.save(fig, "33_rank_bands_b30"); plt.close(fig)
# [그림 3-4] KAMP 전류 × 가압력별 추천 비율
n2_ = i2.타점.unstack().values; M2 = np.where(n2_ < 30, np.nan, i2["추천율%"].unstack().values); fig, ax = plt.subplots(figsize=(Wd, 3.0)); vmax = np.nanmax(M2); ax.imshow(M2, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
for i in range(M2.shape[0]):
    for j in range(M2.shape[1]):
        if np.isnan(M2[i, j]): ax.text(j, i, "자료 없음", ha="center", va="center", fontsize=9, color=GRAY)
        else: ax.text(j, i, f"{M2[i,j]:.1f}%\n({int(n2_[i,j]):,}타점)", ha="center", va="center", fontsize=10, color="white" if M2[i, j] > .6 * vmax else INK, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(list(i2.타점.unstack().columns)); ax.set_yticks(range(4)); ax.set_yticklabels(list(i2.타점.unstack().index)); ax.set_xlabel("가압력 (그날 중앙값 대비, bar)"); ax.set_ylabel("전류 (그날 중앙값 대비, kA)"); ax.grid(False)
for s_ in ax.spines.values(): s_.set_visible(False)
fig.tight_layout(); P.save(fig, "34_kamp_interaction_b30"); plt.close(fig)
# [그림 3-5] KAMP 조건 재현
conds = [("base", "원래 조건"), ("a", "(a) 불량률\n0.4%"), ("b", "(b) 학습 날짜\n8일"), ("c1", "(c) 위치 정보\n제거"), ("abc", "(a)+(b)+(c)\nKAMP 조건")]; fig, ax = plt.subplots(figsize=(Wd, 3.3)); wd = .27
for k_, (m, nm, col) in enumerate([("raw PVC", "raw PVC", PAL[1]), ("D4", "D4", PAL[2]), ("결합", "raw PVC + D4", PAL[0])]):
    v = [t5[(t5["조건"] == c) & (t5["모델"] == m)]["배수"].iloc[0] for c, _ in conds]; ax.bar(np.arange(5) + (k_ - 1) * wd, v, wd * .92, color=col, label=nm)
    for i, vv in enumerate(v): ax.text(i + (k_ - 1) * wd, vv + .04, f"{vv:.1f}", ha="center", fontsize=8.5)
ax.axhline(1, color=INK2, ls="--", lw=1.2); ax.text(4.48, 1.05, "무작위 수준", ha="right", color=INK2, fontsize=8.5); ax.set_xticks(range(5)); ax.set_xticklabels([c[1] for c in conds], fontsize=8.5)
ax.set_ylabel("예산 30%에서 찾은 불량 ÷ 무작위 기대값"); ax.set_ylim(0, 2.6); ax.legend(ncol=3, loc="upper right"); fig.tight_layout(); P.save(fig, "35_conditions_b30"); plt.close(fig)
# [그림 4-3] KAMP 날짜별 안정 공정범위
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "xtick.labelsize": 7.5, "ytick.labelsize": 8.5}); days_ = list(lo["cur"].index); xs = np.arange(len(days_)); fig, axes = plt.subplots(2, 3, figsize=(Wd, 4.5))
for ax, c in zip(axes.flat, V5):
    ax.axhspan(*allr[c], color=PAL[2], alpha=.16, lw=0); ax.bar(xs, hi[c].values - lo[c].values, bottom=lo[c].values, width=.56, color=PAL[0]); ax.plot(xs, med[c].values, "o", color="white", ms=3.6, mec=INK, mew=.9)
    yl, yh = min(lo[c].min(), allr[c][0]), max(hi[c].max(), allr[c][1]); pad = (yh - yl) * .22 or .5; ax.set_ylim(yl - pad, yh + pad); ax.set_xticks(xs); ax.set_xticklabels([d[5:].replace("-", "/") for d in days_], rotation=60)
    f = (lambda v: f"{v:,.0f}") if c == "heat" else (lambda v: f"{v:g}"); ax.set_title(VN[c], loc="left", fontweight="bold"); ax.set_title(f"{f(allr[c][0])}~{f(allr[c][1])}", loc="right", fontsize=9, fontweight="bold", color="#0d6b4a"); ax.grid(axis="x", visible=False)
    if c == "time": ax.set_yticks([70, 71, 72, 73])
ax = axes.flat[5]; ax.axis("off")
ax.add_patch(plt.Rectangle((.04, .74), .12, .11, color=PAL[0], transform=ax.transAxes)); ax.text(.22, .795, "그날 타점의 가운데 95% 구간\n(일별 관리 기준)", transform=ax.transAxes, va="center", fontsize=8.5)
ax.plot([.10], [.56], "o", color="white", ms=4.5, mec=INK, mew=1, transform=ax.transAxes); ax.text(.22, .56, "그날 중앙값", transform=ax.transAxes, va="center", fontsize=8.5)
ax.add_patch(plt.Rectangle((.04, .30), .12, .11, color=PAL[2], alpha=.25, transform=ax.transAxes)); ax.text(.22, .355, "9일 전체의 가운데 95% 구간\n(제안 안정 공정범위)", transform=ax.transAxes, va="center", fontsize=8.5)
fig.tight_layout(h_pad=1.2, w_pad=1.0); P.save(fig, "37_stable_range"); plt.close(fig)
# [그림 2-2] 모델별 포착 수 (10% 선정 기준, 30% 운영 권장)
mpl.rcParams.update({"axes.grid": False, "font.size": 10, "xtick.labelsize": 10, "ytick.labelsize": 9}); fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.3)); names = ["가이드북 AE", "raw PVC", "D4", "raw PVC\n+ D4", "무작위"]
for ax, b, ttl in zip(axes, (.1, B), ("검사예산 10% (모델 선정 기준)", "검사예산 30% (운영 권장)")):
    cv = [caught(a, b) for a, _ in MS.values()] + [rnd(b)]; ho = [ct(c, b) for _, c in MS.values()] + [rndt(b)]; xx = np.arange(5); w = .36
    ax.bar(xx - w/2, cv, w, color="#2a6ebb", label="일자 단위 교차검증 (불량 79건)"); ax.bar(xx + w/2, ho, w, color="#e8743b", label="시간순 홀드아웃 (불량 51건)")
    for i in range(5):
        for xo, v in ((-w/2, cv[i]), (w/2, ho[i])): ax.text(i + xo, v + max(cv) * .015, f"{v:.1f}".rstrip("0").rstrip("."), ha="center", fontsize=10)
    ax.set_xticks(xx); ax.set_xticklabels(names); ax.set_ylim(0, max(cv) * 1.22); ax.set_title(ttl, fontsize=11.5, fontweight="bold"); ax.set_ylabel("포착한 불량 수")
    if b == .1: ax.legend(loc="upper left", frameon=True, fontsize=9)
fig.tight_layout(w_pad=2.5); P.save(fig, "36_model_capture_b10_b30"); plt.close(fig)
