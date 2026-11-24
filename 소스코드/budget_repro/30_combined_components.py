# %% [markdown]
# # 30. raw PVC + D4 결합 모델의 구성 요소별 기여 (보고서 3장 [표 3-2])
# raw PVC(저장된 10시드 평균 점수)에 D4의 세 묶음(수준 이탈 / 이동 중앙값 대비 편차 / 직전 타점 대비 변화량)을 하나씩 결합해
# 예산 10%에서 찾은 불량 수와 AUROC를 낸다. 결합 = 두 점수의 날짜 안 순위 평균(20번·29번 노트북과 같은 식).
# 새 학습은 하지 않고 저장된 점수만 쓴다. numpy·pandas만 있으면 돈다(pandas 3 이상: 저장된 pkl을 읽기 위해).
# AUROC는 2장(25번 노트북)과 같이 날짜마다 점수를 순위 백분위로 바꿔 모은 값으로 계산한다. 동점은 기대값으로 센다.

# %%
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # 스크립트·노트북 둘 다 실행
P0 = pd.read_pickle(HERE.parent / "exp" / "cache" / "polito_summary.pkl")
RAW = pd.read_pickle(HERE / "results" / "same_model_scores.pkl")
E3 = pd.read_pickle(HERE / "results" / "exp3_scores.pkl")
PHYS = ["heat", "dynres", "i_over_f"]

def derived5_absz(P, day, w=20):
    """br.spec.load_polito("derived5_absz")와 같은 계산: 파생 9개 → 날짜 안 |강건 z|."""
    X = pd.DataFrame({"cur": P.cur_mean.values, "vol": P.vol_mean.values, "time": P.weld_len.values, "force": P.force_mean.values})
    X["heat"] = X.cur * X.vol * X.time; X["dynres"] = X.vol / X.cur; X["i_over_f"] = X.cur / X.force
    out = X[PHYS].copy(); g = pd.Series(day)
    for c in PHYS:
        v = X[c]
        out[f"dev_{c}"] = (v - v.groupby(g).transform(lambda s: s.shift(1).rolling(w, min_periods=3).median())).fillna(0.0)
        out[f"dl_{c}"] = (v - v.groupby(g).shift(1)).fillna(0.0)
    out = out.fillna(out.median()); med = out.groupby(g).transform("median"); mad = (out - med).abs().groupby(g).transform("median") * 1.4826
    return ((out - med).abs() / (mad + 1e-9 + 1e-6 * out.abs().groupby(g).transform("median"))).clip(upper=10)

def inclusion_prob(score, k, grp):
    """각 타점이 '그날 상위 k개'에 뽑힐 확률(동점은 균등 분할). wl.metrics.inclusion_prob와 같다."""
    n = len(score); order = np.lexsort((-score, grp)); g, s = grp[order], score[order]
    new_grp = np.r_[True, g[1:] != g[:-1]]; new_tie = new_grp | np.r_[True, s[1:] != s[:-1]]; idx = np.arange(n)
    grp_start = np.maximum.accumulate(np.where(new_grp, idx, 0)); tie_start = np.maximum.accumulate(np.where(new_tie, idx, 0))
    tie_id = np.cumsum(new_tie) - 1; tie_size = np.bincount(tie_id)[tie_id]
    out = np.empty(n); out[order] = np.clip((np.asarray(k, float)[g] - (tie_start - grp_start)) / tie_size, 0.0, 1.0); return out

def auroc(y, s):
    r = pd.Series(s).rank().values; p = y.sum(); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p))

day = P0.date.dt.strftime("%Y-%m-%d").values; y = P0.y.values
Xd = derived5_absz(P0, day)
raw = np.mean([g.sort_values("row").score.values for _, g in RAW[(RAW.data == "polito") & (RAW.model == "PVC MLP")].groupby("seed")], 0)
TE = np.array([d >= "2024-02-01" for d in day])
groups = {"수준 이탈": PHYS, "이동 중앙값 대비 편차": [c for c in Xd if c.startswith("dev_")],
          "직전 타점 대비 변화량": [c for c in Xd if c.startswith("dl_")], "D4(전체 9개)": list(Xd.columns)}
rows = []
for ev, mask, rw in [("전체 20일", np.ones(len(y), bool), raw), ("시간순(2024 시험)", TE, np.asarray(E3["raw_temporal"]))]:
    yt, dt = y[mask], day[mask]; _, gi = np.unique(dt, return_inverse=True); n = np.bincount(gi); k = np.ceil(.1 * n - 1e-9)
    rk = lambda s: pd.Series(s).groupby(dt).rank(pct=True).values
    def add(name, s):
        rows.append({"평가": ev, "구성": name, "10% 포착": round((inclusion_prob(s, k, gi) * yt).sum(), 1), "불량 수": int(yt.sum()),
                     "무작위": round((k / n * np.bincount(gi, weights=yt)).sum(), 1), "AUROC": round(auroc(yt, rk(s)), 3)})
    add("raw PVC 단독", rw)
    for g, cols in groups.items():
        s = Xd.loc[mask, cols].sum(axis=1).values
        add(g + " 단독", s); add("raw PVC + " + g, (rk(rw) + rk(s)) / 2)
    d4 = Xd.loc[mask].sum(axis=1).values; a, b = inclusion_prob(rw, k, gi), inclusion_prob(d4, k, gi)
    print(f"{ev}: raw PVC만 잡은 불량 {(a*(1-b)*yt).sum():.1f}, D4만 {(b*(1-a)*yt).sum():.1f}, 둘 다 {(a*b*yt).sum():.1f}")
R = pd.DataFrame(rows); R.to_csv(HERE / "results" / "combined_components.csv", index=False, encoding="utf-8-sig"); print(R.to_string(index=False))
