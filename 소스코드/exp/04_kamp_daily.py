# %% [markdown]
# # 04. KAMP 실제 데이터: 날짜 단위 검증 (묶음 = 하루, 크기 조정 없음)
# 최종 산출물은 "날짜별 예상 불량 개수"와 "검사 우선순위"다. 이 노트북은 KAMP 원본에서 1단계를 직접 검증한다.
# - 하루를 빼고 학습해 그날의 불량 개수를 예측(일 단위 leave-one-day-out), 라벨 있는 8일 사용
# - 비교선: 고정 불량률(학습일 평균 불량률 × 그날 타점 수)
# - 하한: 불량률을 정확히 아는 이상적 예측기도 우연 변동 때문에 남는 오차
# - 3/27은 라벨이 없는 날이라 8일 전체로 학습한 뒤 예측만 한다
# - 복사 블록(가압력 > 3 bar, 291행×4일)은 기록 오류이므로 `중립화`(해당 행 피처를 비블록 중앙값으로 대체) 변형을 함께 본다. 행 수는 그대로 둬 일별 개수와의 대응을 유지한다.

# %%
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from scipy.stats import binom
from scipy.special import expit
from wl import data, models, runner
pd.set_option("display.width", 220)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

raw = pd.read_excel(data.KAMP_XLSX, sheet_name="Raw data")
res = pd.read_excel(data.KAMP_XLSX, sheet_name="result")
K = pd.DataFrame({
    "date": raw["working time"].dt.normalize().values,
    "force_mean": raw["weld force(bar)"], "cur_mean": raw["weld current(kA)"],
    "vol_mean": raw["weld Voltage(v)"], "weld_len": raw["weld time(ms)"],
})
K["heat"] = K.vol_mean * K.cur_mean * K.weld_len
K["dynres"] = K.vol_mean / K.cur_mean
K["i_over_f"] = K.cur_mean / K.force_mean
K["block"] = K.force_mean > 3.0
FEATS = data.FEATURE_SETS["kamp_full"]
lab = res.groupby(res["working time"].dt.normalize())["defect"].sum()
days = sorted(K.date.unique())
cnt_day = pd.Series({d: lab.get(d, np.nan) for d in days})
n_day = K.groupby("date").size()
print(pd.DataFrame({"타점수": n_day, "불량(라벨)": cnt_day, "블록행": K.groupby("date").block.sum()}))

# %% [markdown]
# ## 입력 준비: 블록 중립화

# %%
def make_X(neutralize):
    X = K[FEATS].copy()
    if neutralize:
        med = X[~K.block].median()
        X.loc[K.block, :] = med.values
    return X.values.astype(float)

# %% [markdown]
# ## 일 단위 leave-one-day-out
# 학습: 남은 7일의 (타점 피처, 일별 개수). 예측: 빠진 날 타점 확률의 합 → 학습 구간 전체 개수에 맞게 보정.

# %%
labeled = [d for d in days if not np.isnan(cnt_day[d])]
MODELS = ["A_dllp_mlp", "B_count_mlp", "C_prior_two", "C_prior_two_only", "D_pseudo_logreg"]
SEEDS = range(3)

def fit_predict(name, Xtr, bag_tr, cnt_tr, Xte, seed):
    mdl = models.make_model(name, dev)
    mdl.fit(Xtr, bag_tr, cnt_tr, None, np.zeros(len(Xtr), int), FEATS, seed)
    _, z_tr = mdl.predict(Xtr, None); _, z_te = mdl.predict(Xte, None)
    sh = runner._calibrate_shift(z_tr, cnt_tr.sum())
    return expit(z_te + sh)

def loo(neutralize):
    X = make_X(neutralize)
    rows = []
    for d in labeled:
        te = (K.date == d).values
        tr = (~te) & K.date.isin(labeled).values
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-9
        Xs = (np.where(np.isnan(X), np.nanmedian(X[tr], 0), X) - mu) / sd
        ub, bag_tr = np.unique(K.date.values[tr], return_inverse=True)
        cnt_tr = np.array([cnt_day[u] for u in ub]).astype(int)
        row = {"날짜": str(d)[:10], "타점수": int(te.sum()), "실제": int(cnt_day[d]),
               "고정불량률": cnt_tr.sum() / tr.sum() * te.sum()}
        for m in MODELS:
            row[m] = np.mean([fit_predict(m, Xs[tr], bag_tr, cnt_tr, Xs[te], s).sum() for s in SEEDS])
        rows.append(row)
    return pd.DataFrame(rows)

out = {}
for nz in (False, True):
    out[nz] = loo(nz)
    t = out[nz]
    mae = {c: float((t[c] - t["실제"]).abs().mean()) for c in ["고정불량률"] + MODELS}
    floor = np.mean([sum(binom.pmf(k, n, a / n) * abs(k - a) for k in range(0, int(n) + 1))
                     for n, a in zip(t["타점수"], t["실제"])])
    print(f"\n=== 블록 중립화 = {nz} ===")
    display(t.round(2))
    print("MAE:", {k: round(v, 3) for k, v in mae.items()})
    print(f"우연 변동 하한(그날 실제 불량률을 정확히 아는 경우) ≈ {floor:.2f}")

# %% [markdown]
# ## 해석 기준
# - 모델 MAE가 고정 불량률보다 작아야 1단계 기준 충족. 8일뿐이라 차이가 우연인지는 일별 오차 차이의 부호 일치로 함께 본다.
# - 우연 변동 하한에 가까우면 "더 줄일 여지가 없다"는 뜻이지 모델이 좋다는 뜻이 아니다.

# %%
for nz in (False, True):
    t = out[nz]
    e0 = (t["고정불량률"] - t["실제"]).abs()
    print(f"\n블록 중립화={nz}: 고정 불량률보다 일별 오차가 작은 날 수")
    for m in MODELS:
        em = (t[m] - t["실제"]).abs()
        print(f"  {m:18s} {(em < e0).sum()}/8일  평균 오차 차이 {float((em - e0).mean()):+.3f}")

# %% [markdown]
# ## 3/27 예측 (라벨 없는 날, 8일 전체로 학습)
# 날짜별 예상 불량 개수와, 그날 타점 중 확률 상위 타점(검사 후보). 위치 정보가 없어 타점 순위는 같은 설비 안의 상대 순위일 뿐이다.

# %%
X = make_X(True)
tr = K.date.isin(labeled).values; te = (K.date == pd.Timestamp("2020-03-27")).values
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
Xs = (X - mu) / sd
ub, bag_tr = np.unique(K.date.values[tr], return_inverse=True)
cnt_tr = np.array([cnt_day[u] for u in ub]).astype(int)
pred = {}
for m in MODELS:
    P = np.mean([fit_predict(m, Xs[tr], bag_tr, cnt_tr, Xs[te], s) for s in SEEDS], axis=0)
    pred[m] = P
print("3/27 타점 수", int(te.sum()), "| 고정 불량률 예상", round(cnt_tr.sum() / tr.sum() * te.sum(), 2))
print({m: round(float(p.sum()), 2) for m, p in pred.items()})
