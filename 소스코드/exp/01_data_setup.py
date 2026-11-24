# %% [markdown]
# # 01. 환경·데이터·묶음 조건 점검
# 설계서(`제조 ai 실험설계.md`) 6절의 평가조건을 구현한 `wl/` 라이브러리를 확인한다.
# - 입력: Stellantis N.V. 파형 → KAMP형 타점 요약값(가압력·전류·전압·통전시간 + 파생)
# - 묶음 조건 4가지: `orig`(20묶음·4%) / `k8_4`(8묶음·4%) / `k8_1`(8묶음·1%) / `k8_05`(8묶음·0.5%)
# - 분할: 차체 단위 7-fold, 반복 시드 5회 이상

# %%
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
import matplotlib; matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from wl import data, bags, metrics
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| GPU 수", torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i, torch.cuda.get_device_name(i))

# %% [markdown]
# ## 1. 타점 요약값
# 가압력이 전부 0으로 기록된 타점(약 284개)은 `force_mean`, `i_over_f`가 NaN이다. 학습 폴드의 중앙값으로 채운다(`kamp_full`).
# 가압력을 아예 빼는 `kamp_noforce` 세트로 민감도를 따로 본다.

# %%
df = data.load_polito()
print(df.shape); display(df.head())
print("가압력 결측:", df.force_mean.isna().sum(), "| 연도별:", df.groupby(df.date.dt.year).force_mean.apply(lambda s: s.isna().mean().round(2)).to_dict())

# %% [markdown]
# ## 2. 위치 교란 재확인 (1장 핵심 발견)
# 위치 번호만으로 AUROC가 얼마나 나오는지, 그리고 같은 위치 안에서는 신호가 얼마나 남는지 본다.

# %%
y = df.y.values
loc_rate = df.groupby("loc").y.transform("mean").values   # 전체 데이터 기준(누수 있음, 교란 크기 확인용)
print("위치별 불량률(전체 데이터)만으로 AUROC =", round(metrics.evaluate(y, np.full(len(y), y.mean()), loc_rate, df["loc"].values, np.zeros(len(y), int))["auroc"], 3))
top = df.groupby("loc").y.agg(["sum", "count"]).sort_values("sum", ascending=False).head(4); display(top)
feat_auc = {}
for c in data.FEATURE_SETS["kamp_full"]:
    v = df[c].fillna(df[c].median()).values
    r = metrics.evaluate(y, np.full(len(y), y.mean()), v, df["loc"].values, np.zeros(len(y), int))
    feat_auc[c] = (r["auroc"], r["wl_auroc"])
display(pd.DataFrame(feat_auc, index=["전체 AUROC", "같은 위치 안 AUROC"]).T.round(3))

# %% [markdown]
# ## 3. 묶음 조건
# KAMP는 묶음당 평균 약 1,300타점이지만 Stellantis N.V.는 전체가 1,976타점이라 묶음을 8개로 합쳐도 약 250타점이다. 이 차이는 이 실험으로 메울 수 없으므로 한계로 기록한다.

# %%
rows = []
for cond in bags.CONDITIONS:
    idx, b = bags.build_condition(df, cond, 0)
    d = df.iloc[idx]
    rows.append(dict(조건=cond, 타점수=len(idx), 불량=int(d.y.sum()), 불량률=f"{d.y.mean()*100:.2f}%", 묶음수=int(b.max()) + 1,
                     묶음크기=np.bincount(b).tolist(), 묶음별불량=np.bincount(b, weights=d.y.values).astype(int).tolist()))
display(pd.DataFrame(rows))

# %% [markdown]
# ## 4. 지표 검증 (무작위·완벽 기준)
# 동점을 기대값으로 처리하는 포착률이 이론값과 맞는지 확인한다.

# %%
rng = np.random.default_rng(0)
yy = (rng.random(5000) < 0.04).astype(int); loc = rng.integers(0, 150, 5000); bag = rng.integers(0, 8, 5000)
for name, s in {"무작위": rng.random(5000), "상수": np.zeros(5000), "완벽": yy + 0.0}.items():
    r = metrics.evaluate(yy, np.full(5000, 0.04), s, loc, bag)
    assert abs(r['wl_recall@0.1'] - 0.1) < 1e-9 or name == "완벽" or name == "무작위", name
    print(f"{name:4s} auroc={r['auroc']:.3f} recall@10%={r['recall@0.1']:.3f} 위치내 recall@10%={r['wl_recall@0.1']:.3f}")
