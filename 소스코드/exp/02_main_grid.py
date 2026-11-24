# %% [markdown]
# # 02. 메인 그리드: 후보 모델 × 묶음 조건 (Stellantis N.V., 차체 단위 CV)
# 설계서 6절의 후보를 같은 조건에서 돌리고, **통과 기준은 결과를 보기 전에 아래와 같이 고정**한다(셀 실행 후 수정하지 않는다).
#
# | # | 기준 | 판정 |
# |---|---|---|
# | 1 | 1단계: 묶음 개수 오차(MAE)가 고정 불량률 기준보다 작다 | 시드 평균 비교 |
# | 2 | 2단계: 예산 10% 포착률의 95% 구간 하한 > 0.10 (무작위 기대값) | 차체 부트스트랩 |
# | 3 | 같은 위치 안: 위치별 예산 10% 포착률 하한 > 0.10 **그리고** 같은 위치 안 AUROC 하한 > 0.5 | 차체 부트스트랩 |
# | 4 | KAMP형 조건(`k8_1`, `k8_05`) 둘 다에서 1~3 유지 | — |
#
# 판정 대상은 우리 모델(A·B·C·D)이다. 기준·상한 모델은 비교선으로만 쓴다.
# `loc_rate`는 학습 폴드에서 차체를 뺀 위치별 불량률이라 같은 위치 안 점수가 leave-one-out 인공물(AUROC < 0.5)이 되므로 위치 안 지표를 "해당 없음"으로 둔다.

# %%
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from joblib import Parallel, delayed
from wl import data, runner, bags
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)

MODELS = ["const_rate", "loc_rate", "iforest", "autoencoder",
          "A_dllp_mlp", "B_count_mlp", "C_prior_two", "D_pseudo_logreg",
          "UB_logreg", "UB_catboost"]
OURS = ["A_dllp_mlp", "B_count_mlp", "C_prior_two", "D_pseudo_logreg"]
CONDS = list(bags.CONDITIONS)
SEEDS = list(range(5))
N_GPU = torch.cuda.device_count()
RERUN = False

# %% [markdown]
# ## 1. 학습·OOF 예측 (GPU 2장에 작업을 나눠 병렬 실행)

# %%
def _task(i, m, c, s):
    return runner.run_one(m, c, s, gpu=i % max(N_GPU, 1))

name = "main_grid"
if (runner.RES / f"{name}.pkl").exists() and not RERUN:
    runs = runner.load_runs(name)
else:
    tasks = [(m, c, s) for m in MODELS for c in CONDS for s in SEEDS]
    t = time.time()
    runs = Parallel(n_jobs=8, verbose=5)(delayed(_task)(i, *k) for i, k in enumerate(tasks))
    print(f"{len(runs)}개 실행, {time.time()-t:.0f}초")
    runner.save_runs(runs, name)
by = {}
for r in runs: by.setdefault((r["model"], r["cond"]), []).append(r)

# %% [markdown]
# ## 2. 지표 + 차체 단위 부트스트랩 95% 구간

# %%
def _boot(key, n_boot=200):
    return key, runner.bootstrap(by[key], n_boot=n_boot)
t = time.time()
boots = dict(Parallel(n_jobs=16)(delayed(_boot)(k) for k in by))
print(f"부트스트랩 {time.time()-t:.0f}초")
WL = ["wl_auroc", "wl_recall@0.05", "wl_recall@0.1", "wl_recall@0.2"]
for (m, c), b in boots.items():
    if m == "loc_rate": b.loc[WL, :] = np.nan

def fmt(m, c, k, d=3):
    r = boots[(m, c)].loc[k]
    return "n/a" if np.isnan(r["mean"]) else f"{r['mean']:.{d}f} [{r['lo']:.{d}f}, {r['hi']:.{d}f}]"

def table(c, keys):
    return pd.DataFrame({k: {m: fmt(m, c, k) for m in MODELS} for k in keys})

# %% [markdown]
# ### 2-1. 원본 조건 (`orig`: 20묶음, 불량률 4%)

# %%
KEYS = ["mae", "auroc", "wl_auroc", "recall@0.1", "wl_recall@0.1", "recall@5x"]
table("orig", KEYS)

# %% [markdown]
# ### 2-2. 묶음만 8개로 줄인 조건 (`k8_4`)

# %%
table("k8_4", KEYS)

# %% [markdown]
# ### 2-3. KAMP형: 8묶음·불량률 1% (`k8_1`)

# %%
table("k8_1", KEYS)

# %% [markdown]
# ### 2-4. KAMP형: 8묶음·불량률 0.5% (`k8_05`)

# %%
table("k8_05", KEYS)

# %% [markdown]
# ## 3. 통과 기준 판정 (위 표의 기준을 그대로 적용)

# %%
def verdict(m, c):
    mae_ok = boots[(m, c)].loc["mae", "mean"] < boots[("const_rate", c)].loc["mae", "mean"]
    r10 = boots[(m, c)].loc["recall@0.1", "lo"] > 0.10
    wl = boots[(m, c)].loc["wl_recall@0.1", "lo"] > 0.10 and boots[(m, c)].loc["wl_auroc", "lo"] > 0.5
    return mae_ok, r10, wl

rows = []
for m in OURS:
    row = {"모델": m}
    for c in CONDS:
        a, b, w = verdict(m, c)
        row[c] = f"{'O' if a else 'X'}{'O' if b else 'X'}{'O' if w else 'X'}"
    kl = [verdict(m, c) for c in ("k8_1", "k8_05")]
    row["기준4(KAMP형 유지)"] = "통과" if all(all(v) for v in kl) else "미통과"
    row["전 조건 통과"] = "통과" if all(all(verdict(m, c)) for c in CONDS) else "미통과"
    rows.append(row)
print("표기: [기준1 MAE][기준2 포착률][기준3 위치안] 순서, O=충족 X=미충족")
pd.DataFrame(rows)
