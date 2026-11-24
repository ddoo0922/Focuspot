# %% [markdown]
# # 03. 입력 변형 민감도 (탐색 실험)
# 02의 결과에서 "같은 위치 안" 기준을 통과한 우리 모델이 없었고, 타점 정답으로 학습한 상한 모델(UB_logreg)도 위치 안 AUROC가 0.5 아래였다.
# 원시 요약값에는 위치 정보가 섞여 있어 모델이 위치 효과를 배우고, 위치 안에서는 그 방향이 뒤집히기 때문이다(01 노트북 2절).
# 설비·품목·두께가 고정된 KAMP에서는 위치 효과가 없으므로, **위치별 평균을 뺀 입력**(`*_locstd`)이 KAMP 조건에 더 가깝다.
#
# 통과 기준은 02와 **같은 기준을 그대로** 적용한다(기준 변경 없음). 이 노트북은 탐색용이며, 최종 입력 조건을 고르는 근거로 쓰려면 새 시드로 확인해야 한다.
# - `kamp_full`: 02와 동일(가압력 포함, 위치 미보정) — 비교선
# - `kamp_noforce`: 가압력 제외
# - `kamp_full_locstd` / `kamp_noforce_locstd`: 위치 평균 제거

# %%
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from joblib import Parallel, delayed
from wl import runner, bags
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)

MODELS = ["const_rate", "iforest", "autoencoder", "A_dllp_mlp", "B_count_mlp", "C_prior_two", "C_prior_two_only",
          "D_pseudo_logreg", "UB_logreg", "UB_catboost"]
OURS = ["A_dllp_mlp", "B_count_mlp", "C_prior_two", "C_prior_two_only", "D_pseudo_logreg"]
CONDS = list(bags.CONDITIONS)
FSETS = ["kamp_noforce", "kamp_full_locstd", "kamp_noforce_locstd"]
SEEDS = list(range(5))
N_GPU = torch.cuda.device_count()
RERUN = False

# %%
def _task(i, m, c, s, fs):
    return runner.run_one(m, c, s, feature_set=fs, gpu=i % max(N_GPU, 1))

name = "feature_ablation"
if (runner.RES / f"{name}.pkl").exists() and not RERUN:
    runs = runner.load_runs(name)
else:
    tasks = [(m, c, s, fs) for fs in FSETS for m in MODELS for c in CONDS for s in SEEDS]
    t = time.time()
    runs = Parallel(n_jobs=8, verbose=2)(delayed(_task)(i, *k) for i, k in enumerate(tasks))
    print(f"{len(runs)}개 실행, {time.time()-t:.0f}초")
    runner.save_runs(runs, name)
by = {}
for r in runs: by.setdefault((r["feature_set"], r["model"], r["cond"]), []).append(r)

# %%
def _boot(key): return key, runner.bootstrap(by[key], n_boot=200)
boots = dict(Parallel(n_jobs=16)(delayed(_boot)(k) for k in by))

def fmt(fs, m, c, k, d=3):
    r = boots[(fs, m, c)].loc[k]
    return f"{r['mean']:.{d}f} [{r['lo']:.{d}f}, {r['hi']:.{d}f}]"

KEYS = ["mae", "auroc", "wl_auroc", "recall@0.1", "wl_recall@0.1"]
for fs in FSETS:
    for c in ("orig", "k8_1"):
        print(f"\n=== {fs} | {c} ==="); display(pd.DataFrame({k: {m: fmt(fs, m, c, k) for m in MODELS} for k in KEYS}))

# %% [markdown]
# ## 통과 기준 판정 (02와 동일 기준)
# 표기: [기준1 MAE][기준2 포착률][기준3 위치안], O=충족 X=미충족. 기준4는 `k8_1`·`k8_05` 둘 다에서 1~3 충족.

# %%
def verdict(fs, m, c):
    b = boots[(fs, m, c)]; bc = boots[(fs, "const_rate", c)]
    return (b.loc["mae", "mean"] < bc.loc["mae", "mean"], b.loc["recall@0.1", "lo"] > 0.10,
            b.loc["wl_recall@0.1", "lo"] > 0.10 and b.loc["wl_auroc", "lo"] > 0.5)

rows = []
for fs in FSETS:
    for m in OURS:
        row = {"입력": fs, "모델": m}
        for c in CONDS:
            row[c] = "".join("O" if v else "X" for v in verdict(fs, m, c))
        row["기준4"] = "통과" if all(all(verdict(fs, m, c)) for c in ("k8_1", "k8_05")) else "미통과"
        rows.append(row)
pd.DataFrame(rows)
