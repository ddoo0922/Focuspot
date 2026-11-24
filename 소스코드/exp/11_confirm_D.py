# %% [markdown]
# # 11. 확인 실험: 새 시드 10개로 D(의사라벨) 고정 여부 판단
# 10에서 쓴 시드(0~4)와 겹치지 않는 새 시드 5~14로 다시 돌린다(결과에 맞춘 선택을 피하려는 확인 실험).
# **판단 규칙(사전 고정)**: D가 4개 조건 평균 같은 위치 안 AUROC에서 A·C 이상이고, 원본·8묶음 4% 조건에서 기준 2·3을 다시 통과하면 최종 약지도 모델로 고정한다.

# %%
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from joblib import Parallel, delayed
from wl import runner, bags
pd.set_option("display.width", 250)
MODELS = ["const_rate", "A_dllp_mlp", "C_prior_two", "D_pseudo_logreg", "UB_logreg"]
CONDS = list(bags.CONDITIONS); SEEDS = list(range(5, 15)); N_GPU = torch.cuda.device_count()
name = "shape_confirm"
def _task(i, m, c, s): return runner.run_one(m, c, s, feature_set="shape_locstd", gpu=i % max(N_GPU, 1))
if (runner.RES / f"{name}.pkl").exists():
    runs = runner.load_runs(name)
else:
    tasks = [(m, c, s) for m in MODELS for c in CONDS for s in SEEDS]
    t = time.time(); runs = Parallel(n_jobs=8, verbose=2)(delayed(_task)(i, *k) for i, k in enumerate(tasks))
    print(f"{len(runs)}개 실행 {time.time()-t:.0f}초"); runner.save_runs(runs, name)
by = {}
for r in runs: by.setdefault((r["model"], r["cond"]), []).append(r)
boots = dict(Parallel(n_jobs=16)(delayed(lambda k: (k, runner.bootstrap(by[k], n_boot=200)))(k) for k in by))
def fmt(m, c, k):
    r = boots[(m, c)].loc[k]; return f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
KEYS = ["auroc", "wl_auroc", "recall@0.1", "wl_recall@0.1", "recall@5x"]
for c in CONDS:
    print(f"\n=== shape_locstd | {c} ==="); display(pd.DataFrame({k: {m: fmt(m, c, k) for m in MODELS} for k in KEYS}))

# %%
rows = []
for m in MODELS[1:]:
    row = {"모델": m}
    for c in CONDS:
        b = boots[(m, c)]
        row[c] = ("O" if b.loc["recall@0.1", "lo"] > .10 else "X") + ("O" if (b.loc["wl_recall@0.1", "lo"] > .10 and b.loc["wl_auroc", "lo"] > .5) else "X")
    rows.append(row)
print("[기준2 포착률][기준3 같은 위치 안]"); pd.DataFrame(rows)

# %%
avg = {m: np.mean([boots[(m, c)].loc["wl_auroc", "mean"] for c in CONDS]) for m in MODELS[1:]}
print("4조건 평균 같은 위치 안 AUROC:", {m: round(v, 3) for m, v in avg.items()})
