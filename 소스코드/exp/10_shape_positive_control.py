# %% [markdown]
# # 10. 양성 대조: 파형 모양 피처에서 약지도 모델이 신호를 잡는가
# 1장에서 위치를 통제하고도 남았던 신호(동저항 하강폭, 전류 후반/초반비, 가압력 최대 등)는 파형 모양에서 나온다. KAMP에는 없는 정보다.
# 목적 1: 평가 코드가 신호가 있을 때 제대로 잡는지(양성 대조). 목적 2: 날짜별 개수만으로 학습하는 우리 모델이 신호가 있을 때 그걸 잡는지.
# 입력은 파형 모양 피처 10개를 위치 평균 제거(`shape_locstd`). 기준은 02와 같다.

# %%
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from joblib import Parallel, delayed
from wl import runner, bags
pd.set_option("display.width", 250)
MODELS = ["const_rate", "autoencoder", "A_dllp_mlp", "B_count_mlp", "C_prior_two", "D_pseudo_logreg", "UB_logreg", "UB_catboost"]
CONDS = list(bags.CONDITIONS); SEEDS = list(range(5)); N_GPU = torch.cuda.device_count()
name = "shape_control"
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
for c in ("orig", "k8_1"):
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
