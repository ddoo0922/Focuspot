# %% [markdown]
# # 05. 순서 피처 점검 (2단계 선별 모델 후보)
# KAMP는 생산 순서의 인접 상관이 매우 높다(전류 0.96, 가압력 0.90, 동저항 0.92). 느린 변화(전극 마모 등)를 빼고 순간 이탈만 보는
# `dev_*`(직전 20타점 중앙값 대비 편차), `dl_*`(직전 타점 대비 변화량) 피처를 만든다.
# Stellantis N.V.는 순서를 파일 순서(차체 → 타점 번호)로 **가정**했고 위치 평균을 먼저 뺐다. 순서 가정이 틀리면 이 검증은 의미가 약하다는 한계가 있다.
# 통과 기준은 02의 기준 2·3(포착률, 같은 위치 안)을 그대로 쓴다.

# %%
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from joblib import Parallel, delayed
from wl import runner, bags
pd.set_option("display.width", 250)
MODELS = ["const_rate", "seq_rule", "iforest", "autoencoder", "A_dllp_mlp", "B_count_mlp", "D_pseudo_logreg", "UB_logreg", "UB_catboost"]
CONDS = list(bags.CONDITIONS); SEEDS = list(range(5)); N_GPU = torch.cuda.device_count()
name = "seq_features"
def _task(i, m, c, s): return runner.run_one(m, c, s, feature_set="seq", gpu=i % max(N_GPU, 1))
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
    print(f"\n=== seq | {c} ==="); display(pd.DataFrame({k: {m: fmt(m, c, k) for m in MODELS} for k in KEYS}))

# %%
rows = []
for m in MODELS[1:]:
    row = {"모델": m}
    for c in CONDS:
        b = boots[(m, c)]
        row[c] = ("O" if b.loc["recall@0.1", "lo"] > .10 else "X") + ("O" if (b.loc["wl_recall@0.1", "lo"] > .10 and b.loc["wl_auroc", "lo"] > .5) else "X")
    rows.append(row)
print("[기준2 포착률][기준3 같은 위치 안]"); pd.DataFrame(rows)
