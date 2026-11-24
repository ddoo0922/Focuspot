# %% [markdown]
# # 07. (A) 상위 타점 쏠림이 복사 블록 경계 효과인가 / (B) 2단계 오토인코더 입력 비교
# **사전 규칙(결과 보기 전 고정)**
# - A: 블록 행을 중립화하지 않고 아예 제거한 뒤 순서 피처를 다시 만들어도 상위 타점의 연속 구간 쏠림이 남으면 경계 효과가 아니라고 본다. 비교 지표는 상위 30개의 겹침, 일 내 순위 상관, 연속 구간 수(무작위 선택 기대값 대비).
# - B: 같은 위치 안 AUROC를 4개 조건 평균으로 비교하고, 가장 높은 입력을 2단계 입력으로 고른다(동률 근처면 구간 하한이 높은 쪽).

# %%
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from wl import data, kamp, models, runner, bags
pd.set_option("display.width", 220)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## A. KAMP 경계 효과 점검

# %%
K, cnt_day = kamp.load_kamp()
for d, g in K[K.block].groupby("date"):
    print(str(d)[:10], "블록 idx 범위", g.idx.min(), "~", g.idx.max(), f"({len(g)}행)")

def ae_risk(Z, seeds=5):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    out = []
    for s in range(seeds):
        ae = models.AutoEncoder(dev); ae.fit(Zs, None, None, None, None, None, s)
        sc, _ = ae.predict(Zs, None); out.append((sc - sc.mean()) / sc.std())
    return np.mean(out, axis=0)

ok = ~K.block.values
Kb = K[ok].reset_index(drop=True)
Xn = pd.DataFrame(kamp.make_X(K, True), columns=kamp.FEATS)
SA = data.seq_features(Xn, K.date.values)                         # 변형 A: 블록 중립화 후 순서 피처
Xraw = K.loc[ok, kamp.FEATS].reset_index(drop=True)
SB = data.seq_features(Xraw, Kb.date.values)                       # 변형 B: 블록 행 제거 후 순서 피처
Kb["A_neutral_seq"] = ae_risk(SA.values[ok])
Kb["B_drop_seq"] = ae_risk(SB.values)
Kb["C_raw"] = ae_risk(Xraw.values)
VAR = ["A_neutral_seq", "B_drop_seq", "C_raw"]

# %%
rows = []
for d, g in Kb.groupby("date"):
    r = {"날짜": str(d)[:10]}
    r["ρ(A,B)"] = spearmanr(g.A_neutral_seq, g.B_drop_seq)[0]
    r["ρ(A,C)"] = spearmanr(g.A_neutral_seq, g.C_raw)[0]
    r["ρ(B,C)"] = spearmanr(g.B_drop_seq, g.C_raw)[0]
    top = {v: set(g.nlargest(30, v).idx) for v in VAR}
    r["상위30 겹침 A∩B"] = len(top["A_neutral_seq"] & top["B_drop_seq"])
    r["상위30 겹침 A∩C"] = len(top["A_neutral_seq"] & top["C_raw"])
    rows.append(r)
print("일 내 순위 상관과 상위 30 겹침"); pd.DataFrame(rows).round(2)

# %%
rng = np.random.default_rng(0)
def n_runs(ix):                      # 연속 idx로 이어진 묶음 수(작을수록 한 구간에 몰림)
    ix = np.sort(ix); return 1 + int((np.diff(ix) > 3).sum())
def block_dist(g, K):
    b = K[K.block & (K.date == g.date.iloc[0])].idx.values
    return np.full(len(g), np.inf) if len(b) == 0 else np.abs(g.idx.values[:, None] - b[None, :]).min(1)
rows = []
for d, g in Kb.groupby("date"):
    dist = block_dist(g, K)
    for v in VAR:
        sel = g.index.get_indexer(g.nlargest(30, v).index)
        base = np.mean([n_runs(g.idx.values[rng.choice(len(g), 30, replace=False)]) for _ in range(500)])
        rows.append({"날짜": str(d)[:10], "변형": v, "상위30 연속구간 수": n_runs(g.idx.values[sel]), "무작위 기대": round(base, 1),
                     "블록 30idx 이내 비율": round(float((dist[sel] <= 30).mean()), 2) if np.isfinite(dist).any() else np.nan,
                     "전체 중 블록 30idx 이내 비율": round(float((dist <= 30).mean()), 2) if np.isfinite(dist).any() else np.nan})
print("연속 구간 수가 무작위 기대보다 훨씬 작으면 상위 타점이 구간에 몰린 것"); pd.DataFrame(rows)

# %% [markdown]
# ## B. 2단계 오토인코더 입력 비교 (Stellantis N.V., 같은 위치 안)
# 입력 3종: `kamp_full`(원시 요약값), `seq`(순서 피처), `kamp_seq`(둘 다). 시드 10개, 4개 조건.

# %%
SETS = ["kamp_full", "seq", "kamp_seq"]; SEEDS = list(range(10)); CONDS = list(bags.CONDITIONS)
N_GPU = torch.cuda.device_count()
def _t(i, c, s, fs): return runner.run_one("autoencoder", c, s, feature_set=fs, gpu=i % max(N_GPU, 1))
name = "ae_input"
if (runner.RES / f"{name}.pkl").exists(): runs = runner.load_runs(name)
else:
    tasks = [(c, s, fs) for fs in SETS for c in CONDS for s in SEEDS]
    runs = Parallel(n_jobs=8, verbose=2)(delayed(_t)(i, *k) for i, k in enumerate(tasks)); runner.save_runs(runs, name)
by = {}
for r in runs: by.setdefault((r["feature_set"], r["cond"]), []).append(r)
boots = dict(Parallel(n_jobs=16)(delayed(lambda k: (k, runner.bootstrap(by[k], n_boot=200)))(k) for k in by))

# %%
rows = []
for fs in SETS:
    row = {"입력": fs}
    for c in CONDS:
        b = boots[(fs, c)]
        row[c] = f"{b.loc['wl_auroc','mean']:.3f} [{b.loc['wl_auroc','lo']:.3f}, {b.loc['wl_auroc','hi']:.3f}]"
    row["4조건 평균 위치안 AUROC"] = round(np.mean([boots[(fs, c)].loc["wl_auroc", "mean"] for c in CONDS]), 3)
    row["4조건 평균 위치안 포착률@10%"] = round(np.mean([boots[(fs, c)].loc["wl_recall@0.1", "mean"] for c in CONDS]), 3)
    rows.append(row)
print("같은 위치 안 AUROC (무작위 0.5), 포착률 무작위 0.10"); pd.DataFrame(rows)
