# %% [markdown]
# # 34. KAMP 공정 신호 t-SNE 시각화
# 33번과 같은 입력(전류·전압·통전시간·가압력 표준화, 복사 블록·저전압 16타점 제외, 10,759타점).
# 타점별 유형 정답이 없으므로 유형으로 색칠할 수 없다. 대신 군집·날짜·각 신호·생산 순서로 색을 입혀 구조가 무엇에서 오는지 본다.
# t-SNE 축은 거리 보존이 아니라 이웃 보존이라 크기·간격을 해석하지 않는다.

# %%
import sys, time, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import br
from br import plot as P
from wl import kamp as wlkamp
warnings.filterwarnings("ignore"); P.setup(); t0 = time.time()

K, _ = wlkamp.load_kamp()
K = K.rename(columns={"cur_mean": "cur", "vol_mean": "vol", "weld_len": "time", "force_mean": "force"})
K["day"] = pd.to_datetime(K.date).dt.strftime("%m-%d")
KB = K[(~K.block) & (K.vol >= 2.6)].reset_index(drop=True)
KB["order"] = KB.groupby("day").cumcount() / KB.groupby("day").idx.transform("size")      # 같은 날 생산 순서(0~1)
RAW4 = ["cur", "vol", "time", "force"]
Z = ((KB[RAW4] - KB[RAW4].mean()) / KB[RAW4].std()).values
print("타점:", len(KB))
km = KMeans(3, n_init=10, random_state=0).fit_predict(Z)

# %%
def tsne(perp, seed=0):
    return TSNE(2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(Z)

E = {50: tsne(50)}; print(f"perplexity 50 완료 [{time.time() - t0:.0f}s]")

# %%
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
def panel(a, c, title, cmap="viridis", cat=None, vmin=None, vmax=None):
    a.grid(False); a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=11)
    if cat is None:
        s = a.scatter(*E[50].T, c=c, s=3, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True); plt.colorbar(s, ax=a, shrink=0.8)
    else:
        for i, (lab, col) in enumerate(cat):
            m = c == lab; a.scatter(*E[50][m].T, s=3, color=col, label=f"{lab}  (n={m.sum():,})", rasterized=True)
        a.legend(markerscale=4, fontsize=8, loc="best")
panel(ax[0, 0], km, "K-평균 k=3 군집", cat=[(0, P.PAL[0]), (1, P.PAL[1]), (2, P.PAL[2])])
days = sorted(KB.day.unique())
panel(ax[0, 1], KB.day.values, "날짜", cat=[(d, c) for d, c in zip(days, P.PAL + ["#52514e"])])
panel(ax[0, 2], KB.order.values, "같은 날 생산 순서 (0=시작, 1=끝)", cmap="viridis")
vc = lambda k: np.nanpercentile(KB[k], [1, 99])
panel(ax[1, 0], KB.cur.values, "전류 (kA)", cmap="coolwarm", vmin=vc("cur")[0], vmax=vc("cur")[1])
panel(ax[1, 1], KB.force.values, "가압력 (bar)", cmap="coolwarm", vmin=vc("force")[0], vmax=vc("force")[1])
tm = KB.time.clip(70, 73).astype(int).values
panel(ax[1, 2], tm, "통전시간 (ms, 70 이하·73 이상은 묶음)", cat=[(70, P.PAL[4]), (71, P.PAL[3]), (72, P.PAL[0]), (73, P.PAL[1])])
fig.suptitle("KAMP 공정 신호 t-SNE (perplexity 50, 10,759타점) — 불량 유형은 타점별 정답이 없어 색칠 불가", fontsize=13, y=0.995)
fig.tight_layout(); P.save(fig, "27_kamp_tsne"); plt.close(fig)

# %% [markdown]
# ## 보조: perplexity·시드를 바꿔도 같은 구조인가

# %%
rows = [(15, 0), (150, 0), (50, 1)]
for p, s in rows:
    E[(p, s)] = tsne(p, s); print(f"perplexity {p} seed {s} 완료 [{time.time() - t0:.0f}s]")
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
for j, key in enumerate([(15, 0), (50, 0), (150, 0)]):
    Y = E[50] if key == (50, 0) else E[key]
    for i, (c, ttl) in enumerate([(km, "K-평균 k=3"), (KB.time.clip(70, 73).values, "통전시간")]):
        a = ax[i, j]; a.grid(False); a.set_xticks([]); a.set_yticks([])
        a.scatter(*Y.T, c=c, s=3, cmap="tab10" if i == 0 else "viridis", rasterized=True); a.set_title(f"perplexity {key[0]} · {ttl}")
fig.tight_layout(); P.save(fig, "28_kamp_tsne_robust"); plt.close(fig)
print(f"전체 {time.time() - t0:.0f}s")
