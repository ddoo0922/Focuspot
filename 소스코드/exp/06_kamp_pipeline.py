# %% [markdown]
# # 06. KAMP 2단계 파이프라인: 날짜별 예상 불량 개수 + 타점별 검사 우선순위
# 두 모델을 따로 쓴다.
# - **개수 모델(1단계)**: C(물리규칙 사전지식, 개수 손실). 입력은 블록 중립화한 `kamp_full` 피처. 04의 일 단위 leave-one-day-out에서 평균 오차가 가장 낮았다.
# - **선별 모델(2단계)**: 오토인코더 이탈 점수. 입력은 원시 요약값(`kamp_full`). 07에서 사전 규칙(같은 위치 안 AUROC 4조건 평균 최고)으로 순서 피처·결합 입력보다 선택했다(0.548 vs 0.535 vs 0.509, 구간은 겹침).
# - 합치기: 그날 예상 개수 N의 m배(m=5, 10)와 타점 수 비율(5·10·20%)만큼 상위 타점을 검사 후보로 표시한다.
# - 복사 블록(가압력 > 3 bar) 행은 순위에서 빼고 `품질확인` 으로 따로 표시한다(1장 전처리 원칙).
#
# **검증 한계(보고서에 그대로 적을 것)**: 2단계 점수는 KAMP에 타점 정답이 없어 확인할 수 없고, Stellantis N.V. 같은 위치 안 검증에서도 무작위와 구분되지 않았다(05).
# 1단계는 고정 불량률 대비 개선이 통계적으로 확인되지 않았다(04, 8일).

# %%
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from scipy.special import expit
from wl import data, kamp, models, runner
pd.set_option("display.width", 220)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
K, cnt_day = kamp.load_kamp()
days = sorted(K.date.unique())
labeled = [d for d in days if not np.isnan(cnt_day[d])]
print("라벨 있는 날:", len(labeled), "| 라벨 없는 날:", [str(d)[:10] for d in days if d not in labeled])

# %% [markdown]
# ## 2. 1단계 개수 모델 학습 (라벨 있는 8일 전체) + 3/27 예측

# %%
Xn = pd.DataFrame(kamp.make_X(K, neutralize=True), columns=kamp.FEATS)   # 블록 행은 비블록 중앙값으로 중립화

def std_fit(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    return (X - mu) / sd

tr = K.date.isin(labeled).values
Xs = std_fit(Xn.values, tr)
ub, bag_tr = np.unique(K.date.values[tr], return_inverse=True)
cnt_tr = np.array([cnt_day[u] for u in ub]).astype(int)
ps = []
for seed in range(5):
    m = models.make_model("C_prior_two", dev)
    m.fit(Xs[tr], bag_tr, cnt_tr, None, np.zeros(tr.sum(), int), kamp.FEATS, seed)
    _, z_all = m.predict(Xs, None)
    _, z_tr = m.predict(Xs[tr], None)
    ps.append(expit(z_all + runner._calibrate_shift(z_tr, cnt_tr.sum())))
K["p_count"] = np.mean(ps, axis=0)
daily = pd.DataFrame({"타점수": K.groupby("date").size(),
                      "실제불량": cnt_day,
                      "C모델 예상": K.groupby("date").p_count.sum().round(2),
                      "고정불량률 예상": (cnt_tr.sum() / tr.sum() * K.groupby("date").size()).round(2)})
daily.index = [str(d)[:10] for d in daily.index]
print("※ 라벨 있는 날의 C모델 값은 학습에 쓰인 날이라 표본 내 값이다(표본 외 성능은 04의 leave-one-day-out 참고).")
daily

# %% [markdown]
# ## 3. 2단계 선별 모델: 원시 요약값 오토인코더 이탈 점수
# 학습에 타점 정답을 쓰지 않는다. 블록 행은 학습에서 제외하고 점수도 매기지 않는다.

# %%
Xraw = K[kamp.FEATS].values.astype(float)
ok = ~K.block.values
Z = (Xraw - Xraw[ok].mean(0)) / (Xraw[ok].std(0) + 1e-9)
scores = []
for seed in range(5):
    ae = models.AutoEncoder(dev)
    ae.fit(Z[ok], None, None, None, None, kamp.FEATS, seed)
    sc, _ = ae.predict(Z, None)
    scores.append((sc - sc[ok].mean()) / sc[ok].std())
K["risk"] = np.mean(scores, axis=0)
K.loc[K.block, "risk"] = np.nan

# %% [markdown]
# ## 4. 파이프라인 결합: 날짜별 상위 타점 선택 → 우선순위

# %%
K["rank_in_day"] = K[~K.block].groupby("date").risk.rank(ascending=False, method="first")
n_ok = K[~K.block].groupby("date").size()
K["pct_in_day"] = K.rank_in_day / K.date.map(n_ok)
for m in (5, 10):
    k = np.ceil(m * K.groupby("date").p_count.transform("sum"))
    K[f"검사후보_{m}x"] = (K.rank_in_day <= k)
for b in (0.05, 0.10, 0.20):
    K[f"검사후보_{int(b*100)}%"] = (K.pct_in_day <= b)
K["상태"] = np.where(K.block, "품질확인(복사 블록)", "순위 부여")
cols = ["date", "idx", "상태", "p_count", "risk", "rank_in_day", "pct_in_day"] + [c for c in K.columns if c.startswith("검사후보")] + kamp.FEATS
out = K[cols].rename(columns={"p_count": "개수모델 확률", "risk": "위험점수", "rank_in_day": "일내순위", "pct_in_day": "일내순위비율"})
(runner.RES / "kamp_priority.csv").parent.mkdir(exist_ok=True)
out.to_csv(runner.RES / "kamp_priority.csv", index=False, encoding="utf-8-sig")
print("저장:", runner.RES / "kamp_priority.csv", out.shape)
summ = K.groupby("date").agg(타점수=("idx", "size"), 블록=("block", "sum"), **{"5x후보수": ("검사후보_5x", "sum"), "10x후보수": ("검사후보_10x", "sum")})
summ.index = [str(d)[:10] for d in summ.index]
summ

# %% [markdown]
# ## 5. 3/27 검사 우선순위 상위 20

# %%
t = K[(K.date == pd.Timestamp("2020-03-27")) & ~K.block].sort_values("rank_in_day").head(20)
t[["idx", "rank_in_day", "risk", "p_count"] + kamp.FEATS].round(3)
