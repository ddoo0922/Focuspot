# %% [markdown]
# # 01. Stellantis N.V. 검사 추천 모델 — 차체 단위 7-fold OOF 점수
# 1차 탐색(Colab)에서 쓴 개별 정답 모델을 같은 분할로 다시 구현한다. 이후 실험의 상한(참고) 성능으로 쓴다.
#
# - 데이터: Stellantis N.V. RSW 1,976타점, 불량 79개, 차체 28대, 위치 158곳
# - 분할: 차체 단위 7-fold (같은 차체는 학습·평가에 섞이지 않음), fold 배정 시드 0~4 (5회)
# - 모델: 위치별 과거 불량률, AE(개별 정답 미사용), 로지스틱·CatBoost·ExtraTrees·TabICLv2(개별 정답 사용)
# - 특징: 기본 16개(전류·전압 통계) / 도메인 90개(본 용접 구간의 전류·전압·상대 저항·상대 전력 모양 지표)
#
# 결과는 `results/polito_oof.pkl`에 저장하고, 검사 예산 곡선은 02에서 그린다.

# %%
import time, warnings
import numpy as np, pandas as pd
import br
from br import features as F, models as M, budget as B
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
df = F.load()
print(df.shape, "불량", df.y.sum(), "차체", df.body.nunique(), "위치", df["loc"].nunique())

# %%
RUNS = [  # (표시 이름, 모델, 특징)
    ("위치별 과거 불량률", "loc_prior", []),
    ("AE·기본16 (정답 미사용)", "autoencoder", F.BASE),
    ("로지스틱·기본16", "logistic", F.BASE),
    ("CatBoost·기본16", "catboost", F.BASE),
    ("TabICLv2·기본16", "tabicl", F.BASE),
    ("로지스틱·도메인90", "logistic", F.DOMAIN),
    ("ExtraTrees·도메인90", "extratrees", F.DOMAIN),
    ("TabICLv2·도메인90", "tabicl", F.DOMAIN),
    ("CatBoost·도메인90", "catboost", F.DOMAIN),
]
SEEDS = range(5)
out = br.RES / "polito_oof.pkl"
br.RES.mkdir(exist_ok=True)
oof = pd.read_pickle(out) if out.exists() else pd.DataFrame()
done = set(map(tuple, oof[["label", "seed"]].drop_duplicates().values)) if len(oof) else set()
for label, name, feats in RUNS:
    for s in SEEDS:
        if (label, s) in done:
            continue
        t = time.time()
        o = M.run_oof(df, name, feats, s).assign(label=label, seed=s)
        oof = pd.concat([oof, o], ignore_index=True)
        n, h = B.hits_at(df.y.values, o.score.values, o.fold.values, 0.10)
        print(f"{label:24s} seed={s}  10%({n}개) 포착 {h:5.1f}  {time.time()-t:5.1f}초", flush=True)
    oof.to_pickle(out)
print("저장:", out, oof.shape)

# %% [markdown]
# ## 10% 검사(약 200개)에서 포착한 불량 수 — 시드별

# %%
y = df.y.values
tab = (oof.groupby(["label", "seed"])
          .apply(lambda g: B.hits_at(y, g.sort_values("row").score.values, g.sort_values("row").fold.values, 0.10)[1])
          .unstack())
tab["평균"] = tab.mean(1); tab["최소"] = tab.iloc[:, :5].min(1); tab["최대"] = tab.iloc[:, :5].max(1)
tab = tab.loc[[r[0] for r in RUNS]]
tab.round(1)
