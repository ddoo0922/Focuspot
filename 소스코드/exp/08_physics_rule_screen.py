# %% [markdown]
# # 08. 유형별 물리 규칙 스크린 (학습 없음, 사전 지정 3개)
# 1장 표 1의 방향을 그대로 부호로 고정한다. 입력은 위치별 표준화 z(라벨 무관, 위치 평균·전체 표준편차).
# - 파임(압흔 과다): +전류 +가압력 +통전시간
# - 용접부족(너깃 미달): −전류 −통전시간 +가압력(과가압)
# - 크랙(응고균열): +전류 −가압력 +통전시간
# Stellantis N.V.는 불량 유형을 모르므로 세 규칙 중 어느 하나라도 맞는지만 본다. **검정 3개**, 다중비교 때문에 구간 하한이 0.5를 확실히 넘을 때만 신호로 인정한다.
# 규칙은 학습하지 않으므로 라벨을 쓰지 않는다(위치 평균은 라벨 무관).

# %%
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from wl import data, runner
pd.set_option("display.width", 200)
df = data.load_polito()
cols = ["cur_mean", "weld_len", "force_mean"]
X = df[cols].copy(); X["force_mean"] = X.force_mean.fillna(X.force_mean.median())
R = X - X.groupby(df["loc"]).transform("mean")
Z = R / R.std()
rules = {
    "파임형(+전류+가압+시간)":      Z.cur_mean + Z.force_mean + Z.weld_len,
    "용접부족형(-전류-시간+가압)":  -Z.cur_mean - Z.weld_len + Z.force_mean,
    "크랙형(+전류-가압+시간)":      Z.cur_mean - Z.force_mean + Z.weld_len,
}
def mk(score):
    return dict(model="rule", cond="orig", seed=0, y=df.y.values, loc=df["loc"].values, body=df.body.values,
                bag=np.zeros(len(df), int), p=np.full(len(df), df.y.mean()), score=np.asarray(score, float))
rows = []
for name, sc in rules.items():
    b = runner.bootstrap([mk(sc)], n_boot=400)
    rows.append({"규칙": name, **{k: f"{b.loc[k,'mean']:.3f} [{b.loc[k,'lo']:.3f}, {b.loc[k,'hi']:.3f}]" for k in ["wl_auroc", "wl_recall@0.1", "wl_recall@0.2"]}})
print("같은 위치 안 지표 (무작위: AUROC 0.5, 포착률 = 예산 비율)")
pd.DataFrame(rows)

# %% [markdown]
# ## 개별 변수 방향 (참고, 사후 탐색이므로 신호로 인정하지 않음)

# %%
rows = []
for c in cols:
    b = runner.bootstrap([mk(Z[c])], n_boot=200)
    rows.append({"변수": c, "위치안 AUROC(+방향)": f"{b.loc['wl_auroc','mean']:.3f} [{b.loc['wl_auroc','lo']:.3f}, {b.loc['wl_auroc','hi']:.3f}]"})
pd.DataFrame(rows)
