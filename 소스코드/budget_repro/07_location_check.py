# %% [markdown]
# # 07. 같은 모델(06)의 Stellantis N.V. 포착은 위치 차이에서 오는가
# 06 점수를 그대로 쓴다(모델 변경 없음). 위치 번호는 모델 입력이 아니다. 하지만 위치마다 전류·전압 수준이 달라 모델이 간접적으로 위치를 구분할 수 있다.
# 1. 위치 73·84(불량 25개)를 빼고 다시 채점
# 2. 점수를 위치 평균 부분과 위치 안 편차 부분으로 나눠, 각각만으로 순위를 매긴다(위치 평균은 같은 위치 다른 날짜 점수로 계산하며 정답은 쓰지 않는다)
# 3. 위치마다 같은 비율을 검사하는 정책
#
# 위치 효과에서 온 포착도 "위험 타점 추천"에는 쓸모가 있다. 다만 공정조건이 불량을 만든다는 해석과 KAMP로의 일반화에는 영향을 준다.

# %%
import numpy as np, pandas as pd
import br
from br import spec as S, budget as B
from wl.metrics import inclusion_prob
from wl import data as wldata
pd.set_option("display.width", 220)
X, day, cnt, y, _ = S.load_polito()
P0 = wldata.load_polito(); body, loc = P0.body.values, P0["loc"].values
sc = pd.read_pickle(br.RES / "same_model_scores.pkl")
pol = sc[sc.data == "polito"].sort_values(["model", "seed", "row"])
rng = np.random.default_rng(1); ub = np.unique(body)
Wb = np.stack([np.bincount(d_, minlength=len(ub)) for d_ in rng.integers(0, len(ub), (5000, len(ub)))])[:, np.searchsorted(ub, body)]
def ci(a): return f"{np.mean(a):+.3f} [{np.quantile(a, .025):+.3f}, {np.quantile(a, .975):+.3f}]"

def loc_parts(s):
    """위치 평균 = 같은 위치·다른 날짜 점수 평균(없으면 전체 평균), 위치 안 편차 = 점수 − 위치 평균."""
    df = pd.DataFrame({"s": s, "l": loc, "d": day})
    tot = df.groupby("l").s.agg(["sum", "count"])
    same = df.groupby(["l", "d"]).s.agg(["sum", "count"])
    a = tot.loc[df.l].values - same.loc[list(zip(df.l, df.d))].values
    lm = np.where(a[:, 1] > 0, a[:, 0] / np.maximum(a[:, 1], 1), s.mean())
    return lm, s - lm

def eval_policy(scores_by_seed, mask, b, grp_fn=lambda m: day[m], frac=False):
    """mask 행만으로 그룹별 상위 b 검사 → 포착률과 무작위 대비 차이 구간."""
    g = grp_fn(mask); ug, gi = np.unique(g, return_inverse=True); n = np.bincount(gi)
    k = b * n if frac else np.ceil(b * n - 1e-9)
    ps = np.mean([inclusion_prob(s[mask], k, gi) for s in scores_by_seed], 0)
    pr = (k / n)[gi]
    W, yy = Wb[:, mask], y[mask]
    r = (W @ (ps * yy)) / (W @ yy); r0 = (W @ (pr * yy)) / (W @ yy)
    return (ps * yy).sum() / yy.sum(), ci(r - r0)

# %%
rows = []
hot = np.isin(loc, [73, 84]); allm = np.ones(len(y), bool)
for m in ["PVC MLP", "비율맞춤 MLP"]:
    S_ = [g.score.values for _, g in pol[pol.model == m].groupby("seed")]
    parts = [loc_parts(s) for s in S_]
    for b in (0.10, 0.20):
        for nm, ss, mask, kw in [
            ("원래 점수 · 전체", S_, allm, {}),
            ("원래 점수 · 73·84 제외", S_, ~hot, {}),
            ("위치 평균 부분만", [p[0] for p in parts], allm, {}),
            ("위치 안 편차 부분만", [p[1] for p in parts], allm, {}),
            ("위치별 동일 비율 정책", S_, allm, dict(grp_fn=lambda m_: np.char.add(day[m_].astype(str), "_" + loc[m_].astype(str)), frac=True)),
        ]:
            r, d = eval_policy(ss, mask, b, **kw)
            rows.append({"모델": m, "예산": f"{int(b*100)}%", "평가": nm, "포착률": round(r, 3), "무작위 대비 [95%]": d})
res = pd.DataFrame(rows); res.to_csv(br.RES / "same_model_location_check.csv", index=False, encoding="utf-8-sig")
res
