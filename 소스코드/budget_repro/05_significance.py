# %% [markdown]
# # 05. 결과가 우연인가 — 차체 단위 부트스트랩과 같은 위치 안 순열검정
# - 검사 정책(점수 상위 ceil(b·n))은 고정하고, 각 기록이 뽑힐 확률을 5시드 평균한 뒤 **차체를 복원추출**해 포착률의 불확실성을 본다.
#   같은 부트스트랩 표본에서 두 모델의 차이를 계산한다(대응 비교).
# - 같은 위치 안 판별력: 같은 fold·위치 그룹 안에서 점수를 섞는 순열검정. 위치 간 차이를 빼고도 신호가 남는지 본다.
# - 시간순 시험(04): 시험 차체 9대를 복원추출.

# %%
import numpy as np, pandas as pd
import br
from br import features as F, budget as B
from wl.metrics import inclusion_prob, within_loc_auroc
pd.set_option("display.width", 220)
df = F.load(); y = df.y.values; loc = df["loc"].values; body = df.body.values
oof = pd.read_pickle(br.RES / "polito_oof.pkl").sort_values(["label", "seed", "row"])
NB = 5000; rng = np.random.default_rng(0)
ub = np.unique(body)
draws = rng.integers(0, len(ub), (NB, len(ub)))
W = np.stack([np.bincount(d_, minlength=len(ub)) for d_ in draws])[:, np.searchsorted(ub, body)]  # (NB, n) 행 가중치

def incl(lab, b):
    """5시드 평균 선택 확률."""
    ps = []
    for s, g in oof[oof.label == lab].groupby("seed"):
        ug, k = B.pick_counts(g.fold.values, b)
        ps.append(inclusion_prob(g.score.values, k.astype(float), np.searchsorted(ug, g.fold.values)))
    return np.mean(ps, 0)

def boot_recall(p):
    return (W @ (p * y)) / (W @ y)

def ci(a): return f"{np.mean(a):.3f} [{np.quantile(a, .025):.3f}, {np.quantile(a, .975):.3f}]"

# %% [markdown]
# ## 1. 10%·20% 검사 포착률의 95% 구간, 그리고 모델 간 차이

# %%
LABS = ["위치별 과거 불량률", "CatBoost·기본16", "TabICLv2·기본16", "CatBoost·도메인90", "TabICLv2·도메인90"]
PAIRS = [("CatBoost·도메인90", "위치별 과거 불량률"), ("TabICLv2·도메인90", "위치별 과거 불량률"),
         ("CatBoost·도메인90", "CatBoost·기본16"), ("TabICLv2·도메인90", "TabICLv2·기본16"),
         ("TabICLv2·도메인90", "CatBoost·도메인90")]
out1, out2 = [], []
for b in (0.10, 0.20):
    R = {lab: boot_recall(incl(lab, b)) for lab in LABS}
    for lab in LABS:
        out1.append({"예산": f"{int(b*100)}%", "모델": lab, "포착률 [95% 구간]": ci(R[lab])})
    for a, c in PAIRS:
        d_ = R[a] - R[c]
        out2.append({"예산": f"{int(b*100)}%", "비교": f"{a} − {c}", "차이 [95% 구간]": ci(d_), "차이≤0 비율": f"{(d_ <= 0).mean():.3f}"})
display(pd.DataFrame(out1).pivot(index="모델", columns="예산", values="포착률 [95% 구간]").loc[LABS])
t2 = pd.DataFrame(out2); t2.to_csv(br.RES / "significance_pairs.csv", index=False, encoding="utf-8-sig"); t2

# %% [markdown]
# ## 2. 위치 73·84를 뺀 나머지 불량(54개)에서도 위치 기준보다 나은가 (10%)

# %%
hot = np.isin(loc, [73, 84])
def incl_sub(lab, b, m):
    ps = []
    for s, g in oof[oof.label == lab].groupby("seed"):
        f = g.fold.values[m]; ug, k = B.pick_counts(f, b)
        ps.append(inclusion_prob(g.score.values[m], k.astype(float), np.searchsorted(ug, f)))
    return np.mean(ps, 0)
Wm, ym = W[:, ~hot], y[~hot]
R = {lab: (Wm @ (incl_sub(lab, .1, ~hot) * ym)) / (Wm @ ym) for lab in ["위치별 과거 불량률", "CatBoost·도메인90", "TabICLv2·도메인90"]}
for lab, r in R.items(): print(f"{lab:20s} {ci(r)}")
for a in ["CatBoost·도메인90", "TabICLv2·도메인90"]:
    d_ = R[a] - R["위치별 과거 불량률"]; print(f"{a} − 위치 기준: {ci(d_)}, 차이≤0 비율 {(d_<=0).mean():.3f}")

# %% [markdown]
# ## 3. 같은 fold·위치 안 판별력 — 순열검정 (시드별)
# 정상·불량이 함께 있는 fold×위치 그룹에서만 계산된다. 귀무가설: 그룹 안 점수 순서가 무작위.

# %%
NP_ = 2000
rows = []
for lab in ["위치별 과거 불량률", "CatBoost·기본16", "CatBoost·도메인90", "TabICLv2·도메인90"]:
    for s, g in oof[oof.label == lab].groupby("seed"):
        grp = np.char.add(g.fold.values.astype(str), "_" + loc.astype(str))
        sc = g.score.values
        obs = within_loc_auroc(y, sc, grp)
        gi = pd.Series(grp).groupby(grp).indices
        mixed = [ix for ix in gi.values() if 0 < y[ix].sum() < len(ix)]
        npair = sum(int(y[ix].sum() * (len(ix) - y[ix].sum())) for ix in mixed)
        null = []
        for _ in range(NP_):
            sp = sc.copy()
            for ix in mixed: sp[ix] = sc[rng.permutation(ix)]
            null.append(within_loc_auroc(y, sp, grp))
        null = np.array(null)
        rows.append(dict(모델=lab, seed=s, AUROC=obs, p값=(np.sum(null >= obs) + 1) / (NP_ + 1),
                         불량_포함그룹=len(mixed), 비교쌍=npair))
perm = pd.DataFrame(rows); perm.to_csv(br.RES / "significance_withinloc.csv", index=False, encoding="utf-8-sig")
perm.groupby("모델")[["AUROC", "p값", "불량_포함그룹", "비교쌍"]].agg(["mean", "min", "max"]).round(3)

# %% [markdown]
# ## 4. 시간순 시험(2024년 5일, 불량 51개) — 무작위보다 나은가 (10%)
# 시험 차체 9대를 복원추출. 무작위 기대 포착률은 날짜별 ceil 때문에 약 11%.

# %%
pr = pd.read_pickle(br.RES / "daily_aggregate_preds.pkl")
te = np.sort(pr.row.unique()); yte = y[te]; dte = df.date.dt.strftime("%Y-%m-%d").values[te]; bte = body[te]
ubt = np.unique(bte); dr = rng.integers(0, len(ubt), (NB, len(ubt)))
Wt = np.stack([np.bincount(x, minlength=len(ubt)) for x in dr])[:, np.searchsorted(ubt, bte)]
ug, k = B.pick_counts(dte, .10); gi = np.searchsorted(ug, dte)
p_rand = (k / np.bincount(gi))[gi]
r_rand = (Wt @ (p_rand * yte)) / (Wt @ yte)
rows = []
for (proto, lab), g in pr.groupby(["proto", "label"]):
    if lab == "일정 불량률": continue
    ps = [inclusion_prob(gg.set_index("row").loc[te].p.values, k.astype(float), gi) for _, gg in g.groupby("seed")]
    r = (Wt @ (np.mean(ps, 0) * yte)) / (Wt @ yte)
    d_ = r - r_rand
    rows.append({"모델": f"{lab} [{proto}]", "포착률 [95%]": ci(r), "무작위 대비 차이 [95%]": ci(d_), "차이≤0 비율": f"{(d_<=0).mean():.3f}"})
print("무작위 기대 포착률", ci(r_rand))
pd.DataFrame(rows)
