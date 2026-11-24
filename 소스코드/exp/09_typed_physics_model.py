# %% [markdown]
# # 09. 유형별 물리 방향 고정 모델 (개발 모델) — KAMP 일 단위 검증
# 지금까지 버렸던 정보: 날짜 × 유형(파임·용접부족·크랙)별 불량 개수(24칸).
# 유형 k마다 logit_k = b_k + Σ_j s_kj · w_kj · z_j, 부호 s는 1장 표 1로 고정하고 크기 w ≥ 0만 개수 손실로 학습(유형당 3개 + 절편).
# 비교: 부호를 고정하지 않은 같은 구조(`free`), 유형별 고정 불량률.
# 평가: 하루를 빼고 학습해 그날 유형별 개수 예측(leave-one-day-out, 8일). Stellantis N.V.는 유형 구분이 없어 이 모델을 검증할 수 없다.

# %%
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from scipy.special import expit
from wl import data, kamp, runner
from wl.models import pb_logpmf_masked, _bag_structs
pd.set_option("display.width", 220)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
K, _ = kamp.load_kamp()
res = pd.read_excel(data.KAMP_XLSX, sheet_name="result")
T = res.pivot_table(index=res["working time"].dt.normalize(), columns="defect type", values="defect", aggfunc="sum").fillna(0).astype(int)
labeled = list(T.index)
COLS = ["cur_mean", "force_mean", "weld_len"]
SIGNS = np.array([[+1, +1, +1],    # 파임: +전류 +가압 +시간
                  [-1, +1, -1],    # 용접부족: -전류 +가압(과가압) -시간
                  [+1, -1, +1]])   # 크랙: +전류 -가압 +시간
TYPES = ["파임", "용접부족", "크랙"]
Xn = pd.DataFrame(kamp.make_X(K, neutralize=True), columns=kamp.FEATS)[COLS].values
dates = K.date.values

# %%
def fit_typed(Z, bag, C3, constrained, seed, epochs=400, lr=0.05, lam=0.01):
    torch.manual_seed(seed)
    Zt = torch.tensor(Z, dtype=torch.float32, device=dev)
    mat, msk, _ = _bag_structs(bag, dev)
    r = C3.sum(0) / len(Z)
    b = torch.tensor(np.log(r / (1 - r)), dtype=torch.float32, device=dev, requires_grad=True)
    u = torch.full((3, 3), -2.0 if constrained else 0.0, device=dev, requires_grad=True)
    S = torch.tensor(SIGNS, dtype=torch.float32, device=dev)
    C = torch.tensor(C3, dtype=torch.long, device=dev)
    W = (lambda: S * F.softplus(u)) if constrained else (lambda: u)
    opt = torch.optim.Adam([b, u], lr)
    for _ in range(epochs):
        opt.zero_grad()
        P = torch.sigmoid(b + Zt @ W().T)
        L = 0
        for k in range(3):
            Kk = max(int(C3[:, k].max()), 1)
            L = L - pb_logpmf_masked(P[:, k], mat, msk, Kk).gather(1, C[:, k:k + 1]).mean()
        (L + lam * (W() ** 2).sum()).backward(); opt.step()
    return b.detach().cpu().numpy(), W().detach().cpu().numpy()

def predict_counts(Ztr, bag_tr, C3, Zte, constrained, seeds=3):
    out = []
    for s in range(seeds):
        b, W = fit_typed(Ztr, bag_tr, C3, constrained, s)
        ztr, zte = b + Ztr @ W.T, b + Zte @ W.T
        out.append([expit(zte[:, k] + runner._calibrate_shift(ztr[:, k], C3[:, k].sum())).sum() for k in range(3)])
    return np.mean(out, 0), W

rows = []
for d in labeled:
    te = dates == d; tr = np.isin(dates, labeled) & ~te
    mu, sd = Xn[tr].mean(0), Xn[tr].std(0) + 1e-9
    Ztr, Zte = (Xn[tr] - mu) / sd, (Xn[te] - mu) / sd
    ub, bag_tr = np.unique(dates[tr], return_inverse=True)
    C3 = T.loc[ub].values
    fixed = C3.sum(0) / tr.sum() * te.sum()
    pc, _ = predict_counts(Ztr, bag_tr, C3, Zte, True)
    pf, _ = predict_counts(Ztr, bag_tr, C3, Zte, False)
    act = T.loc[d].values
    for k in range(3):
        rows.append({"날짜": str(d)[:10], "유형": TYPES[k], "실제": act[k], "고정": fixed[k], "부호고정": pc[k], "자유": pf[k]})
R = pd.DataFrame(rows)

# %%
print("유형별 MAE (8일 × 유형)")
display(R.groupby("유형")[["고정", "부호고정", "자유"]].apply(lambda g: (g.sub(R.loc[g.index, "실제"], axis=0)).abs().mean()).round(3))
tot = R.groupby("날짜")[["실제", "고정", "부호고정", "자유"]].sum()
print("합계 개수 MAE:", {c: round(float((tot[c] - tot["실제"]).abs().mean()), 3) for c in ["고정", "부호고정", "자유"]})
print("참고: 04의 합계 MAE — 고정 불량률 2.19, C 모델 1.55~1.70")
tot.round(2)

# %% [markdown]
# ## 8일 전체로 학습한 부호고정 모델의 가중치 (해석)

# %%
tr = np.isin(dates, labeled)
mu, sd = Xn[tr].mean(0), Xn[tr].std(0) + 1e-9
ub, bag_tr = np.unique(dates[tr], return_inverse=True)
_, W = predict_counts((Xn[tr] - mu) / sd, bag_tr, T.loc[ub].values, (Xn[tr] - mu) / sd, True)
pd.DataFrame(W, index=TYPES, columns=COLS).round(3)
