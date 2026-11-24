"""후보 모델. 약지도 모델은 fit()에서 타점 라벨 y를 받지 않고 묶음 번호와 묶음별 개수만 받는다."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


# ---------------------------------------------------------------- 개수 손실(Poisson-binomial 정확 확률)
def pb_logpmf(p_pad, K):
    """p_pad: (B, N) 불량 확률(패딩은 마스크로 제외), 반환 (B, K+1) = log P(묶음 내 불량 수 = k).
    로그 공간 이진 트리 합성곱 — 수치적으로 안정적이고 GPU에서 한 번에 계산."""
    raise NotImplementedError  # 아래 pb_logpmf_masked 사용


def pb_logpmf_masked(p, bag_mat, mask, K):
    """p: (n,) 확률, bag_mat: (B, N) 각 묶음의 타점 인덱스(패딩 0), mask: (B, N) 유효 여부."""
    pm = p[bag_mat].clamp(1e-6, 1 - 1e-6)
    B, N = pm.shape
    A = torch.full((B, N, K + 1), NEG, device=p.device, dtype=p.dtype)
    A[..., 0] = torch.where(mask, torch.log1p(-pm), torch.zeros_like(pm))
    if K >= 1:
        A[..., 1] = torch.where(mask, torch.log(pm), torch.full_like(pm, NEG))
    n2 = 1
    while n2 < N:
        n2 *= 2
    if n2 > N:
        pad = torch.full((B, n2 - N, K + 1), NEG, device=p.device, dtype=p.dtype)
        pad[..., 0] = 0.0
        A = torch.cat([A, pad], 1)
    J = torch.arange(K + 1, device=p.device)
    idx = J[:, None] - J[None, :]
    valid = idx >= 0
    idx = idx.clamp(min=0)
    while A.shape[1] > 1:
        X, Y = A[:, 0::2], A[:, 1::2]
        Ys = torch.where(valid, Y[..., idx], torch.full_like(Y[..., idx], NEG))
        A = torch.logsumexp(X[..., None, :] + Ys, dim=-1)
    return A[:, 0]


# ---------------------------------------------------------------- 공통 유틸
def _t(x, dev, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)


def _bag_structs(bag, dev):
    B = int(bag.max()) + 1
    sizes = np.bincount(bag, minlength=B)
    N = int(sizes.max())
    mat = np.zeros((B, N), int)
    msk = np.zeros((B, N), bool)
    for b in range(B):
        ii = np.flatnonzero(bag == b)
        mat[b, :len(ii)] = ii
        msk[b, :len(ii)] = True
    return torch.as_tensor(mat, device=dev), torch.as_tensor(msk, device=dev), sizes


def _mlp(d, h, depth=2):
    layers, i = [], d
    for _ in range(depth):
        layers += [nn.Linear(i, h), nn.ReLU()]
        i = h
    layers.append(nn.Linear(i, 1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------- 기준 모델
class ConstRate:
    name, needs_strong = "const_rate", False

    def fit(self, X, bag, cnt, y, loc, feats, seed): return self
    def predict(self, X, loc): return np.zeros(len(X)), None


class LocRate:
    """위치별 과거 불량률(타점 라벨 집계 사용) — 파형을 쓰지 않는 단순 기준."""
    name, needs_strong = "loc_rate", True

    def __init__(self, strength=2.0): self.a = strength

    def fit(self, X, bag, cnt, y, loc, feats, seed):
        r0 = y.mean()
        n = {l: 0 for l in np.unique(loc)}; k = dict(n)
        for l, v in zip(loc, y): n[l] += 1; k[l] += v
        self.rate = {l: (k[l] + self.a * r0) / (n[l] + self.a) for l in n}
        self.r0 = r0
        return self

    def predict(self, X, loc):
        r = np.array([self.rate.get(l, self.r0) for l in loc])
        return r, np.log(r / (1 - r))


class SeqRule:
    """순서 피처의 이탈 크기 규칙 점수: 표준화한 |dev|·|dl|의 평균. 학습 없음(표준화 통계만 사용)."""
    name, needs_strong = "seq_rule", False

    def fit(self, X, bag, cnt, y, loc, feats, seed): return self
    def predict(self, X, loc): return np.abs(X).mean(1), None


class IsoForest:
    name, needs_strong = "iforest", False

    def fit(self, X, bag, cnt, y, loc, feats, seed):
        from sklearn.ensemble import IsolationForest
        self.m = IsolationForest(n_estimators=300, random_state=seed).fit(X)
        return self

    def predict(self, X, loc): return -self.m.score_samples(X), None


class AutoEncoder:
    name, needs_strong = "autoencoder", False

    def __init__(self, device="cpu", hidden=8, epochs=300): self.dev, self.h, self.ep = device, hidden, epochs

    def fit(self, X, bag, cnt, y, loc, feats, seed):
        torch.manual_seed(seed)
        d = X.shape[1]
        self.net = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, self.h), nn.ReLU(),
                                 nn.Linear(self.h, 16), nn.ReLU(), nn.Linear(16, d)).to(self.dev)
        Xt = _t(X, self.dev)
        opt = torch.optim.Adam(self.net.parameters(), 1e-2, weight_decay=1e-4)
        for _ in range(self.ep):
            opt.zero_grad(); F.mse_loss(self.net(Xt), Xt).backward(); opt.step()
        return self

    def predict(self, X, loc):
        Xt = _t(X, self.dev)
        with torch.no_grad():
            return ((self.net(Xt) - Xt) ** 2).mean(1).cpu().numpy(), None


# ---------------------------------------------------------------- 약지도 모델 (A: 비율손실, B: 개수손실, C: 개수손실+물리 사전지식)
def _prior_score(Xs, feats, kind):
    ix = [feats.index(c) for c in ("heat", "dynres", "i_over_f") if c in feats]
    if not ix:   # 물리 파생 변수가 없는 입력(순서 피처 등)은 전체 피처의 이탈 크기를 쓴다
        return np.abs(Xs).mean(1)
    z = Xs[:, ix]
    if kind == "two":
        return np.abs(z).mean(1)
    if kind == "over":
        return z[:, 0]
    if kind == "under":
        return -z[:, 0]
    raise ValueError(kind)


def _train_weak(net_fn, params_fn, Xt, bag, cnt, loss, epochs, lr, wd, dev, extra_loss=None):
    mat, msk, sizes = _bag_structs(bag, dev)
    cnt_t = _t(cnt, dev, torch.long)
    sizes_t = _t(sizes, dev)
    K = int(cnt.max())
    bag_t = torch.as_tensor(bag, device=dev)
    opt = torch.optim.Adam(params_fn(), lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        logit = net_fn(Xt)
        p = torch.sigmoid(logit)
        if loss == "dllp":
            m = torch.zeros(len(sizes), device=dev).index_add_(0, bag_t, p) / sizes_t
            L = F.binary_cross_entropy(m.clamp(1e-6, 1 - 1e-6), (cnt_t / sizes_t))
        else:
            lp = pb_logpmf_masked(p, mat, msk, K)
            L = -lp.gather(1, cnt_t[:, None]).mean()
        if extra_loss is not None:
            L = L + extra_loss()
        L.backward(); opt.step()


class WeakMLP:
    needs_strong = False

    def __init__(self, loss="count", device="cpu", hidden=32, epochs=300, lr=1e-2, wd=1e-3, depth=2):
        self.loss, self.dev, self.h, self.ep, self.lr, self.wd, self.depth = loss, device, hidden, epochs, lr, wd, depth
        self.name = {"dllp": "A_dllp_mlp", "count": "B_count_mlp"}[loss]

    def fit(self, X, bag, cnt, y, loc, feats, seed):
        torch.manual_seed(seed)
        self.net = _mlp(X.shape[1], self.h, self.depth).to(self.dev)
        r = max(cnt.sum() / len(X), 1e-4)
        with torch.no_grad():
            self.net[-1].bias.fill_(float(np.log(r / (1 - r))))
        _train_weak(lambda x: self.net(x).squeeze(-1), lambda: self.net.parameters(),
                    _t(X, self.dev), bag, cnt, self.loss, self.ep, self.lr, self.wd, self.dev)
        return self

    def predict(self, X, loc):
        with torch.no_grad():
            z = self.net(_t(X, self.dev)).squeeze(-1).cpu().numpy()
        return z, z


class PriorLogit:
    """C: logit = b + a*prior(x) + w·x, 개수 손실로 학습. w는 0으로 당기는 L2(사전지식 중심)."""
    needs_strong = False

    def __init__(self, prior="two", lam=1.0, device="cpu", epochs=300, lr=5e-2, use_x=True):
        self.prior, self.lam, self.dev, self.ep, self.lr, self.use_x = prior, lam, device, epochs, lr, use_x
        self.name = f"C_prior_{prior}" + ("" if use_x else "_only")

    def _logit(self, Xt, pr):
        z = self.b + self.a * pr
        return z + Xt @ self.w if self.use_x else z

    def fit(self, X, bag, cnt, y, loc, feats, seed):
        torch.manual_seed(seed)
        self.feats = feats
        Xt = _t(X, self.dev)
        pr = _t(_prior_score(X, feats, self.prior), self.dev)
        r = max(cnt.sum() / len(X), 1e-4)
        self.b = torch.tensor(float(np.log(r / (1 - r))), device=self.dev, requires_grad=True)
        self.a = torch.zeros((), device=self.dev, requires_grad=True)
        self.w = torch.zeros(X.shape[1], device=self.dev, requires_grad=True)
        _train_weak(lambda x: self._logit(x, pr), lambda: [self.b, self.a, self.w],
                    Xt, bag, cnt, "count", self.ep, self.lr, 0.0, self.dev,
                    extra_loss=lambda: self.lam * (self.w ** 2).sum() / len(X) * 100)
        return self

    def predict(self, X, loc):
        pr = _t(_prior_score(X, self.feats, self.prior), self.dev)
        with torch.no_grad():
            z = self._logit(_t(X, self.dev), pr).cpu().numpy()
        return z, z


class PseudoLabel:
    """D: 묶음 개수 제약을 만족하는 의사확률 → 가중 지도학습을 반복(EM형). Havaldar 외 BP의 단순화 버전."""
    needs_strong = False
    name = "D_pseudo_logreg"

    def __init__(self, rounds=6, C=0.1, prior="two"): self.rounds, self.C, self.prior = rounds, C, prior

    @staticmethod
    def _bag_shift(s, bag, cnt):
        q = np.zeros_like(s)
        for b in range(len(cnt)):
            m = bag == b
            lo, hi = -30.0, 30.0
            for _ in range(50):
                mid = (lo + hi) / 2
                if (1 / (1 + np.exp(-(s[m] + mid)))).sum() < cnt[b]: lo = mid
                else: hi = mid
            q[m] = 1 / (1 + np.exp(-(s[m] + (lo + hi) / 2)))
        return q

    def fit(self, X, bag, cnt, y, loc, feats, seed):
        from sklearn.linear_model import LogisticRegression
        s = _prior_score(X, feats, self.prior)
        s = (s - s.mean()) / (s.std() + 1e-9)
        for _ in range(self.rounds):
            q = self._bag_shift(s, bag, cnt)
            Xd = np.vstack([X, X]); yd = np.r_[np.ones(len(X)), np.zeros(len(X))]
            wd = np.r_[q, 1 - q]
            self.m = LogisticRegression(C=self.C, max_iter=500).fit(Xd, yd, sample_weight=wd)
            s = self.m.decision_function(X)
        return self

    def predict(self, X, loc):
        z = self.m.decision_function(X)
        return z, z


# ---------------------------------------------------------------- 상한 모델 (타점 정답 사용)
class CatBoostUB:
    name, needs_strong = "UB_catboost", True

    def __init__(self, iters=300, depth=4): self.iters, self.depth = iters, depth

    def fit(self, X, bag, cnt, y, loc, feats, seed):
        from catboost import CatBoostClassifier
        self.m = CatBoostClassifier(iterations=self.iters, depth=self.depth, learning_rate=0.05,
                                    random_seed=seed, verbose=0, thread_count=4).fit(X, y)
        return self

    def predict(self, X, loc):
        p = self.m.predict_proba(X)[:, 1].clip(1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        return z, z


class LogregUB:
    name, needs_strong = "UB_logreg", True

    def fit(self, X, bag, cnt, y, loc, feats, seed):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=0.3, max_iter=1000).fit(X, y)
        return self

    def predict(self, X, loc):
        z = self.m.decision_function(X)
        return z, z


def make_model(name, device="cpu"):
    return {
        "const_rate": lambda: ConstRate(),
        "loc_rate": lambda: LocRate(),
        "iforest": lambda: IsoForest(),
        "seq_rule": lambda: SeqRule(),
        "autoencoder": lambda: AutoEncoder(device),
        "A_dllp_mlp": lambda: WeakMLP("dllp", device),
        "B_count_mlp": lambda: WeakMLP("count", device),
        "C_prior_two": lambda: PriorLogit("two", device=device),
        "C_prior_over": lambda: PriorLogit("over", device=device),
        "C_prior_under": lambda: PriorLogit("under", device=device),
        "C_prior_two_only": lambda: PriorLogit("two", device=device, use_x=False),
        "D_pseudo_logreg": lambda: PseudoLabel(),
        "UB_catboost": lambda: CatBoostUB(),
        "UB_logreg": lambda: LogregUB(),
    }[name]()
