"""날짜별 불량 합계만으로 학습하는 모델. fit()은 개별 정답을 받지 않는다(날짜 번호와 날짜별 합계만)."""
import numpy as np
import torch
from scipy.special import expit
from wl.models import WeakMLP


class AggLogistic:
    """기록별 로지스틱 점수의 날짜 합계를 Poisson 개수 손실로 날짜별 불량 수에 맞춘다."""

    def __init__(self, l2=1e-2, epochs=500, lr=5e-2): self.l2, self.ep, self.lr = l2, epochs, lr

    def fit(self, X, bag, cnt, seed):
        torch.manual_seed(seed)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        bt = torch.as_tensor(bag)
        ct = torch.as_tensor(cnt, dtype=torch.float32)
        r = max(cnt.sum() / len(X), 1e-4)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.tensor(float(np.log(r / (1 - r))), requires_grad=True)
        opt = torch.optim.Adam([w, b], self.lr)
        for _ in range(self.ep):
            opt.zero_grad()
            p = torch.sigmoid(Xt @ w + b)
            lam = torch.zeros(len(cnt)).index_add_(0, bt, p).clamp_min(1e-6)
            loss = (lam - ct * torch.log(lam)).mean() + self.l2 * (w ** 2).sum()
            loss.backward(); opt.step()
        self.w, self.b = w.detach().numpy(), float(b)
        return self

    def logit(self, X): return X @ self.w + self.b


class Weak:
    """WeakMLP(count = PVC 방식 Poisson-binomial 개수 손실, dllp = 비율 맞춤) 래퍼."""

    def __init__(self, loss): self.m = WeakMLP(loss=loss, device="cpu")

    def fit(self, X, bag, cnt, seed):
        self.m.fit(X, bag, cnt, None, None, None, seed); return self

    def logit(self, X): return self.m.predict(X, None)[1]


class ConstRate:
    def fit(self, X, bag, cnt, seed):
        r = cnt.sum() / len(X); self.z = np.log(r / (1 - r)); return self

    def logit(self, X): return np.full(len(X), self.z)


def make(name):
    return {"pvc": lambda: Weak("count"), "ratio": lambda: Weak("dllp"),
            "agglogit": AggLogistic, "const": ConstRate}[name]()


def shift_to(z, target):
    """sigmoid(z + s)의 합이 target이 되는 절편 s (순위는 그대로)."""
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expit(z + mid).sum() < target else (lo, mid)
    return (lo + hi) / 2


class MonoCount:
    """단조 제약 개수 모델: logit = b + softplus(w)·x (모든 가중치 ≥ 0, 입력이 클수록 위험이 커지거나 같다).
    손실은 PVC와 같은 Poisson-binomial 개수 손실. 방향은 학습이 아니라 사전지식(이탈 크기 ↑ = 위험 ↑)으로 고정한다."""

    def __init__(self, epochs=300, lr=5e-2, device="cpu"): self.ep, self.lr, self.dev = epochs, lr, device

    def fit(self, X, bag, cnt, seed):
        from wl.models import _train_weak, _t
        torch.manual_seed(seed)
        d = X.shape[1]
        self.w = torch.nn.Parameter(torch.randn(d, device=self.dev) * 0.1 - 2.0)
        r = max(cnt.sum() / len(X), 1e-4)
        self.b = torch.nn.Parameter(torch.tensor(float(np.log(r / (1 - r))), device=self.dev))
        f = lambda x: x @ torch.nn.functional.softplus(self.w) + self.b
        _train_weak(f, lambda: [self.w, self.b], _t(X, self.dev), bag, cnt, "count", self.ep, self.lr, 0.0, self.dev)
        return self

    def logit(self, X):
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
            return (Xt @ torch.nn.functional.softplus(self.w) + self.b).cpu().numpy()


class RulePrior:
    """D5: logit = b + softplus(a)·규칙 + softplus(w)·x. 규칙 = 입력 열의 합(이탈 크기 합). a ≥ 0이라 규칙 방향은 뒤집히지 않는다.
    w는 L2로 0 쪽으로 당겨 규칙에서 크게 벗어나지 않게 한다. 개수 손실(PVC와 같은 Poisson-binomial)로 학습한다."""

    def __init__(self, lam=1.0, epochs=300, lr=5e-2, device="cpu"): self.lam, self.ep, self.lr, self.dev = lam, epochs, lr, device

    def fit(self, X, bag, cnt, seed):
        from wl.models import _train_weak, _t
        torch.manual_seed(seed)
        d = X.shape[1]
        sp = torch.nn.functional.softplus
        self.a = torch.nn.Parameter(torch.tensor(0.5, device=self.dev))
        self.w = torch.nn.Parameter(torch.full((d,), -4.0, device=self.dev) + torch.randn(d, device=self.dev) * 0.1)
        r = max(cnt.sum() / len(X), 1e-4)
        self.b = torch.nn.Parameter(torch.tensor(float(np.log(r / (1 - r))), device=self.dev))
        Xt = _t(X, self.dev)
        self._f = lambda x: sp(self.a) * x.sum(1) / np.sqrt(d) + x @ sp(self.w) + self.b
        _train_weak(self._f, lambda: [self.a, self.w, self.b], Xt, bag, cnt, "count", self.ep, self.lr, 0.0, self.dev,
                    extra_loss=lambda: self.lam * (sp(self.w) ** 2).sum())
        return self

    def logit(self, X):
        with torch.no_grad():
            return self._f(torch.as_tensor(X, dtype=torch.float32, device=self.dev)).cpu().numpy()
