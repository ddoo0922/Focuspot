"""묶음(하루) 조건 구성: 원본 / KAMP형(묶음 수 축소, 불량률 솎기)과 차체 단위 폴드."""
import numpy as np

# 설계서 6절 평가조건 3가지 + 중간 단계(k8_4: 묶음만 줄임)
CONDITIONS = {
    "orig":   dict(n_groups=None, rate=None),
    "k8_4":   dict(n_groups=8, rate=None),
    "k8_1":   dict(n_groups=8, rate=0.01),
    "k8_05":  dict(n_groups=8, rate=0.005),
}


def date_bag_ids(dates):
    u, inv = np.unique(dates, return_inverse=True)
    return inv


def merge_chronological(bag, n_groups):
    """날짜 묶음을 시간순으로 이어 붙여 타점 수가 비슷한 n_groups개로 합친다."""
    sizes = np.bincount(bag)
    cum = np.cumsum(sizes) - sizes / 2.0
    grp_of_bag = np.minimum((cum / sizes.sum() * n_groups).astype(int), n_groups - 1)
    return grp_of_bag[bag]


def thin_defects(y, rate, rng):
    """불량 타점 행을 무작위로 제거해 불량률을 rate에 맞춘다. 유지할 행 마스크 반환."""
    n_norm = int((y == 0).sum())
    n_keep = int(round(rate * n_norm / (1 - rate)))
    pos = np.flatnonzero(y == 1)
    keep_pos = rng.choice(pos, size=min(n_keep, len(pos)), replace=False)
    keep = (y == 0)
    keep[keep_pos] = True
    return keep


def build_condition(meta, cond, seed):
    """(kept_idx, bag) — kept_idx는 meta의 행 위치, bag은 같은 길이의 묶음 번호(0..B-1)."""
    cfg = CONDITIONS[cond]
    rng = np.random.default_rng(seed)
    y = meta["y"].values
    keep = np.ones(len(y), bool) if cfg["rate"] is None else thin_defects(y, cfg["rate"], rng)
    bag = date_bag_ids(meta["date"].values)
    if cfg["n_groups"] is not None:
        bag = merge_chronological(bag, cfg["n_groups"])
    idx = np.flatnonzero(keep)
    _, b = np.unique(bag[idx], return_inverse=True)
    return idx, b


def body_folds(body, k, seed):
    """차체 단위 K-fold: 같은 차체는 같은 폴드. 반환: 행별 폴드 번호."""
    rng = np.random.default_rng(seed)
    ub = np.unique(body)
    rng.shuffle(ub)
    fold_of = {b: i % k for i, b in enumerate(ub)}
    return np.array([fold_of[b] for b in body])
