import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, os, json
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "..", "..", "학습용 데이터")
OUT = HERE
FIG = f"{OUT}/figures"
RES = f"{OUT}/results"
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic" if "Malgun Gothic" in {f.name for f in fm.fontManager.ttflist} else "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 130
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

# 색: 정상=회청, 불량=주황(강조), 보조=청록/보라
C_OK, C_BAD, C_A, C_B, C_GR = "#5B7C99", "#E8743B", "#2A9D8F", "#8E6BBF", "#9AA5B1"

def save(fig, name):
    p = f"{FIG}/{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p

def dump(name, obj):
    with open(f"{RES}/{name}.json", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

def quality_index(df, key_cols=None, valid_rules=None, name=""):
    """수상작(Absolute A·Family AI Kamp)이 쓴 KDATA 5축 품질지수 방식.
    완전성=결측 없음 / 유일성=중복 없음 / 유효성=범위 만족 / 일관성=타입 일관 / 정확성(무결성)=위 지수 100% 여부"""
    n = len(df); cells = df.size
    comp = 100 * (1 - df.isna().sum().sum() / cells) if cells else np.nan
    dup = df.duplicated().sum()
    uniq = 100 * (1 - dup / n) if n else np.nan
    keydup = int(df.duplicated(subset=key_cols).sum()) if key_cols else None
    valid_n, valid_tot = 0, 0
    if valid_rules:
        for c, (lo, hi) in valid_rules.items():
            s = df[c].dropna(); valid_tot += len(s)
            valid_n += ((s >= lo) & (s <= hi)).sum()
    valid = 100 * valid_n / valid_tot if valid_tot else np.nan
    return dict(name=name, rows=n, cols=df.shape[1], 완전성=round(comp, 3), 유일성=round(uniq, 3),
                중복행=int(dup), 키중복=keydup, 유효성=round(valid, 3) if valid == valid else None)
