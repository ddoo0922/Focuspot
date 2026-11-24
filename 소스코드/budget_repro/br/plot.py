"""그림 공통 설정. 범주 색은 고정 순서(파랑·주황·청록·노랑·자홍·초록·보라·빨강), 무작위 기준은 회색 점선."""
import matplotlib as mpl
import matplotlib.pyplot as plt

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e6e5e0"


def setup():
    mpl.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 110, "savefig.dpi": 160,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False, "font.size": 10,
    })


def save(fig, name):
    from . import FIG
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
