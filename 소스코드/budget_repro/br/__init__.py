"""검사 예산 재현 실험. 데이터 로더·동점 처리·개수 손실은 ../exp/wl 을 재사용한다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "exp"))
RES = ROOT / "results"
FIG = ROOT / "figures"
