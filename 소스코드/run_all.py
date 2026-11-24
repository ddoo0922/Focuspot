"""전체 실행: python run_all.py [시작 번호]
2차 실험(budget_repro) 노트북을 번호 순서대로 실행한다. 1차 실험(exp)은 --exp 를 붙이면 먼저 실행한다.
예) python run_all.py 17   → 17번부터 실행 (앞 번호의 결과 파일이 results/ 에 있어야 함)"""
import subprocess, sys
from pathlib import Path

here = Path(__file__).resolve().parent
BR = here / "budget_repro"
args = [a for a in sys.argv[1:] if a != "--exp"]
start = int(args[0]) if args else 0

steps = []
if "--exp" in sys.argv:
    steps += [f"../exp/{p.stem}" for p in sorted((here / "exp").glob("[0-9][0-9]_*.py"))]
steps += [p.stem for p in sorted(BR.glob("[0-9][0-9]_*.py")) if int(p.stem[:2]) >= start]

for s in steps:
    print(f"\n===== {s} =====", flush=True)
    r = subprocess.run([sys.executable, "run_nb.py", s], cwd=BR)
    if r.returncode:
        print(f"[중단] {s} 실행 실패"); sys.exit(1)
print("\n완료")
