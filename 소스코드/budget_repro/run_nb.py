"""노트북 실행: python run_nb.py 06_same_model
.py(jupytext)를 .ipynb로 바꿔 처음부터 끝까지 실행하고 출력과 그림을 저장한다.
다른 폴더의 노트북은 경로로 지정한다: python run_nb.py ../exp/04_kamp_daily"""
import json, subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
here = Path(__file__).resolve().parent
target = (here / sys.argv[1]).with_suffix(".py").resolve()
cwd, name = target.parent, target.stem
py = sys.executable
subprocess.run([py, "-m", "jupytext", "--to", "ipynb", f"{name}.py", "-q"], cwd=cwd, check=True)
r = subprocess.run([py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", f"{name}.ipynb"],
                   cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode:
    print(r.stderr[-4000:]); sys.exit(1)
for c in json.loads((cwd / f"{name}.ipynb").read_text(encoding="utf-8"))["cells"]:
    for o in c.get("outputs", []):
        if "text" in o:
            print("".join(o["text"]))
        elif "data" in o:
            d = o["data"]
            print("[그림]" if "image/png" in d else "".join(d.get("text/plain", "")))
