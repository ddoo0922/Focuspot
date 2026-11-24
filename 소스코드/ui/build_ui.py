"""검사 우선순위 모니터(시연용 화면) 만들기.
KAMP 예측결과(results/final_kamp_recommendations_combined.csv)를 읽어 ui_template.html에 넣고 한 파일짜리 HTML로 저장한다.
KAMP에는 타점별 라벨이 없으므로 화면의 불량 위치는 가상 라벨이다: 날짜별 불량 개수는 실제 값을 쓰고(3/27은 예측 개수),
어느 타점인지는 Stellantis N.V. 일자 단위 교차검증의 예산별 포착률(err_budget_table.csv)이 그대로 나온다고 가정해 배치한다(난수 없음, 결정적).
부품 형상과 타점 위치(가이드북의 4지점에 맞춰 생산 순서대로 타점 4개를 부품 1개로 묶어 부품 번호와 타점 1~4를 배정)도 시연을 위한 가상 값이다. 안정 공정범위는 그날 타점의 가운데 95% 구간이다.
실행: python build_ui.py
"""
import json, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; R = HERE.parent / "budget_repro" / "results"; out = HERE / "점용접_검사우선순위_모니터.html"
d = pd.read_csv(f"{R}/final_kamp_recommendations_combined.csv", encoding="utf-8-sig")
plan = pd.read_csv(f"{R}/kamp_daily_inspection_plan.csv", encoding="utf-8-sig", index_col=0)
sus = pd.read_csv(f"{R}/kamp_recommendations_suspect_type.csv", encoding="utf-8-sig")
bud = pd.read_csv(f"{R}/err_budget_table.csv", encoding="utf-8-sig")
d = d.rename(columns={"weld current(kA)": "c", "weld Voltage(v)": "v", "weld time(ms)": "t", "weld force(bar)": "f"})
d["blk"] = d["상태"] != "순위 부여"; d["h"] = d.v * d.c * d.t
print(d["상태"].value_counts().to_dict())
# 검증 포착률 곡선 (Stellantis N.V. 시간순 홀드아웃, 표 4-1 과 같은 값) : 예산 b% -> 포착률
xs = [0] + list(bud["예산%"] / 100) + [1.0]; ys = [0] + list(bud["포착률%"] / 100) + [1.0]
print("curve", list(zip(xs, ys)))
days = sorted(d.date.unique()); cnt = {}
for day in days:
    a = plan.loc[day, "실제 불량 수"]; cnt[day] = int(a) if a == a else int(plan.loc[day, "의사라벨 A"])
N = sum(cnt.values()); u = (np.arange(N) + 0.5) / N
q = np.interp(u, ys, xs)            # 각 가상 불량의 '일내 위험 순위 상위 비율'
# 정렬된 순위를 날짜에 차례로 나눠 줌 (날짜별 개수는 실제 값 유지)
slots = [day for day in days for _ in range(cnt[day])]
order = []
left = dict(cnt)
while sum(left.values()):
    for day in days:
        if left[day]: order.append(day); left[day] -= 1
assign = {day: [] for day in days}
for day, qq in zip(order, q): assign[day].append(float(qq))
OUT = {"days": [], "curve": {"b": [5, 10, 20, 30], "rate": [float(bud.set_index("예산%").loc[b, "포착률%"]) for b in (5, 10, 20, 30)]}}
tot = {10: 0, 20: 0, 30: 0}
for day in days:
    g = d[d.date == day].sort_values("idx").reset_index(drop=True); e = g[~g.blk]
    band = {}
    for k in ["c", "v", "t", "f", "h"]:
        med = float(e[k].median()); mad = float((e[k] - med).abs().median()) * 1.4826
        sc = mad if mad > 0 else float(e[k].std())
        band[k] = dict(med=med, sc=sc, lo=float(e[k].quantile(.025)), hi=float(e[k].quantile(.975)), min=float(e[k].min()), max=float(e[k].max()), madzero=mad == 0)
    rk = e["결합점수"].rank(ascending=False, method="first"); m = len(e)
    sim = []
    for qq in assign[day]:
        r = min(m, max(1, int(np.ceil(qq * m)))); pos = int(rk[rk == r].index[0])
        while pos in sim: r = r + 1 if r < m else r - 1; pos = int(rk[rk == r].index[0])
        sim.append(pos)
    for b in tot: tot[b] += sum(1 for p in sim if rk[p] <= np.ceil(b / 100 * m))
    sg = sus[sus.date == day]; smap = {int(r.idx): r["주 의심 유형(입열 방향)"] for r in sg.itertuples(index=False) if False}
    smap = {int(i): (t if isinstance(t, str) else "") for i, t in zip(sg["idx"], sg["주 의심 유형(입열 방향)"])}
    OUT["days"].append(dict(date=day, real=None if plan.loc[day, "실제 불량 수"] != plan.loc[day, "실제 불량 수"] else int(plan.loc[day, "실제 불량 수"]), nsim=cnt[day],
        idx=g.idx.astype(int).tolist(), s=[None if b else round(float(x), 4) for x, b in zip(g["결합점수"], g.blk)],
        c=g.c.round(3).tolist(), v=g.v.round(3).tolist(), t=g.t.astype(int).tolist(), f=g.f.round(3).tolist(), band=band, sim=sorted(sim), sus=smap))
    anyout = np.zeros(len(e), bool)
    for k in band: anyout |= ((e[k] < band[k]["lo"]) | (e[k] > band[k]["hi"])).values
    m30 = int(np.ceil(.3 * m)); h30 = sum(1 for p_ in sim if rk[p_] <= m30)
    print("  이탈", int(anyout.sum()), "/", len(e), round(100 * anyout.mean(), 1), "% | 30% 포착", h30, "미탐지", cnt[day] - h30)
    print(day, len(g), int(g.blk.sum()), cnt[day], {k: (round(b_["lo"], 3), round(b_["hi"], 3), b_["madzero"]) for k, b_ in band.items()})
print("총 가상 불량", N, "예산별 포착(전체일 기준)", tot, {b: round(100 * v / N, 1) for b, v in tot.items()})
html = (HERE / "ui_template.html").read_text(encoding="utf-8").replace("/*DATA*/", json.dumps(OUT, ensure_ascii=False, separators=(",", ":")))
out.write_text(html, encoding="utf-8"); print("저장:", out, len(html))
