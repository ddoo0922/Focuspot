from common import *
raw=pd.read_pickle("kamp_raw.pkl"); raw["d"]=raw.date.dt.strftime("%m-%d"); K=json.load(open(f"{RES}/kamp.json"))
print((raw.force>4).sum()/4, ((raw.force>3)&(raw.force<=4.07)).sum()/4)
days=sorted(raw.d.unique())
fig,ax=plt.subplots(2,2,figsize=(14,9))
a=ax[0,0]; n=raw.groupby("d").size(); nb=raw[raw.force>3].groupby("d").size().reindex(days).fillna(0)
a.bar(days,n.reindex(days)-nb,color=C_OK,label="정상 범위 타점"); a.bar(days,nb,bottom=n.reindex(days)-nb,color=C_BAD,label="복사된 고가압력 블록(291행)")
for i,dv in enumerate(K["daily"]): a.text(i,n.reindex(days).iloc[i]+30,f"불량 {int(dv['defect'])}건" if dv["date"]!="03-27" else "불량 기록 없음",ha="center",fontsize=8)
a.set_title("① 9일치 11,939타점, 일별 불량 0~8건(약 0.33%)",loc="left",fontsize=11,fontweight="bold"); a.legend(loc="upper right",fontsize=9); a.set_ylabel("타점 수"); a.set_ylim(0,2350)
a=ax[0,1]
for d in days:
    g=raw[raw.d==d]; a.plot(g.idx.values,g.force.values,lw=0.4,color=C_BAD if (g.force>3).any() else C_GR,alpha=0.8)
a.set_yscale("log"); a.set_title("② 가압력: 4일에 동일한 291행 블록이 복사돼 있다 (나머지는 1.74~2.94 bar)",loc="left",fontsize=11,fontweight="bold"); a.set_xlabel("생산순번(일 내)"); a.set_ylabel("가압력 (bar, log)")
core=raw[raw.force<=3]
a=ax[1,0]; labs=["가압력(bar)","전류(kA)","전압(V)","통전시간"]; cols=["force","current","voltage","wtime"]
C=core[cols].corr(method="spearman").values; im=a.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1); a.grid(False); a.set_xticks(range(4)); a.set_xticklabels(labs,fontsize=9); a.set_yticks(range(4)); a.set_yticklabels(labs,fontsize=9)
for i in range(4):
    for j in range(4): a.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",color="white" if abs(C[i,j])>0.5 else "black",fontweight="bold")
a.set_title("③ 블록 제외 상관: 가압력↔전류 −0.55만 뚜렷",loc="left",fontsize=11,fontweight="bold")
a=ax[1,1]
for c,col,lab in [("current",C_A,"전류"),("wtime",C_B,"통전시간")]:
    pass
d_moms=core.groupby("d").current.mean(); a.plot(range(len(days)),d_moms.reindex(days).values,"o-",color=C_A,label="전류 일평균(kA)"); a.set_ylabel("전류 일평균 (kA)"); a2=a.twinx(); a2.plot(range(len(days)),core.groupby("d").force.std().reindex(days).values,"s--",color=C_BAD,label="가압력 일 표준편차"); a2.set_ylabel("가압력 표준편차 (bar)"); a2.grid(False)
a.set_xticks(range(len(days))); a.set_xticklabels(days); a.set_title("④ 일별 평균은 거의 변하지 않는다 (전류 ±0.05kA)",loc="left",fontsize=11,fontweight="bold")
fig.suptitle("KAMP 제공 용접 데이터(참조 기준) — 설비·품목·두께 고정, 행 단위 라벨 없음",fontsize=13,fontweight="bold",y=1.0); fig.tight_layout(); save(fig,"kamp_01_overview")
