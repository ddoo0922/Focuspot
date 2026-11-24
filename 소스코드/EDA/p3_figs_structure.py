from common import *
df=pd.read_csv(f"{RES}/polito_features.csv",parse_dates=["date"])
S=json.load(open(f"{RES}/polito_structure.json"))
# ---------- 그림 1: 데이터 구조 개요 (4패널)
fig,ax=plt.subplots(2,2,figsize=(13,8))
a=ax[0,0]
df["ym"]=df.date.dt.to_period("M").astype(str)
g=df.groupby("ym").agg(n=("y","size"),bad=("y","sum"))
a.bar(g.index,g.n-g.bad,color=C_OK,label="정상"); a.bar(g.index,g.bad,bottom=g.n-g.bad,color=C_BAD,label="불량")
for i,(k,r) in enumerate(g.iterrows()): a.text(i,r.n+15,f"{int(r.bad)}",ha="center",fontsize=8,color=C_BAD)
a.set_title("① 수집 시기: 2022년은 소량, 2023-06·2024-01~03에 집중",loc="left",fontsize=11,fontweight="bold"); a.set_ylabel("타점 수"); a.legend(); a.tick_params(axis="x",rotation=45)
a=ax[0,1]
cb=pd.DataFrame(S["cb_table"]); cc=[C_OK if n>=100 else C_GR for n in cb.n]
a.bar(cb.cb.astype(str),cb.n,color=cc); a.bar(cb.cb.astype(str),cb.bad,color=C_BAD)
a.axhline(100,ls="--",color="k",lw=0.8); a.text(27.5,104,"100타점",fontsize=8,ha="right")
a.set_title("② 차체 28대 중 14대만 전 타점(≈120~138) 검사, 나머지는 일부만",loc="left",fontsize=11,fontweight="bold"); a.set_xlabel("차체 번호"); a.set_ylabel("타점 수(주황=불량)"); a.tick_params(axis="x",labelsize=7)
a=ax[1,0]
sc=df.groupby("sp").size()
a.hist(sc,bins=range(0,sc.max()+3),color=C_OK,edgecolor="white")
a.set_title(f"③ 타점 위치 158곳당 표본 수: 중앙값 {int(sc.median())}개 (위치별 개별 모델은 표본 부족)",loc="left",fontsize=11,fontweight="bold"); a.set_xlabel("위치당 표본 수"); a.set_ylabel("위치 수")
a=ax[1,1]
dd=df.groupby("date").agg(n=("y","size"),bad=("y","sum"))
a.scatter(dd.n,dd.bad,s=45,color=C_BAD,edgecolor="k",lw=0.4)
a.set_title("④ 날짜별 묶음: 묶음 크기 1~268, 불량 0~19 (KAMP 일 묶음과 유사)",loc="left",fontsize=11,fontweight="bold"); a.set_xlabel("그날 검사된 타점 수"); a.set_ylabel("그날 불량 수")
for k,r in dd.iterrows():
    if r.bad>=8 or (r.n<10 and r.bad>=5): a.annotate(str(k.date()),(r.n,r.bad),fontsize=7,xytext=(3,3),textcoords="offset points")
fig.suptitle("Stellantis N.V. 데이터 구조 한눈에 보기",fontsize=14,fontweight="bold",y=1.01); fig.tight_layout(); save(fig,"polito_01_structure")
# ---------- 그림 2: 불량 위치 편중
fig,ax=plt.subplots(1,3,figsize=(15,4.8),gridspec_kw={"width_ratios":[1.1,1.1,1.6]})
spot=df.groupby("sp").agg(n=("y","size"),bad=("y","sum")).sort_values("bad",ascending=False)
a=ax[0]; cum=np.r_[0,np.cumsum(spot.bad.values)/spot.bad.sum()]; x=np.arange(len(cum))/len(spot)
a.plot(x,cum,color=C_BAD,lw=2); a.plot([0,1],[0,1],"--",color=C_GR)
a.axvline(2/158,color=C_GR,lw=0.6); a.text(0.03,0.33,"위치 2곳(1.3%)이\n불량 32% 차지",fontsize=9)
a.set_xlabel("위치 누적 비율(불량 많은 순)"); a.set_ylabel("불량 누적 비율"); a.set_title(f"편중도: 지니 {S['labels']['gini_bad_over_spots']:.2f}",loc="left",fontsize=11,fontweight="bold")
a=ax[1]; t=spot.head(12).reset_index(); a.barh(t.sp.astype(str)[::-1],(t.bad/t.n)[::-1]*100,color=[C_BAD if s in (84,73) else C_OK for s in t.sp[::-1]])
for i,(n,b) in enumerate(zip(t.n[::-1],t.bad[::-1])): a.text(1,i,f"{b}/{n}",va="center",fontsize=8,color="white" if b/n>0.15 else "k")
a.set_xlabel("위치별 불량률(%)"); a.set_ylabel("타점 위치 번호"); a.set_title("불량률 상위 12곳 (표기: 불량/표본)",loc="left",fontsize=11,fontweight="bold")
a=ax[2]
M=np.full((df.cb.nunique(),158),np.nan)
for _,r in df.drop_duplicates(["cb","sp"],keep="last").iterrows(): M[int(r.cb),int(r.sp)]=r.y
# 위치를 불량 많은 순으로 재배열
order=spot.index.values; M2=M[:,order]
from matplotlib.colors import ListedColormap
cm=ListedColormap(["#E9EDF2",C_BAD]); 
mm=np.ma.masked_invalid(M2); a.imshow(np.where(np.isnan(M2),-1,M2),aspect="auto",cmap=ListedColormap(["#FFFFFF","#DDE3EA",C_BAD]),vmin=-1,vmax=1,interpolation="nearest")
a.set_xlabel("타점 위치 (불량 많은 순으로 정렬, 158곳)"); a.set_ylabel("차체 번호"); a.grid(False)
a.set_title("차체×위치 불량 지도 (주황=불량, 회색=정상, 흰색=미검사)",loc="left",fontsize=11,fontweight="bold")
fig.suptitle("불량은 소수 위치에 몰려 있다 — 위치 번호만 알아도 불량을 어느 정도 맞힌다(교란)",fontsize=13,fontweight="bold",y=1.03); fig.tight_layout(); save(fig,"polito_02_label_concentration")
# ---------- 그림 3: 라벨 무작위 재배치(순열) 검정
fig,ax=plt.subplots(figsize=(6.5,4))
rng=np.random.default_rng(1); y=df.y.values; sp=df.sp.values
top2=[]
for _ in range(3000):
    s=pd.Series(rng.permutation(y)).groupby(sp).sum().sort_values(ascending=False); top2.append(s.head(2).sum()/s.sum()*100)
ax.hist(top2,bins=40,color=C_GR); ax.axvline(31.6,color=C_BAD,lw=2); ax.text(31.0,ax.get_ylim()[1]*0.85,"실제 31.6%",color=C_BAD,ha="right",fontweight="bold")
ax.set_xlabel("불량 상위 2개 위치가 차지하는 비율(%)"); ax.set_ylabel("빈도")
ax.set_title("불량이 위치와 무관하게 무작위라면 상위 2곳 몫은 평균 7.6%\n→ 실제 31.6%는 우연이 아님(p<0.001)",loc="left",fontsize=10.5,fontweight="bold"); save(fig,"polito_03_permutation")
print("ok")
