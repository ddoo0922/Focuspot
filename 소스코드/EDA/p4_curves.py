from common import *
df=pd.read_csv(f"{RES}/polito_features.csv",parse_dates=["date"])
d=np.load(f"{OUT}/polito.npz",allow_pickle=True)
V,I,F=d["V"],d["I"],d["F"]
y=df.y.values; sp=df.sp.values
# ---- 그림 4: 대표 곡선 (정상 3 + 불량 3) 3채널, 2단계 표시
rng=np.random.default_rng(3)
ok=rng.choice(np.where((y==0)&(df.F_zero==0))[0],4,replace=False); bd=rng.choice(np.where((y==1)&(df.sp.isin([84,73])==False)&(df.F_zero==0))[0],4,replace=False)
fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
for a,(X,nm) in zip(ax,[(V,"전압 V (0~1 정규화)"),(I,"전류 I (0~1 정규화)"),(F,"가압력 F (0~1 정규화)")]):
    for i in ok: a.plot(X[i],color=C_OK,alpha=0.8,lw=1.2)
    for i in bd: a.plot(X[i],color=C_BAD,alpha=0.9,lw=1.2)
    a.axvspan(0,100,color="#EEF2F6",zorder=0); a.set_ylabel(nm)
ax[0].plot([],[],color=C_OK,label="정상(4개)"); ax[0].plot([],[],color=C_BAD,label="불량(4개, 84·73번 위치 제외)"); ax[0].legend(loc="upper right")
ax[0].text(50,ax[0].get_ylim()[1]*0.92,"① 예열\n펄스",ha="center",fontsize=9); ax[0].text(300,ax[0].get_ylim()[1]*0.92,"② 본 용접(길이는 위치마다 다름)",fontsize=9)
ax[2].set_xlabel("시간 스텝 (0~1000, 뒤는 0으로 채움)")
fig.suptitle("시계열은 '예열 펄스(≈100스텝) → 10스텝 무신호 → 본 용접' 2단계 구조",fontsize=13,fontweight="bold",y=0.995); fig.tight_layout(); save(fig,"polito_04_examples")
# ---- 본 용접 구간을 200점으로 리샘플
def resample_main(X,n=200):
    out=np.zeros((len(X),n))
    for i in range(len(X)):
        a,b=int(df.start2[i]),int(df.end2[i]); seg=X[i,a:b]
        out[i]=np.interp(np.linspace(0,len(seg)-1,n),np.arange(len(seg)),seg)
    return out
Vr,Ir,Fr=resample_main(V),resample_main(I),resample_main(F)
np.savez_compressed(f"{OUT}/polito_resampled.npz",Vr=Vr,Ir=Ir,Fr=Fr)
Rr=np.where(Ir>0.02,Vr/np.maximum(Ir,1e-6),np.nan)
# ---- 그림 5: 평균 곡선 정상 vs 불량 (전체 / 84·73 제외 / 84·73만)
fig,ax=plt.subplots(2,4,figsize=(17,7),sharex=True)
t=np.linspace(0,1,200)
groups=[("전체",np.ones(len(df),bool)),("84·73 제외",~df.sp.isin([84,73]).values)]
for r,(gn,mask) in enumerate(groups):
    for c,(X,nm) in enumerate([(Vr,"전압"),(Ir,"전류"),(Rr,"동저항 V/I(상대값)"),(Fr,"가압력")]):
        a=ax[r,c]
        for lab_,col,m in [("정상",C_OK,(y==0)&mask),("불량",C_BAD,(y==1)&mask)]:
            if X is Fr: m=m&(df.F_zero.values==0)
            q=np.nanpercentile(X[m],[25,50,75],axis=0)
            a.fill_between(t,q[0],q[2],color=col,alpha=0.22); a.plot(t,q[1],color=col,lw=2,label=f"{lab_} (n={m.sum()})")
        a.set_title(f"{gn} · {nm}",fontsize=10,fontweight="bold",loc="left"); a.legend(fontsize=8)
        if r==1: a.set_xlabel("본 용접 구간 (0=시작, 1=끝)")
fig.suptitle("평균 곡선(중앙값+사분위 밴드): 불량과 정상이 육안으로는 거의 겹친다",fontsize=13,fontweight="bold",y=1.0); fig.tight_layout(); save(fig,"polito_05_mean_curves")
print("done")
