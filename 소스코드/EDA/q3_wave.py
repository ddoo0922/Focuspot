from common import *
from sklearn.metrics import roc_auc_score
s=pd.read_csv(f"{RES}/queretaro_samples.csv",index_col=0); rows=pd.read_csv(f"{RES}/queretaro_rows_clean.csv")
p=s[(~s.DOE)&s.Imean.notna()].copy(); rows=rows[rows.sid.isin(p.index)]
grid=np.linspace(0,1,20); W={}
for sid,g in rows.groupby("sid"):
    x=g.I.values
    if len(x)<4: continue
    W[sid]=np.interp(grid,np.linspace(0,1,len(x)),x/x.mean())
Wd=pd.DataFrame(W).T; Wd["cat"]=p.cat.reindex(Wd.index); Wd["t"]=p.t.reindex(Wd.index); Wd["psi"]=p.psi.reindex(Wd.index)
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
a=ax[0]
for c,col in [("Good",C_OK),("Explode",C_B),("Bad",C_BAD)]:
    m=Wd[Wd.cat==c][list(range(20))].values if False else Wd[Wd.cat==c].drop(columns=["cat","t","psi"]).values
    if len(m)==0: continue
    q=np.percentile(m,[25,50,75],axis=0); a.fill_between(grid,q[0],q[2],color=col,alpha=0.18); a.plot(grid,q[1],color=col,lw=2,label=f"{c} (n={len(m)})")
a.set_xlabel("통전 구간 (0=시작, 1=끝)"); a.set_ylabel("전류 ÷ 타점 평균 전류"); a.legend(); a.set_title("생산형 실험 B: 날림(Explode)은 전류가 초반에 정점을 찍고 후반에 처진다",loc="left",fontsize=10.5,fontweight="bold")
# 통전 800~1000ms 만 (같은 길이 조건)
a=ax[1]
for c,col in [("Good",C_OK),("Explode",C_B)]:
    m=Wd[(Wd.cat==c)&(Wd.t.isin([600,800]))].drop(columns=["cat","t","psi"]).values
    q=np.percentile(m,[25,50,75],axis=0); a.fill_between(grid,q[0],q[2],color=col,alpha=0.18); a.plot(grid,q[1],color=col,lw=2,label=f"{c} (n={len(m)})")
a.set_xlabel("통전 구간"); a.legend(); a.set_title("같은 통전 시간(600·800ms)만 비교해도 같은 경향",loc="left",fontsize=10.5,fontweight="bold")
fig.tight_layout(); save(fig,"quer_06_waveform")
# 요약값 저장
y=(p.cat=="Explode").astype(int); out={}
rows2=rows.groupby("sid")
p["Ipeak_pos"]=rows2.I.apply(lambda x: np.argmax(x.values)/max(len(x)-1,1)); p["Ilast_over_mean"]=p.Ilast/p.Imean
for c in ["Imean","Ipeak_pos","Ilast_over_mean","Idrop"]:
    z=p.groupby(["psi","t"])[c].transform(lambda x:(x-x.mean())/(x.std()+1e-9)); m=z.notna()
    out[c]=dict(auc_raw=float(roc_auc_score(y,p[c].fillna(p[c].median()))),auc_within_cell=float(roc_auc_score(y[m],z[m])))
dump("queretaro_wave",out); print(out)
