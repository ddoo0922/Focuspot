from common import *
from scipy import stats
d=np.load(f"{OUT}/polito.npz",allow_pickle=True)
V,I,F=d["V"].astype(np.float64),d["I"].astype(np.float64),d["F"].astype(np.float64)
lab=pd.DataFrame({"cb":d["cb"],"sp":d["sp"],"date":pd.to_datetime(d["date"]),"y":d["y"]})
lab["year"]=lab.date.dt.year
def vlen(X):
    nz=X>0
    return np.where(nz.any(1), X.shape[1]-np.argmax(nz[:,::-1],axis=1),0)
LV=vlen(V); LF=vlen(F)
rows=[]
for i in range(len(V)):
    v,c,f=V[i],I[i],F[i]
    m=v>0.02
    dm=np.diff(np.r_[0,m.astype(int),0]); s=np.where(dm==1)[0]; e=np.where(dm==-1)[0]
    r=dict(idx=i,n_runs=len(s))
    if len(s)>=2:
        p1=(s[0],e[0]); p2=(s[-1],e[-1])
    else:
        p1=None; p2=(s[0],e[0])
    a,b=p2; r["D"]=b-a; r["start2"]=a; r["end2"]=b
    if p1: r["D1"]=p1[1]-p1[0]; r["gap"]=a-p1[1]
    vm,cm,fm=v[a:b],c[a:b],f[a:b]
    R=np.where(cm>0.02,vm/np.maximum(cm,1e-6),np.nan)
    r.update(V_mean=vm.mean(),V_max=vm.max(),V_std=vm.std(),I_mean=cm.mean(),I_max=cm.max(),I_std=cm.std())
    n=len(vm); k=max(int(n*0.2),5)
    r["V_late_early"]=vm[-k:].mean()/max(vm[:k].mean(),1e-9)
    r["I_late_early"]=cm[-k:].mean()/max(cm[:k].mean(),1e-9)
    r["V_slope"]=np.polyfit(np.arange(n),vm,1)[0]*n
    r["I_slope"]=np.polyfit(np.arange(n),cm,1)[0]*n
    r["energy"]=(vm*cm).sum()
    r["energy_norm"]=(vm*cm).mean()
    Rv=R[np.isfinite(R)]
    if len(Rv)>10:
        # 초기 과도 구간 제외(첫 10%)
        Rs=Rv[int(len(Rv)*0.1):]
        r.update(R_mean=np.nanmean(Rs),R_std=np.nanstd(Rs),R_min=np.nanmin(Rs),R_max=np.nanmax(Rs),
                 R_late_early=np.nanmean(Rs[-k:])/max(np.nanmean(Rs[:k]),1e-9),
                 R_argmin=np.nanargmin(Rs)/len(Rs))
        sm=np.convolve(Rs,np.ones(5)/5,mode="valid")
        dd=np.diff(sm)
        r["R_maxdrop"]=dd.min(); r["R_maxrise"]=dd.max()
        r["R_dip_depth"]=(np.nanmax(sm[:len(sm)//2])-np.nanmin(sm[len(sm)//2:])) if len(sm)>4 else np.nan
    if p1:
        v1,c1=v[p1[0]:p1[1]],c[p1[0]:p1[1]]
        r.update(V1_mean=v1.mean(),I1_mean=c1.mean(),V1_std=v1.std(),I1_std=c1.std(),E1=(v1*c1).sum())
    r["F_mean_main"]=fm.mean(); r["F_max"]=f.max(); r["F_len"]=LF[i]
    r["F_zero"]=int(f.max()==0)
    r["F_start"]=f[:20].mean(); r["F_main_std"]=fm.std()
    r["F_main_late_early"]=fm[-k:].mean()/max(fm[:k].mean(),1e-9) if fm[:k].mean()>0 else np.nan
    r["Ve_len"]=LV[i]
    rows.append(r)
ft=pd.DataFrame(rows)
df=pd.concat([lab.reset_index(drop=True),ft.drop(columns=["idx"])],axis=1)
df.to_csv(f"{RES}/polito_features.csv",index=False)
print(df.shape); print(df.describe().T[["count","mean","std","min","max"]].to_string())
