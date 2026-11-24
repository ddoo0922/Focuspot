from common import *
from scipy import stats
from sklearn.linear_model import LogisticRegression
d=pd.read_csv(f"{B}/02_Queretaro_RSW/data/Queretaro_RSW_v3/Data_RSW.csv")
d=d.rename(columns={"Sample ID":"sid","Pressure (PSI)":"psi","Welding Time (ms)":"t","Angle (Deg)":"ang","Force (N)":"F","Current (A)":"I","Thickness A (mm)":"thA","Thickness B (mm)":"thB","PullTest (N)":"pull","NuggetDiameter (mm)":"nug","Category":"cat","Comments":"cm"})
bad_mask=(d.F==-99)|(d.I==-99)
R={}
R["shape"]=d.shape; R["n_samples"]=int(d.sid.nunique()); R["sentinel_rows"]=int(bad_mask.sum()); R["sentinel_samples"]=d[bad_mask].sid.unique().tolist()
dc=d[~bad_mask].copy()
g=dc.groupby("sid")
s=d.groupby("sid").first()[["psi","t","ang","thA","thB","pull","nug","cat","cm"]].copy()
s["nrow"]=d.groupby("sid").size(); s["thick"]=(s.thA+s.thB)/2; s["DOE"]=(s.cm=="DOE"); s["thick_flag"]=s.thick>0.8
a=g.agg(Imean=("I","mean"),Imax=("I","max"),Fmean=("F","mean"),Fmax=("F","max"),Fstart=("F","first"))
a["Istd"]=g.I.std(); a["Ifirst"]=g.I.first(); a["Ilast"]=g.I.last(); a["Idrop"]=(a.Ilast-a.Ifirst)/a.Ifirst
s=s.join(a)
R["design"]=pd.crosstab([s.psi,s.t],[s.DOE]).reset_index().to_dict("records")
# 두 실험 분리
R["groups"]=dict(DOE_n=int(s.DOE.sum()),prod_n=int((~s.DOE).sum()),
   DOE_cat=s[s.DOE].cat.value_counts().to_dict(),prod_cat=s[~s.DOE].cat.value_counts().to_dict(),
   DOE_thick=float(s[s.DOE].thick.mean()),prod_thick=float(s[~s.DOE].thick.mean()))
# 200ms 주장 재검
R["t200"]=dict(n=int((s.t==200).sum()),bad=int(((s.t==200)&(s.cat=="Bad")).sum()),all_in_DOE=bool(s[s.t==200].DOE.all()))
p=s[~s.DOE]
# 생산형(얇은판)에서 Explode: 압력·시간·각도
def rate(df,by): 
    r=df.groupby(by).apply(lambda x:pd.Series(dict(n=len(x),n_explode=int((x.cat=="Explode").sum()),bad=int((x.cat=="Bad").sum()))),include_groups=False); r["explode_rate"]=r["n_explode"]/r["n"]; return r
print(rate(p,"psi")); print(rate(p,"t")); print(rate(p,"ang")); print(rate(p,["psi","ang"]))
R["prod_by_psi"]=rate(p,"psi").reset_index().to_dict("records"); R["prod_by_t"]=rate(p,"t").reset_index().to_dict("records"); R["prod_by_ang"]=rate(p,"ang").reset_index().to_dict("records")
# 피셔 검정
tab=pd.crosstab(p.psi,p.cat=="Explode"); print(tab); R["fisher_psi60_80"]=stats.fisher_exact(tab.values)
tab=pd.crosstab(p.ang,p.cat=="Explode"); R["fisher_angle"]=stats.fisher_exact(tab.values); print(tab, R["fisher_angle"])
# 로지스틱: Explode ~ psi + t + ang (생산형)
X=np.column_stack([(p.psi-p.psi.mean())/p.psi.std(),(p.t-p.t.mean())/p.t.std(),p.ang/15]); yv=(p.cat=="Explode").astype(int).values
lr=LogisticRegression(C=1e6,max_iter=1000).fit(X,yv); R["logit_explode_prod"]=dict(zip(["psi_z","t_z","ang15"],lr.coef_[0].round(3).tolist()))
print(R["logit_explode_prod"])
# 주요 관계 (생산형): 너깃, 인장 vs 시간·압력
for c in ["nug","pull"]:
    for v in ["t","psi","Imean","ang"]:
        r,pv=stats.spearmanr(p[c],p[v],nan_policy="omit"); R[f"spearman_{c}_{v}_prod"]=(float(r),float(pv))
print({k:tuple(round(x,3) for x in v) for k,v in R.items() if k.startswith("spearman")})
R["nug_pull_corr_all"]=float(s.nug.corr(s.pull)); R["nug_pull_corr_prod"]=float(p.nug.corr(p.pull))
# 카테고리 vs 너깃/인장 (규칙 역추적)
R["cat_stats"]={f"{c}|{k}":v for (c,k),v in s.groupby("cat")[["nug","pull"]].agg(["min","median","max"]).round(2).stack(level=0,future_stack=True).to_dict("index").items()} if False else s.groupby("cat")[["nug","pull"]].median().round(2).to_dict()
print(s.groupby("cat")[["nug","pull"]].agg(["min","median","max"]).round(2))
print(p.groupby("cat")[["nug","pull","Imean","Imax"]].agg(["median"]).round(2))
# 전류: 압력/시간과의 관계
print(p.groupby("psi").Imean.describe().round(0)); print(p.groupby("t").Imean.describe().round(0))
# 샘플 순서(실험 순서)와 라벨
p2=p.reset_index(); p2["blk"]=pd.cut(p2.sid,bins=range(0,520,50))
print(p2.groupby("blk",observed=True).agg(n=("cat","size"),exp=("cat",lambda x:(x=="Explode").mean()),I=("Imean","mean"),t=("t","mean"),psi=("psi","mean")).round(2))
s.to_csv(f"{RES}/queretaro_samples.csv"); dc.to_csv(f"{RES}/queretaro_rows_clean.csv",index=False)
dump("queretaro",R)
