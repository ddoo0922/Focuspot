from common import *
from scipy import stats
df=pd.read_csv(f"{RES}/polito_features.csv",parse_dates=["date"])
d=np.load(f"{OUT}/polito.npz",allow_pickle=True)
V,I,F=d["V"],d["I"],d["F"]
R={}
# ---- 품질 5축
keyc=["cb","sp"]
q=quality_index(pd.DataFrame(V),name="voltage")
R["quality"]={
 "rows":len(df),"cols_series":1000,
 "결측(NaN) 셀 수":int(np.isnan(V).sum()+np.isnan(I).sum()+np.isnan(F).sum()),
 "라벨 값":sorted(df.y.unique().tolist()),
 "키(차체,타점) 중복 쌍":int(df.duplicated(keyc).sum()),
 "키 중복 그룹 수":int(df[df.duplicated(keyc,keep=False)].groupby(keyc).ngroups),
 "전압 시계열 완전 중복 행":int(pd.DataFrame(V).duplicated().sum()),
 "전류 시계열 완전 중복 행":int(pd.DataFrame(I).duplicated().sum()),
 "범위 [0,1] 만족 비율":float(((V>=0)&(V<=1)).mean()),
 "force 전부 0인 타점":int(df.F_zero.sum()),
 "phase1 없는 타점(구간 1개)":int((df.n_runs==1).sum()),
}
# 중복 세부
dk=df[df.duplicated(keyc,keep=False)].copy()
grp=[];
for (cb,sp),g in dk.groupby(keyc):
    ii=g.index.values
    same=np.allclose(V[ii[0]],V[ii[1]]) and np.allclose(I[ii[0]],I[ii[1]]) and np.allclose(F[ii[0]],F[ii[1]])
    grp.append(dict(cb=cb,sp=sp,n=len(ii),identical=bool(same),y=list(g.y.values),date=[str(x.date()) for x in g.date]))
gd=pd.DataFrame(grp)
R["dup_groups"]=dict(total=len(gd),identical=int(gd.identical.sum()),nonidentical=int((~gd.identical).sum()),
     label_conflict=int((gd.y.apply(lambda x:len(set(x))>1)).sum()), n_gt2=int((gd.n>2).sum()))
print(R["quality"]); print(R["dup_groups"]); print(gd[~gd.identical].head(20))
# 라벨 충돌 있는 쌍
print(gd[gd.y.apply(lambda x:len(set(x))>1)])
# ---- 라벨 구조
spot=df.groupby("sp").agg(n=("y","size"),bad=("y","sum")).assign(rate=lambda x:x.bad/x.n).sort_values("bad",ascending=False)
R["labels"]=dict(n=len(df),bad=int(df.y.sum()),rate=float(df.y.mean()),
   spots=int(df.sp.nunique()),spots_with_bad=int((spot.bad>0).sum()),
   top2=spot.head(2).reset_index().to_dict("records"),
   top2_share=float(spot.head(2).bad.sum()/spot.bad.sum()),
   top5_share=float(spot.head(5).bad.sum()/spot.bad.sum()),top10_share=float(spot.head(10).bad.sum()/spot.bad.sum()))
# 지니계수(불량의 위치 편중)
b=np.sort(spot.bad.values)  # 오름차순
cum=np.cumsum(b)/b.sum(); lor=np.r_[0,cum]; x=np.linspace(0,1,len(lor))
gini=1-2*np.trapezoid(lor,x); R["labels"]["gini_bad_over_spots"]=float(gini)
# 기대 대비: 위치별 불량률 동일 가정(귀무)에서 34곳 이상 몰림? 순열검정: 불량 라벨을 무작위 재배치했을 때 top2 몫
rng=np.random.default_rng(0); y=df.y.values; sp=df.sp.values
top2s=[]; nspots=[]
for _ in range(2000):
    yp=rng.permutation(y)
    s=pd.Series(yp).groupby(sp).sum().sort_values(ascending=False)
    top2s.append(s.head(2).sum()/s.sum()); nspots.append((s>0).sum())
R["labels"]["perm_top2_share_mean"]=float(np.mean(top2s)); R["labels"]["perm_top2_share_p99"]=float(np.percentile(top2s,99))
R["labels"]["perm_spots_with_bad_mean"]=float(np.mean(nspots))
# 위치별 불량률의 이질성: chi2
tab=pd.crosstab(df.sp,df.y); chi2,p,dof,_=stats.chi2_contingency(tab)
R["labels"]["chi2_spot_heterogeneity"]=dict(chi2=float(chi2),dof=int(dof),p=float(p))
# 84·73 제외
ex=df[~df.sp.isin([84,73])]
R["labels"]["excl_84_73"]=dict(n=len(ex),bad=int(ex.y.sum()),rate=float(ex.y.mean()))
# 차체별
cbt=df.groupby("cb").agg(n=("y","size"),bad=("y","sum"),date=("date","min")).assign(rate=lambda x:x.bad/x.n)
R["cb_table"]=cbt.reset_index().assign(date=lambda x:x.date.astype(str)).to_dict("records")
# 차체 크기>=100 (전체 타점 검사한 차체)
full=cbt[cbt.n>=100]
R["cb_full"]=dict(n_cb=len(full),samples=int(full.n.sum()),bad=int(full.bad.sum()),rate=float(full.bad.sum()/full.n.sum()),
                  mean_bad=float(full.bad.mean()),var_bad=float(full.bad.var()))
# 부분 차체(<100)
part=cbt[cbt.n<100]
R["cb_partial"]=dict(n_cb=len(part),samples=int(part.n.sum()),bad=int(part.bad.sum()),rate=float(part.bad.sum()/part.n.sum()))
# 연도
yr=df.groupby("year").agg(n=("y","size"),bad=("y","sum"),cbs=("cb","nunique")).assign(rate=lambda x:x.bad/x.n)
R["year"]=yr.reset_index().to_dict("records")
print(yr)
# 날짜별 (KAMP 일 단위 묶음 시뮬)
dd=df.groupby("date").agg(n=("y","size"),bad=("y","sum")).assign(rate=lambda x:x.bad/x.n)
R["date_bags"]=dict(n_days=len(dd),size_min=int(dd.n.min()),size_med=float(dd.n.median()),size_max=int(dd.n.max()),
                    bad_min=int(dd.bad.min()),bad_med=float(dd.bad.median()),bad_max=int(dd.bad.max()),days_zero_bad=int((dd.bad==0).sum()))
print(dd.to_string())
# 차체 내 순서: spot 번호 인접에서의 불량 군집성
adj=[];pos=[]
for cb,g in df.groupby("cb"):
    g=g.sort_values("sp"); yy=g.y.values; ss=g.sp.values
    for k in range(len(g)-1):
        if ss[k+1]-ss[k]==1: adj.append((yy[k],yy[k+1]))
adj=np.array(adj); p_next=adj[adj[:,0]==1][:,1].mean(); p_base=adj[:,1].mean()
R["adjacent"]=dict(pairs=len(adj),p_bad_next_given_bad=float(p_next),p_bad_next_overall=float(p_base),n_bad_first=int((adj[:,0]==1).sum()))
# 위치별 불량 횟수 과산포
R["labels"]["spot_table_top15"]=spot.head(15).reset_index().to_dict("records")
dump("polito_structure",R)
print(json.dumps(R["labels"],ensure_ascii=False,indent=1)); print(R["cb_full"],R["cb_partial"],R["adjacent"],R["date_bags"])
gd.to_csv(f"{RES}/polito_dup_groups.csv",index=False)
