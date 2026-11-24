from common import *
from scipy import stats
from sklearn.metrics import roc_auc_score
df=pd.read_csv(f"{RES}/polito_features.csv",parse_dates=["date"])
y=df.y.values
feats=["D","V_mean","V_max","V_std","I_mean","I_max","I_std","V_late_early","I_late_early","V_slope","I_slope","energy_norm","energy","R_mean","R_std","R_min","R_late_early","R_argmin","R_maxdrop","R_dip_depth","V1_mean","I1_mean","E1","D1","gap","F_mean_main","F_max","F_main_std","F_start"]
names={"D":"본 용접 길이","V_mean":"전압 평균","V_max":"전압 최대","V_std":"전압 표준편차","I_mean":"전류 평균","I_max":"전류 최대","I_std":"전류 표준편차","V_late_early":"전압 후반/초반","I_late_early":"전류 후반/초반","V_slope":"전압 기울기","I_slope":"전류 기울기","energy_norm":"에너지(평균 V·I)","energy":"에너지 합(ΣV·I)","R_mean":"동저항 평균","R_std":"동저항 변동","R_min":"동저항 최소","R_late_early":"동저항 후반/초반","R_argmin":"동저항 최저 시점","R_maxdrop":"동저항 최대 급락","R_dip_depth":"동저항 하강폭","V1_mean":"예열 전압 평균","I1_mean":"예열 전류 평균","E1":"예열 에너지","D1":"예열 길이","gap":"무신호 간격","F_mean_main":"가압력 평균","F_max":"가압력 최대","F_main_std":"가압력 변동","F_start":"가압력 초기값"}
df["spot_year"]=df.sp.astype(str)+"_"+df.year.astype(str)
def auc(x,y):
    m=np.isfinite(x)
    if y[m].sum()<3: return np.nan
    return roc_auc_score(y[m],x[m])
def boot_ci(x,y,n=400,seed=0):
    rng=np.random.default_rng(seed); m=np.isfinite(x); x=x[m]; yy=y[m]; a=[]
    ip=np.where(yy==1)[0]; iN=np.where(yy==0)[0]
    for _ in range(n):
        s=np.r_[rng.choice(ip,len(ip)),rng.choice(iN,len(iN))]
        a.append(roc_auc_score(yy[s],x[s]))
    return np.percentile(a,[2.5,97.5])
# 위치(spot) 표준화: spot 내 평균·표준편차로 z (표본 ≥5인 위치만; 아니면 NaN)
def zspot(col,key="sp"):
    g=df.groupby(key)[col]; n=g.transform("count"); mu=g.transform("mean"); sd=g.transform("std")
    z=(df[col]-mu)/sd; z[(n<5)|(sd==0)]=np.nan; return z
rows=[]
for f in feats:
    x=df[f].values.astype(float)
    z=zspot(f).values; zy=zspot(f,"spot_year").values
    # η²: 위치가 설명하는 분산 비율
    m=np.isfinite(x); g=df.loc[m].groupby("sp")[f]
    ssb=((g.transform("mean")-x[m].mean())**2).sum(); sst=((x[m]-x[m].mean())**2).sum()
    # 위치 안 비교: 두 클래스가 모두 있는 위치만
    both=df.groupby("sp").y.agg(lambda s:(s.sum()>0)&(s.sum()<len(s)))
    mb=df.sp.map(both).values&np.isfinite(z)
    mw=stats.mannwhitneyu(x[(y==1)&m],x[(y==0)&m])[1] if (y[m]==1).sum()>2 else np.nan
    mw2=stats.mannwhitneyu(z[mb&(y==1)],z[mb&(y==0)])[1]
    ci=boot_ci(x,y); ci2=boot_ci(np.where(mb,z,np.nan),y)
    rows.append(dict(feature=f,name=names[f],eta2_spot=ssb/sst,
        auc_raw=auc(x,y),auc_raw_lo=ci[0],auc_raw_hi=ci[1],p_raw=mw,
        auc_within=auc(np.where(mb,z,np.nan),y),auc_within_lo=ci2[0],auc_within_hi=ci2[1],p_within=mw2,
        auc_within_spotyear=auc(np.where(mb,zy,np.nan),y),
        med_bad=np.nanmedian(x[y==1]),med_ok=np.nanmedian(x[y==0]),
        z_bad=np.nanmean(z[mb&(y==1)]),z_ok=np.nanmean(z[mb&(y==0)]),n_within=int((mb&np.isfinite(z)).sum()),bad_within=int((mb&(y==1)&np.isfinite(z)).sum())))
T=pd.DataFrame(rows); T.to_csv(f"{RES}/polito_feature_table.csv",index=False)
pd.set_option("display.width",250)
print(T[["name","eta2_spot","auc_raw","p_raw","auc_within","p_within","z_bad","n_within","bad_within"]].round(3).to_string())
# 위치 84·73 제외 버전
ex=~df.sp.isin([84,73]).values
rows2=[]
for f in feats:
    x=df[f].values.astype(float); z=zspot(f).values
    both=df.groupby("sp").y.agg(lambda s:(s.sum()>0)&(s.sum()<len(s))); mb=df.sp.map(both).values&np.isfinite(z)&ex
    rows2.append(dict(feature=f,auc_raw_ex=auc(np.where(ex,x,np.nan),y),auc_within_ex=auc(np.where(mb,z,np.nan),y),bad_within_ex=int((mb&(y==1)).sum())))
T2=pd.DataFrame(rows2); T2.to_csv(f"{RES}/polito_feature_table_ex8473.csv",index=False); print(T2.round(3).to_string())
