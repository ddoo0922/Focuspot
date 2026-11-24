from common import *
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
df=pd.read_csv(f"{RES}/polito_features.csv",parse_dates=["date"]); y=df.y.values
T=pd.read_csv(f"{RES}/polito_feature_table.csv"); T2=pd.read_csv(f"{RES}/polito_feature_table_ex8473.csv")
R={}
# 1) 위치 번호만 쓰는 예측: 차체 단위 GroupKFold, 위치별 불량률(타 차체)로 점수
def spot_only_auc(mask=None):
    sc=np.full(len(df),np.nan)
    gkf=GroupKFold(n_splits=7)
    for tr,te in gkf.split(df,y,df.cb):
        r=df.iloc[tr].groupby("sp").y.agg(["sum","count"]); prior=y[tr].mean()
        m=(r["sum"]+prior*2)/(r["count"]+2)
        sc[te]=df.sp.iloc[te].map(m).fillna(prior).values
    if mask is None: mask=np.ones(len(df),bool)
    return roc_auc_score(y[mask],sc[mask]),sc
a_all,sc=spot_only_auc(); ex=~df.sp.isin([84,73]).values
a_ex,_=spot_only_auc(ex)
R["spot_only_auc_all"]=a_all; R["spot_only_auc_ex8473"]=a_ex
print("위치 번호만으로 AUC (차체 단위 교차검증):",round(a_all,3),"84·73 제외 채점:",round(a_ex,3))
# 2) 시계열 요약만 vs 시계열+위치정보 (로지스틱, GroupKFold)
feats=["D","V_mean","I_mean","I_std","V_std","energy_norm","R_mean","R_std","R_min","R_late_early","R_dip_depth","R_maxdrop","V_late_early","I_late_early","F_max","F_mean_main"]
X=df[feats].fillna(df[feats].median()).values
def cv_auc(X,mask=None):
    p=np.zeros(len(df))
    for tr,te in GroupKFold(n_splits=7).split(X,y,df.cb):
        sc_=StandardScaler().fit(X[tr]); clf=LogisticRegression(C=0.3,max_iter=2000,class_weight="balanced").fit(sc_.transform(X[tr]),y[tr]); p[te]=clf.predict_proba(sc_.transform(X[te]))[:,1]
    m=np.ones(len(df),bool) if mask is None else mask
    return roc_auc_score(y[m],p[m])
R["logit_summary_raw_auc"]=cv_auc(X); R["logit_summary_raw_auc_ex"]=cv_auc(X,ex)
# 위치 표준화 피처
def zs(col):
    g=df.groupby("sp")[col]; z=(df[col]-g.transform("mean"))/g.transform("std"); z[(g.transform("count")<5)|(g.transform("std")==0)]=0; return z.fillna(0)
Xz=np.column_stack([zs(c) for c in feats])
R["logit_summary_spotz_auc"]=cv_auc(Xz); R["logit_summary_spotz_auc_ex"]=cv_auc(Xz,ex)
print({k:round(v,3) for k,v in R.items()})
# 3) 곡선 임베딩: PCA/t-SNE
rs=np.load(f"{OUT}/polito_resampled.npz"); Vr,Ir=rs["Vr"],rs["Ir"]
Z=np.hstack([Vr,Ir]); Zs=StandardScaler().fit_transform(Z)
pca=PCA(n_components=10).fit(Zs); P=pca.transform(Zs); R["pca_var10"]=pca.explained_variance_ratio_.round(3).tolist()
tsne=TSNE(n_components=2,perplexity=35,random_state=0,init="pca").fit_transform(P)
np.save("polito_tsne.npy",tsne)
# 클러스터가 위치를 반영하는가: 각 표본의 최근접 5개 이웃 중 같은 위치 비율 (랜덤 기대 ≈ 1/158)
from sklearn.neighbors import NearestNeighbors
nn=NearestNeighbors(n_neighbors=6).fit(P); dist,idx=nn.kneighbors(P)
same_sp=(df.sp.values[idx[:,1:]]==df.sp.values[:,None]).mean()
same_y=(y[idx[:,1:]]==y[:,None])[y==1].mean()
R["knn_same_spot_share"]=float(same_sp); R["knn_bad_neighbor_bad_share"]=float(same_y)
# 불량인 표본 이웃 중 불량 비율(기준 4%)
print("최근접 이웃 5개 중 같은 위치 비율:",round(same_sp,3),"(무작위 기대 ≈0.008) / 불량 표본의 이웃이 불량인 비율:",round(same_y,3),"(기준 0.04)")
# 4) 연도 드리프트
yrs=df.groupby("year")[["D","energy_norm","I_mean","V_mean","R_mean","F_zero"]].mean().round(3); print(yrs)
R["year_means"]=yrs.reset_index().to_dict("records")
# 같은 위치에서 연도 간 차이 (위치 통제): spot 평균 대비 편차의 연도별 평균
for c in ["I_mean","V_mean","R_mean","energy_norm"]:
    df[c+"_dz"]=df[c]-df.groupby("sp")[c].transform("mean")
R["year_within_spot_dev"]=df.groupby("year")[[c+"_dz" for c in ["I_mean","V_mean","R_mean","energy_norm"]]].mean().round(4).reset_index().to_dict("records")
print(df.groupby("year")[[c+"_dz" for c in ["I_mean","V_mean","R_mean","energy_norm"]]].mean())
# force zero share by year/cb
fz=df.groupby("year").F_zero.mean(); R["force_zero_by_year"]=fz.to_dict(); print(fz)
fz_cb=df.groupby("cb").F_zero.mean(); print("force zero by cb:", fz_cb[fz_cb>0].round(2).to_dict())
R["force_zero_bad_rate"]=dict(zero=float(df[df.F_zero==1].y.mean()),nonzero=float(df[df.F_zero==0].y.mean()))
print(R["force_zero_bad_rate"])
dump("polito_confound",R)
