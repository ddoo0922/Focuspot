from common import *
from scipy import stats
df=pd.read_csv(f"{RES}/polito_features.csv",parse_dates=["date"]); y=df.y.values
T=pd.read_csv(f"{RES}/polito_feature_table.csv"); RC=json.load(open(f"{RES}/polito_confound.json"))
# ---- Fig6: η² + AUC 덤벨
T=T.sort_values("auc_raw",ascending=True)
sel=T[~T.feature.isin(["energy","D1","gap","F_start","V_max","I_max","R_std"])].copy()
fig,ax=plt.subplots(1,2,figsize=(14,8),gridspec_kw={"width_ratios":[1,1.5]},sharey=True)
yy=np.arange(len(sel))
ax[0].barh(yy,sel.eta2_spot,color=[C_BAD if v>0.8 else C_OK for v in sel.eta2_spot]); ax[0].axvline(0.8,ls="--",color="k",lw=0.7)
ax[0].set_yticks(yy); ax[0].set_yticklabels(sel.name); ax[0].set_xlabel("타점 위치가 설명하는 분산 비율 (η²)"); ax[0].set_title("① 대부분의 피처는 '어느 위치냐'로 거의 결정된다",loc="left",fontsize=11,fontweight="bold")
a=ax[1]
for i,(_,r) in enumerate(sel.iterrows()):
    a.plot([r.auc_raw,r.auc_within],[i,i],color=C_GR,lw=1.5,zorder=1)
a.scatter(sel.auc_raw,yy,color=C_OK,s=45,label="그냥 비교 (위치 섞임)",zorder=2)
a.scatter(sel.auc_within,yy,color=C_BAD,s=45,label="같은 위치 안에서 비교",zorder=3)
a.axvline(0.5,color="k",lw=0.8); a.set_xlim(0.28,0.8); a.set_xlabel("불량 판별 AUC (0.5=무작위, 0.5 미만=방향이 반대)"); a.legend(loc="lower right")
a.set_title("② 위치를 통제하면 겉보기 판별력(파랑)이 사라지거나 뒤집힌다(주황)",loc="left",fontsize=11,fontweight="bold")
fig.suptitle("Stellantis N.V. 피처별: 위치 교란 진단",fontsize=14,fontweight="bold",y=0.995); fig.tight_layout(); save(fig,"polito_06_confounding_features")
# ---- Fig7: 세 피처 분포 raw vs within
fig,ax=plt.subplots(2,3,figsize=(15,7))
def zspot(col):
    g=df.groupby("sp")[col]; z=(df[col]-g.transform("mean"))/g.transform("std"); z[(g.transform("count")<5)|(g.transform("std")==0)]=np.nan; return z
both=df.groupby("sp").y.agg(lambda s:(s.sum()>0)&(s.sum()<len(s))); mb=df.sp.map(both).values
for c,(col,nm) in enumerate([("energy","에너지 합(ΣV·I)"),("I_mean","전류 평균(본 용접)"),("R_dip_depth","동저항 하강폭")]):
    a=ax[0,c]; x=df[col]
    bins=np.linspace(x.min(),x.max(),40)
    a.hist(x[y==0],bins=bins,density=True,color=C_OK,alpha=0.6,label="정상"); a.hist(x[y==1],bins=bins,density=True,color=C_BAD,alpha=0.7,label="불량")
    a.set_title(f"{nm} — 그냥 비교 ",loc="left",fontsize=10.5,fontweight="bold"); a.legend(fontsize=8)
    a=ax[1,c]; z=zspot(col); m=mb&np.isfinite(z)
    bins=np.linspace(-3.5,3.5,36)
    a.hist(z[m&(y==0)],bins=bins,density=True,color=C_OK,alpha=0.6); a.hist(z[m&(y==1)],bins=bins,density=True,color=C_BAD,alpha=0.7)
    a.axvline(0,color="k",lw=0.7); a.set_title(f"{nm} — 같은 위치 안에서(표준화)",loc="left",fontsize=10.5,fontweight="bold"); a.set_xlabel("위치 평균 대비 편차 (σ)")
fig.suptitle("에너지·전류의 불량-정상 차이는 위치를 맞추면 사라지거나 반대가 된다. 동저항 하강폭만 위치를 맞춰도 불량이 큼",fontsize=12.5,fontweight="bold",y=1.0); fig.tight_layout(); save(fig,"polito_07_feature_dist")
# ---- Fig8: t-SNE
ts=np.load("polito_tsne.npy")
fig,ax=plt.subplots(1,3,figsize=(17,5.3))
a=ax[0]; a.scatter(ts[y==0,0],ts[y==0,1],s=5,color=C_GR,alpha=0.5); a.scatter(ts[y==1,0],ts[y==1,1],s=22,color=C_BAD,edgecolor="k",lw=0.3); a.set_title("불량(주황)이 흩어져 있지 않고 뭉쳐 있다",loc="left",fontsize=10.5,fontweight="bold")
a=ax[1]; a.scatter(ts[:,0],ts[:,1],s=5,color=C_GR,alpha=0.4)
for s_,col in [(84,"#D62828"),(73,"#1D4E89")]:
    m=df.sp.values==s_; a.scatter(ts[m,0],ts[m,1],s=30,color=col,edgecolor="k",lw=0.3,label=f"위치 {s_}번")
a.legend(); a.set_title("뭉친 곳 = 84·73번 위치의 곡선 모양",loc="left",fontsize=10.5,fontweight="bold")
a=ax[2]; cols={2022:"#8E6BBF",2023:C_A,2024:C_OK}
for yr,cc in cols.items(): m=df.year.values==yr; a.scatter(ts[m,0],ts[m,1],s=5,color=cc,alpha=0.6,label=str(yr))
a.legend(markerscale=3); a.set_title("연도별 (2022 = 별도 섬)",loc="left",fontsize=10.5,fontweight="bold")
for a in ax: a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.suptitle("전압·전류 곡선 t-SNE(고차원 → 2차원 지도): 곡선 모양은 '위치의 지문'",fontsize=13,fontweight="bold",y=1.02); fig.tight_layout(); save(fig,"polito_08_tsne")
# ---- Fig9: AUC 정직 비교
vals=[("위치 번호만\n(타 차체 불량률 조회)",RC["spot_only_auc_all"],RC["spot_only_auc_ex8473"]),
      ("시계열 요약 16개\n(그대로)",RC["logit_summary_raw_auc"],RC["logit_summary_raw_auc_ex"]),
      ("시계열 요약 16개\n(위치별 표준화)",RC["logit_summary_spotz_auc"],RC["logit_summary_spotz_auc_ex"])]
fig,ax=plt.subplots(figsize=(8.5,4.8)); w=0.35; xx=np.arange(3)
ax.bar(xx-w/2,[v[1] for v in vals],w,color=C_OK,label="전체 채점"); ax.bar(xx+w/2,[v[2] for v in vals],w,color=C_BAD,label="84·73번 제외하고 채점")
for i,v in enumerate(vals): ax.text(i-w/2,v[1]+0.01,f"{v[1]:.2f}",ha="center",fontsize=9); ax.text(i+w/2,v[2]+0.01,f"{v[2]:.2f}",ha="center",fontsize=9)
ax.axhline(0.5,color="k",lw=0.8); ax.set_ylim(0.4,0.9); ax.set_xticks(xx); ax.set_xticklabels([v[0] for v in vals]); ax.set_ylabel("AUC (차체 단위 교차검증)"); ax.legend(loc="upper right")
ax.set_title("위치 번호만 알아도 AUC 0.79 — 시계열 모델 0.82의 대부분은 '위치 맞히기'",loc="left",fontsize=11,fontweight="bold"); save(fig,"polito_09_honest_auc")
# ---- Fig10: 연도·force 결측
fig,ax=plt.subplots(1,3,figsize=(15,4.3))
g=df.groupby("year").agg(n=("y","size"),fz=("F_zero","mean"),D=("D","mean"))
ax[0].bar(g.index.astype(str),g.fz*100,color=[C_BAD,C_BAD,C_OK]); ax[0].set_ylabel("가압력 신호가 전부 0인 타점(%)"); ax[0].set_title("가압력 센서: 2022 전부 결측, 2023 16%, 2024 2%",loc="left",fontsize=10.5,fontweight="bold")
for i,(k,r) in enumerate(g.iterrows()): ax[0].text(i,r.fz*100+2,f"{r.fz*100:.0f}%",ha="center")
ax[1].boxplot([df[df.year==k].D for k in [2022,2023,2024]],labels=["2022","2023","2024"],patch_artist=True,boxprops=dict(facecolor="#DDE3EA")); ax[1].set_ylabel("본 용접 길이(스텝)"); ax[1].set_title("본 용접 길이 분포: 2022만 크게 다름(공정 조건 변경)",loc="left",fontsize=10.5,fontweight="bold")
cbt=df.groupby("cb").agg(fz=("F_zero","mean"),n=("y","size")); ax[2].bar(cbt.index.astype(str),cbt.fz*100,color=C_OK); ax[2].set_xlabel("차체 번호"); ax[2].set_ylabel("force=0 비율(%)"); ax[2].tick_params(axis="x",labelsize=7); ax[2].set_title("차체별: 6~16번(2022) 전부, 0~5번(2023-06) 16%",loc="left",fontsize=10.5,fontweight="bold")
fig.tight_layout(); save(fig,"polito_10_year_force")
# ---- Fig11: 상관 히트맵 (spearman)
cols=["D","V_mean","I_mean","energy_norm","R_mean","R_std","R_min","R_late_early","R_dip_depth","R_maxdrop","V_late_early","I_late_early","I_std","F_mean_main","F_max"]
nm=dict(zip(T.feature,T.name)); C=df[cols].rank().corr()
fig,ax=plt.subplots(figsize=(9,7.5)); im=ax.imshow(C.values,cmap="RdBu_r",vmin=-1,vmax=1); ax.grid(False)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([nm[c] for c in cols],rotation=60,ha="right",fontsize=8); ax.set_yticks(range(len(cols))); ax.set_yticklabels([nm[c] for c in cols],fontsize=8)
for i in range(len(cols)):
    for j in range(len(cols)): ax.text(j,i,f"{C.values[i,j]:.1f}",ha="center",va="center",fontsize=6.5,color="white" if abs(C.values[i,j])>0.6 else "k")
plt.colorbar(im,fraction=0.04); ax.set_title("피처 간 순위상관(Spearman): 에너지·전압·길이가 한 덩어리",loc="left",fontsize=11,fontweight="bold"); save(fig,"polito_11_corr")
print("ok")
