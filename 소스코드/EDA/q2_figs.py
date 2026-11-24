from common import *
from scipy import stats
s=pd.read_csv(f"{RES}/queretaro_samples.csv",index_col=0); rows=pd.read_csv(f"{RES}/queretaro_rows_clean.csv")
rows=rows.rename(columns={})
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return c-h,c+h
cc={"Good":C_OK,"Bad":C_BAD,"Explode":C_B}
# Fig Q1: 설계 지도
fig,ax=plt.subplots(1,2,figsize=(14,4.8),gridspec_kw={"width_ratios":[1,1.5]})
for a,(nm,m) in zip(ax,[("실험 A: 두꺼운 판(≈0.93mm) 33타점 — 극단 조건 2×2",s.DOE),("실험 B: 얇은 판(≈0.63mm) 462타점 — 생산형 조건",~s.DOE)]):
    sub=s[m]; cells=sub.groupby(["psi","t"])
    tv=sorted(sub.t.unique()); pv=sorted(sub.psi.unique())
    a.set_xticks(range(len(tv))); a.set_xticklabels(tv); a.set_yticks(range(len(pv))); a.set_yticklabels([f"{p} PSI" for p in pv]); a.set_xlim(-0.6,len(tv)-0.4); a.set_ylim(-0.6,len(pv)-0.4)
    for (p_,t_),g in cells:
        x=tv.index(t_); y_=pv.index(p_); n=len(g); nb=(g.cat=="Bad").sum(); ne=(g.cat=="Explode").sum(); ng=n-nb-ne
        w=0.8; x0=x-w/2
        for lab,cnt,col in [("Good",ng,C_OK),("Explode",ne,C_B),("Bad",nb,C_BAD)]:
            if cnt: a.barh(y_,w*cnt/n,left=x0,height=0.55,color=col,edgecolor="white"); x0+=w*cnt/n
        a.text(x,y_+0.36,f"n={n}",ha="center",fontsize=8)
    a.set_xlabel("통전 시간 (ms)"); a.set_title(nm,loc="left",fontsize=10.5,fontweight="bold"); a.grid(False)
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color=C_OK,label="Good"),Patch(color=C_B,label="Explode(날림)"),Patch(color=C_BAD,label="Bad(약한 용접)")],loc="lower center",bbox_to_anchor=(0.5,-0.35),ncol=3,fontsize=8)
fig.suptitle("Querétaro는 '서로 다른 두 실험'이 섞여 있다 — Bad 21건 중 17건이 실험 A(200ms 극단 조건)",fontsize=13,fontweight="bold",y=1.03); fig.tight_layout(); save(fig,"quer_01_design")
# Fig Q2: 너깃-인장 산점
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
a=ax[0]
for c in ["Good","Explode","Bad"]:
    for doe,mk in [(False,"o"),(True,"^")]:
        m=(s.cat==c)&(s.DOE==doe); a.scatter(s.nug[m],s.pull[m],s=22 if not doe else 40,marker=mk,color=cc[c],alpha=0.65,edgecolor="k" if doe else "none",lw=0.5)
a.set_xlabel("너깃 지름 (mm)"); a.set_ylabel("인장 시험 강도 (N)"); a.set_title(f"너깃 지름과 강도는 함께 움직인다 (r={s.nug.corr(s.pull):.2f})\n○ 실험 B, △ 실험 A",loc="left",fontsize=10.5,fontweight="bold")
from matplotlib.lines import Line2D
a.legend(handles=[Line2D([],[],marker="o",ls="",color=cc[k],label=k) for k in cc],fontsize=8)
a=ax[1]
data=[s[s.cat==c].nug for c in ["Bad","Good","Explode"]]; bp=a.boxplot(data,tick_labels=["Bad","Good","Explode"],patch_artist=True,widths=0.5)
for p_,c in zip(bp["boxes"],["Bad","Good","Explode"]): p_.set_facecolor(cc[c]); p_.set_alpha(0.7)
a.axhline(3.0,ls="--",color="k",lw=0.7); a.set_ylabel("너깃 지름 (mm)"); a.set_title("Bad는 너깃이 작다(중앙 2.7mm). Explode는 Good과 너깃 크기가 비슷 → 크기가 아닌 '날림' 여부로 분류",loc="left",fontsize=10,fontweight="bold")
fig.tight_layout(); save(fig,"quer_02_labels")
# Fig Q3: 실험 B 조건별 Explode/Bad 비율 + Wilson CI
p=s[~s.DOE]
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
for a,(col,nm,vals) in zip(ax,[("psi","가압력 (PSI)",[60,80]),("t","통전 시간 (ms)",[400,600,800,1000,1200]),("ang","전극 각도 (°)",[0,15])]):
    xs=np.arange(len(vals)); 
    for j,(cat,colr,off) in enumerate([("Explode",C_B,-0.18),("Bad",C_BAD,0.18)]):
        rs=[];lo=[];hi=[]
        for v in vals:
            g=p[p[col]==v]; k=(g.cat==cat).sum(); n=len(g); l,h=wilson(k,n); rs.append(100*k/n); lo.append(100*l); hi.append(100*h)
        a.bar(xs+off,rs,0.34,color=colr,label=cat if col=="psi" else None); a.errorbar(xs+off,rs,yerr=[np.array(rs)-np.array(lo),np.array(hi)-np.array(rs)],fmt="none",ecolor="k",lw=0.8,capsize=3)
    for x,v in zip(xs,vals): a.text(x,-2.3,f"n={int((p[col]==v).sum())}",ha="center",fontsize=8)
    a.set_xticks(xs); a.set_xticklabels(vals); a.set_xlabel(nm); a.set_ylabel("비율(%) · 막대=95% 신뢰구간"); a.set_ylim(-3,26)
ax[0].legend(); ax[0].set_title("60 PSI에서 날림 7.2% vs 80 PSI 1.5%",loc="left",fontsize=10.5,fontweight="bold"); ax[1].set_title("통전 시간: 1000ms만 0%, 나머지는 6~10%(단조 아님)",loc="left",fontsize=10.5,fontweight="bold"); ax[2].set_title("각도 15°: 날림 8.3% vs 0° 3.0%",loc="left",fontsize=10.5,fontweight="bold")
fig.suptitle("실험 B(얇은 판 462타점)만 보면 — Bad는 4건뿐, 날림(Explode)이 주된 불량",fontsize=13,fontweight="bold",y=1.03); fig.tight_layout(); save(fig,"quer_03_prod_rates")
# Fig Q4: 실험 순서(샘플 번호)에 따른 전류 수준과 날림
fig,ax=plt.subplots(2,1,figsize=(13,6.5),sharex=True)
pp=p.reset_index()
ax[0].scatter(pp.sid,pp.Imean,s=8,color=C_GR); m=pp.cat=="Explode"; ax[0].scatter(pp.sid[m],pp.Imean[m],s=28,color=C_B,edgecolor="k",lw=0.4,label="Explode"); m=pp.cat=="Bad"; ax[0].scatter(pp.sid[m],pp.Imean[m],s=36,color=C_BAD,edgecolor="k",lw=0.4,label="Bad")
ax[0].set_ylabel("타점 평균 전류 (A)"); ax[0].legend(); ax[0].set_title("샘플 번호(=실험 순서) 250~400 구간에서 전류 수준이 올라가고 날림이 몰린다",loc="left",fontsize=10.5,fontweight="bold")
blk=pd.cut(pp.sid,bins=range(0,520,25)); g=pp.groupby(blk,observed=True).agg(exp=("cat",lambda x:(x=="Explode").mean()*100),sid=("sid","mean"))
ax[1].bar(g.sid,g.exp,width=20,color=C_B); ax[1].set_ylabel("25개 구간별 날림 비율(%)"); ax[1].set_xlabel("샘플 번호")
fig.tight_layout(); save(fig,"quer_04_run_order")
# Fig Q5: Force 컬럼 정체 + 전류 파형
fig,ax=plt.subplots(1,3,figsize=(16,4.5))
ex_ids={"실험 A 35 PSI":s[(s.DOE)&(s.psi==35)&(s.t==1500)].index[0],"실험 A 95 PSI":s[(s.DOE)&(s.psi==95)&(s.t==1500)].index[0],"실험 B 60 PSI":s[(~s.DOE)&(s.psi==60)&(s.t==1000)].index[0],"실험 B 80 PSI":s[(~s.DOE)&(s.psi==80)&(s.t==1000)].index[0]}
for k,i in ex_ids.items():
    g=rows[rows.sid==i]; ax[0].plot(np.arange(len(g))*100,g["F"],marker="o",ms=3,label=k); ax[1].plot(np.arange(len(g))*100,g["I"],marker="o",ms=3,label=k)
ax[0].set_xlabel("경과 시간 (ms, 10Hz 가정)"); ax[0].set_ylabel("Force (N) 열"); ax[0].legend(fontsize=8); ax[0].set_title("실험 A의 Force는 0에서 직선 증가 — 실측 가압력이 아닌 것으로 의심",loc="left",fontsize=10,fontweight="bold")
ax[1].set_xlabel("경과 시간 (ms)"); ax[1].set_ylabel("전류 (A)"); ax[1].set_title("전류 파형(타점당 3~16점, 10Hz)",loc="left",fontsize=10,fontweight="bold")
rr=s.groupby("psi").apply(lambda g: pd.Series(dict(Fmean=g.Fmean.mean(),Fstart=g.Fstart.mean())),include_groups=False); 
pf=s.assign(pc=s.psi.astype(str)).boxplot(column="Fmean",by="pc",ax=ax[2],grid=False); ax[2].set_xlabel("가압력 설정 (PSI)"); ax[2].set_ylabel("Force (N) 열 평균"); ax[2].set_title("PSI별 평균 Force 열 값 (35·95는 직선 증가로 왜곡)",loc="left",fontsize=10,fontweight="bold"); plt.suptitle("")
fig.tight_layout(); save(fig,"quer_05_force_current")
print(rr)
