from common import *
from scipy import stats
f=f"{B}/01. KAMP_용접기 AI 데이터셋/2. 용접기 AI 데이터셋/Welding Data Set_01.xlsx"
raw=pd.read_excel(f,sheet_name="Raw data"); res=pd.read_excel(f,sheet_name="result").iloc[:,:6]; dd=pd.read_excel(f,sheet_name="data set")
raw.columns=["idx","machine","item","date","th1","th2","force","current","voltage","wtime"]
res.columns=["idx","machine","item","date","defect","dtype"]
R={}
R["shape"]=raw.shape; R["nan"]=raw.isna().sum().to_dict(); R["unique"]={c:int(raw[c].nunique()) for c in ["machine","item","th1","th2"]}
R["dates"]=raw.date.dt.strftime("%m-%d").value_counts().sort_index().to_dict()
print(R["dates"])
R["desc"]=raw[["force","current","voltage","wtime"]].describe().round(3).to_dict()
print(raw[["force","current","voltage","wtime"]].describe().round(3))
print(dd.to_string())
# 중복
R["dup_rows_full"]=int(raw.drop(columns="idx").duplicated().sum())
R["consec_dup_pairs"]=int((raw[["force","current","voltage","wtime"]].diff().abs().sum(axis=1)==0).sum())
# 복사 블록: 가압력 >3
hi=raw[raw.force>3]; R["force_gt3"]=dict(n=len(hi),by_date=hi.date.dt.strftime("%m-%d").value_counts().sort_index().to_dict(),fmin=float(hi.force.min()),fmax=float(hi.force.max()))
print(R["force_gt3"])
# 블록 동일성: 날짜별 hi 블록 동일?
blocks={k:g[["force","current","voltage","wtime"]].values for k,g in hi.groupby("date")}
ks=list(blocks); R["blocks_identical"]={f"{str(a)[5:10]}~{str(b)[5:10]}":bool(blocks[a].shape==blocks[b].shape and np.allclose(blocks[a],blocks[b])) for i,a in enumerate(ks) for b in ks[i+1:]}
print(R["blocks_identical"])
# 결과 시트
res["date"]=pd.to_datetime(res.date); res["dtype"]=res.dtype if False else res["dtype"]
piv=res.pivot_table(index="date",columns="idx",values="defect",aggfunc="sum")
tot=res.groupby("date").defect.sum(); n=raw.groupby("date").size()
tab=pd.DataFrame({"n_raw":n,"defect":tot}).fillna(0); tab["rate_pct"]=100*tab.defect/tab.n_raw
print(tab)
R["daily"]=tab.reset_index().assign(date=lambda x:x.date.dt.strftime("%m-%d")).to_dict("records")
# 유형 별 합 (defect type 컬럼: 1,2,3 for each date)
res["type"]=res.groupby("date").cumcount()%3+1
tt=res.pivot_table(index="date",columns="type",values="defect",aggfunc="sum"); print(tt); R["daily_by_type"]=tt.reset_index().assign(date=lambda x:x.date.dt.strftime("%m-%d")).to_dict("records")
R["total_defects"]=int(res.defect.sum()); R["total_rate_pct"]=float(100*res.defect.sum()/len(raw)); R["by_type_total"]=tt.sum().to_dict()
# 블록 제외 후 일별 표본
nb=raw[raw.force<=3].groupby("date").size(); R["n_no_block"]=nb.to_dict() if False else {str(k)[5:10]:int(v) for k,v in nb.items()}
# 상관, 스케일
core=raw[raw.force<=3][["force","current","voltage","wtime"]]; R["corr_noblock"]=core.corr(method="spearman").round(2).to_dict()
print(core.corr(method="spearman").round(2))
# 일별 분포 드리프트
dr=raw[raw.force<=3].groupby(raw.date.dt.strftime("%m-%d"))[["force","current","voltage","wtime"]].agg(["mean","std"]).round(3); print(dr)
R["daily_moments"]=json.loads(dr.to_json())
# 동저항/에너지
raw["R"]=raw.voltage/raw.current*1000; raw["E"]=raw.voltage*raw.current*raw.wtime
R["R_mOhm"]=raw.R[raw.force<=3].describe().round(4).to_dict()
# scaled 데이터와 대조
sc=pd.read_csv(f"{B}/01. KAMP_용접기 AI 데이터셋/2. 용접기 AI 데이터셋/scaled_data.csv",encoding="cp949"); R["scaled_shape"]=sc.shape; R["scaled_nan"]=sc.isna().sum().to_dict(); print(sc.describe().T)
raw.to_pickle("kamp_raw.pkl")
dump("kamp",R)
