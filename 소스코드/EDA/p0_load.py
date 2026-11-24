from common import *
P=f"{B}/03_Stellantis N.V._RSW/data"
lab=pd.read_csv(f"{P}/labels.csv")
V=pd.read_csv(f"{P}/voltage.csv"); I=pd.read_csv(f"{P}/current.csv"); F=pd.read_csv(f"{P}/force.csv")
print(lab.shape,V.shape,I.shape,F.shape)
print(lab.dtypes)
key=["Car Body","Welding Spot","Date"]
for n,d in [("V",V),("I",I),("F",F)]:
    print(n,(d[key].values==lab[key].values).all(), d.columns[:5].tolist(), d.columns[-2:].tolist())
tv=V.drop(columns=key).values.astype(np.float32); ti=I.drop(columns=key).values.astype(np.float32); tf=F.drop(columns=key).values.astype(np.float32)
np.savez_compressed(f"{OUT}/polito.npz",V=tv,I=ti,F=tf,cb=lab["Car Body"].values,sp=lab["Welding Spot"].values,date=lab["Date"].values.astype(str),y=lab["Fault"].values)
print(tv.shape, np.nanmin(tv),np.nanmax(tv),np.isnan(tv).sum(),np.isnan(ti).sum(),np.isnan(tf).sum())
print(lab.Fault.value_counts()); print(lab.Date.min(),lab.Date.max())
