from pathlib import Path
import csv,sys
from collections import Counter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import rows_to_samples,slice_window,window_features,ZONE_MAP,ZONE_NAMES
def wrap(x):return (x+180)%360-180
roll0={"0812":{"L":22.41,"R":-42.68},"0805":{"L":24.58,"R":-16.55}}
def load(path,key):
    with path.open(encoding="utf-8-sig",newline="") as f:rr=list(csv.DictReader(f))
    out=[]
    for r in rr:
        q=dict(r);q["roll_deg"]=str(wrap(float(r["roll_deg"])-roll0[key][r["hand"]]));out.append(q)
    return rows_to_samples(out)
s12=load(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv","0812")
with (ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv").open(encoding="utf-8",newline="") as f:g12=list(csv.DictReader(f))
X=[];y=[];h=[];g=[]
for r in g12:
 hh=r["final_hand"];d=r["drum"]
 if hh not in ("L","R"):continue
 c=float(r["csv_center_ms"]);ft=window_features(slice_window(s12[hh],c,80),c)
 if ft is not None:X.append(ft);y.append(ZONE_MAP[d]);h.append(hh);g.append(int(c//4000))
X=np.asarray(X);y=np.asarray(y);h=np.asarray(h);g=np.asarray(g)
s5=load(ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv","0805")
with (ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv").open(encoding="utf-8-sig",newline="") as f:g5=list(csv.DictReader(f))
X5=[];y5=[];h5=[]
for r in g5:
 hh=r["assigned_hand"];d=r["zone_name"];c=float(r["aligned_csv_time_ms"]);ft=window_features(slice_window(s5[hh],c,80),c)
 if ft is not None:X5.append(ft);y5.append(ZONE_MAP[d]);h5.append(hh)
X5=np.asarray(X5);y5=np.asarray(y5);h5=np.asarray(h5)
def model():return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1600,random_state=42))])
def report(tag,yy,pp):
 print("\n",tag,"n",len(yy),"acc",round(accuracy_score(yy,pp),4),"bal",round(balanced_accuracy_score(yy,pp),4),"F1",round(f1_score(yy,pp,average="macro"),4))
 for k,nm in enumerate(ZONE_NAMES):
  ix=np.where(yy==k)[0]
  if len(ix):print(k,nm,"n",len(ix),"acc",round(float(np.mean(pp[ix]==k)),3),"pred",Counter(pp[ix].tolist()))
# mixed
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
p=np.full(len(y),-1,int)
for tr,te in cv.split(X,y,g):
 m=model();m.fit(X[tr],y[tr]);p[te]=m.predict(X[te])
report("relative_roll_mixed_CV",y,p)
m=model();m.fit(X,y);report("relative_roll_mixed_0805",y5,m.predict(X5))
# + handflag
Xh=np.hstack([X,np.where(h=="R",1.0,-1.0)[:,None]])
X5h=np.hstack([X5,np.where(h5=="R",1.0,-1.0)[:,None]])
p=np.full(len(y),-1,int)
for tr,te in cv.split(Xh,y,g):
 m=model();m.fit(Xh[tr],y[tr]);p[te]=m.predict(Xh[te])
report("relative_roll_handflag_CV",y,p)
m=model();m.fit(Xh,y);report("relative_roll_handflag_0805",y5,m.predict(X5h))
# separate
p=np.full(len(y),-1,int)
for hh in ("L","R"):
 idx=np.where(h==hh)[0]
 cvh=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
 for tr0,te0 in cvh.split(X[idx],y[idx],g[idx]):
  tr=idx[tr0];te=idx[te0];m=model();m.fit(X[tr],y[tr]);p[te]=m.predict(X[te])
report("relative_roll_separate_CV",y,p)
p5=np.full(len(y5),-1,int)
for hh in ("L","R"):
 tr=np.where(h==hh)[0];te=np.where(h5==hh)[0];m=model();m.fit(X[tr],y[tr]);p5[te]=m.predict(X5[te])
report("relative_roll_separate_0805",y5,p5)
