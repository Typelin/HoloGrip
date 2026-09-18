from pathlib import Path
import csv,sys
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"; sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows,rows_to_samples,slice_window,window_features,ZONE_MAP,ZONE_NAMES
SEED=42

# 0812
s12=rows_to_samples(load_raw_rows(str(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv")))
with (ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv").open(encoding="utf-8",newline="") as f:g12=list(csv.DictReader(f))
X=[];y=[];hand=[];grp=[]
for r in g12:
    h=r["final_hand"]; d=r["drum"]
    if h not in ("L","R"):continue
    c=float(r["csv_center_ms"])
    ft=window_features(slice_window(s12[h],c,80.0),c)
    if ft is None:continue
    X.append(ft); y.append(ZONE_MAP[d]); hand.append(h); grp.append(int(c//4000))
X=np.asarray(X,float); y=np.asarray(y); hand=np.asarray(hand); grp=np.asarray(grp)

# 0805 test
s5=rows_to_samples(load_raw_rows(str(ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv")))
with (ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv").open(encoding="utf-8-sig",newline="") as f:g5=list(csv.DictReader(f))
X5=[];y5=[];h5=[]
for r in g5:
    h=r["assigned_hand"]; d=r["zone_name"]; c=float(r["aligned_csv_time_ms"])
    if h not in ("L","R") or d not in ZONE_MAP:continue
    ft=window_features(slice_window(s5[h],c,80.0),c)
    if ft is None:continue
    X5.append(ft); y5.append(ZONE_MAP[d]); h5.append(h)
X5=np.asarray(X5,float); y5=np.asarray(y5); h5=np.asarray(h5)

def make():
    return Pipeline([("scaler",StandardScaler()),("mlp",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1600,random_state=SEED))])

def report(name, truth,pred):
    print(name,"n",len(truth),"acc",round(accuracy_score(truth,pred),4),"bal",round(balanced_accuracy_score(truth,pred),4),"macroF1",round(f1_score(truth,pred,average="macro"),4))
    # numeric confusion summary per class
    for k,nm in enumerate(ZONE_NAMES):
        ix=np.where(truth==k)[0]
        if len(ix):
            ok=np.mean(pred[ix]==k)
            print(" ",k,nm,"n",len(ix),"acc",round(float(ok),3),"pred",Counter(pred[ix].tolist()))

# A original grouped CV
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
pred=np.full(len(y),-1,int)
for tr,te in cv.split(X,y,grp):
    m=make(); m.fit(X[tr],y[tr]); pred[te]=m.predict(X[te])
report("A_original_CV",y,pred)
m=make();m.fit(X,y);report("A_original_0805",y5,m.predict(X5))

# B + hand flag
hf=np.where(hand=="R",1.0,-1.0)[:,None]; hf5=np.where(h5=="R",1.0,-1.0)[:,None]
Xh=np.hstack([X,hf]); X5h=np.hstack([X5,hf5])
pred=np.full(len(y),-1,int)
for tr,te in cv.split(Xh,y,grp):
    m=make();m.fit(Xh[tr],y[tr]);pred[te]=m.predict(Xh[te])
report("B_handflag_CV",y,pred)
m=make();m.fit(Xh,y);report("B_handflag_0805",y5,m.predict(X5h))

# C separate models by hand
pred=np.full(len(y),-1,int)
for h in ("L","R"):
    idx=np.where(hand==h)[0]
    # group CV within hand
    cvh=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    for tr0,te0 in cvh.split(X[idx],y[idx],grp[idx]):
        tr=idx[tr0]; te=idx[te0]
        m=make();m.fit(X[tr],y[tr]);pred[te]=m.predict(X[te])
report("C_separate_CV",y,pred)

pred5=np.full(len(y5),-1,int)
for h in ("L","R"):
    tr=np.where(hand==h)[0]; te=np.where(h5==h)[0]
    m=make();m.fit(X[tr],y[tr]);pred5[te]=m.predict(X5[te])
report("C_separate_0805",y5,pred5)
