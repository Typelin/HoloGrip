from pathlib import Path
import csv, json, sys
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
S=LAB/"protocol_sessions/20260918_155836"
sys.path.insert(0,str(LAB))
from vnext_core import make_relative_sample, feature_from_samples

STAGE_TO_COARSE={2:0,4:1,6:2,8:0,10:2,12:2}
COARSE_NAMES=["LEFT","CENTER","RIGHT"]

def readcsv(n):
    with (S/n).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
raw=readcsv("raw_100hz.csv"); hits=readcsv("hits.csv")
cal=json.loads((S/"calibration.json").read_text(encoding="utf-8"))

R0={}
for h in ("L","R"):
    y,p,r=cal[h]["zero"]
    R0[h]=Rot.from_euler("ZYX",[y,p,r],degrees=True).as_matrix()

samples={"L":[],"R":[]}
for row in raw:
    idx=int(row["stage_index"])
    if idx not in STAGE_TO_COARSE or row["frame_valid"]!="1":continue
    h=row["hand"]
    sm=make_relative_sample(R0[h],float(row["protocol_elapsed_s"])*1000.0,
        float(row["ax"]),float(row["ay"]),float(row["az"]),
        float(row["yaw"]),float(row["pitch"]),float(row["roll"]))
    sm["stage"]=idx
    samples[h].append(sm)
for h in samples:samples[h].sort(key=lambda x:x["t"])

hh={"L":defaultdict(list),"R":defaultdict(list)}
for row in hits:
    idx=int(row["stage_index"])
    if idx not in STAGE_TO_COARSE:continue
    x=dict(row);x["pe"]=float(row["protocol_elapsed_s"])*1000.0
    hh[row["hand"]][idx].append(x)
for h in hh:
    for idx in hh[h]:hh[h][idx].sort(key=lambda x:x["pe"])

def dedup(lst,min_gap=180):
    out=[];last=-1e18
    for x in lst:
        if x["pe"]-last>=min_gap:
            out.append(x);last=x["pe"]
    return out

def build(hand):
    X=[];y=[];stage=[];t=[]
    for idx,c in STAGE_TO_COARSE.items():
        hs=dedup(hh[hand][idx])
        ss=[s for s in samples[hand] if s["stage"]==idx]
        for hit in hs:
            center=hit["pe"]-80.0
            f=feature_from_samples(ss,center,80.0)
            if f is None:continue
            X.append(f);y.append(c);stage.append(idx);t.append(hit["pe"])
    return np.asarray(X),np.asarray(y),np.asarray(stage),np.asarray(t)

def make():
    return Pipeline([("s",StandardScaler()),("m",SVC(C=2.0,kernel="rbf",gamma="scale",probability=True,random_state=42))])

out=LAB/"coarse_direction_models"
out.mkdir(parents=True,exist_ok=True)
meta={"coarse_names":COARSE_NAMES,"stage_to_coarse":STAGE_TO_COARSE,"hands":{}}

for hand in ("L","R"):
    X,y,stage,t=build(hand)
    print("\nHAND",hand,"n",len(y),"counts",Counter(y.tolist()),"stagecounts",Counter(stage.tolist()))
    # chronological 60/40 split within each original stage to avoid training/testing same hit neighborhoods indiscriminately
    tr=[];te=[]
    for idx in sorted(set(stage)):
        ix=np.where(stage==idx)[0]
        ix=ix[np.argsort(t[ix])]
        n=max(1,int(round(len(ix)*0.6)))
        if n>=len(ix): n=max(1,len(ix)-1)
        tr.extend(ix[:n]);te.extend(ix[n:])
    tr=np.asarray(tr);te=np.asarray(te)
    m=make();m.fit(X[tr],y[tr]);p=m.predict(X[te])
    acc=float(accuracy_score(y[te],p));bal=float(balanced_accuracy_score(y[te],p));f1=float(f1_score(y[te],p,average="macro"))
    print("3WAY chronological test",len(te),"acc",acc,"bal",bal,"f1",f1)
    print(confusion_matrix(y[te],p,labels=[0,1,2]))
    # catastrophic opposite-side error: true LEFT->RIGHT or RIGHT->LEFT
    opp=int(np.sum(((y[te]==0)&(p==2))|((y[te]==2)&(p==0))))
    side_n=int(np.sum((y[te]==0)|(y[te]==2)))
    print("OPPOSITE_SIDE",opp,"/",side_n, "rate",opp/side_n if side_n else None)

    # Fit full data for demo
    mf=make();mf.fit(X,y)
    path=out/f"coarse_{hand}.joblib";joblib.dump(mf,path)
    meta["hands"][hand]={
        "n":int(len(y)),"counts":{COARSE_NAMES[k]:int(np.sum(y==k)) for k in range(3)},
        "chronological_acc":acc,"chronological_bal":bal,"chronological_f1":f1,
        "opposite_side_errors":opp,"opposite_side_denominator":side_n,
        "model_path":str(path)
    }

(out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nSAVED",out)
