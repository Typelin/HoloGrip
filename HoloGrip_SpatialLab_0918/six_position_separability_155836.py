from pathlib import Path
import csv,json,sys
from collections import Counter,defaultdict
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
S=LAB/"protocol_sessions/20260918_155836"
sys.path.insert(0,str(LAB))
from vnext_core import make_relative_sample, feature_from_samples, r6

TARGETS={2:"左上",4:"中上",6:"右上",8:"左中",10:"右中",12:"右中偏高"}
cal=json.loads((S/"calibration.json").read_text(encoding="utf-8"))

def readcsv(n):
    with (S/n).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
raw=readcsv("raw_100hz.csv"); hits=readcsv("hits.csv")

R0={}
for h in ("L","R"):
    y,p,r=cal[h]["zero"]
    R0[h]=Rot.from_euler("ZYX",[y,p,r],degrees=True).as_matrix()

# relative sample timeline from valid target-stage raw
SAMP={"L":[],"R":[]}
for row in raw:
    idx=int(row["stage_index"])
    if idx not in TARGETS or row["frame_valid"]!="1":continue
    h=row["hand"]
    pe=float(row["protocol_elapsed_s"])*1000.0
    sm=make_relative_sample(R0[h],pe,
        float(row["ax"]),float(row["ay"]),float(row["az"]),
        float(row["yaw"]),float(row["pitch"]),float(row["roll"]))
    sm["stage"]=idx
    SAMP[h].append(sm)
for h in SAMP:SAMP[h].sort(key=lambda s:s["t"])

HITS={"L":defaultdict(list),"R":defaultdict(list)}
for row in hits:
    idx=int(row["stage_index"])
    if idx not in TARGETS:continue
    x=dict(row);x["pe"]=float(row["protocol_elapsed_s"])*1000.0;x["dt"]=datetime.fromisoformat(row["wall_time"])
    HITS[row["hand"]][idx].append(x)
for h in HITS:
    for idx in HITS[h]:HITS[h][idx].sort(key=lambda x:x["pe"])

def dedup(lst,min_gap_ms=180):
    out=[];last=-1e18
    for x in lst:
        if x["pe"]-last>=min_gap_ms:
            out.append(x);last=x["pe"]
    return out

def seq_feature(samples,center):
    # exact-ish 100Hz trajectory: 17 offsets [-80,+80], local accel 3 + rotation6
    arr=[]
    for off in range(-80,81,10):
        target=center+off
        # local search brute force small enough
        n=min(samples,key=lambda s:abs(s["t"]-target))
        if abs(n["t"]-target)>12:return None
        arr.extend(n["alocal"].tolist());arr.extend(r6(n["Rrel"]).tolist())
    return np.asarray(arr,float)

def build(hand,use_dedup):
    X31=[];Xseq=[];y=[];times=[]
    for cls,idx in enumerate(TARGETS):
        hh=HITS[hand][idx]
        if use_dedup:hh=dedup(hh)
        ss=[s for s in SAMP[hand] if s["stage"]==idx]
        for hit in hh:
            center=hit["pe"]-80.0 # logged when +80ms window available
            f=feature_from_samples(ss,center,80.0)
            q=seq_feature(ss,center)
            if f is None or q is None:continue
            X31.append(f);Xseq.append(q);y.append(cls);times.append(hit["pe"])
    return np.array(X31),np.array(Xseq),np.array(y),np.array(times)

def chrono_split(y,times):
    tr=[];te=[]
    for c in sorted(set(y)):
        ix=np.where(y==c)[0]
        ix=ix[np.argsort(times[ix])]
        n=max(1,int(round(len(ix)*0.6)))
        if n>=len(ix):n=len(ix)-1
        tr.extend(ix[:n]);te.extend(ix[n:])
    return np.array(tr),np.array(te)

def eval_model(tag,X,y,times,kind):
    tr,te=chrono_split(y,times)
    if kind=="svc":
        m=Pipeline([("s",StandardScaler()),("m",SVC(C=3.0,gamma="scale"))])
    elif kind=="linear":
        m=Pipeline([("s",StandardScaler()),("m",SVC(C=1.0,kernel="linear"))])
    else:
        m=RandomForestClassifier(n_estimators=400,max_depth=None,min_samples_leaf=1,random_state=42,class_weight="balanced")
    m.fit(X[tr],y[tr]);p=m.predict(X[te])
    print(tag,kind,"train",len(tr),"test",len(te),"acc",round(accuracy_score(y[te],p),3),"bal",round(balanced_accuracy_score(y[te],p),3))
    print(" test counts",Counter(y[te]),"pred",Counter(p))
    print(" cm rows true 0..5")
    print(confusion_matrix(y[te],p,labels=list(range(6))))
    return accuracy_score(y[te],p)

for ded in (False,True):
    print("\n================ DEDUP",ded,"================")
    for hand in ("L","R"):
        X31,Xseq,y,t=build(hand,ded)
        print("\nHAND",hand,"n",len(y),"class counts",{TARGETS[list(TARGETS)[c]]:int(np.sum(y==c)) for c in range(6)})
        for kind in ("svc","linear","rf"):
            eval_model(hand+" 31D",X31,y,t,kind)
        for kind in ("svc","linear"):
            eval_model(hand+" SEQ153",Xseq,y,t,kind)
