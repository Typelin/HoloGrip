from __future__ import annotations
from pathlib import Path
import csv, sys, json
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"
LAB=ROOT/"HoloGrip_SpatialLab_0918"
sys.path.insert(0,str(APP))
sys.path.insert(0,str(LAB))
from product_hit_and_zone import ZONE_MAP, ZONE_NAMES
from vnext_core import compute_r0, make_relative_sample, feature_from_samples, r6

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
RAW5=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
GT5=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"

OFFSETS=list(range(-80,81,10))  # 17 points

def readcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

def prep_session(raw_path):
    rows=readcsv(raw_path)
    out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        times=np.array([float(r["song_time_ms"]) for r in rr])
        t0=float(times.min())
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        R0,zero=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
        ss=[]
        for r in rr:
            sm=make_relative_sample(
                R0,float(r["song_time_ms"]),
                float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
                float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])
            )
            ss.append(sm)
        out[h]={"R0":R0,"zero":zero,"samples":ss}
    return out

def nearest_sample(ss,target):
    # binary search-ish
    ts=np.fromiter((x["t"] for x in ss),float)
    j=int(np.searchsorted(ts,target))
    cand=[]
    for k in (j-1,j,j+1):
        if 0<=k<len(ss):cand.append(ss[k])
    if not cand:return None
    n=min(cand,key=lambda s:abs(s["t"]-target))
    if abs(n["t"]-target)>12.5:return None
    return n

def seq153(ss,center):
    vals=[]
    for off in OFFSETS:
        s=nearest_sample(ss,center+off)
        if s is None:return None
        vals.extend(s["alocal"].tolist())
        vals.extend(r6(s["Rrel"]).tolist())
    return np.asarray(vals,float)

def seq306(ss,center):
    base=seq153(ss,center)
    if base is None:return None
    A=base.reshape(len(OFFSETS),9)
    D=np.diff(A,axis=0,prepend=A[[0]])
    return np.concatenate([base,D.reshape(-1)])

def build_12(data, kind):
    gt=readcsv(GT12,"utf-8")
    X=[];y=[];h=[];groups=[]
    for r in gt:
        hh=r["final_hand"]; d=r["drum"]
        if hh not in ("L","R") or d not in ZONE_MAP:continue
        c=float(r["csv_center_ms"])
        ft=seq153(data[hh]["samples"],c) if kind=="seq153" else seq306(data[hh]["samples"],c)
        if ft is None:continue
        X.append(ft);y.append(ZONE_MAP[d]);h.append(hh);groups.append(int(c//4000))
    return np.asarray(X),np.asarray(y),np.asarray(h),np.asarray(groups)

def build_5(data, kind):
    gt=readcsv(GT5)
    X=[];y=[];h=[];meta=[]
    for r in gt:
        hh=r["assigned_hand"]; d=r["zone_name"]
        if hh not in ("L","R") or d not in ZONE_MAP:continue
        c=float(r["aligned_csv_time_ms"])
        ft=seq153(data[hh]["samples"],c) if kind=="seq153" else seq306(data[hh]["samples"],c)
        if ft is None:continue
        X.append(ft);y.append(ZONE_MAP[d]);h.append(hh);meta.append(r)
    return np.asarray(X),np.asarray(y),np.asarray(h),meta

def make_model(name):
    if name=="rbf":
        return Pipeline([("s",StandardScaler()),("m",SVC(C=3.0,gamma="scale",probability=True,random_state=42))])
    if name=="linear":
        return Pipeline([("s",StandardScaler()),("m",SVC(C=1.0,kernel="linear",probability=True,random_state=42))])
    if name=="pca_rbf":
        return Pipeline([("s",StandardScaler()),("p",PCA(n_components=0.95,random_state=42)),("m",SVC(C=3.0,gamma="scale",probability=True,random_state=42))])
    if name=="mlp":
        return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(96,48),max_iter=2200,random_state=42))])
    raise ValueError(name)

def metrics(y,p):
    return {
        "accuracy":float(accuracy_score(y,p)),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,average="macro")),
    }

def report(tag,y,p):
    m=metrics(y,p)
    print(f"{tag}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} macroF1={m['macro_f1']:.4f} n={len(y)}")
    for k,nm in enumerate(ZONE_NAMES):
        ix=np.where(y==k)[0]
        if len(ix):
            print(" ",k,nm,"n",len(ix),"acc",round(float(np.mean(p[ix]==k)),3),"pred",dict(Counter(p[ix].tolist())))
    return m

D12=prep_session(RAW12)
D5=prep_session(RAW5)
print("ZERO12",{h:[round(x,2) for x in D12[h]["zero"]] for h in ("L","R")})
print("ZERO5",{h:[round(x,2) for x in D5[h]["zero"]] for h in ("L","R")})

results={}
best=None
for kind in ("seq153","seq306"):
    X,y,h,g=build_12(D12,kind)
    X5,y5,h5,meta5=build_5(D5,kind)
    print("\n================",kind,X.shape,X5.shape,"================")
    results[kind]={}
    for model_name in ("rbf","linear","pca_rbf","mlp"):
        print("\n--",model_name,"--")
        # per-hand grouped CV
        p=np.full(len(y),-1,int)
        ok=True
        for hh in ("L","R"):
            idx=np.where(h==hh)[0]
            try:
                cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
                for tr0,te0 in cv.split(X[idx],y[idx],g[idx]):
                    m=make_model(model_name);m.fit(X[idx][tr0],y[idx][tr0]);p[idx[te0]]=m.predict(X[idx][te0])
            except Exception as e:
                print("CV_FAIL",hh,repr(e));ok=False;break
        if not ok:continue
        cvm=report("0812_GroupCV",y,p)

        # cross-session per-hand
        p5=np.full(len(y5),-1,int)
        models={}
        try:
            for hh in ("L","R"):
                tr=np.where(h==hh)[0];te=np.where(h5==hh)[0]
                m=make_model(model_name);m.fit(X[tr],y[tr]);p5[te]=m.predict(X5[te]);models[hh]=m
        except Exception as e:
            print("CROSS_FAIL",repr(e));continue
        xsm=report("0812_to_0805",y5,p5)
        results[kind][model_name]={"cv":cvm,"cross":xsm}

        score=xsm["balanced_accuracy"] + 0.25*xsm["accuracy"]
        if best is None or score>best[0]:
            best=(score,kind,model_name,xsm,cvm,models)

print("\n=== BEST ===")
print(best[1],best[2],"cross",best[3],"cv",best[4])

# Save best candidate
_,kind,model_name,xsm,cvm,models=best
out=LAB/"vnext_sequence_models"
out.mkdir(parents=True,exist_ok=True)
import joblib
for hh,m in models.items():
    joblib.dump(m,out/f"hologrip_vnext_seq_{hh}_{kind}_{model_name}.joblib")
meta={
    "version":"vNext-sequence-0918",
    "feature_kind":kind,
    "model_kind":model_name,
    "offsets_ms":OFFSETS,
    "channels":["local_ax","local_ay","local_az","r6_0","r6_1","r6_2","r6_3","r6_4","r6_5"],
    "cross_0805":xsm,
    "group_cv_0812":cvm,
    "all_results":results,
}
(out/"sequence_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
print("SAVED",out)
