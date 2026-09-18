from pathlib import Path
import csv,sys,math
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows,rows_to_samples,ZONE_MAP,ZONE_NAMES
SEED=42

def circmean(a):
    r=np.deg2rad(np.asarray(a,float))
    return float(np.arctan2(np.mean(np.sin(r)),np.mean(np.cos(r)))) # radians

def features(samples,center,hand,mode):
    s=[x for x in samples if center-80<=x["song_time_ms"]<=center+80]
    if not s:return None
    mag=np.asarray([x["mag"] for x in s],float)
    yaw=np.asarray([x["yaw"] for x in s],float)
    pitch=np.asarray([x["pitch"] for x in s],float)
    yu=np.rad2deg(np.unwrap(np.deg2rad(yaw)))
    near=min(s,key=lambda x:abs(x["song_time_ms"]-center))
    # coordinate-invariant motion primitives
    base=[
        np.max(mag),np.mean(mag),np.std(mag),np.sum(np.abs(mag-1.0)),
        np.max(np.abs(np.diff(mag))) if len(mag)>1 else 0.0,
        np.std(yu),np.std(pitch),np.max(yu)-np.min(yu),np.max(pitch)-np.min(pitch),
    ]
    ang=circmean(yaw)
    cy=math.radians(float(near["yaw"]))
    pose=[
        math.sin(ang),math.cos(ang),np.mean(pitch),
        math.sin(cy),math.cos(cy),float(near["pitch"]),
    ]
    hf=[1.0 if hand=="R" else -1.0]
    if mode=="motion": return np.asarray(base+hf,float)
    if mode=="pose": return np.asarray(pose+hf,float)
    if mode=="motion_pose": return np.asarray(base+pose+hf,float)
    raise ValueError(mode)

# datasets
s12=rows_to_samples(load_raw_rows(str(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv")))
with (ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv").open(encoding="utf-8",newline="") as f:g12=list(csv.DictReader(f))
s5=rows_to_samples(load_raw_rows(str(ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv")))
with (ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv").open(encoding="utf-8-sig",newline="") as f:g5=list(csv.DictReader(f))

def build12(mode):
    X=[];y=[];g=[];h=[]
    for r in g12:
        hh=r["final_hand"];d=r["drum"]
        if hh not in ("L","R"):continue
        c=float(r["csv_center_ms"]);ft=features(s12[hh],c,hh,mode)
        if ft is not None:X.append(ft);y.append(ZONE_MAP[d]);g.append(int(c//4000));h.append(hh)
    return np.asarray(X),np.asarray(y),np.asarray(g),np.asarray(h)

def build5(mode):
    X=[];y=[];h=[];meta=[]
    for r in g5:
        hh=r["assigned_hand"];d=r["zone_name"]
        if hh not in ("L","R") or d not in ZONE_MAP:continue
        c=float(r["aligned_csv_time_ms"]);ft=features(s5[hh],c,hh,mode)
        if ft is not None:X.append(ft);y.append(ZONE_MAP[d]);h.append(hh);meta.append(r)
    return np.asarray(X),np.asarray(y),np.asarray(h),meta

def model():
    return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1800,random_state=SEED))])

def score(tag,y,p):
    print(tag,"acc",round(accuracy_score(y,p),4),"bal",round(balanced_accuracy_score(y,p),4),"F1",round(f1_score(y,p,average="macro"),4))
    for k,n in enumerate(ZONE_NAMES):
        ix=np.where(y==k)[0]
        if len(ix): print(" ",k,n,"n",len(ix),"acc",round(float(np.mean(p[ix]==k)),3))

for mode in ("motion","pose","motion_pose"):
    print("\n###",mode)
    X,y,g,h=build12(mode);X5,y5,h5,meta=build5(mode)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    p=np.full(len(y),-1,int)
    for tr,te in cv.split(X,y,g):
        m=model();m.fit(X[tr],y[tr]);p[te]=m.predict(X[te])
    score("CV",y,p)
    m=model();m.fit(X,y);p5=m.predict(X5);score("0805",y5,p5)
    # separate hand models with same feature set minus redundant hand flag harmless
    ps=np.full(len(y5),-1,int)
    for hh in ("L","R"):
        tr=np.where(h==hh)[0];te=np.where(h5==hh)[0]
        mm=model();mm.fit(X[tr],y[tr]);ps[te]=mm.predict(X5[te])
    score("0805_separate",y5,ps)
    # clean 0805 subset: isolated and very dominant
    clean=np.array([
        (float(r["dominant_hand_share"])>=0.95 and
         (not r["previous_onset_gap_ms"] or float(r["previous_onset_gap_ms"])>=300) and
         (not r["next_onset_gap_ms"] or float(r["next_onset_gap_ms"])>=300))
        for r in meta
    ])
    if clean.any():
        score("0805_clean",y5[clean],ps[clean])
        print(" clean_n",int(clean.sum()))
