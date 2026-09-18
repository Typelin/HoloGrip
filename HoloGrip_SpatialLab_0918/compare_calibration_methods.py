from __future__ import annotations
from pathlib import Path
import csv,sys,math
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP))
from product_hit_and_zone import ZONE_MAP

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
RAW5=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
GT5=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"

def readcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

def circ_mean_deg(vals):
    a=np.asarray(vals,float);r=np.deg2rad(a)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)),np.mean(np.cos(r)))))

def compute_r0(eulers,method):
    a=np.asarray(eulers,float)
    if method=="component":
        return Rot.from_euler("ZYX",[circ_mean_deg(a[:,0]),float(np.median(a[:,1])),circ_mean_deg(a[:,2])],degrees=True).as_matrix()
    if method=="rotmean":
        return Rot.from_euler("ZYX",a,degrees=True).mean().as_matrix()
    raise ValueError(method)

def prep(raw_path,dur_s,method):
    rows=readcsv(raw_path)
    out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+dur_s*1000]
        R0=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal],method)
        ss=[]
        for r in rr:
            t=float(r["song_time_ms"]);Rc=Rot.from_euler("ZYX",[float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])],degrees=True).as_matrix()
            Rrel=R0.T@Rc
            acc=np.array([float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"])])
            alocal=R0.T@Rc@acc
            ss.append({"t":t,"mag":float(np.linalg.norm(acc)),"alocal":alocal,"Rrel":Rrel})
        out[h]=ss
    return out

def r6(M):return M[:,:2].reshape(-1)
def feat(ss,c):
    w=[s for s in ss if c-80<=s["t"]<=c+80]
    if not w:return None
    mag=np.array([s["mag"] for s in w]);A=np.vstack([s["alocal"] for s in w]);mats=np.stack([s["Rrel"] for s in w])
    near=min(w,key=lambda s:abs(s["t"]-c));r6s=np.stack([r6(M) for M in mats]);angles=np.array([Rot.from_matrix(M).magnitude() for M in mats])
    return np.asarray([
      np.max(mag),np.mean(mag),np.std(mag),np.sum(np.abs(mag-1)),
      np.max(np.abs(np.diff(mag))) if len(mag)>1 else 0,
      *np.max(np.abs(A),axis=0),*np.mean(A,axis=0),
      float(np.std(angles)),float(np.max(angles)-np.min(angles)),
      *r6(near["Rrel"]),*np.mean(r6s,axis=0),*np.std(r6s,axis=0)
    ],float)

def build12(D):
    gt=readcsv(GT12,"utf-8");X=[];y=[];h=[];g=[]
    for r in gt:
        hh=r["final_hand"];d=r["drum"]
        if hh not in ("L","R"):continue
        c=float(r["csv_center_ms"]);f=feat(D[hh],c)
        if f is not None:X.append(f);y.append(ZONE_MAP[d]);h.append(hh);g.append(int(c//4000))
    return np.array(X),np.array(y),np.array(h),np.array(g)
def build5(D):
    gt=readcsv(GT5);X=[];y=[];h=[]
    for r in gt:
        hh=r["assigned_hand"];d=r["zone_name"];c=float(r["aligned_csv_time_ms"]);f=feat(D[hh],c)
        if f is not None:X.append(f);y.append(ZONE_MAP[d]);h.append(hh)
    return np.array(X),np.array(y),np.array(h)

def model():return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1800,random_state=42))])

for dur in (0.5,1.0,2.0,3.0):
  for method in ("component","rotmean"):
    D12=prep(RAW12,dur,method);D5=prep(RAW5,dur,method)
    X,y,h,g=build12(D12);X5,y5,h5=build5(D5)
    # cross
    p5=np.full(len(y5),-1,int)
    for hh in ("L","R"):
        tr=np.where(h==hh)[0];te=np.where(h5==hh)[0];m=model();m.fit(X[tr],y[tr]);p5[te]=m.predict(X5[te])
    cross=(accuracy_score(y5,p5),balanced_accuracy_score(y5,p5),f1_score(y5,p5,average="macro"))
    # CV
    p=np.full(len(y),-1,int)
    for hh in ("L","R"):
        idx=np.where(h==hh)[0];cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        for tr0,te0 in cv.split(X[idx],y[idx],g[idx]):
            m=model();m.fit(X[idx][tr0],y[idx][tr0]);p[idx[te0]]=m.predict(X[idx][te0])
    cvm=(accuracy_score(y,p),balanced_accuracy_score(y,p),f1_score(y,p,average="macro"))
    print("dur",dur,"method",method,
          "CROSS acc/bal/f1",*[round(x,4) for x in cross],
          "CV",*[round(x,4) for x in cvm])
