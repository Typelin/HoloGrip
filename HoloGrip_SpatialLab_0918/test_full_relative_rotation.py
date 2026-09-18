from pathlib import Path
import csv,sys,math
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import ZONE_MAP,ZONE_NAMES

def load_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))

def cmean_deg(v):
    r=np.deg2rad(np.asarray(v,float))
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)),np.mean(np.cos(r)))))

def prep(path):
    rows=load_rows(path)
    by={h:[r for r in rows if r["hand"]==h] for h in ("L","R")}
    out={}
    for h,rr in by.items():
        t=np.array([float(r["song_time_ms"]) for r in rr])
        m=t<t.min()+1000
        y=np.array([float(r["yaw_deg"]) for r in rr])
        p=np.array([float(r["pitch_deg"]) for r in rr])
        ro=np.array([float(r["roll_deg"]) for r in rr])
        y0=cmean_deg(y[m]); p0=float(np.median(p[m])); r0=cmean_deg(ro[m])
        R0=Rot.from_euler("ZYX",[y0,p0,r0],degrees=True).as_matrix()
        samples=[]
        for r in rr:
            yy=float(r["yaw_deg"]);pp=float(r["pitch_deg"]);rrr=float(r["roll_deg"])
            Rc=Rot.from_euler("ZYX",[yy,pp,rrr],degrees=True).as_matrix()
            Rrel=R0.T@Rc
            a=np.array([float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"])],float)
            a0=R0.T@Rc@a
            samples.append({
                "t":float(r["song_time_ms"]),
                "mag":float(r["accel_magnitude_g"]),
                "arel":a0,
                "Rrel":Rrel,
            })
        out[h]={"R0":R0,"zero":(y0,p0,r0),"samples":samples}
    return out

def feat(ss,center,variant):
    w=[s for s in ss if center-80<=s["t"]<=center+80]
    if not w:return None
    mag=np.array([s["mag"] for s in w])
    A=np.vstack([s["arel"] for s in w])
    mats=np.stack([s["Rrel"] for s in w])
    near=min(w,key=lambda s:abs(s["t"]-center))
    def six(M): return M[:,:2].reshape(-1)
    center6=six(near["Rrel"])
    mean6=np.mean(np.stack([six(M) for M in mats]),axis=0)
    std6=np.std(np.stack([six(M) for M in mats]),axis=0)
    angles=np.array([Rot.from_matrix(M).magnitude() for M in mats])
    base=[
        np.max(mag),np.mean(mag),np.std(mag),np.sum(np.abs(mag-1.0)),
        np.max(np.abs(np.diff(mag))) if len(mag)>1 else 0.0,
        *np.max(np.abs(A),axis=0).tolist(),
        *np.mean(A,axis=0).tolist(),
        float(np.std(angles)),float(np.max(angles)-np.min(angles)),
    ]
    if variant=="relrot":
        return np.asarray(base+center6.tolist()+mean6.tolist(),float)
    if variant=="relrot_std":
        return np.asarray(base+center6.tolist()+mean6.tolist()+std6.tolist(),float)
    if variant=="noacc":
        return np.asarray([
            np.max(mag),np.mean(mag),np.std(mag),np.sum(np.abs(mag-1.0)),
            np.max(np.abs(np.diff(mag))) if len(mag)>1 else 0.0,
            float(np.std(angles)),float(np.max(angles)-np.min(angles)),
            *center6.tolist(),*mean6.tolist(),*std6.tolist()
        ],float)
    raise ValueError(variant)

D12=prep(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv")
D5=prep(ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv")
print("zero 0812",{h:D12[h]["zero"] for h in D12})
print("zero 0805",{h:D5[h]["zero"] for h in D5})

with (ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv").open(encoding="utf-8",newline="") as f:G12=list(csv.DictReader(f))
with (ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv").open(encoding="utf-8-sig",newline="") as f:G5=list(csv.DictReader(f))

def build12(var):
    X=[];y=[];h=[];g=[]
    for r in G12:
        hh=r["final_hand"];d=r["drum"]
        if hh not in ("L","R"):continue
        c=float(r["csv_center_ms"]);ft=feat(D12[hh]["samples"],c,var)
        if ft is not None:X.append(ft);y.append(ZONE_MAP[d]);h.append(hh);g.append(int(c//4000))
    return np.asarray(X),np.asarray(y),np.asarray(h),np.asarray(g)
def build5(var):
    X=[];y=[];h=[]
    for r in G5:
        hh=r["assigned_hand"];d=r["zone_name"];c=float(r["aligned_csv_time_ms"]);ft=feat(D5[hh]["samples"],c,var)
        if ft is not None:X.append(ft);y.append(ZONE_MAP[d]);h.append(hh)
    return np.asarray(X),np.asarray(y),np.asarray(h)
def model(kind):
    if kind=="mlp": return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1800,random_state=42))])
    return Pipeline([("s",StandardScaler()),("m",SVC(C=3.0,gamma="scale"))])
def score(tag,y,p):
    print(tag,"acc",round(accuracy_score(y,p),4),"bal",round(balanced_accuracy_score(y,p),4),"F1",round(f1_score(y,p,average="macro"),4))
    for k,nm in enumerate(ZONE_NAMES):
        ix=np.where(y==k)[0]
        if len(ix): print(" ",k,nm,len(ix),round(float(np.mean(p[ix]==k)),3))

for var in ("relrot","relrot_std","noacc"):
  for kind in ("mlp","svc"):
    print("\n###",var,kind)
    X,y,h,g=build12(var);X5,y5,h5=build5(var)
    # separate hand cross-session
    p5=np.full(len(y5),-1,int)
    for hh in ("L","R"):
        tr=np.where(h==hh)[0];te=np.where(h5==hh)[0]
        m=model(kind);m.fit(X[tr],y[tr]);p5[te]=m.predict(X5[te])
    score("0805",y5,p5)
    # CV separate hands
    p=np.full(len(y),-1,int)
    for hh in ("L","R"):
        idx=np.where(h==hh)[0]
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        for tr0,te0 in cv.split(X[idx],y[idx],g[idx]):
            tr=idx[tr0];te=idx[te0];m=model(kind);m.fit(X[tr],y[tr]);p[te]=m.predict(X[te])
    score("0812_CV",y,p)
