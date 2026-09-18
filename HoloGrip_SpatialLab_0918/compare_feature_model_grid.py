from __future__ import annotations
from pathlib import Path
import csv,sys,math,json
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
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

NAMES=[
"max_mag","mean_mag","std_mag","energy","jerk_max",
"max_lax","max_lay","max_laz","mean_lax","mean_lay","mean_laz",
"rot_std","rot_range",
*["center_r6_"+str(i) for i in range(6)],
*["mean_r6_"+str(i) for i in range(6)],
*["std_r6_"+str(i) for i in range(6)],
]

def read(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
def cmean(v):
    a=np.deg2rad(np.asarray(v,float));return float(np.rad2deg(np.arctan2(np.mean(np.sin(a)),np.mean(np.cos(a)))))
def r6(M):return M[:,:2].reshape(-1)
def prep(p):
    rows=read(p);out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h];t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+500]
        a=np.array([[float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])] for r in cal])
        R0=Rot.from_euler("ZYX",[cmean(a[:,0]),float(np.median(a[:,1])),cmean(a[:,2])],degrees=True).as_matrix()
        ss=[]
        for r in rr:
            t=float(r["song_time_ms"]);Rc=Rot.from_euler("ZYX",[float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])],degrees=True).as_matrix()
            Rrel=R0.T@Rc;A=np.array([float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"])]);al=R0.T@Rc@A
            ss.append((t,float(np.linalg.norm(A)),al,Rrel))
        out[h]=ss
    return out
def feat(ss,c):
    w=[s for s in ss if c-80<=s[0]<=c+80]
    if not w:return None
    mag=np.array([s[1] for s in w]);A=np.vstack([s[2] for s in w]);M=np.stack([s[3] for s in w])
    near=min(w,key=lambda s:abs(s[0]-c));R6=np.stack([r6(x) for x in M]);ang=np.array([Rot.from_matrix(x).magnitude() for x in M])
    return np.asarray([np.max(mag),np.mean(mag),np.std(mag),np.sum(np.abs(mag-1)),np.max(np.abs(np.diff(mag))) if len(mag)>1 else 0,
      *np.max(np.abs(A),axis=0),*np.mean(A,axis=0),np.std(ang),np.max(ang)-np.min(ang),
      *r6(near[3]),*np.mean(R6,axis=0),*np.std(R6,axis=0)],float)
def build12(D):
    gt=read(GT12,"utf-8");X=[];y=[];h=[];g=[]
    for r in gt:
        hh=r["final_hand"];d=r["drum"]
        if hh not in ("L","R"):continue
        c=float(r["csv_center_ms"]);f=feat(D[hh],c)
        if f is not None:X.append(f);y.append(ZONE_MAP[d]);h.append(hh);g.append(int(c//4000))
    return np.array(X),np.array(y),np.array(h),np.array(g)
def build5(D):
    gt=read(GT5);X=[];y=[];h=[]
    for r in gt:
        hh=r["assigned_hand"];d=r["zone_name"];c=float(r["aligned_csv_time_ms"]);f=feat(D[hh],c)
        if f is not None:X.append(f);y.append(ZONE_MAP[d]);h.append(hh)
    return np.array(X),np.array(y),np.array(h)

D12=prep(RAW12);D5=prep(RAW5);X,y,h,g=build12(D12);X5,y5,h5=build5(D5)
print("shape",X.shape,X5.shape)

idx={
"full":np.arange(31),
"no_local_accel":np.r_[0:5,11:31],
"no_mean_local_accel":np.r_[0:8,11:31],
"rotation_only":np.r_[11:31],
"pose_no_std":np.r_[11:25],
}

def make(name):
    if name=="mlp":return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1800,random_state=42))])
    if name=="linear":return Pipeline([("s",StandardScaler()),("m",SVC(C=1.0,kernel="linear",probability=False,random_state=42))])
    if name=="rbf":return Pipeline([("s",StandardScaler()),("m",SVC(C=3.0,gamma="scale",probability=False,random_state=42))])
    if name=="logreg":return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1.0,max_iter=3000,class_weight=None,random_state=42))])
    if name=="extra":return ExtraTreesClassifier(n_estimators=400,min_samples_leaf=1,max_features="sqrt",class_weight="balanced",random_state=42,n_jobs=-1)
def metrics(a,b):return accuracy_score(a,b),balanced_accuracy_score(a,b),f1_score(a,b,average="macro")

res=[]
for fn,cols in idx.items():
  for mn in ("mlp","linear","rbf","logreg","extra"):
    A=X[:,cols];B=X5[:,cols]
    # cross
    p5=np.full(len(y5),-1,int)
    for hh in ("L","R"):
      tr=np.where(h==hh)[0];te=np.where(h5==hh)[0];m=make(mn);m.fit(A[tr],y[tr]);p5[te]=m.predict(B[te])
    cross=metrics(y5,p5)
    # group CV
    p=np.full(len(y),-1,int)
    for hh in ("L","R"):
      ii=np.where(h==hh)[0];cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
      for tr0,te0 in cv.split(A[ii],y[ii],g[ii]):
        m=make(mn);m.fit(A[ii][tr0],y[ii][tr0]);p[ii[te0]]=m.predict(A[ii][te0])
    cvm=metrics(y,p)
    score=cross[1]+0.25*cross[0]+0.15*cvm[1]
    res.append((score,fn,mn,cross,cvm,len(cols)))
    print(fn,mn,"nfeat",len(cols),"cross",*[round(v,4) for v in cross],"cv",*[round(v,4) for v in cvm])
print("\n=== TOP ===")
for r in sorted(res,reverse=True)[:10]:print(r)
