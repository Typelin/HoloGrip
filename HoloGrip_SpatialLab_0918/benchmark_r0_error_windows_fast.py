from pathlib import Path
import csv,sys,joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.metrics import accuracy_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"; APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,ZONE_NAMES
ZMAP={n:i for i,n in enumerate(ZONE_NAMES)}
M={h:joblib.load(LAB/"vnext_timing_robust_aug20"/f"hologrip_timing_robust_{h}.joblib") for h in ("L","R")}

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"

def rcsv(p,enc="utf-8-sig"):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=rcsv(RAW);D={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h];t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]

events=[]
for g in rcsv(GT):
    h=g["assigned_hand"];d=g["zone_name"]
    if h not in D or d not in ZMAP:continue
    t=float(g["aligned_csv_time_ms"]); w=[s for s in D[h] if t-80<=s["t"]<=t+80]
    if w:events.append((h,ZMAP[d],t,w))

def r6(M): return M[:,:2].reshape(-1)
def feat(w,t,Q):
    QT=Q.T
    mag=np.array([s["mag"] for s in w])
    A=np.stack([QT@s["alocal"] for s in w])
    mats=np.stack([QT@s["Rrel"] for s in w])
    near=min(range(len(w)),key=lambda i:abs(w[i]["t"]-t))
    r6s=np.stack([r6(x) for x in mats])
    angles=np.array([Rot.from_matrix(x).magnitude() for x in mats])
    return np.array([
        np.max(mag),np.mean(mag),np.std(mag),np.sum(np.abs(mag-1)),
        np.max(np.abs(np.diff(mag))) if len(mag)>1 else 0,
        *np.max(np.abs(A),axis=0),*np.mean(A,axis=0),
        np.std(angles),np.max(angles)-np.min(angles),
        *r6(mats[near]),*np.mean(r6s,axis=0),*np.std(r6s,axis=0)
    ],float)

for axis_name,axis in [("yaw/Z",[0,0,1]),("pitch/Y",[0,1,0]),("roll/X",[1,0,0])]:
    print("\n",axis_name)
    for deg in (0,5,8,10,15):
        vals=[]
        for sign in ((1,) if deg==0 else (-1,1)):
            Q=Rot.from_rotvec(np.array(axis)*np.deg2rad(deg*sign)).as_matrix()
            yt=[];yp=[];hh=[]
            FX={"L":[],"R":[]}; YY={"L":[],"R":[]}
            for h,y,t,w in events:
                FX[h].append(feat(w,t,Q));YY[h].append(y)
            corr=tot=0;per={}
            for h in ("L","R"):
                X=np.asarray(FX[h]);y=np.asarray(YY[h]);p=M[h].predict(X)
                per[h]=accuracy_score(y,p);corr+=int(np.sum(p==y));tot+=len(y)
            vals.append((corr/tot,per,sign))
        best=max(vals,key=lambda x:x[0]);worst=min(vals,key=lambda x:x[0])
        print(deg,"worst",round(worst[0],4),"sign",worst[2],"LR",{k:round(v,4) for k,v in worst[1].items()},
              "best",round(best[0],4),"sign",best[2])
