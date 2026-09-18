from pathlib import Path
import csv, sys, joblib, itertools
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.metrics import accuracy_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES
ZMAP={n:i for i,n in enumerate(ZONE_NAMES)}
MODELS={h:joblib.load(LAB/"vnext_timing_robust_aug20"/f"hologrip_timing_robust_{h}.joblib") for h in ("L","R")}

SETS={
 "0812":(
   ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv",
   ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv",
   "final_hand","drum","csv_center_ms","utf-8"),
 "0805":(
   ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv",
   ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv",
   "assigned_hand","zone_name","aligned_csv_time_ms","utf-8-sig"),
}

def rcsv(p,enc):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

def base_r0(rawp):
    rows=rcsv(rawp,"utf-8-sig"); out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        out[h],_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    return rows,out

def eval_set(tag,angle,axis,side):
    rawp,gtp,hk,dk,tk,enc=SETS[tag]
    rows,R0s=base_r0(rawp)
    D={}
    for h in ("L","R"):
        Q=Rot.from_rotvec(np.eye(3)[axis]*np.deg2rad(angle*side)).as_matrix()
        # local neutral-frame perturbation
        R0p=R0s[h] @ Q
        rr=[r for r in rows if r["hand"]==h]
        D[h]=[make_relative_sample(R0p,float(r["song_time_ms"]),
             float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
             float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
    yt=[];yp=[];hh=[]
    for g in rcsv(gtp,enc):
        h=g[hk]; d=g[dk]
        if h not in ("L","R") or d not in ZMAP:continue
        ft=feature_from_samples(D[h],float(g[tk]),80)
        if ft is None:continue
        yt.append(ZMAP[d]);yp.append(int(MODELS[h].predict([ft])[0]));hh.append(h)
    yt=np.asarray(yt);yp=np.asarray(yp);hh=np.asarray(hh)
    acc=accuracy_score(yt,yp)
    per={}
    for h in ("L","R"):
        ix=np.where(hh==h)[0]; per[h]=accuracy_score(yt[ix],yp[ix]) if len(ix) else float("nan")
    return acc,per

axes=["X","Y","Z"]
for tag in ("0812","0805"):
    print("\n===",tag,"===")
    base=eval_set(tag,0,0,1);print("0deg",round(base[0],4),{k:round(v,4) for k,v in base[1].items()})
    for ang in (5,7.5,10,12.5,15,20):
        vals=[]
        for ai,an in enumerate(axes):
            for side in (-1,1):
                a,p=eval_set(tag,ang,ai,side)
                vals.append((a,an,side,p))
        accs=np.array([x[0] for x in vals])
        worst=min(vals,key=lambda x:x[0]);best=max(vals,key=lambda x:x[0])
        print(f"{ang:>4.1f}deg mean={accs.mean():.4f} worst={worst[0]:.4f}({worst[1]}{worst[2]:+d}) best={best[0]:.4f}({best[1]}{best[2]:+d})",
              "worstLR",{k:round(v,4) for k,v in worst[3].items()})
