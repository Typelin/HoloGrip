from pathlib import Path
import csv,sys,joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.metrics import accuracy_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"; APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES
ZMAP={n:i for i,n in enumerate(ZONE_NAMES)}
M={h:joblib.load(LAB/"vnext_timing_robust_aug20"/f"hologrip_timing_robust_{h}.joblib") for h in ("L","R")}
SETS={
"0812":(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv",ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv","final_hand","drum","csv_center_ms","utf-8"),
"0805":(ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv",ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv","assigned_hand","zone_name","aligned_csv_time_ms","utf-8-sig")
}
def rcsv(p,enc):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

cache={}
for tag,(rawp,gtp,hk,dk,tk,enc) in SETS.items():
    rows=rcsv(rawp,"utf-8-sig"); D={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h];t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
        D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
    G=[g for g in rcsv(gtp,enc) if g[hk] in ("L","R") and g[dk] in ZMAP]
    cache[tag]=(D,G,hk,dk,tk)

def transformed_samples(samples,Q):
    QT=Q.T
    out=[]
    for s in samples:
        Rr=QT@s["Rrel"]; al=QT@s["alocal"]
        out.append({"t":s["t"],"mag":s["mag"],"alocal":al,"Rrel":Rr})
    return out

def evalp(tag,deg,axis,sgn):
    D,G,hk,dk,tk=cache[tag]
    Q=Rot.from_rotvec(np.eye(3)[axis]*np.deg2rad(deg*sgn)).as_matrix()
    DD={h:transformed_samples(D[h],Q) for h in ("L","R")}
    yt=[];yp=[];hh=[]
    for g in G:
        h=g[hk];ft=feature_from_samples(DD[h],float(g[tk]),80)
        if ft is None:continue
        yt.append(ZMAP[g[dk]]);yp.append(int(M[h].predict([ft])[0]));hh.append(h)
    yt=np.array(yt);yp=np.array(yp);hh=np.array(hh)
    per={}
    for h in ("L","R"):
        ix=hh==h;per[h]=accuracy_score(yt[ix],yp[ix])
    return accuracy_score(yt,yp),per

for tag in ("0812","0805"):
    print("\n===",tag,"===")
    a,p=evalp(tag,0,0,1);print("0deg",round(a,4),{k:round(v,4) for k,v in p.items()})
    for deg in (5,7.5,10,12.5,15,20):
        vals=[]
        for ax,n in enumerate("XYZ"):
            for s in (-1,1):
                a,p=evalp(tag,deg,ax,s);vals.append((a,n,s,p))
        acc=np.array([v[0] for v in vals]);w=min(vals,key=lambda v:v[0]);b=max(vals,key=lambda v:v[0])
        print(f"{deg:>4.1f}° mean={acc.mean():.3f} worst={w[0]:.3f}({w[1]}{w[2]:+d}) best={b[0]:.3f}({b[1]}{b[2]:+d}) worstLR={{{'L'}:{w[3]['L']:.3f},{'R'}:{w[3]['R']:.3f}}}")
