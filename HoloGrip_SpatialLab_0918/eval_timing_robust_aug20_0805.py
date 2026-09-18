from pathlib import Path
import csv,sys,joblib
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES
RAW5=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
GT5=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"
ZMAP={n:i for i,n in enumerate(ZONE_NAMES)}
SIDE={"Crash":0,"Hi-Hat":0,"小鼓":1,"高音 Tom":1,"中音 Tom":1,"Ride":2,"落地 Tom":2}
def rcsv(p,enc="utf-8-sig"):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=rcsv(RAW5);D={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h];t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
        float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
M={h:joblib.load(LAB/"vnext_timing_robust_aug20"/f"hologrip_timing_robust_{h}.joblib") for h in ("L","R")}
yt=[];yp=[];hh=[];dr=[]
for r in rcsv(GT5):
    h=r["assigned_hand"];d=r["zone_name"]
    if h not in ("L","R") or d not in ZMAP:continue
    ft=feature_from_samples(D[h],float(r["aligned_csv_time_ms"]),80)
    if ft is None:continue
    yt.append(ZMAP[d]);yp.append(int(M[h].predict([ft])[0]));hh.append(h);dr.append(d)
yt=np.array(yt);yp=np.array(yp);hh=np.array(hh);dr=np.array(dr)
print("overall",len(yt),accuracy_score(yt,yp))
for h in ("L","R"):
    ix=np.where(hh==h)[0];print(h,len(ix),accuracy_score(yt[ix],yp[ix]))
print("CM\n",confusion_matrix(yt,yp,labels=range(7)))
opp=0;side_n=0
for d,p in zip(dr,yp):
    a=SIDE[d];b=SIDE[ZONE_NAMES[p]]
    if a in (0,2):side_n+=1
    if (a==0 and b==2) or (a==2 and b==0):opp+=1
print("opposite_side",opp,"/",side_n)
for k,nm in enumerate(ZONE_NAMES):
    ix=np.where(yt==k)[0]
    if len(ix):print(nm,len(ix),"correct",int(np.sum(yp[ix]==k)),"preds",dict(Counter(ZONE_NAMES[x] for x in yp[ix])))
