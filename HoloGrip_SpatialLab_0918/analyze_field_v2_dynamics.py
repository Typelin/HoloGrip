from pathlib import Path
import csv,sys
from collections import defaultdict
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,FEATURE_NAMES
RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
LIVE=LAB/"field_v2_sessions/20260918_210839/resolved_hits.csv"
def rcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=rcsv(RAW);D={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h];t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
        float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
tr={"L":[],"R":[]}
for g in rcsv(GT,"utf-8"):
    h=g["final_hand"]
    if h not in tr:continue
    ft=feature_from_samples(D[h],float(g["csv_center_ms"]),80)
    if ft is not None:tr[h].append(ft)
live=rcsv(LIVE,"utf-8")
ix={n:i for i,n in enumerate(FEATURE_NAMES)}
for h in ("L","R"):
    T=np.asarray(tr[h]);q=[r for r in live if r["hand"]==h]
    if not q:continue
    X=np.asarray([[float(r[n]) for n in FEATURE_NAMES] for r in q])
    peak=np.asarray([float(r["peak_g"]) for r in q])
    print("HAND",h)
    print("train max_mag feature p10/50/90",np.round(np.percentile(T[:,0],[10,50,90]),2))
    print("live detector peak_g p10/50/90",np.round(np.percentile(peak,[10,50,90]),2))
    print("train rot_std p10/50/90",np.round(np.percentile(T[:,11],[10,50,90]),3))
    print("live rot_std p10/50/90",np.round(np.percentile(X[:,11],[10,50,90]),3))
    print("train rot_range p10/50/90",np.round(np.percentile(T[:,12],[10,50,90]),3))
    print("live rot_range p10/50/90",np.round(np.percentile(X[:,12],[10,50,90]),3))
    for marker in sorted(set(r["marker"] for r in q)):
        z=[r for r in q if r["marker"]==marker]
        p=np.array([float(r["peak_g"]) for r in z]);XX=np.array([[float(r[n]) for n in FEATURE_NAMES] for r in z])
        print(" ",marker,"n",len(z),"peak med",round(float(np.median(p)),2),"rot_range med",round(float(np.median(XX[:,12])),3))
