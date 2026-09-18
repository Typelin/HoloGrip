from pathlib import Path
import csv,sys,json,hashlib
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
PKG=LAB/"HoloGrip_Production_vNext_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(PKG));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples
from product_hit_and_zone import ZONE_MAP,ZONE_NAMES

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"

def readcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=readcsv(RAW)
session={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,zero=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    ss=[make_relative_sample(R0,float(r["song_time_ms"]),
        float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
        float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
    session[h]={"zero":zero,"samples":ss}

X=[];y=[];hand=[];time_ms=[]
for r in readcsv(GT,"utf-8"):
    h=r["final_hand"];d=r["drum"]
    if h not in ("L","R") or d not in ZONE_MAP:continue
    c=float(r["csv_center_ms"])
    ft=feature_from_samples(session[h]["samples"],c,80)
    if ft is None:continue
    X.append(ft);y.append(ZONE_MAP[d]);hand.append(h);time_ms.append(c)
X=np.asarray(X,float);y=np.asarray(y,int);hand=np.asarray(hand);time_ms=np.asarray(time_ms,float)

out=PKG/"models"/"base_training_features.npz"
np.savez_compressed(out,X=X,y=y,hand=hand,time_ms=time_ms,zone_names=np.asarray(ZONE_NAMES,dtype=object))
meta={
    "n":int(len(y)),
    "feature_count":int(X.shape[1]),
    "L":int(np.sum(hand=="L")),
    "R":int(np.sum(hand=="R")),
    "zone_names":ZONE_NAMES,
    "source_raw":str(RAW),
    "source_gt":str(GT),
}
(PKG/"models"/"training_archive_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(meta,ensure_ascii=False,indent=2))
print(out)
