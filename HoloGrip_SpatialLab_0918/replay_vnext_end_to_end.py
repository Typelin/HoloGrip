from __future__ import annotations
import csv, sys, joblib, math
from pathlib import Path
from collections import deque
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP))
sys.path.insert(0,str(LAB))

from song_collection_server import HitDetector, SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS, PEAK_LAG_MS, WINDOW_HALF_MS, MATCH_TOL_MS
from vnext_core import compute_r0, make_relative_sample, feature_from_samples, ZONE_NAMES

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
MODELS={
    h:joblib.load(LAB/"vnext_models"/f"hologrip_vnext_{h}_relative_rotation.joblib")
    for h in ("L","R")
}

with RAW.open(encoding="utf-8-sig",newline="") as f:
    rows=list(csv.DictReader(f))
with GT.open(encoding="utf-8",newline="") as f:
    gt=list(csv.DictReader(f))

R0={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0[h],_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])

det={h:HitDetector(**PRODUCT_DETECTOR_KWARGS) for h in ("L","R")}
buf={h:deque(maxlen=500) for h in ("L","R")}
pending={h:[] for h in ("L","R")}
preds=[]

for r in rows:
    h=r["hand"]
    if h not in det: continue
    song=float(r["song_time_ms"])
    pkt=SensorPacket(
        hand=h,
        ax=float(r["ax_g"]),ay=float(r["ay_g"]),az=float(r["az_g"]),
        yaw=float(r["yaw_deg"]),pitch=float(r["pitch_deg"]),roll=float(r["roll_deg"]),
        packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),
        received_time_ms=int(float(r["received_time_ms"])),received_monotonic=song/1000.0,
    )
    sm=make_relative_sample(R0[h],song,pkt.ax,pkt.ay,pkt.az,pkt.yaw,pkt.pitch,pkt.roll)
    buf[h].append(sm)
    ev=det[h].add_packet(pkt,sm["rel_yaw"],sm["rel_pitch"])
    if ev:
        pending[h].append({"peak_ms":song-PEAK_LAG_MS,"trigger_ms":song,"peak_accel_g":ev["peak_accel_g"]})
    remain=[]
    for item in pending[h]:
        if song >= item["peak_ms"]+WINDOW_HALF_MS:
            ft=feature_from_samples(list(buf[h]),item["peak_ms"],WINDOW_HALF_MS)
            if ft is None:
                continue
            m=MODELS[h]
            proba=m.predict_proba([ft])[0]
            cls=int(m.classes_[int(np.argmax(proba))])
            preds.append({
                "hand":h,
                "trigger_ms":item["trigger_ms"],
                "peak_ms":item["peak_ms"],
                "pred_drum":ZONE_NAMES[cls],
                "prob":float(np.max(proba)),
            })
        else:
            remain.append(item)
    pending[h]=remain

gt_single=[g for g in gt if g["final_hand"] in ("L","R")]
used=set(); matched=[]; misses=[]
for g in gt_single:
    h=g["final_hand"]; t=float(g["csv_center_ms"])
    best=None;bd=None
    for i,p in enumerate(preds):
        if i in used or p["hand"]!=h: continue
        d=abs(p["trigger_ms"]-t)
        if d<=MATCH_TOL_MS and (bd is None or d<bd):
            best=i;bd=d
    if best is None:
        misses.append(g);continue
    used.add(best)
    p=preds[best]
    matched.append((g,p,bd))
extras=[p for i,p in enumerate(preds) if i not in used]
joint=sum(g["drum"]==p["pred_drum"] for g,p,_ in matched)
print("gt",len(gt_single),"pred",len(preds),"matched",len(matched),"misses",len(misses),"extras",len(extras))
print("hit_recall",len(matched)/len(gt_single))
print("hit_precision",len(matched)/len(preds))
print("zone_acc_given_hit",joint/len(matched))
print("product_recall",joint/len(gt_single))
for name in ZONE_NAMES:
    rr=[(g,p) for g,p,_ in matched if g["drum"]==name]
    if rr:
        ok=sum(g["drum"]==p["pred_drum"] for g,p in rr)
        print(name,ok,"/",len(rr),round(ok/len(rr),3))
