from __future__ import annotations
from pathlib import Path
import csv,sys,joblib
from collections import deque,Counter
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
PKG=LAB/"HoloGrip_Production_vNext_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP));sys.path.insert(0,str(PKG))
from song_collection_server import HitDetector,SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS,PEAK_LAG_MS,WINDOW_HALF_MS,MATCH_TOL_MS
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES
from production_core import FrameValidator,RefractoryGate

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
MODELS={h:joblib.load(PKG/"models"/f"base_{h}.joblib") for h in ("L","R")}

with RAW.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
with GT.open(encoding="utf-8",newline="") as f:gt=[r for r in csv.DictReader(f) if r["final_hand"] in ("L","R")]

R0={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0[h],_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])

det={h:HitDetector(**PRODUCT_DETECTOR_KWARGS) for h in ("L","R")}
gate={h:RefractoryGate(120.0) for h in ("L","R")}
val={h:FrameValidator() for h in ("L","R")}
buf={h:deque(maxlen=500) for h in ("L","R")}
pending={h:[] for h in ("L","R")}
preds=[]

for r in rows:
    h=r["hand"];t=float(r["song_time_ms"])
    p={
        "ax":float(r["ax_g"]),"ay":float(r["ay_g"]),"az":float(r["az_g"]),
        "yaw":float(r["yaw_deg"]),"pitch":float(r["pitch_deg"]),"roll":float(r["roll_deg"])
    }
    ok,reason=val[h].check(p)
    if not ok:
        det[h]=HitDetector(**PRODUCT_DETECTOR_KWARGS);pending[h].clear();buf[h].clear();gate[h].reset()
        continue
    pkt=SensorPacket(hand=h,ax=p["ax"],ay=p["ay"],az=p["az"],yaw=p["yaw"],pitch=p["pitch"],roll=p["roll"],
        packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),
        received_time_ms=int(float(r["received_time_ms"])),received_monotonic=t/1000.0)
    sm=make_relative_sample(R0[h],t,p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"])
    buf[h].append(sm)
    ev=det[h].add_packet(pkt,sm["rel_yaw"],sm["rel_pitch"])
    if ev:
        peak=t-PEAK_LAG_MS
        if gate[h].accept(peak):
            pending[h].append({"peak_ms":peak,"peak_g":float(ev["peak_accel_g"])})
    remain=[]
    for item in pending[h]:
        if t>=item["peak_ms"]+WINDOW_HALF_MS:
            ft=feature_from_samples(list(buf[h]),item["peak_ms"],WINDOW_HALF_MS)
            if ft is None:continue
            m=MODELS[h];proba=m.predict_proba([ft])[0];j=int(np.argmax(proba));cls=int(m.classes_[j])
            preds.append({"hand":h,"trigger_ms":item["peak_ms"]+PEAK_LAG_MS,"peak_ms":item["peak_ms"],
                          "pred_drum":ZONE_NAMES[cls],"prob":float(proba[j])})
        else:remain.append(item)
    pending[h]=remain

used=set();matched=[]
for g in gt:
    h=g["final_hand"];t=float(g["csv_center_ms"])
    best=None;bd=None
    for i,p in enumerate(preds):
        if i in used or p["hand"]!=h:continue
        d=abs(p["trigger_ms"]-t)
        if d<=MATCH_TOL_MS and (bd is None or d<bd):
            best=i;bd=d
    if best is not None:
        used.add(best);matched.append((g,preds[best],bd))
tp=len(matched);fp=len(preds)-tp;fn=len(gt)-tp
p=tp/(tp+fp);r=tp/(tp+fn);f=2*p*r/(p+r)
correct=sum(g["drum"]==pp["pred_drum"] for g,pp,_ in matched)
joint_tp=correct
joint_precision=joint_tp/len(preds)
joint_recall=joint_tp/len(gt)
joint_f1=2*joint_precision*joint_recall/(joint_precision+joint_recall)
print("GT",len(gt),"preds",len(preds),"matched",tp,"FP",fp,"FN",fn)
print("hit P/R/F1",p,r,f)
print("zone acc given matched hit",correct/tp)
print("joint exact P/R/F1",joint_precision,joint_recall,joint_f1)
print("suppressed", {h:gate[h].suppressed for h in ("L","R")})
print("invalid", {h:val[h].invalid for h in ("L","R")})
for d in ZONE_NAMES:
    rr=[(g,pp) for g,pp,_ in matched if g["drum"]==d]
    if rr:
        print(d,sum(g["drum"]==pp["pred_drum"] for g,pp in rr),"/",len(rr))
