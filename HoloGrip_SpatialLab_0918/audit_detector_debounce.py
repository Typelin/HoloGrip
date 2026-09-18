from pathlib import Path
import csv,sys
from collections import Counter
import numpy as np
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows, MATCH_TOL_MS, PEAK_LAG_MS
from song_collection_server import HitDetector, SensorPacket

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"

with GT.open(encoding="utf-8",newline="") as f: gt=[r for r in csv.DictReader(f) if r["final_hand"] in ("L","R")]
rows=load_raw_rows(str(RAW))

print("=== GT SAME-HAND INTERVALS ===")
for h in ("L","R"):
    ts=sorted(float(r["csv_center_ms"]) for r in gt if r["final_hand"]==h)
    d=np.diff(ts)
    print(h,"n",len(ts),"min",float(np.min(d)),"p1/5/10/25/50",np.percentile(d,[1,5,10,25,50]).round(1).tolist())
    for th in (80,100,120,150,180,200,250):
        print(" <",th,"ms",int(np.sum(d<th)))

def replay(cfg):
    det={h:HitDetector(**cfg) for h in ("L","R")}
    out=[]
    for r in rows:
        h=r["hand"]
        if h not in det: continue
        song=float(r["song_time_ms"])
        pkt=SensorPacket(
            hand=h,ax=float(r["ax_g"]),ay=float(r["ay_g"]),az=float(r["az_g"]),
            yaw=float(r["yaw_deg"]),pitch=float(r["pitch_deg"]),roll=float(r["roll_deg"]),
            packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),
            received_time_ms=int(float(r["received_time_ms"])),received_monotonic=song/1000.0
        )
        ev=det[h].add_packet(pkt,float(r["cal_yaw_deg"]),float(r["cal_pitch_deg"]))
        if ev:
            out.append({"hand":h,"trigger_ms":song,"peak_ms":song-PEAK_LAG_MS,"peak_g":float(ev["peak_accel_g"])})
    return out

def match(preds,tol=90):
    used=set(); matched=[]; misses=[]
    for g in sorted(gt,key=lambda x:float(x["csv_center_ms"])):
        h=g["final_hand"]; t=float(g["csv_center_ms"])
        best=None;bd=None
        for i,p in enumerate(preds):
            if i in used or p["hand"]!=h:continue
            d=abs(p["trigger_ms"]-t)
            if d<=tol and (bd is None or d<bd):
                best=i;bd=d
        if best is None:misses.append(g)
        else:
            used.add(best);matched.append((g,preds[best],bd))
    extras=[p for i,p in enumerate(preds) if i not in used]
    return matched,misses,extras

configs=[]
for heavy in (0.08,0.10,0.12,0.14,0.15,0.18):
  for light in (0.12,0.15,0.18,0.20,0.22,0.25):
    if light<heavy: continue
    configs.append((heavy,light))

print("\n=== DETECTOR GRID ===")
best=[]
for heavy,light in configs:
    cfg={"mag_min":2.3,"debounce_heavy_s":heavy,"debounce_light_s":light,"motion_bypass_mag":4.0}
    p=replay(cfg)
    m,miss,extra=match(p,MATCH_TOL_MS)
    recall=len(m)/len(gt)
    prec=len(m)/len(p) if p else 0
    f1=2*recall*prec/(recall+prec) if recall+prec else 0
    # count same-hand predicted intervals under thresholds
    short={}
    for th in (80,100,120,150):
        n=0
        for h in ("L","R"):
            ts=sorted(x["trigger_ms"] for x in p if x["hand"]==h)
            n+=int(np.sum(np.diff(ts)<th))
        short[th]=n
    best.append((f1,recall,prec,len(p),len(extra),heavy,light,short))
for row in sorted(best,reverse=True)[:20]:
    print("heavy",row[5],"light",row[6],"pred",row[3],"extra",row[4],
          "recall",round(row[1],4),"prec",round(row[2],4),"f1",round(row[0],4),"short",row[7])

# current detail
cfg={"mag_min":2.3,"debounce_heavy_s":0.10,"debounce_light_s":0.15,"motion_bypass_mag":4.0}
p=replay(cfg);m,miss,extra=match(p,MATCH_TOL_MS)
print("\nCURRENT matched",len(m),"miss",len(miss),"extra",len(extra),"pred",len(p))
print("miss by drum",Counter(g["drum"] for g in miss))
print("extra by hand",Counter(x["hand"] for x in extra))
for h in ("L","R"):
    ts=sorted(x["trigger_ms"] for x in p if x["hand"]==h)
    print(h,"pred interval min/p1/5/10/50",np.percentile(np.diff(ts),[0,1,5,10,50]).round(1).tolist())
