from __future__ import annotations
from pathlib import Path
import csv,sys,math,joblib
from collections import Counter,deque
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP));sys.path.insert(0,str(LAB))
from song_collection_server import HitDetector,SensorPacket
from product_hit_and_zone import load_raw_rows,MATCH_TOL_MS,PEAK_LAG_MS,WINDOW_HALF_MS
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
MODELS={h:joblib.load(LAB/"vnext_models"/f"hologrip_vnext_{h}_relative_rotation.joblib") for h in ("L","R")}

rows=load_raw_rows(str(RAW))
with GT.open(encoding="utf-8",newline="") as f:
    gt=[r for r in csv.DictReader(f) if r["final_hand"] in ("L","R")]

# calibration R0 from first second
R0={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0[h],_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])

# precompute relative samples
rel={"L":[],"R":[]}
for r in rows:
    h=r["hand"];t=float(r["song_time_ms"])
    rel[h].append(make_relative_sample(R0[h],t,float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])))

def classify(h,peak_ms):
    ft=feature_from_samples(rel[h],peak_ms,WINDOW_HALF_MS)
    if ft is None:return None,0
    m=MODELS[h];pr=m.predict_proba([ft])[0];j=int(np.argmax(pr));cls=int(m.classes_[j])
    return ZONE_NAMES[cls],float(pr[j])

def replay(heavy,light):
    det={h:HitDetector(mag_min=2.3,debounce_heavy_s=heavy,debounce_light_s=light,motion_bypass_mag=4.0) for h in ("L","R")}
    out=[]
    for r in rows:
        h=r["hand"];t=float(r["song_time_ms"])
        pkt=SensorPacket(hand=h,ax=float(r["ax_g"]),ay=float(r["ay_g"]),az=float(r["az_g"]),yaw=float(r["yaw_deg"]),pitch=float(r["pitch_deg"]),roll=float(r["roll_deg"]),packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),received_time_ms=int(float(r["received_time_ms"])),received_monotonic=t/1000.0)
        ev=det[h].add_packet(pkt,float(r["cal_yaw_deg"]),float(r["cal_pitch_deg"]))
        if ev:
            peak=t-PEAK_LAG_MS
            pred,prob=classify(h,peak)
            out.append({"hand":h,"trigger_ms":t,"peak_ms":peak,"pred":pred,"prob":prob,"peak_g":float(ev["peak_accel_g"])})
    return out

def evaluate(preds):
    used=set(); matched=[]; misses=[]
    for g in sorted(gt,key=lambda x:float(x["csv_center_ms"])):
        h=g["final_hand"];t=float(g["csv_center_ms"])
        cand=[(abs(p["trigger_ms"]-t),i) for i,p in enumerate(preds) if i not in used and p["hand"]==h and abs(p["trigger_ms"]-t)<=MATCH_TOL_MS]
        if not cand:misses.append(g);continue
        d,i=min(cand);used.add(i);matched.append((g,preds[i],d))
    extras=[p for i,p in enumerate(preds) if i not in used]
    correct=sum(g["drum"]==p["pred"] for g,p,_ in matched)
    hitrec=len(matched)/len(gt);hitprec=len(matched)/len(preds)
    prodrec=correct/len(gt)
    # product precision: correct / all emitted predictions
    prodprec=correct/len(preds)
    prodf1=2*prodrec*prodprec/(prodrec+prodprec) if prodrec+prodprec else 0
    return {
        "preds":len(preds),"matched":len(matched),"misses":len(misses),"extras":len(extras),
        "correct":correct,"hitrec":hitrec,"hitprec":hitprec,
        "prodrec":prodrec,"prodprec":prodprec,"prodf1":prodf1,
        "miss_by_drum":Counter(g["drum"] for g in misses),
        "wrong_by_true":Counter(g["drum"] for g,p,_ in matched if g["drum"]!=p["pred"]),
        "extra_by_hand":Counter(p["hand"] for p in extras)
    }

for heavy,light in [(0.10,0.15),(0.12,0.18),(0.14,0.18),(0.15,0.18),(0.15,0.20)]:
    p=replay(heavy,light);m=evaluate(p)
    print("\n",heavy,light)
    for k,v in m.items():print(k,v)
