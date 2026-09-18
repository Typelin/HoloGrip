from pathlib import Path
import csv,sys,math
from collections import Counter
import numpy as np
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows, MATCH_TOL_MS, PEAK_LAG_MS, PRODUCT_DETECTOR_KWARGS
from song_collection_server import HitDetector, SensorPacket

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
with GT.open(encoding="utf-8",newline="") as f:gt=[r for r in csv.DictReader(f) if r["final_hand"] in ("L","R")]
rows=load_raw_rows(str(RAW))
byraw={"L":[],"R":[]}
for r in rows:
    h=r["hand"]
    t=float(r["song_time_ms"])
    mag=math.sqrt(float(r["ax_g"])**2+float(r["ay_g"])**2+float(r["az_g"])**2)
    byraw[h].append((t,mag))

def replay():
    det={h:HitDetector(**PRODUCT_DETECTOR_KWARGS) for h in ("L","R")}
    out=[]
    for r in rows:
        h=r["hand"];song=float(r["song_time_ms"])
        pkt=SensorPacket(hand=h,ax=float(r["ax_g"]),ay=float(r["ay_g"]),az=float(r["az_g"]),
            yaw=float(r["yaw_deg"]),pitch=float(r["pitch_deg"]),roll=float(r["roll_deg"]),
            packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),
            received_time_ms=int(float(r["received_time_ms"])),received_monotonic=song/1000)
        ev=det[h].add_packet(pkt,float(r["cal_yaw_deg"]),float(r["cal_pitch_deg"]))
        if ev: out.append({"hand":h,"trigger_ms":song,"peak_ms":song-PEAK_LAG_MS,"peak_g":float(ev["peak_accel_g"])})
    return out

preds=replay()
# greedy match
used=set()
for g in sorted(gt,key=lambda x:float(x["csv_center_ms"])):
    h=g["final_hand"];t=float(g["csv_center_ms"])
    cand=[(abs(p["trigger_ms"]-t),i) for i,p in enumerate(preds) if i not in used and p["hand"]==h and abs(p["trigger_ms"]-t)<=MATCH_TOL_MS]
    if cand:
        d,i=min(cand);used.add(i);preds[i]["matched"]=True;preds[i]["gt_t"]=t;preds[i]["gt_drum"]=g["drum"];preds[i]["dt_gt"]=d
for i,p in enumerate(preds):
    p.setdefault("matched",False)

def minmag_between(h,t1,t2):
    vals=[m for t,m in byraw[h] if t1<=t<=t2]
    return min(vals) if vals else np.nan

pairs=[]
for h in ("L","R"):
    pp=sorted([p for p in preds if p["hand"]==h],key=lambda p:p["trigger_ms"])
    for a,b in zip(pp[:-1],pp[1:]):
        dt=b["trigger_ms"]-a["trigger_ms"]
        if dt<180:
            valley=minmag_between(h,a["peak_ms"]+20,b["peak_ms"]-20)
            pairs.append({
                "hand":h,"dt":dt,"a_match":a["matched"],"b_match":b["matched"],
                "a_g":a["peak_g"],"b_g":b["peak_g"],"ratio":b["peak_g"]/a["peak_g"] if a["peak_g"] else np.nan,
                "valley":valley,
                "a_gt":a.get("gt_drum",""),"b_gt":b.get("gt_drum",""),
                "a_t":a["trigger_ms"],"b_t":b["trigger_ms"]
            })

print("pairs<180",len(pairs))
for label,sel in [
    ("SECOND_MATCHED",[x for x in pairs if x["b_match"]]),
    ("SECOND_EXTRA",[x for x in pairs if not x["b_match"]]),
]:
    print("\n",label,"n",len(sel))
    if sel:
        for k in ("dt","a_g","b_g","ratio","valley"):
            arr=np.array([x[k] for x in sel],float)
            print(k,"p10/25/50/75/90",np.round(np.percentile(arr,[10,25,50,75,90]),3).tolist())
        print("dt<150",sum(x["dt"]<150 for x in sel))
        print("valley<1.5",sum(x["valley"]<1.5 for x in sel),"valley<1.8",sum(x["valley"]<1.8 for x in sel))
        print("examples")
        for x in sel[:30]:
            print(x)

# Try rules: suppress second hit when dt<T AND valley > V (never rearmed)
print("\n=== POST RULE GRID ===")
base=sorted(preds,key=lambda p:(p["hand"],p["trigger_ms"]))
for T in (120,130,140,150,160):
  for V in (1.2,1.4,1.6,1.8,2.0):
    kept=[]
    last_by={}
    for p in base:
        h=p["hand"]
        prev=last_by.get(h)
        keep=True
        if prev is not None:
            dt=p["trigger_ms"]-prev["trigger_ms"]
            if dt<T:
                valley=minmag_between(h,prev["peak_ms"]+20,p["peak_ms"]-20)
                if valley>V: keep=False
        if keep:
            kept.append(p);last_by[h]=p
    # match kept from scratch
    used=set();m=0
    for g in sorted(gt,key=lambda x:float(x["csv_center_ms"])):
        h=g["final_hand"];t=float(g["csv_center_ms"])
        cand=[(abs(p["trigger_ms"]-t),i) for i,p in enumerate(kept) if i not in used and p["hand"]==h and abs(p["trigger_ms"]-t)<=MATCH_TOL_MS]
        if cand:
            d,i=min(cand);used.add(i);m+=1
    extra=len(kept)-m
    rec=m/len(gt);prec=m/len(kept)
    f1=2*rec*prec/(rec+prec)
    if rec>=0.975 or f1>=0.955:
        print("T",T,"V",V,"kept",len(kept),"m",m,"extra",extra,"rec",round(rec,4),"prec",round(prec,4),"f1",round(f1,4))
