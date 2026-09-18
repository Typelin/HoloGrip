from pathlib import Path
import csv,sys,joblib,math,json
from collections import Counter,defaultdict
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
PROD=LAB/"HoloGrip_Production_vNext_0918"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP));sys.path.insert(0,str(PROD))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES
from song_collection_server import HitDetector,SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS,PEAK_LAG_MS
from production_core import FrameValidator,RefractoryGate

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
MODELS={h:joblib.load(LAB/"vnext_models"/f"hologrip_vnext_{h}_relative_rotation.joblib") for h in ("L","R")}
ZONE_MAP={n:i for i,n in enumerate(ZONE_NAMES)}

def rcsv(p,enc="utf-8-sig"):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=rcsv(RAW)
gt=[r for r in rcsv(GT,"utf-8") if r["final_hand"] in ("L","R") and r["drum"] in ZONE_MAP]

# Build relative samples per hand.
D={}
R0={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0[h],_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0[h],float(r["song_time_ms"]),
        float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
        float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]

print("=== FEATURE CENTER TIME-SHIFT SENSITIVITY, 8/12 GT ===")
shifts=[-120,-100,-80,-60,-40,-30,-20,-10,0,10,20,30,40,60,80,100,120]
pred_by_shift={}
for shift in shifts:
    correct=0;n=0;preds=[]
    hand_corr=Counter();hand_n=Counter()
    for g in gt:
        h=g["final_hand"];t=float(g["csv_center_ms"])+shift
        ft=feature_from_samples(D[h],t,80)
        if ft is None:continue
        pr=int(MODELS[h].predict([ft])[0]);truth=ZONE_MAP[g["drum"]]
        preds.append(pr);n+=1;hand_n[h]+=1
        if pr==truth:correct+=1;hand_corr[h]+=1
    pred_by_shift[shift]=np.array(preds,dtype=int)
    print(f"shift {shift:+4d}ms acc {correct/n:.4f}  L {hand_corr['L']}/{hand_n['L']}={hand_corr['L']/hand_n['L']:.3f}  R {hand_corr['R']}/{hand_n['R']}={hand_corr['R']/hand_n['R']:.3f}")

base=pred_by_shift[0]
print("\n=== PREDICTION FLIP RATE VS 0ms ===")
for shift in shifts:
    if shift==0:continue
    p=pred_by_shift[shift]
    m=min(len(p),len(base))
    print(f"{shift:+4d}ms flip {np.mean(p[:m]!=base[:m]):.3f}")

# Event-specific stability: how many of shifts -40..+40 preserve same answer, and true answer.
local_shifts=[-40,-30,-20,-10,0,10,20,30,40]
unstable=Counter();wrong_any=Counter()
for i,g in enumerate(gt):
    h=g["final_hand"];truth=ZONE_MAP[g["drum"]]
    pp=[]
    for s in local_shifts:
        ft=feature_from_samples(D[h],float(g["csv_center_ms"])+s,80)
        if ft is not None:pp.append(int(MODELS[h].predict([ft])[0]))
    if len(set(pp))>1:unstable[(h,g["drum"])]+=1
    if any(x!=truth for x in pp):wrong_any[(h,g["drum"])]+=1
print("\nunstable within +/-40ms by hand/drum:")
for k,v in sorted(unstable.items()):print(k,v)

# Replay detector, raw and with 120ms production gate.
det={h:HitDetector(**PRODUCT_DETECTOR_KWARGS) for h in ("L","R")}
gate={h:RefractoryGate(120.0) for h in ("L","R")}
rawdet=[];gated=[]
for r in rows:
    h=r["hand"];t=float(r["song_time_ms"])
    pkt=SensorPacket(hand=h,ax=float(r["ax_g"]),ay=float(r["ay_g"]),az=float(r["az_g"]),
        yaw=float(r["yaw_deg"]),pitch=float(r["pitch_deg"]),roll=float(r["roll_deg"]),
        packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),
        received_time_ms=int(float(r["received_time_ms"])),received_monotonic=t/1000.0)
    # Detector only uses cal_yaw/cal_pitch for raise filter; use original calibrated values if present.
    cy=float(r.get("cal_yaw_deg",r["yaw_deg"]));cp=float(r.get("cal_pitch_deg",r["pitch_deg"]))
    ev=det[h].add_packet(pkt,cy,cp)
    if ev:
        peak=t-PEAK_LAG_MS
        x={"hand":h,"peak":peak,"trigger":t,"g":float(ev["peak_accel_g"])}
        rawdet.append(x)
        if gate[h].accept(peak):gated.append(x)

print("\n=== DETECTIONS ===")
print("raw detector",len(rawdet),Counter(x["hand"] for x in rawdet))
print("with 120ms gate",len(gated),Counter(x["hand"] for x in gated))

def match(dets,tol=90,use="peak"):
    used=set();res=[];miss=[]
    for gi,g in enumerate(gt):
        h=g["final_hand"];t=float(g["csv_center_ms"])
        opts=[]
        for i,d in enumerate(dets):
            if i in used or d["hand"]!=h:continue
            dt=d[use]-t
            if abs(dt)<=tol:opts.append((abs(dt),i,dt))
        if not opts:
            miss.append(gi);continue
        _,i,dt=min(opts);used.add(i);res.append((gi,i,dt))
    extras=[i for i in range(len(dets)) if i not in used]
    return res,miss,extras

for name,dets in [("raw",rawdet),("gate120",gated)]:
  for use in ("peak","trigger"):
    m,miss,extra=match(dets,90,use)
    dt=np.array([x[2] for x in m],float)
    print(f"\n{name} matched using {use}: matched={len(m)} miss={len(miss)} extra={len(extra)}")
    if len(dt):
      print(" dt mean/median/p10/p90/min/max",np.round([dt.mean(),np.median(dt),*np.percentile(dt,[10,90]),dt.min(),dt.max()],2))
      for h in ("L","R"):
        q=np.array([x[2] for x in m if gt[x[0]]["final_hand"]==h],float)
        print(" ",h,"n",len(q),"median",round(float(np.median(q)),2),"p10/90",np.round(np.percentile(q,[10,90]),2))

# how many detections around each GT in wider windows; identify doublets after GT.
print("\n=== MULTIPLE DETECTIONS AROUND SAME GT (raw detector) ===")
for window in (100,140,180,220):
    counts=[]
    for g in gt:
        h=g["final_hand"];t=float(g["csv_center_ms"])
        counts.append(sum(d["hand"]==h and abs(d["peak"]-t)<=window for d in rawdet))
    c=Counter(counts)
    print("window",window,"ms",dict(sorted(c.items())),"GT with >=2",sum(x>=2 for x in counts))

# detection inter-arrival distributions.
print("\n=== DETECTION INTERVALS ===")
for name,dets in [("raw",rawdet),("gate120",gated)]:
  for h in ("L","R"):
    ts=np.array([d["peak"] for d in dets if d["hand"]==h])
    di=np.diff(ts)
    print(name,h,"n",len(ts),"interval <130",int(np.sum(di<130)),"130-180",int(np.sum((di>=130)&(di<180))),"<250",int(np.sum(di<250)),
          "p1/5/10/50",np.round(np.percentile(di,[1,5,10,50]),1) if len(di) else None)
