from __future__ import annotations
import csv, json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

LAB=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918")
S=LAB/"live_sessions"/"20260918_141423"
REF=json.loads((LAB/"spatial_reference_0826.json").read_text(encoding="utf-8"))

# Parse calibration events
events=[]
for line in (S/"events.jsonl").read_text(encoding="utf-8").splitlines():
    try: o=json.loads(line)
    except: continue
    events.append(o)
cal=[o for o in events if o.get("kind")=="CALIBRATE_OK"]
cal_begin=[o for o in events if o.get("kind")=="CALIBRATE_BEGIN"]
hits_ev=[o for o in events if o.get("kind")=="HIT"]

print("=== CALIBRATIONS ===")
for i,(b,o) in enumerate(zip(cal_begin,cal),1):
    print(i,b["time"],"origins=",b["origins"],"offsets=",o["offsets"])

if cal_begin:
    first=cal_begin[0]["origins"]; last=cal_begin[-1]["origins"]
    print("ZERO_ORIGIN_SHIFT first->last")
    for h in ("L","R"):
        dy=((last[h]["raw_yaw"]-first[h]["raw_yaw"]+180)%360)-180
        dp=last[h]["raw_pitch"]-first[h]["raw_pitch"]
        dr=((last[h]["raw_roll"]-first[h]["raw_roll"]+180)%360)-180
        print(h, "yaw",round(dy,2),"pitch",round(dp,2),"roll",round(dr,2))

# hits CSV
with (S/"hits.csv").open(encoding="utf-8",newline="") as f:
    hits=list(csv.DictReader(f))
print("\n=== HITS ===")
print("count",len(hits))
print("pred",Counter(r["pred_drum"] for r in hits))
print("band",Counter(r["nn_band"] for r in hits))
for hand in ("L","R"):
    rr=[r for r in hits if r["hand"]==hand]
    print(hand,"n",len(rr),"band",Counter(r["nn_band"] for r in rr),"pred",Counter(r["pred_drum"] for r in rr))

# After final calibration time based on elapsed_s near event elapsed
last_cal_elapsed=max(float(o["elapsed_s"]) for o in cal) if cal else 0
post=[r for r in hits if float(r["elapsed_s"])>=last_cal_elapsed]
print("\nAFTER_FINAL_CAL elapsed",last_cal_elapsed,"n",len(post))
print("band",Counter(r["nn_band"] for r in post))
for hand in ("L","R"):
    rr=[r for r in post if r["hand"]==hand]
    print(hand,"n",len(rr),"band",Counter(r["nn_band"] for r in rr),"pred",Counter(r["pred_drum"] for r in rr))

# training class centers/ranges
print("\n=== TRAINING ANGLE CLASSES ===")
for d in REF["drum_order"]:
    c=REF["classes"][d]
    print(d, "n",c["n"], "yaw med",round(c["yaw_median"],1),"p10-90",round(c["yaw_p10"],1),round(c["yaw_p90"],1),
          "pitch med",round(c["pitch_median"],1),"p10-90",round(c["pitch_p10"],1),round(c["pitch_p90"],1))

# Current hit angle summary after final cal
print("\n=== CURRENT HIT ANGLES AFTER FINAL CAL ===")
for hand in ("L","R"):
    rr=[r for r in post if r["hand"]==hand]
    if not rr: continue
    y=np.array([float(r["hit_yaw"]) for r in rr]); p=np.array([float(r["hit_pitch"]) for r in rr])
    print(hand,"yaw med",round(float(np.median(y)),1),"p05-95",round(float(np.percentile(y,5)),1),round(float(np.percentile(y,95)),1),
          "pitch med",round(float(np.median(p)),1),"p05-95",round(float(np.percentile(p,5)),1),round(float(np.percentile(p,95)),1))

# count hits inside each training class 10-90 ellipse-ish rectangle (simple axis-aligned)
print("\n=== FRACTION OF CURRENT HITS INSIDE EACH TRAINING 10-90 BOX ===")
for d in REF["drum_order"]:
    c=REF["classes"][d]
    cnt=0
    for r in post:
        y=float(r["hit_yaw"]); p=float(r["hit_pitch"])
        # yaw ranges don't cross wrap in current reference
        if c["yaw_p10"] <= y <= c["yaw_p90"] and c["pitch_p10"] <= p <= c["pitch_p90"]:
            cnt+=1
    print(d,cnt)

# Crash-specific: all current hits close to Crash training 10-90
cr=REF["classes"]["Crash"]
crash_like=[r for r in post if cr["yaw_p10"]<=float(r["hit_yaw"])<=cr["yaw_p90"] and cr["pitch_p10"]<=float(r["hit_pitch"])<=cr["pitch_p90"]]
print("\nCrash 10-90 box current hits",len(crash_like),"of",len(post))
print("their model preds",Counter(r["pred_drum"] for r in crash_like))

# raw packet analysis after final calibration
with (S/"raw_packets.csv").open(encoding="utf-8",newline="") as f:
    raw=list(csv.DictReader(f))
raw_post=[r for r in raw if r["cal_yaw"]!="" and float(r["elapsed_s"])>=last_cal_elapsed]
print("\n=== RAW AFTER FINAL CAL ===",len(raw_post))
for h in ("L","R"):
    rr=[r for r in raw_post if r["hand"]==h]
    if not rr: continue
    y=np.array([float(r["cal_yaw"]) for r in rr]); p=np.array([float(r["cal_pitch"]) for r in rr]); mag=np.array([float(r["mag_g"]) for r in rr])
    print(h,"N",len(rr),"yaw p01/50/99",*[round(float(x),1) for x in np.percentile(y,[1,50,99])],
          "pitch p01/50/99",*[round(float(x),1) for x in np.percentile(p,[1,50,99])])

    # Find stationary-ish 0.5s windows: |mag-1|<0.08 and std yaw/pitch <2.5°, then report centers farthest from zero.
    # Work in consecutive chunks of 50 samples and use circular unwrap for yaw locally.
    chunks=[]
    for i in range(0,len(rr)-50,25):
        w=rr[i:i+50]
        yy=np.array([float(x["cal_yaw"]) for x in w])
        pp=np.array([float(x["cal_pitch"]) for x in w])
        mm=np.array([float(x["mag_g"]) for x in w])
        yu=np.rad2deg(np.unwrap(np.deg2rad(yy)))
        if np.mean(np.abs(mm-1.0))<0.12 and np.std(yu)<3.0 and np.std(pp)<3.0:
            center_y=((np.mean(yu)+180)%360)-180
            center_p=float(np.mean(pp))
            chunks.append((float(w[len(w)//2]["elapsed_s"]),center_y,center_p,float(np.std(yu)),float(np.std(pp)),float(np.mean(np.abs(mm-1.0)))))
    print(h,"stationary_windows",len(chunks))
    for z in sorted(chunks,key=lambda x:abs(x[1])+abs(x[2]),reverse=True)[:8]:
        print("  t=%.1fs yaw=%+.1f pitch=%+.1f std=(%.2f,%.2f) amagerr=%.3f"%z)

# Original raw low-motion local stability, just quantify local step/noise not zero-return.
orig=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Raw\Song_Collection_COM\S20260812_P01_song02_raw_100hz_20260812_112101.csv")
with orig.open(encoding="utf-8",newline="") as f:
    old=list(csv.DictReader(f))
print("\n=== ORIGINAL 8/12 LOCAL STABILITY ===")
for h in ("L","R"):
    rr=[r for r in old if r["hand"]==h]
    y=np.array([float(r["cal_yaw_deg"]) for r in rr]); p=np.array([float(r["cal_pitch_deg"]) for r in rr])
    dy=np.rad2deg(np.angle(np.exp(1j*np.deg2rad(y[1:]-y[:-1]))))
    dp=np.diff(p)
    print(h,"N",len(rr),"step yaw |p50,p95,p99|",*[round(float(x),3) for x in np.percentile(np.abs(dy),[50,95,99])],
          "pitch",*[round(float(x),3) for x in np.percentile(np.abs(dp),[50,95,99])])
