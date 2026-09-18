from pathlib import Path
import csv, math, sys
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
S=ROOT/"HoloGrip_SpatialLab_0918/protocol_sessions/20260918_155836"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP))
from song_collection_server import HitDetector, SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS

def readcsv(n):
    with (S/n).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
raw=readcsv("raw_100hz.csv");hits=readcsv("hits.csv")
for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"])
    r["stage"]=int(r["stage_index"])
    r["valid"]=int(r["frame_valid"])
    for k in ("ax","ay","az","yaw","pitch","roll"):
        r[k]=float(r[k])
    for k in ("rel_yaw","rel_pitch","rel_roll"):
        try:r[k]=float(r[k])
        except:r[k]=np.nan
for h in hits:
    h["dt"]=datetime.fromisoformat(h["wall_time"]);h["stage"]=int(h["stage_index"])

def mean_rotation(mats):
    return Rot.from_matrix(np.stack(mats)).mean().as_matrix()

def angle_deg(A,B):
    return float(np.rad2deg(Rot.from_matrix(A.T@B).magnitude()))

print("=== STAGE ORIENTATION CENTERS AT HITS ===")
centers={"L":{},"R":{}}
spread={"L":{},"R":{}}
for hand in ("L","R"):
    print("\nHAND",hand)
    for idx in (2,4,6,8,10,12):
        rr=[r for r in raw if r["hand"]==hand and r["stage"]==idx and r["valid"] and not np.isnan(r["rel_yaw"])]
        hh=[h for h in hits if h["hand"]==hand and h["stage"]==idx]
        mats=[]
        for h in hh:
            t=h["dt"]-timedelta(milliseconds=80)
            if not rr:continue
            n=min(rr,key=lambda r:abs((r["dt"]-t).total_seconds()))
            if abs((n["dt"]-t).total_seconds())<=0.05:
                M=Rot.from_euler("ZYX",[n["rel_yaw"],n["rel_pitch"],n["rel_roll"]],degrees=True).as_matrix()
                mats.append(M)
        if mats:
            C=mean_rotation(mats);centers[hand][idx]=C
            ds=np.array([angle_deg(C,M) for M in mats])
            spread[hand][idx]=(float(np.median(ds)),float(np.percentile(ds,90)),len(ds))
            print(idx,"n",len(ds),"within angle med",round(np.median(ds),1),"p90",round(np.percentile(ds,90),1))

    print("\nPAIRWISE CENTER ANGLES deg")
    ids=sorted(centers[hand])
    print("    "+" ".join(f"{i:>6}" for i in ids))
    for i in ids:
        row=[]
        for j in ids:
            row.append(angle_deg(centers[hand][i],centers[hand][j]))
        print(f"{i:>3} "+" ".join(f"{x:6.1f}" for x in row))

# Nearest stage-center separability relative to within spread
for hand in ("L","R"):
    print("\nSEPARABILITY",hand)
    ids=sorted(centers[hand])
    for i in ids:
        nearest=min((angle_deg(centers[hand][i],centers[hand][j]),j) for j in ids if j!=i)
        med,p90,n=spread[hand][i]
        print(i,"nearest stage",nearest[1],"center_dist",round(nearest[0],1),"within med/p90",round(med,1),round(p90,1),
              "ratio center/within_p90",round(nearest[0]/p90,2) if p90>0 else None)

print("\n=== STRESS B BEFORE / DURING / AFTER R CORRUPTION ===")
bad=[r for r in raw if r["hand"]=="R" and r["stage"]==16 and not r["valid"]]
if bad:
    b0=min(r["dt"] for r in bad);b1=max(r["dt"] for r in bad)
    # long episode start: find from 16:09:55.431 hard from previous analysis
    long0=datetime.fromisoformat("2026-09-18T16:09:55.431+08:00")
    long1=datetime.fromisoformat("2026-09-18T16:10:01.205+08:00")
    print("bad overall",b0.time(),b1.time(),"long",long0.time(),long1.time())
    for label,a,b in [
        ("pre", datetime.fromisoformat("2026-09-18T16:09:48.597+08:00"), long0),
        ("corrupt",long0,long1),
        ("post",long1,datetime.fromisoformat("2026-09-18T16:10:04.700+08:00"))
    ]:
        hh=sorted([h for h in hits if h["stage"]==16 and a<=h["dt"]<b],key=lambda x:x["dt"])
        seq="".join(h["hand"] for h in hh)
        alt=sum(seq[i]!=seq[i-1] for i in range(1,len(seq)))/(len(seq)-1) if len(seq)>1 else 0
        print(label,"n",len(hh),"L",seq.count("L"),"R",seq.count("R"),"alternation",round(alt*100,1),"%","seq",seq[:100])

print("\n=== STRESS B 2-SECOND BINS ===")
start=datetime.fromisoformat("2026-09-18T16:09:48.597+08:00")
for bi in range(8):
    a=start+timedelta(seconds=2*bi);b=a+timedelta(seconds=2)
    print("bin",bi,f"{2*bi}-{2*bi+2}s")
    for hand in ("L","R"):
        hh=[h for h in hits if h["stage"]==16 and h["hand"]==hand and a<=h["dt"]<b]
        c=Counter(h["pred"] for h in hh)
        print(" ",hand,len(hh),dict(c))

print("\n=== OFFLINE FINAL STATIC DETECTOR REPLAY (valid frames only) ===")
for hand in ("L","R"):
    rr=sorted([r for r in raw if r["hand"]==hand and r["stage"]==17 and r["valid"]],key=lambda x:x["dt"])
    det=HitDetector(**PRODUCT_DETECTOR_KWARGS)
    evs=[]
    if rr:
        t0=rr[0]["dt"]
        for r in rr:
            mono=(r["dt"]-t0).total_seconds()
            pkt=SensorPacket(hand=hand,ax=r["ax"],ay=r["ay"],az=r["az"],
                             yaw=r["yaw"],pitch=r["pitch"],roll=r["roll"],
                             packet_id=int(r["packet_id"]),sensor_time_ms=int(r["sensor_ms"]),
                             received_time_ms=int(r["dt"].timestamp()*1000),received_monotonic=mono)
            ev=det.add_packet(pkt,
                0.0 if np.isnan(r["rel_yaw"]) else r["rel_yaw"],
                0.0 if np.isnan(r["rel_pitch"]) else r["rel_pitch"])
            if ev:
                evs.append((mono,ev))
    print(hand,"would_trigger",len(evs),[(round(t,3),round(float(e["peak_accel_g"]),3)) for t,e in evs])

print("\n=== FINAL STATIC R HIGH MAG SAMPLES ===")
rr=[r for r in raw if r["hand"]=="R" and r["stage"]==17 and r["valid"]]
if rr:
    t0=rr[0]["dt"]
    highs=[r for r in rr if math.sqrt(r["ax"]**2+r["ay"]**2+r["az"]**2)>2.3]
    for r in highs[:30]:
        mag=math.sqrt(r["ax"]**2+r["ay"]**2+r["az"]**2)
        print(round((r["dt"]-t0).total_seconds(),3),round(mag,3),r["ax"],r["ay"],r["az"],r["yaw"],r["pitch"],r["roll"])
    print("high count",len(highs))
