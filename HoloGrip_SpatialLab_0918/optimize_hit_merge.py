from __future__ import annotations
from pathlib import Path
import csv,sys,math,joblib
from collections import Counter
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP));sys.path.insert(0,str(LAB))
from song_collection_server import HitDetector,SensorPacket
from product_hit_and_zone import load_raw_rows,MATCH_TOL_MS,PEAK_LAG_MS,WINDOW_HALF_MS,PRODUCT_DETECTOR_KWARGS
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
MODELS={h:joblib.load(LAB/"vnext_models"/f"hologrip_vnext_{h}_relative_rotation.joblib") for h in ("L","R")}
rows=load_raw_rows(str(RAW))
with GT.open(encoding="utf-8",newline="") as f:gt=[r for r in csv.DictReader(f) if r["final_hand"] in ("L","R")]

R0={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h];t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0[h],_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
rel={"L":[],"R":[]}
for r in rows:
    h=r["hand"];t=float(r["song_time_ms"])
    rel[h].append(make_relative_sample(R0[h],t,float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])))

def classify(h,peak):
    ft=feature_from_samples(rel[h],peak,WINDOW_HALF_MS)
    m=MODELS[h];pr=m.predict_proba([ft])[0];j=int(np.argmax(pr));cls=int(m.classes_[j])
    return ZONE_NAMES[cls],float(pr[j])

# current detector only
D={h:HitDetector(**PRODUCT_DETECTOR_KWARGS) for h in ("L","R")}
pred=[]
for r in rows:
    h=r["hand"];t=float(r["song_time_ms"])
    pkt=SensorPacket(hand=h,ax=float(r["ax_g"]),ay=float(r["ay_g"]),az=float(r["az_g"]),yaw=float(r["yaw_deg"]),pitch=float(r["pitch_deg"]),roll=float(r["roll_deg"]),packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),received_time_ms=int(float(r["received_time_ms"])),received_monotonic=t/1000)
    ev=D[h].add_packet(pkt,float(r["cal_yaw_deg"]),float(r["cal_pitch_deg"]))
    if ev:
        peak=t-PEAK_LAG_MS;lab,prob=classify(h,peak)
        pred.append({"hand":h,"trigger_ms":t,"peak_ms":peak,"peak_g":float(ev["peak_accel_g"]),"pred":lab,"prob":prob})

def evaluate(P):
    used=set();correct=0;matched=0
    for g in sorted(gt,key=lambda x:float(x["csv_center_ms"])):
        h=g["final_hand"];t=float(g["csv_center_ms"])
        c=[(abs(p["trigger_ms"]-t),i) for i,p in enumerate(P) if i not in used and p["hand"]==h and abs(p["trigger_ms"]-t)<=MATCH_TOL_MS]
        if c:
            d,i=min(c);used.add(i);matched+=1;correct+=(P[i]["pred"]==g["drum"])
    extra=len(P)-matched;rec=correct/len(gt);prec=correct/len(P)
    f1=2*rec*prec/(rec+prec) if rec+prec else 0
    return correct,matched,extra,rec,prec,f1,len(P)

def process_cluster(P,T,choice):
    out=[]
    for h in ("L","R"):
        pp=sorted([p for p in P if p["hand"]==h],key=lambda x:x["trigger_ms"])
        clusters=[];cur=[]
        for p in pp:
            if cur and p["trigger_ms"]-cur[-1]["trigger_ms"]>=T:
                clusters.append(cur);cur=[]
            cur.append(p)
        if cur:clusters.append(cur)
        for c in clusters:
            if len(c)==1:out.append(c[0]);continue
            if choice=="max_peak":out.append(max(c,key=lambda x:x["peak_g"]))
            elif choice=="max_prob":out.append(max(c,key=lambda x:x["prob"]))
            elif choice=="first":out.append(c[0])
            elif choice=="last":out.append(c[-1])
    return sorted(out,key=lambda x:(x["hand"],x["trigger_ms"]))

def process_pair_rule(P,T,ratio,same_only=False):
    out=[]
    for h in ("L","R"):
        pp=sorted([p for p in P if p["hand"]==h],key=lambda x:x["trigger_ms"])
        kept=[]
        for p in pp:
            if not kept:
                kept.append(p);continue
            prev=kept[-1];dt=p["trigger_ms"]-prev["trigger_ms"]
            if dt<T and (not same_only or p["pred"]==prev["pred"]):
                # If the new pulse is clearly weaker, suppress it.
                if p["peak_g"] < ratio*prev["peak_g"]:
                    continue
                # If new is clearly stronger, replace previous close rebound.
                if prev["peak_g"] < ratio*p["peak_g"]:
                    kept[-1]=p;continue
            kept.append(p)
        out.extend(kept)
    return sorted(out,key=lambda x:(x["hand"],x["trigger_ms"]))

print("BASE",evaluate(pred))
best=[]
for T in (120,130,140,150,160):
    for choice in ("max_peak","max_prob","first","last"):
        P=process_cluster(pred,T,choice);m=evaluate(P);best.append((m[5],m,T,choice,"cluster"))
for T in (120,130,140,150,160):
    for ratio in (0.35,0.45,0.55,0.65,0.75,0.85):
        for same in (False,True):
            P=process_pair_rule(pred,T,ratio,same);m=evaluate(P);best.append((m[5],m,T,ratio,f"pair same={same}"))
for b in sorted(best,reverse=True)[:30]:
    print(b)
