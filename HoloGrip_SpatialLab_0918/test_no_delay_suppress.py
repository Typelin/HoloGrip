from pathlib import Path
import csv,sys,math,joblib
import numpy as np
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918";APP=ROOT/"Apps"/"Song_Collection_COM"
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
 rr=[r for r in rows if r["hand"]==h];t0=min(float(r["song_time_ms"]) for r in rr);cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
 R0[h],_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
rel={"L":[],"R":[]}
for r in rows:
 h=r["hand"];t=float(r["song_time_ms"]);rel[h].append(make_relative_sample(R0[h],t,float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])))
def predlab(h,peak):
 ft=feature_from_samples(rel[h],peak,WINDOW_HALF_MS);m=MODELS[h];pr=m.predict_proba([ft])[0];j=int(np.argmax(pr));return ZONE_NAMES[int(m.classes_[j])],float(pr[j])
D={h:HitDetector(**PRODUCT_DETECTOR_KWARGS) for h in ("L","R")};P=[]
for r in rows:
 h=r["hand"];t=float(r["song_time_ms"])
 pkt=SensorPacket(hand=h,ax=float(r["ax_g"]),ay=float(r["ay_g"]),az=float(r["az_g"]),yaw=float(r["yaw_deg"]),pitch=float(r["pitch_deg"]),roll=float(r["roll_deg"]),packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),received_time_ms=int(float(r["received_time_ms"])),received_monotonic=t/1000)
 ev=D[h].add_packet(pkt,float(r["cal_yaw_deg"]),float(r["cal_pitch_deg"]))
 if ev:
  peak=t-PEAK_LAG_MS;lab,prob=predlab(h,peak);P.append({"hand":h,"t":t,"peak":float(ev["peak_accel_g"]),"pred":lab,"prob":prob})
def evaluate(Q):
 used=set();m=c=0
 for g in sorted(gt,key=lambda x:float(x["csv_center_ms"])):
  h=g["final_hand"];t=float(g["csv_center_ms"]);cand=[(abs(p["t"]-t),i) for i,p in enumerate(Q) if i not in used and p["hand"]==h and abs(p["t"]-t)<=MATCH_TOL_MS]
  if cand:
   _,i=min(cand);used.add(i);m+=1;c+=(Q[i]["pred"]==g["drum"])
 rec=c/len(gt);prec=c/len(Q);f1=2*rec*prec/(rec+prec)
 return c,m,len(Q)-m,rec,prec,f1,len(Q)
print("BASE",evaluate(P))
for T in (120,130,140,150,160):
 for ratio in (.25,.30,.35,.40,.45,.50,.55,.60):
  out=[]
  for h in ("L","R"):
   last=None
   for p in sorted([x for x in P if x["hand"]==h],key=lambda x:x["t"]):
    if last is not None and p["t"]-last["t"]<T and p["peak"]<ratio*last["peak"]:
     continue
    out.append(p);last=p
  m=evaluate(out)
  if m[5]>=0.915 or (m[0]>=290 and m[2]<=30):
   print("T",T,"ratio",ratio,m)
