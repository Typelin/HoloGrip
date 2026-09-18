from pathlib import Path
import csv,sys,joblib
from collections import Counter
import numpy as np
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import rows_to_samples,slice_window,window_features,predict_zone,ZONE_MAP

def wrap(x): return (x+180)%360-180
# 0812 known offsets inferred exactly from raw-cal columns
target={"L":{"yaw":-175.78,"pitch":9.33},"R":{"yaw":3.57,"pitch":29.55}}

p=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
with p.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
adj=[]
for r in rows:
    h=r["hand"]
    q=dict(r)
    q["cal_yaw_deg"]=str(wrap(float(r["yaw_deg"])-target[h]["yaw"]))
    q["cal_pitch_deg"]=str(float(r["pitch_deg"])-target[h]["pitch"])
    adj.append(q)
samples=rows_to_samples(adj)
with (ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv").open(encoding="utf-8-sig",newline="") as f:ev=list(csv.DictReader(f))
model=joblib.load(ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/hologrip_song2_七鼓點模型_真正驗證版_0826.joblib")
pred=[]
for e in ev:
    h=e["assigned_hand"];d=e["zone_name"];c=float(e["aligned_csv_time_ms"])
    ft=window_features(slice_window(samples[h],c,80),c)
    if ft is None:continue
    pr,pro=predict_zone(model,ft)
    pred.append((d,pr,h))
print("n",len(pred),"acc",sum(a==b for a,b,_ in pred)/len(pred))
for d in ["小鼓","高音 Tom","中音 Tom","落地 Tom","Hi-Hat","Crash","Ride"]:
    rr=[x for x in pred if x[0]==d]
    if rr:print(d,sum(a==b for a,b,_ in rr),"/",len(rr),"pred",Counter(b for _,b,_ in rr))
