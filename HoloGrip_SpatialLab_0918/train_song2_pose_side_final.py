from pathlib import Path
import csv,sys,json,joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
SIDE={"Crash":0,"Hi-Hat":0,"Ride":1,"落地 Tom":1}
IDX=list(range(19,25)) # mean_r6 only

def rcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=rcsv(RAW)
D={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
        float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
out=LAB/"song2_pose_side_models"
out.mkdir(parents=True,exist_ok=True)
meta={"source":"8/12 Song2 only","labels":{"0":"LEFT","1":"RIGHT"},"feature_indices":IDX,"feature_kind":"mean_r6","drums":{"LEFT":["Crash","Hi-Hat"],"RIGHT":["Ride","落地 Tom"]},"hands":{}}
for h in ("L","R"):
    X=[];y=[]
    for r in rcsv(GT,"utf-8"):
        if r["final_hand"]!=h or r["drum"] not in SIDE:continue
        ft=feature_from_samples(D[h],float(r["csv_center_ms"]),80)
        if ft is not None:X.append(ft[IDX]);y.append(SIDE[r["drum"]])
    X=np.asarray(X);y=np.asarray(y)
    m=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=3000,random_state=42))])
    m.fit(X,y);joblib.dump(m,out/f"side_{h}.joblib")
    meta["hands"][h]={"n":int(len(y)),"left":int(np.sum(y==0)),"right":int(np.sum(y==1))}
(out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(meta,ensure_ascii=False,indent=2))
