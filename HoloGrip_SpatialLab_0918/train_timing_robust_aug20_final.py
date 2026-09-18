from pathlib import Path
import csv,sys,json,joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
OUT=LAB/"vnext_timing_robust_aug20"
OUT.mkdir(parents=True,exist_ok=True)
ZMAP={n:i for i,n in enumerate(ZONE_NAMES)}

def rcsv(p,enc="utf-8-sig"):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=rcsv(RAW)
D={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,zero=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
        float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]

gt=[r for r in rcsv(GT,"utf-8") if r["final_hand"] in ("L","R") and r["drum"] in ZMAP]
meta={"source_raw":str(RAW),"source_gt":str(GT),"offsets_ms":[-20,0,20],"hands":{}}
for h in ("L","R"):
    X=[];y=[];events=0
    for r in gt:
        if r["final_hand"]!=h:continue
        events+=1;t=float(r["csv_center_ms"])
        for o in (-20,0,20):
            ft=feature_from_samples(D[h],t+o,80)
            if ft is not None:X.append(ft);y.append(ZMAP[r["drum"]])
    X=np.asarray(X);y=np.asarray(y)
    m=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),alpha=.001,max_iter=1200,random_state=42))])
    m.fit(X,y)
    path=OUT/f"hologrip_timing_robust_{h}.joblib";joblib.dump(m,path)
    meta["hands"][h]={"events":events,"train_windows":len(y),"model_path":str(path)}
(OUT/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(meta,ensure_ascii=False,indent=2))
