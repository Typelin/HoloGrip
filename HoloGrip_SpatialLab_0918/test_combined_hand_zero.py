from pathlib import Path
import csv,sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows,rows_to_samples,slice_window,window_features,ZONE_MAP
def wrap(x):return (x+180)%360-180
target={"L":{"yaw":-175.78,"pitch":9.33},"R":{"yaw":3.57,"pitch":29.55}}
# 0812 train
s12=rows_to_samples(load_raw_rows(str(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv")))
with (ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv").open(encoding="utf-8",newline="") as f:g12=list(csv.DictReader(f))
X=[];y=[];h=[]
for r in g12:
 hh=r["final_hand"];d=r["drum"]
 if hh not in ("L","R"):continue
 c=float(r["csv_center_ms"]);ft=window_features(slice_window(s12[hh],c,80),c)
 if ft is not None:X.append(ft);y.append(ZONE_MAP[d]);h.append(hh)
X=np.asarray(X);y=np.asarray(y);h=np.asarray(h)
# 0805 adjusted
with (ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv").open(encoding="utf-8-sig",newline="") as f:rr=list(csv.DictReader(f))
adj=[]
for r in rr:
 q=dict(r);hh=r["hand"];q["cal_yaw_deg"]=str(wrap(float(r["yaw_deg"])-target[hh]["yaw"]));q["cal_pitch_deg"]=str(float(r["pitch_deg"])-target[hh]["pitch"]);adj.append(q)
s5=rows_to_samples(adj)
with (ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv").open(encoding="utf-8-sig",newline="") as f:g5=list(csv.DictReader(f))
X5=[];y5=[];h5=[]
for r in g5:
 hh=r["assigned_hand"];d=r["zone_name"];c=float(r["aligned_csv_time_ms"]);ft=window_features(slice_window(s5[hh],c,80),c)
 if ft is not None:X5.append(ft);y5.append(ZONE_MAP[d]);h5.append(hh)
X5=np.asarray(X5);y5=np.asarray(y5);h5=np.asarray(h5)
def model(): return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1600,random_state=42))])
p=np.full(len(y5),-1,int)
for hh in ("L","R"):
 tr=np.where(h==hh)[0];te=np.where(h5==hh)[0];m=model();m.fit(X[tr],y[tr]);p[te]=m.predict(X5[te])
print("separate+0812zero",len(y5),"acc",accuracy_score(y5,p),"bal",balanced_accuracy_score(y5,p),"F1",f1_score(y5,p,average="macro"))
