from pathlib import Path
import csv,sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import rows_to_samples,slice_window,window_features,ZONE_MAP
def wrap(x):return (x+180)%360-180
# inferred from first 1 s
roll0_12={"L":22.41,"R":-42.68}
roll0_5={"L":24.58,"R":-16.55}
zero12={"L":{"yaw":-175.78,"pitch":9.33},"R":{"yaw":3.57,"pitch":29.55}}

def load_adjust(path,roll0,force_zero12=False):
    with path.open(encoding="utf-8-sig",newline="") as f:rr=list(csv.DictReader(f))
    out=[]
    for r in rr:
        q=dict(r);h=r["hand"]
        q["roll_deg"]=str(wrap(float(r["roll_deg"])-roll0[h]))
        if force_zero12:
            q["cal_yaw_deg"]=str(wrap(float(r["yaw_deg"])-zero12[h]["yaw"]))
            q["cal_pitch_deg"]=str(float(r["pitch_deg"])-zero12[h]["pitch"])
        out.append(q)
    return rows_to_samples(out)

def build12(samples):
    with (ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv").open(encoding="utf-8",newline="") as f:g=list(csv.DictReader(f))
    X=[];y=[];h=[]
    for r in g:
        hh=r["final_hand"];d=r["drum"]
        if hh not in ("L","R"):continue
        c=float(r["csv_center_ms"]);ft=window_features(slice_window(samples[hh],c,80),c)
        if ft is not None:X.append(ft);y.append(ZONE_MAP[d]);h.append(hh)
    return np.asarray(X),np.asarray(y),np.asarray(h)

def build5(samples):
    with (ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv").open(encoding="utf-8-sig",newline="") as f:g=list(csv.DictReader(f))
    X=[];y=[];h=[]
    for r in g:
        hh=r["assigned_hand"];d=r["zone_name"];c=float(r["aligned_csv_time_ms"]);ft=window_features(slice_window(samples[hh],c,80),c)
        if ft is not None:X.append(ft);y.append(ZONE_MAP[d]);h.append(hh)
    return np.asarray(X),np.asarray(y),np.asarray(h)

def model():return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1600,random_state=42))])

s12=load_adjust(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv",roll0_12)
X,y,h=build12(s12)
for name,force in [("relative_roll",False),("relative_roll_plus_0812zero",True)]:
    s5=load_adjust(ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv",roll0_5,force)
    X5,y5,h5=build5(s5)
    p=np.full(len(y5),-1,int)
    for hh in ("L","R"):
        tr=np.where(h==hh)[0];te=np.where(h5==hh)[0]
        m=model();m.fit(X[tr],y[tr]);p[te]=m.predict(X5[te])
    print(name,"acc",accuracy_score(y5,p),"bal",balanced_accuracy_score(y5,p),"F1",f1_score(y5,p,average="macro"))
