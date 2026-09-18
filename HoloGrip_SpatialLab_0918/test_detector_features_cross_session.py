from pathlib import Path
import csv,sys,math
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows,ZONE_MAP,ZONE_NAMES,PRODUCT_DETECTOR_KWARGS,PEAK_LAG_MS,MATCH_TOL_MS
from song_collection_server import HitDetector,SensorPacket
SEED=42

def replay_full(path):
    rows=load_raw_rows(str(path))
    det={"L":HitDetector(**PRODUCT_DETECTOR_KWARGS),"R":HitDetector(**PRODUCT_DETECTOR_KWARGS)}
    out=[]
    for r in rows:
        h=r["hand"]
        if h not in det:continue
        song=float(r["song_time_ms"])
        p=SensorPacket(
            hand=h,ax=float(r["ax_g"]),ay=float(r["ay_g"]),az=float(r["az_g"]),
            yaw=float(r["yaw_deg"]),pitch=float(r["pitch_deg"]),roll=float(r["roll_deg"]),
            packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),
            received_time_ms=int(float(r["received_time_ms"])),received_monotonic=song/1000.0)
        ev=det[h].add_packet(p,float(r["cal_yaw_deg"]),float(r["cal_pitch_deg"]))
        if ev:
            out.append({"hand":h,"trigger_ms":song,"peak_ms":song-PEAK_LAG_MS,**ev})
    return out

def greedy_match(events,dets,timekey,handkey,labelkey,tol=80):
    used=set();out=[]
    for e in events:
        h=e[handkey]
        if h not in ("L","R"):continue
        t=float(e[timekey])
        bi=None;bd=None
        for i,d in enumerate(dets):
            if i in used or d["hand"]!=h:continue
            dt=abs(d["trigger_ms"]-t)
            if dt<=tol and (bd is None or dt<bd):
                bi=i;bd=dt
        if bi is not None:
            used.add(bi);out.append((e,dets[bi],bd))
    return out

d12=replay_full(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv")
with (ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv").open(encoding="utf-8",newline="") as f:g12=list(csv.DictReader(f))
m12=greedy_match(g12,d12,"csv_center_ms","final_hand","drum")
print("0812 detections",len(d12),"matched",len(m12),"/",len([x for x in g12 if x["final_hand"] in ("L","R")]))

d5=replay_full(ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv")
with (ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv").open(encoding="utf-8-sig",newline="") as f:g5=list(csv.DictReader(f))
m5=greedy_match(g5,d5,"aligned_csv_time_ms","assigned_hand","zone_name")
print("0805 detections",len(d5),"matched",len(m5),"/",len(g5))

def vec(ev,mode):
    core=[
        float(ev["peak_accel_g"]),float(ev["v_score"]),float(ev["v_score_long"]),
        float(ev["pitch_diff_deg"]),float(ev["vertical_ratio"]),
        float(ev["swing_depth_deg"]),
    ]
    posture=[float(ev["feature_yaw_deg"]),float(ev["feature_pitch_deg"])]
    hand=[1.0 if ev["hand"]=="R" else -1.0]
    if mode=="motion":return core+hand
    if mode=="motion_posture":return core+posture+hand
    raise ValueError

def build(matched,session,mode):
    X=[];y=[];g=[];h=[]
    for e,d,dt in matched:
        lab=e["drum"] if session=="12" else e["zone_name"]
        X.append(vec(d,mode));y.append(ZONE_MAP[lab]);h.append(d["hand"])
        t=float(e["csv_center_ms"] if session=="12" else e["aligned_csv_time_ms"])
        g.append(int(t//4000))
    return np.asarray(X,float),np.asarray(y),np.asarray(g),np.asarray(h)

def model():
    return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(48,24),max_iter=1800,random_state=SEED))])
def report(tag,y,p):
    print(tag,"n",len(y),"acc",round(accuracy_score(y,p),4),"bal",round(balanced_accuracy_score(y,p),4),"F1",round(f1_score(y,p,average="macro"),4))
    for k,nm in enumerate(ZONE_NAMES):
        ix=np.where(y==k)[0]
        if len(ix):print(" ",k,nm,"n",len(ix),"acc",round(float(np.mean(p[ix]==k)),3))

for mode in ("motion","motion_posture"):
    print("\n###",mode)
    X,y,g,h=build(m12,"12",mode);X5,y5,g5,h5=build(m5,"5",mode)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    p=np.full(len(y),-1,int)
    for tr,te in cv.split(X,y,g):
        mm=model();mm.fit(X[tr],y[tr]);p[te]=mm.predict(X[te])
    report("0812_CV",y,p)
    mm=model();mm.fit(X,y);p5=mm.predict(X5);report("0805_cross",y5,p5)
    # separate hands
    ps=np.full(len(y5),-1,int)
    for hh in ("L","R"):
        tr=np.where(h==hh)[0];te=np.where(h5==hh)[0]
        mm=model();mm.fit(X[tr],y[tr]);ps[te]=mm.predict(X5[te])
    report("0805_separate",y5,ps)
