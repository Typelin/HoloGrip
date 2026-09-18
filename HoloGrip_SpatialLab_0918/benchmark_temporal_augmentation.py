from pathlib import Path
import csv,sys,warnings,joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

warnings.filterwarnings("ignore")
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
RAW5=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
GT5=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"
ZMAP={n:i for i,n in enumerate(ZONE_NAMES)}

def rcsv(p,enc="utf-8-sig"):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
def prep(rawp):
    rows=rcsv(rawp);out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
        out[h]=[make_relative_sample(R0,float(r["song_time_ms"]),
            float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
            float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
    return out

D12=prep(RAW12);D5=prep(RAW5)
G12=[r for r in rcsv(GT12,"utf-8") if r["final_hand"] in ("L","R") and r["drum"] in ZMAP]
G5=[r for r in rcsv(GT5) if r["assigned_hand"] in ("L","R") and r["zone_name"] in ZMAP]

def make():
    return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),alpha=.001,max_iter=1200,random_state=42))])

def feature_event(D,h,t):
    return feature_from_samples(D[h],t,80)

# center evaluation matrices
E12=[]
for r in G12:
    ft=feature_event(D12,r["final_hand"],float(r["csv_center_ms"]))
    if ft is not None:E12.append((r,ft))
E5=[]
for r in G5:
    ft=feature_event(D5,r["assigned_hand"],float(r["aligned_csv_time_ms"]))
    if ft is not None:E5.append((r,ft))

configs={
    "center_only":[0],
    "aug20":[-20,0,20],
    "aug40":[-40,-20,0,20,40],
    "aug60":[-60,-40,-20,0,20,40,60],
}
print("events 8/12",len(E12),"8/5",len(E5))

for name,offs in configs.items():
    print("\n===",name,offs,"===")
    models={}
    cv_all_true=[];cv_all_pred=[]
    for h in ("L","R"):
        events=[r for r in G12 if r["final_hand"]==h]
        # build per-event augmented training features
        event_feats=[];event_y=[];groups=[]
        for ei,r in enumerate(events):
            t=float(r["csv_center_ms"]);fs=[]
            for o in offs:
                ft=feature_event(D12,h,t+o)
                if ft is not None:fs.append(ft)
            event_feats.append(fs);event_y.append(ZMAP[r["drum"]]);groups.append(int(t//4000))
        yev=np.array(event_y);grp=np.array(groups)
        pred=np.full(len(events),-1,int)
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        for tr,te in cv.split(np.zeros((len(events),1)),yev,grp):
            Xtr=[];ytr=[]
            for i in tr:
                for ft in event_feats[i]:Xtr.append(ft);ytr.append(yev[i])
            m=make();m.fit(np.asarray(Xtr),np.asarray(ytr))
            for i in te:
                # test ONLY at true center, not augmented
                ft=feature_event(D12,h,float(events[i]["csv_center_ms"]))
                pred[i]=int(m.predict([ft])[0])
        cv_all_true+=yev.tolist();cv_all_pred+=pred.tolist()

        # final full model
        Xtr=[];ytr=[]
        for i in range(len(events)):
            for ft in event_feats[i]:Xtr.append(ft);ytr.append(yev[i])
        m=make();m.fit(np.asarray(Xtr),np.asarray(ytr));models[h]=m
    print("8/12 grouped-CV center acc",round(accuracy_score(cv_all_true,cv_all_pred),4),
          "bal",round(balanced_accuracy_score(cv_all_true,cv_all_pred),4),
          "f1",round(f1_score(cv_all_true,cv_all_pred,average="macro"),4))

    # 8/5 center and jitter evaluation
    for shift in (-60,-40,-20,0,20,40,60):
        yt=[];yp=[]
        for r in G5:
            h=r["assigned_hand"];t=float(r["aligned_csv_time_ms"])+shift
            ft=feature_event(D5,h,t)
            if ft is None:continue
            yt.append(ZMAP[r["zone_name"]]);yp.append(int(models[h].predict([ft])[0]))
        print("8/5 shift",f"{shift:+d}","acc",round(accuracy_score(yt,yp),4),
              "bal",round(balanced_accuracy_score(yt,yp),4),
              "f1",round(f1_score(yt,yp,average="macro"),4))
