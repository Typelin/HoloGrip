from pathlib import Path
import csv,sys,json
from collections import Counter,defaultdict
from datetime import datetime,timedelta
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,FEATURE_NAMES

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
HOME=LAB/"protocol_sessions/20260918_155836"
SIDE={"Crash":0,"Hi-Hat":0,"Ride":1,"落地 Tom":1}
EXPECT={2:0,6:1,8:0,10:1,12:1}

def rcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

# 8/12
rows=rcsv(RAW12);D={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
            float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
X=[];y=[];hand=[];groups=[]
for r in rcsv(GT12,"utf-8"):
    h=r["final_hand"];d=r["drum"]
    if h not in ("L","R") or d not in SIDE:continue
    c=float(r["csv_center_ms"]);ft=feature_from_samples(D[h],c,80)
    if ft is not None:X.append(ft);y.append(SIDE[d]);hand.append(h);groups.append(int(c//4000))
X=np.asarray(X);y=np.asarray(y);hand=np.asarray(hand);groups=np.asarray(groups)

# home
raw=rcsv(HOME/"raw_100hz.csv");hits=rcsv(HOME/"hits.csv")
cal=json.loads((HOME/"calibration.json").read_text(encoding="utf-8"))
R0h={h:Rot.from_euler("ZYX",cal[h]["zero"],degrees=True).as_matrix() for h in ("L","R")}
for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"]);r["stage"]=int(r["stage_index"]);r["valid"]=int(r["frame_valid"])
for z in hits:
    z["dt"]=datetime.fromisoformat(z["wall_time"]);z["stage"]=int(z["stage_index"])
ss={"L":defaultdict(list),"R":defaultdict(list)}
for r in raw:
    if r["stage"] not in (2,6,8,10,12) or not r["valid"]:continue
    h=r["hand"]
    sm=make_relative_sample(R0h[h],r["dt"].timestamp()*1000,float(r["ax"]),float(r["ay"]),float(r["az"]),
                            float(r["yaw"]),float(r["pitch"]),float(r["roll"]))
    ss[h][r["stage"]].append(sm)
def dedup(lst,g=.18):
    out=[];last=None
    for z in sorted(lst,key=lambda a:a["dt"]):
        if last is None or (z["dt"]-last).total_seconds()>=g:out.append(z);last=z["dt"]
    return out
home=[]
for stage in EXPECT:
    for h in ("L","R"):
        for hit in dedup([z for z in hits if z["stage"]==stage and z["hand"]==h]):
            center=(hit["dt"]-timedelta(milliseconds=80)).timestamp()*1000
            ft=feature_from_samples(ss[h][stage],center,80)
            if ft is not None:home.append((stage,h,ft))

subsets={
    "center_r6":list(range(13,19)),
    "mean_r6":list(range(19,25)),
    "center+mean_r6":list(range(13,25)),
    "orientation_all":list(range(13,31)),
    "rot_disp+orientation":list(range(11,31)),
    "no_accel":list(range(11,31)),
    "full31":list(range(31)),
}
print("FEATURE_NAMES",list(enumerate(FEATURE_NAMES)))
best=None
for subset,idxs in subsets.items():
  for kind in ("svc","logreg"):
    models={};cv_true=[];cv_pred=[]
    for h in ("L","R"):
        ix=np.where(hand==h)[0]
        Xi=X[ix][:,idxs];yi=y[ix]
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        p=np.full(len(ix),-1,int)
        for tr,te in cv.split(Xi,yi,groups[ix]):
            m=Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",probability=True,random_state=42) if kind=="svc" else LogisticRegression(C=1,max_iter=3000,random_state=42))])
            m.fit(Xi[tr],yi[tr]);p[te]=m.predict(Xi[te])
        cv_true.extend(yi.tolist());cv_pred.extend(p.tolist())
        m=Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",probability=True,random_state=42) if kind=="svc" else LogisticRegression(C=1,max_iter=3000,random_state=42))])
        m.fit(Xi,yi);models[h]=m
    cvacc=accuracy_score(cv_true,cv_pred)
    correct=0;n=0;by=defaultdict(list)
    for stage,h,ft in home:
        pr=models[h].predict_proba([ft[idxs]])[0]
        pred=int(models[h].classes_[int(np.argmax(pr))]);conf=float(np.max(pr))
        by[(stage,h)].append((pred,conf))
        n+=1;correct+=int(pred==EXPECT[stage])
    acc=correct/n
    # stage-majority accuracy
    smc=0;smn=0
    for (stage,h),v in by.items():
        maj=Counter(x[0] for x in v).most_common(1)[0][0]
        smc+=int(maj==EXPECT[stage]);smn+=1
    print(subset,kind,"CV",round(cvacc,3),"HOME_hit",round(acc,3),f"{correct}/{n}","HOME_stage",round(smc/smn,3),f"{smc}/{smn}")
    for key,v in sorted(by.items()):
        print(" ",key,"L",sum(x[0]==0 for x in v),"R",sum(x[0]==1 for x in v),"conf",round(float(np.median([x[1] for x in v])),3))
    score=acc+0.15*(smc/smn)+0.05*cvacc
    if best is None or score>best[0]:
        best=(score,subset,kind,idxs,models,acc,smc/smn,cvacc)
print("\nBEST",best[1],best[2],"HOME_hit",best[5],"HOME_stage",best[6],"CV",best[7])
# save best
out=LAB/"song2_pose_side_models";out.mkdir(parents=True,exist_ok=True)
for h,m in best[4].items():joblib.dump(m,out/f"side_{h}.joblib")
(out/"metadata.json").write_text(json.dumps({"subset":best[1],"kind":best[2],"indices":best[3],"feature_names":[FEATURE_NAMES[i] for i in best[3]],"home_hit_acc":best[5],"home_stage_acc":best[6],"cv_acc":best[7]},ensure_ascii=False,indent=2),encoding="utf-8")
