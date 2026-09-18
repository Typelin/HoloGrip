from pathlib import Path
import csv,sys,json
from collections import Counter,defaultdict
from datetime import datetime,timedelta
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
HOME=LAB/"protocol_sessions/20260918_155836"

# From user's physical layout image. Left = negative, Right = positive.
XPOS={
    "Crash":-1.00,
    "Hi-Hat":-1.00,
    "高音 Tom":-0.35,
    "小鼓":0.00,
    "中音 Tom":0.35,
    "Ride":1.00,
    "落地 Tom":1.00,
}
EXTREME_SIDE={"Crash":0,"Hi-Hat":0,"Ride":1,"落地 Tom":1} # 0 LEFT, 1 RIGHT

def readcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
def prep12():
    rows=readcsv(RAW12);out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
        ss=[make_relative_sample(R0,float(r["song_time_ms"]),
            float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
            float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
        out[h]=ss
    return out
D12=prep12()
gt=readcsv(GT12,"utf-8")

X=[];hand=[];drum=[];group=[];xpos=[]
for r in gt:
    h=r["final_hand"];d=r["drum"]
    if h not in ("L","R") or d not in XPOS:continue
    c=float(r["csv_center_ms"]);ft=feature_from_samples(D12[h],c,80)
    if ft is None:continue
    X.append(ft);hand.append(h);drum.append(d);group.append(int(c//4000));xpos.append(XPOS[d])
X=np.asarray(X);hand=np.asarray(hand);drum=np.asarray(drum);group=np.asarray(group);xpos=np.asarray(xpos,float)

# home features
raw=readcsv(HOME/"raw_100hz.csv");hits=readcsv(HOME/"hits.csv")
cal=json.loads((HOME/"calibration.json").read_text(encoding="utf-8"))
R0h={h:Rot.from_euler("ZYX",cal[h]["zero"],degrees=True).as_matrix() for h in ("L","R")}
for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"]);r["stage"]=int(r["stage_index"]);r["valid"]=int(r["frame_valid"])
for q in hits:
    q["dt"]=datetime.fromisoformat(q["wall_time"]);q["stage"]=int(q["stage_index"])
sshome={"L":defaultdict(list),"R":defaultdict(list)}
for r in raw:
    if r["stage"] not in (2,4,6,8,10,12) or not r["valid"]:continue
    h=r["hand"]
    sm=make_relative_sample(R0h[h],r["dt"].timestamp()*1000.0,
        float(r["ax"]),float(r["ay"]),float(r["az"]),
        float(r["yaw"]),float(r["pitch"]),float(r["roll"]))
    sshome[h][r["stage"]].append(sm)
def dedup(lst,gap=.18):
    out=[];last=None
    for x in sorted(lst,key=lambda z:z["dt"]):
        if last is None or (x["dt"]-last).total_seconds()>=gap:
            out.append(x);last=x["dt"]
    return out
home=[]
for stage in (2,4,6,8,10,12):
    for h in ("L","R"):
        rr=sshome[h][stage]
        for hit in dedup([q for q in hits if q["stage"]==stage and q["hand"]==h]):
            center=(hit["dt"]-timedelta(milliseconds=80)).timestamp()*1000.0
            ft=feature_from_samples(rr,center,80)
            if ft is not None:home.append((stage,h,ft))
EXPECT_SIDE={2:0,6:1,8:0,10:1,12:1} # stage4=center, excluded

def side_model(kind):
    if kind=="svc":return Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",probability=True,random_state=42))])
    return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=3000,random_state=42))])
def reg_model(kind):
    if kind=="svr":return Pipeline([("s",StandardScaler()),("m",SVR(C=3,epsilon=.08,gamma="scale"))])
    return Pipeline([("s",StandardScaler()),("m",Ridge(alpha=3.0))])

print("=== EXTREME BINARY LEFT/RIGHT, TRAIN 8/12 ONLY ===")
for kind in ("svc","logreg"):
    models={}
    cvpred=[];cvtrue=[]
    for h in ("L","R"):
        idx=np.where((hand==h)&np.isin(drum,list(EXTREME_SIDE)))[0]
        y=np.asarray([EXTREME_SIDE[d] for d in drum[idx]],int)
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        p=np.full(len(idx),-1,int)
        for tr,te in cv.split(X[idx],y,group[idx]):
            m=side_model(kind);m.fit(X[idx][tr],y[tr]);p[te]=m.predict(X[idx][te])
        cvpred.extend(p.tolist());cvtrue.extend(y.tolist())
        m=side_model(kind);m.fit(X[idx],y);models[h]=m
        print(kind,h,"n",len(idx),"CVacc",accuracy_score(y,p),"cm",Counter(zip(y.tolist(),p.tolist())))
    print(kind,"pooled CV",accuracy_score(cvtrue,cvpred),balanced_accuracy_score(cvtrue,cvpred))

    correct=0;n=0;by=defaultdict(list)
    for stage,h,ft in home:
        if stage not in EXPECT_SIDE:continue
        m=models[h]
        prob=m.predict_proba([ft])[0]
        pred=int(m.classes_[int(np.argmax(prob))])
        by[(stage,h)].append((pred,float(np.max(prob))))
        n+=1;correct+=int(pred==EXPECT_SIDE[stage])
    print(kind,"HOME side acc",correct/n,correct,"/",n)
    for key,v in sorted(by.items()):
        print(" ",key,"LEFT",sum(x[0]==0 for x in v),"RIGHT",sum(x[0]==1 for x in v),
              "medconf",round(float(np.median([x[1] for x in v])),3))

print("\n=== CONTINUOUS X REGRESSION, TRAIN 8/12 ONLY ===")
for kind in ("svr","ridge"):
    models={}
    for h in ("L","R"):
        idx=np.where(hand==h)[0]
        # group CV MAE
        gkf=GroupKFold(n_splits=5);pred=np.zeros(len(idx))
        for tr,te in gkf.split(X[idx],xpos[idx],group[idx]):
            m=reg_model(kind);m.fit(X[idx][tr],xpos[idx][tr]);pred[te]=m.predict(X[idx][te])
        print(kind,h,"CV MAE",round(float(np.mean(np.abs(pred-xpos[idx]))),3),
              "sign extreme acc",round(float(np.mean(np.sign(pred[np.abs(xpos[idx])>.8])==np.sign(xpos[idx][np.abs(xpos[idx])>.8]))),3))
        m=reg_model(kind);m.fit(X[idx],xpos[idx]);models[h]=m

    corr=0;n=0;by=defaultdict(list)
    for stage,h,ft in home:
        x=float(models[h].predict([ft])[0]);by[(stage,h)].append(x)
        if stage in EXPECT_SIDE:
            exp=EXPECT_SIDE[stage]
            pr=0 if x<0 else 1
            n+=1;corr+=int(pr==exp)
    print(kind,"HOME sign acc",corr/n,corr,"/",n)
    for key,v in sorted(by.items()):
        print(" ",key,"x median",round(float(np.median(v)),3),"p10/90",np.round(np.percentile(v,[10,90]),3))
