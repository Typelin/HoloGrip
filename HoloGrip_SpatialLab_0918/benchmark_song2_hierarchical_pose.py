from pathlib import Path
import csv,sys,json
from collections import Counter,defaultdict
from datetime import datetime,timedelta
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,FEATURE_NAMES

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
HOME=LAB/"protocol_sessions/20260918_155836"

HMAP={"Crash":0,"Hi-Hat":0,"高音 Tom":1,"中音 Tom":1,"小鼓":1,"Ride":2,"落地 Tom":2}
VMAP={"Crash":1,"Hi-Hat":0,"高音 Tom":1,"中音 Tom":1,"小鼓":0,"Ride":1,"落地 Tom":0}
HN=["LEFT","CENTER","RIGHT"]; VN=["LOW","UP"]
EXPECT={2:(0,1),4:(1,1),6:(2,1),8:(0,0),10:(2,0),12:(2,1)}

def rcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=rcsv(RAW);D={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
        float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
X=[];hand=[];grp=[];yh=[];yv=[];drum=[]
for r in rcsv(GT,"utf-8"):
    h=r["final_hand"];d=r["drum"]
    if h not in ("L","R") or d not in HMAP:continue
    ft=feature_from_samples(D[h],float(r["csv_center_ms"]),80)
    if ft is not None:X.append(ft);hand.append(h);grp.append(int(float(r["csv_center_ms"])//4000));yh.append(HMAP[d]);yv.append(VMAP[d]);drum.append(d)
X=np.asarray(X);hand=np.asarray(hand);grp=np.asarray(grp);yh=np.asarray(yh);yv=np.asarray(yv);drum=np.asarray(drum)

# home
raw=rcsv(HOME/"raw_100hz.csv");hits=rcsv(HOME/"hits.csv")
cal=json.loads((HOME/"calibration.json").read_text(encoding="utf-8"))
R0h={h:Rot.from_euler("ZYX",cal[h]["zero"],degrees=True).as_matrix() for h in ("L","R")}
for r in raw:r["dt"]=datetime.fromisoformat(r["wall_time"]);r["stage"]=int(r["stage_index"]);r["valid"]=int(r["frame_valid"])
for z in hits:z["dt"]=datetime.fromisoformat(z["wall_time"]);z["stage"]=int(z["stage_index"])
ss={"L":defaultdict(list),"R":defaultdict(list)}
for r in raw:
    if r["stage"] not in EXPECT or not r["valid"]:continue
    h=r["hand"]
    ss[h][r["stage"]].append(make_relative_sample(R0h[h],r["dt"].timestamp()*1000,float(r["ax"]),float(r["ay"]),float(r["az"]),float(r["yaw"]),float(r["pitch"]),float(r["roll"])))
def dedup(lst,g=.18):
    out=[];last=None
    for z in sorted(lst,key=lambda a:a["dt"]):
        if last is None or (z["dt"]-last).total_seconds()>=g:out.append(z);last=z["dt"]
    return out
home=[]
for st in EXPECT:
    for h in ("L","R"):
        for hit in dedup([z for z in hits if z["stage"]==st and z["hand"]==h]):
            c=(hit["dt"]-timedelta(milliseconds=80)).timestamp()*1000
            ft=feature_from_samples(ss[h][st],c,80)
            if ft is not None:home.append((st,h,ft))

subsets={
 "center_r6":list(range(13,19)),
 "mean_r6":list(range(19,25)),
 "center_mean":list(range(13,25)),
 "orientation_all":list(range(13,31)),
}
def model():
    return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=3000,class_weight="balanced",random_state=42))])

for sub,idxs in subsets.items():
    # H three-way per hand
    hm={};vm={};cvh=[];cvht=[];cvv=[];cvvt=[]
    for h in ("L","R"):
        ix=np.where(hand==h)[0]
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        ph=np.full(len(ix),-1);pv=np.full(len(ix),-1)
        for tr,te in cv.split(X[ix][:,idxs],yh[ix],grp[ix]):
            mh=model();mh.fit(X[ix][tr][:,idxs],yh[ix][tr]);ph[te]=mh.predict(X[ix][te][:,idxs])
            mv=model();mv.fit(X[ix][tr][:,idxs],yv[ix][tr]);pv[te]=mv.predict(X[ix][te][:,idxs])
        cvh.extend(ph.tolist());cvht.extend(yh[ix].tolist());cvv.extend(pv.tolist());cvvt.extend(yv[ix].tolist())
        mh=model();mh.fit(X[ix][:,idxs],yh[ix]);hm[h]=mh
        mv=model();mv.fit(X[ix][:,idxs],yv[ix]);vm[h]=mv
    hc=vc=joint=0;n=0;stage=defaultdict(list)
    for st,h,ft in home:
        hp=int(hm[h].predict([ft[idxs]])[0]);vp=int(vm[h].predict([ft[idxs]])[0])
        he,ve=EXPECT[st]
        hc+=hp==he;vc+=vp==ve;joint+=(hp==he and vp==ve);n+=1
        stage[st].append((hp,vp))
    print("\n",sub,"CV_H",accuracy_score(cvht,cvh),"CV_V",accuracy_score(cvvt,cvv),
          "HOME_H",hc/n,"HOME_V",vc/n,"HOME_joint",joint/n)
    for st,v in sorted(stage.items()):
        hc2=Counter(HN[a] for a,b in v);vc2=Counter(VN[b] for a,b in v);jc=Counter((HN[a],VN[b]) for a,b in v)
        print(st,"exp",HN[EXPECT[st][0]],VN[EXPECT[st][1]],"H",dict(hc2),"V",dict(vc2),"J",dict(jc))
