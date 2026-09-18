from pathlib import Path
import csv,sys,json,joblib,warnings
from collections import Counter,defaultdict
from datetime import datetime,timedelta
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

warnings.filterwarnings("ignore")
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
HOME=LAB/"protocol_sessions/20260918_155836"

DRUM_POS={
    "小鼓":     {"x": 0.22,"z":0.36},
    "高音 Tom": {"x": 0.16,"z":0.62},
    "中音 Tom": {"x":-0.16,"z":0.62},
    "落地 Tom": {"x":-0.48,"z":0.36},
    "Hi-Hat":   {"x": 0.56,"z":0.42},
    "Crash":    {"x": 0.54,"z":0.68},
    "Ride":     {"x":-0.56,"z":0.58},
}
# screen/drummer perspective: +x = LEFT, -x = RIGHT
def hlabel(d):
    x=DRUM_POS[d]["x"]
    return 0 if x>0.30 else (2 if x<-0.30 else 1)
def vlabel(d):
    return 1 if DRUM_POS[d]["z"]>=0.52 else 0  # 1=UP,0=LOW
H_NAMES=["LEFT","CENTER","RIGHT"]
V_NAMES=["LOW","UP"]
SECTOR_NAMES=["LEFT_LOW","LEFT_UP","CENTER_LOW","CENTER_UP","RIGHT_LOW","RIGHT_UP"]
def sector_id(d):
    return hlabel(d)*2 + vlabel(d)

def readcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

def prep12():
    rows=readcsv(RAW12)
    data={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        R0,zero=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
        ss=[make_relative_sample(R0,float(r["song_time_ms"]),
            float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
            float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
        data[h]={"R0":R0,"zero":zero,"samples":ss}
    return data

D12=prep12()
gt=readcsv(GT12,"utf-8")
X=[];hand=[];groups=[];drums=[];yh=[];yv=[];ys=[]
for r in gt:
    h=r["final_hand"];d=r["drum"]
    if h not in ("L","R") or d not in DRUM_POS:continue
    c=float(r["csv_center_ms"])
    ft=feature_from_samples(D12[h]["samples"],c,80)
    if ft is None:continue
    X.append(ft);hand.append(h);groups.append(int(c//4000));drums.append(d)
    yh.append(hlabel(d));yv.append(vlabel(d));ys.append(sector_id(d))
X=np.asarray(X);hand=np.asarray(hand);groups=np.asarray(groups)
yh=np.asarray(yh);yv=np.asarray(yv);ys=np.asarray(ys)

def make(kind):
    if kind=="svc":
        return Pipeline([("s",StandardScaler()),("m",SVC(C=2.0,gamma="scale",probability=True,random_state=42))])
    if kind=="mlp":
        return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(48,24),max_iter=1800,random_state=42))])
    if kind=="extra":
        return ExtraTreesClassifier(n_estimators=500,max_features="sqrt",class_weight="balanced",random_state=42)
    raise ValueError

def cv_score(target,names,kind):
    pred=np.full(len(target),-1,int)
    for h in ("L","R"):
        idx=np.where(hand==h)[0]
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        for tr0,te0 in cv.split(X[idx],target[idx],groups[idx]):
            m=make(kind);m.fit(X[idx][tr0],target[idx][tr0]);pred[idx[te0]]=m.predict(X[idx][te0])
    return {
        "acc":float(accuracy_score(target,pred)),
        "bal":float(balanced_accuracy_score(target,pred)),
        "f1":float(f1_score(target,pred,average="macro")),
        "pred":pred,
    }

print("=== 8/12 TARGET COUNTS ===")
for h in ("L","R"):
    idx=np.where(hand==h)[0]
    print(h,"horizontal",Counter(yh[idx].tolist()),"vertical",Counter(yv[idx].tolist()),"sector",Counter(ys[idx].tolist()))

bench={}
for task,target,names in [("H",yh,H_NAMES),("V",yv,V_NAMES),("S",ys,SECTOR_NAMES)]:
    print("\nTASK",task)
    bench[task]={}
    for kind in ("svc","mlp","extra"):
        try:
            s=cv_score(target,names,kind);bench[task][kind]={k:v for k,v in s.items() if k!="pred"}
            print(kind,bench[task][kind])
            print(confusion_matrix(target,s["pred"],labels=list(range(len(names)))))
        except Exception as e:
            print(kind,"FAIL",repr(e))

# choose by balanced + macroF1
choice={}
targets={"H":yh,"V":yv,"S":ys}
namesmap={"H":H_NAMES,"V":V_NAMES,"S":SECTOR_NAMES}
for task in ("H","V","S"):
    choice[task]=max(bench[task],key=lambda k:bench[task][k]["bal"]+bench[task][k]["f1"])
print("\nCHOICE",choice)

out=LAB/"song2_spatial_models"
out.mkdir(parents=True,exist_ok=True)
models={}
for task,target in targets.items():
    models[task]={}
    for h in ("L","R"):
        idx=np.where(hand==h)[0]
        m=make(choice[task]);m.fit(X[idx],target[idx])
        joblib.dump(m,out/f"song2_{task}_{h}.joblib")
        models[task][h]=m

# HOME ZERO-SHOT reconstruct features around logged hits
raw=readcsv(HOME/"raw_100hz.csv","utf-8")
hits=readcsv(HOME/"hits.csv","utf-8")
cal=json.loads((HOME/"calibration.json").read_text(encoding="utf-8"))
R0h={h:Rot.from_euler("ZYX",cal[h]["zero"],degrees=True).as_matrix() for h in ("L","R")}

for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"])
    r["stage"]=int(r["stage_index"])
    r["valid"]=int(r["frame_valid"])
for h in hits:
    h["dt"]=datetime.fromisoformat(h["wall_time"]);h["stage"]=int(h["stage_index"])

home_samples={"L":defaultdict(list),"R":defaultdict(list)}
for r in raw:
    if r["stage"] not in (2,4,6,8,10,12) or not r["valid"]:continue
    hh=r["hand"]
    sm=make_relative_sample(R0h[hh],r["dt"].timestamp()*1000.0,
        float(r["ax"]),float(r["ay"]),float(r["az"]),
        float(r["yaw"]),float(r["pitch"]),float(r["roll"]))
    home_samples[hh][r["stage"]].append(sm)

def dedup(lst,min_gap=.18):
    out=[];last=None
    for x in sorted(lst,key=lambda x:x["dt"]):
        if last is None or (x["dt"]-last).total_seconds()>=min_gap:
            out.append(x);last=x["dt"]
    return out

# expected stage sectors based on user's instructed geometry
EXPECT={
    2:("LEFT","UP"),
    4:("CENTER","UP"),
    6:("RIGHT","UP"),
    8:("LEFT",None),
    10:("RIGHT",None),
    12:("RIGHT","UP"),
}
records=[]
print("\n=== 9/18 HOME ZERO-SHOT (trained ONLY 8/12) ===")
for idx in (2,4,6,8,10,12):
    print("\nSTAGE",idx,EXPECT[idx])
    for hh in ("L","R"):
        Hs=dedup([x for x in hits if x["stage"]==idx and x["hand"]==hh])
        ss=home_samples[hh][idx]
        preds=[]
        for hit in Hs:
            center=(hit["dt"]-timedelta(milliseconds=80)).timestamp()*1000.0
            ft=feature_from_samples(ss,center,80)
            if ft is None:continue
            ph=models["H"][hh].predict_proba([ft])[0]; hv=int(models["H"][hh].classes_[int(np.argmax(ph))])
            pv=models["V"][hh].predict_proba([ft])[0]; vv=int(models["V"][hh].classes_[int(np.argmax(pv))])
            ps=models["S"][hh].predict_proba([ft])[0]; sv=int(models["S"][hh].classes_[int(np.argmax(ps))])
            preds.append((hv,float(np.max(ph)),vv,float(np.max(pv)),sv,float(np.max(ps))))
            records.append([idx,hh,*preds[-1]])
        if preds:
            hc=Counter(H_NAMES[p[0]] for p in preds);vc=Counter(V_NAMES[p[2]] for p in preds);sc=Counter(SECTOR_NAMES[p[4]] for p in preds)
            eh,ev=EXPECT[idx]
            hacc=np.mean([H_NAMES[p[0]]==eh for p in preds]) if eh else np.nan
            vacc=np.mean([V_NAMES[p[2]]==ev for p in preds]) if ev else np.nan
            print(hh,"n",len(preds),"H",dict(hc),"Hacc",round(float(hacc),3) if eh else None,
                  "V",dict(vc),"Vacc",round(float(vacc),3) if ev else None,
                  "S",dict(sc))

# Overall horizontal catastrophic side score on stages with explicit left/right
side_records=[r for r in records if r[0] in (2,6,8,10,12)]
wrong=0;n=0
for r in side_records:
    idx,hh,hv,hprob,vv,vprob,sv,sprob=r
    exp=EXPECT[idx][0]
    if exp in ("LEFT","RIGHT"):
        n+=1
        if H_NAMES[hv]!=exp:wrong+=1
print("\nHOME_SIDE_ACCURACY", (n-wrong)/n if n else None, "wrong",wrong,"n",n)

meta={
    "source":"8/12 Song2 307 single-hand GT only",
    "drum_positions":DRUM_POS,
    "horizontal_names":H_NAMES,
    "vertical_names":V_NAMES,
    "sector_names":SECTOR_NAMES,
    "model_choice":choice,
    "cv":bench,
    "home_zero_shot_side_accuracy":(n-wrong)/n if n else None,
    "home_zero_shot_side_wrong":wrong,
    "home_zero_shot_side_n":n,
}
(out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
with (out/"home_zero_shot_predictions.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.writer(f);w.writerow(["stage","hand","h_id","h_prob","v_id","v_prob","sector_id","sector_prob"]);w.writerows(records)
print("SAVED",out)
