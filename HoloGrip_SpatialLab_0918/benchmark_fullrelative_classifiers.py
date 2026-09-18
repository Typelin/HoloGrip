from pathlib import Path
import csv,sys,json,warnings
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

warnings.filterwarnings("ignore")
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
sys.path.insert(0,str(LAB))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import ZONE_MAP,ZONE_NAMES

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
RAW5=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
GT5=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"

def readcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

def prep(rawp):
    rows=readcsv(rawp)
    out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        R0,zero=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
        ss=[make_relative_sample(R0,float(r["song_time_ms"]),
            float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
            float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
        out[h]={"zero":zero,"samples":ss}
    return out

D12=prep(RAW12);D5=prep(RAW5)

def build12():
    gt=readcsv(GT12,"utf-8")
    X=[];y=[];h=[];g=[]
    for r in gt:
        hh=r["final_hand"];d=r["drum"]
        if hh not in ("L","R") or d not in ZONE_MAP:continue
        c=float(r["csv_center_ms"]);ft=feature_from_samples(D12[hh]["samples"],c,80)
        if ft is None:continue
        X.append(ft);y.append(ZONE_MAP[d]);h.append(hh);g.append(int(c//4000))
    return np.asarray(X),np.asarray(y),np.asarray(h),np.asarray(g)
def build5():
    gt=readcsv(GT5)
    X=[];y=[];h=[];meta=[]
    for r in gt:
        hh=r["assigned_hand"];d=r["zone_name"]
        if hh not in ("L","R") or d not in ZONE_MAP:continue
        c=float(r["aligned_csv_time_ms"]);ft=feature_from_samples(D5[hh]["samples"],c,80)
        if ft is None:continue
        X.append(ft);y.append(ZONE_MAP[d]);h.append(hh);meta.append(r)
    return np.asarray(X),np.asarray(y),np.asarray(h),meta

X,y,h,g=build12();X5,y5,h5,meta=build5()

def make(name):
    if name=="mlp":
        return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1800,random_state=42))])
    if name=="mlp_small":
        return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(32,),alpha=0.003,max_iter=1800,random_state=42))])
    if name=="linear_svc":
        return Pipeline([("s",StandardScaler()),("m",SVC(C=1.0,kernel="linear",probability=True,random_state=42))])
    if name=="rbf_svc":
        return Pipeline([("s",StandardScaler()),("m",SVC(C=2.0,gamma="scale",probability=True,random_state=42))])
    if name=="logreg":
        return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1.0,max_iter=3000,class_weight="balanced",random_state=42))])
    if name=="lda":
        return Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
    if name=="extra":
        return ExtraTreesClassifier(n_estimators=600,max_features="sqrt",min_samples_leaf=1,class_weight="balanced",random_state=42)
    if name=="rf":
        return RandomForestClassifier(n_estimators=600,max_features="sqrt",class_weight="balanced",random_state=42)
    if name=="hgb":
        return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    raise ValueError(name)

def mets(yy,pp):
    return {
        "acc":float(accuracy_score(yy,pp)),
        "bal":float(balanced_accuracy_score(yy,pp)),
        "f1":float(f1_score(yy,pp,average="macro")),
    }

results={}
for name in ["mlp","linear_svc","logreg","extra"]:
    print("\n===",name,"===")
    pred=np.full(len(y),-1,int)
    fail=False
    for hh in ("L","R"):
        idx=np.where(h==hh)[0]
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        try:
            for tr0,te0 in cv.split(X[idx],y[idx],g[idx]):
                m=make(name);m.fit(X[idx][tr0],y[idx][tr0]);pred[idx[te0]]=m.predict(X[idx][te0])
        except Exception as e:
            print("CV_FAIL",hh,repr(e));fail=True;break
    if fail:continue
    cvm=mets(y,pred)
    print("CV",cvm)

    p5=np.full(len(y5),-1,int)
    models={}
    try:
        for hh in ("L","R"):
            tr=np.where(h==hh)[0];te=np.where(h5==hh)[0]
            m=make(name);m.fit(X[tr],y[tr]);p5[te]=m.predict(X5[te]);models[hh]=m
    except Exception as e:
        print("CROSS_FAIL",repr(e));continue
    xsm=mets(y5,p5)
    print("CROSS",xsm)
    for k,nm in enumerate(ZONE_NAMES):
        ix=np.where(y5==k)[0]
        if len(ix):print(" ",nm,len(ix),round(float(np.mean(p5[ix]==k)),3),dict(Counter(p5[ix].tolist())))
    results[name]={"cv":cvm,"cross":xsm}

print("\n=== RANK BY CROSS BAL + 0.25 ACC ===")
rank=sorted(results.items(),key=lambda kv:kv[1]["cross"]["bal"]+0.25*kv[1]["cross"]["acc"],reverse=True)
for n,r in rank:
    print(n,r)
(LAB/"classifier_benchmark_results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")

