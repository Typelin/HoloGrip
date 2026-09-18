from pathlib import Path
import csv,json,sys,joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
SRC=LAB/"count6_round3_fresh_r0_sessions/20260918_194947"
OUT=LAB/"count6_calibration_models_round3_194947"
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES

POSITIONS=["左上","中上","右上","左中","正中","右中"]
HCLS={0:0,1:1,2:2,3:0,4:1,5:2}
VCLS={0:0,1:0,2:0,3:1,4:1,5:1}
HNAME=["LEFT","CENTER","RIGHT"]; VNAME=["UP","MID"]

with (SRC/"predictions.csv").open(encoding="utf-8",newline="") as f:
    rows=list(csv.DictReader(f))

def make_models_for_hand(h):
    rr=[r for r in rows if r["hand"]==h]
    X=np.asarray([[float(r[n]) for n in FEATURE_NAMES] for r in rr],float)
    y=np.asarray([int(r["stage_index"]) for r in rr],int)
    rep=np.asarray([(int(r["count_after"])-1) for r in rr],int)
    yh=np.asarray([HCLS[int(v)] for v in y],int)
    yv=np.asarray([VCLS[int(v)] for v in y],int)

    candidates={
      "H_mean_lda": (list(range(19,25)), lambda:Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])),
      "H_mean_svc": (list(range(19,25)), lambda:Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",probability=True,random_state=42))])),
      "H_orient_log": (list(range(13,31)), lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=5000,class_weight="balanced",random_state=42))])),
      "V_full_lda": (list(range(31)), lambda:Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])),
      "V_rot_log": (list(range(11,31)), lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=5000,class_weight="balanced",random_state=42))])),
      "V_orient_svc": (list(range(13,31)), lambda:Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",probability=True,random_state=42))])),
      "S_full_lda": (list(range(31)), lambda:Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])),
      "S_mean_log": (list(range(19,25)), lambda:Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=5000,class_weight="balanced",random_state=42))])),
      "S_full_svc": (list(range(31)), lambda:Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",probability=True,random_state=42))])),
    }

    logo=LeaveOneGroupOut()
    scores={}
    for name,(idxs,factory) in candidates.items():
        if name.startswith("H_"): target=yh
        elif name.startswith("V_"): target=yv
        else: target=y
        p=np.full(len(target),-1,int)
        for tr,te in logo.split(X[:,idxs],target,rep):
            m=factory();m.fit(X[tr][:,idxs],target[tr]);p[te]=m.predict(X[te][:,idxs])
        scores[name]=float(accuracy_score(target,p))

    hbest=max((k for k in scores if k.startswith("H_")),key=lambda k:scores[k])
    vbest=max((k for k in scores if k.startswith("V_")),key=lambda k:scores[k])
    sbest=max((k for k in scores if k.startswith("S_")),key=lambda k:scores[k])

    result={}
    for task,best,target in [("H",hbest,yh),("V",vbest,yv),("S",sbest,y)]:
        idxs,factory=candidates[best]
        m=factory();m.fit(X[:,idxs],target)
        joblib.dump(m,OUT/f"{task}_{h}.joblib")
        result[task]={"candidate":best,"cv_acc":scores[best],"idx":idxs}
    result["n"]=len(y)
    result["all_scores"]=scores
    return result

meta={"source_session":str(SRC),"positions":POSITIONS,"H_names":HNAME,"V_names":VNAME,"hands":{}}
for h in ("L","R"):
    meta["hands"][h]=make_models_for_hand(h)
meta["source_r0"]=json.loads((SRC/"fresh_r0_calibration.json").read_text(encoding="utf-8"))
(OUT/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(meta,ensure_ascii=False,indent=2))
