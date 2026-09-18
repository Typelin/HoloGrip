from pathlib import Path
import csv,json,sys,joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
SRC=LAB/"count6_sessions/20260918_185504"
OUT=LAB/"count6_calibration_models_185504"
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES

POSITIONS=["左上","中上","右上","左中","正中","右中"]
HCLS={0:0,1:1,2:2,3:0,4:1,5:2}
VCLS={0:0,1:0,2:0,3:1,4:1,5:1}
HNAME=["LEFT","CENTER","RIGHT"]
VNAME=["UP","MID"]

with (SRC/"hit_candidates.csv").open(encoding="utf-8",newline="") as f:
    rows=[r for r in csv.DictReader(f) if r["counted"]=="1"]

def X_for(hand):
    rr=[r for r in rows if r["hand"]==hand]
    X=np.asarray([[float(r[n]) for n in FEATURE_NAMES] for r in rr],float)
    y=np.asarray([int(r["stage_index"]) for r in rr],int)
    return X,y

models={}
for h in ("L","R"):
    X,y=X_for(h)
    # Horizontal model chosen from leave-one-repetition-out benchmark.
    if h=="L":
        h_idx=list(range(19,25))  # mean_r6
        h_model=Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
        v_idx=list(range(31))
        v_model=Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
        s_idx=list(range(31))
        s_model=Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
    else:
        h_idx=list(range(19,25))
        h_model=Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",kernel="rbf",probability=True,random_state=42))])
        v_idx=list(range(11,31))  # rot dispersion + orientation
        v_model=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=5000,class_weight="balanced",random_state=42))])
        s_idx=list(range(19,25))
        s_model=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=5000,class_weight="balanced",random_state=42))])

    yh=np.asarray([HCLS[int(v)] for v in y],int)
    yv=np.asarray([VCLS[int(v)] for v in y],int)

    h_model.fit(X[:,h_idx],yh)
    v_model.fit(X[:,v_idx],yv)
    s_model.fit(X[:,s_idx],y)

    joblib.dump(h_model,OUT/f"H_{h}.joblib")
    joblib.dump(v_model,OUT/f"V_{h}.joblib")
    joblib.dump(s_model,OUT/f"S_{h}.joblib")
    models[h]={
        "H_idx":h_idx,"V_idx":v_idx,"S_idx":s_idx,
        "n":int(len(y)),
        "stage_counts":{POSITIONS[i]:int(np.sum(y==i)) for i in range(6)}
    }

cal=json.loads((SRC/"calibration.json").read_text(encoding="utf-8"))
meta={
    "source_session":str(SRC),
    "positions":POSITIONS,
    "horizontal_names":HNAME,
    "vertical_names":VNAME,
    "models":models,
    "source_calibration":cal,
    "note":"Models are frozen from first-round counted hits only. Round-2 data must never be used to refit during evaluation."
}
(OUT/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(meta,ensure_ascii=False,indent=2))
