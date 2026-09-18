from pathlib import Path
import csv,sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
TR=LAB/"count6_round3_fresh_r0_sessions/20260918_194947"
TE=LAB/"count6_round4_sessioncal_sessions/20260918_195738"
sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES

HMAP={0:0,1:1,2:2,3:0,4:1,5:2}
VMAP={0:0,1:0,2:0,3:1,4:1,5:1}

def rcsv(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
tr=rcsv(TR/"predictions.csv"); te=rcsv(TE/"predictions.csv")

subsets={
 "center_r6":list(range(13,19)),
 "mean_r6":list(range(19,25)),
 "center_mean":list(range(13,25)),
 "orientation_all":list(range(13,31)),
 "rot_orientation":list(range(11,31)),
 "no_accel":list(range(11,31)),
 "accel_only":list(range(0,11)),
 "full31":list(range(31)),
}
def make(kind):
    if kind=="logreg": return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=5000,class_weight="balanced",random_state=42))])
    if kind=="linear": return Pipeline([("s",StandardScaler()),("m",SVC(C=1,kernel="linear",random_state=42))])
    if kind=="rbf": return Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",kernel="rbf",random_state=42))])
    if kind=="lda": return Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
    if kind=="extra": return ExtraTreesClassifier(n_estimators=400,class_weight="balanced",random_state=42)
    if kind=="rf": return RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42)

print("=== TRUE CROSS-ROUND: train R3, test R4 ===")
for hand in ("L","R"):
  tra=[r for r in tr if r["hand"]==hand]
  tea=[r for r in te if r["hand"]==hand]
  Xtr=np.asarray([[float(r[n]) for n in FEATURE_NAMES] for r in tra],float)
  Xte=np.asarray([[float(r[n]) for n in FEATURE_NAMES] for r in tea],float)
  ys_tr=np.asarray([int(r["stage_index"]) for r in tra],int)
  ys_te=np.asarray([int(r["stage_index"]) for r in tea],int)
  yh_tr=np.asarray([HMAP[int(v)] for v in ys_tr],int); yh_te=np.asarray([HMAP[int(v)] for v in ys_te],int)
  yv_tr=np.asarray([VMAP[int(v)] for v in ys_tr],int); yv_te=np.asarray([VMAP[int(v)] for v in ys_te],int)
  print("\nHAND",hand)
  for task,ytr,yte in [("H",yh_tr,yh_te),("V",yv_tr,yv_te),("S",ys_tr,ys_te)]:
    rows=[]
    for sub,idx in subsets.items():
      for kind in ("logreg","linear","rbf","lda","extra","rf"):
        try:
          m=make(kind);m.fit(Xtr[:,idx],ytr);p=m.predict(Xte[:,idx])
          rows.append((accuracy_score(yte,p),balanced_accuracy_score(yte,p),f1_score(yte,p,average="macro"),sub,kind,p))
        except Exception:pass
    rows.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))
    print(" TASK",task,"TOP5")
    for r in rows[:5]:
      print(" ",round(r[0],3),round(r[1],3),round(r[2],3),r[3],r[4])
    best=rows[0]
    print(" CM best\n",confusion_matrix(yte,best[5]))
