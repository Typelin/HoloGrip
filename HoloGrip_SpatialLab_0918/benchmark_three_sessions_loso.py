from pathlib import Path
import csv,sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,f1_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES

SESS={
 "R1":(LAB/"count6_sessions/20260918_185504"/"hit_candidates.csv","counted"),
 "R3":(LAB/"count6_round3_fresh_r0_sessions/20260918_194947"/"predictions.csv","pred"),
 "R4":(LAB/"count6_round4_sessioncal_sessions/20260918_195738"/"predictions.csv","pred"),
}
HMAP={0:0,1:1,2:2,3:0,4:1,5:2}
VMAP={0:0,1:0,2:0,3:1,4:1,5:1}

def load(path,kind,sess):
    with Path(path).open(encoding="utf-8",newline="") as f:r=list(csv.DictReader(f))
    if kind=="counted": r=[x for x in r if x["counted"]=="1"]
    out=[]
    for x in r:
        out.append((sess,x["hand"],int(x["stage_index"]),[float(x[n]) for n in FEATURE_NAMES]))
    return out
rows=[]
for s,(p,k) in SESS.items(): rows+=load(p,k,s)

subsets={
 "center_r6":list(range(13,19)),
 "mean_r6":list(range(19,25)),
 "orientation_all":list(range(13,31)),
 "rot_orientation":list(range(11,31)),
 "accel_only":list(range(0,11)),
 "full31":list(range(31)),
}
def make(k):
    if k=="logreg":return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=5000,class_weight="balanced",random_state=42))])
    if k=="linear":return Pipeline([("s",StandardScaler()),("m",SVC(C=1,kernel="linear",class_weight="balanced",random_state=42))])
    if k=="rbf":return Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",class_weight="balanced",random_state=42))])
    if k=="lda":return Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
    if k=="extra":return ExtraTreesClassifier(n_estimators=500,class_weight="balanced",random_state=42)

print("=== LEAVE ONE SESSION OUT ===")
for hand in ("L","R"):
    print("\nHAND",hand)
    for task in ("H","V","S"):
        print(" TASK",task)
        for test_s in SESS:
            tr=[r for r in rows if r[0]!=test_s and r[1]==hand]
            te=[r for r in rows if r[0]==test_s and r[1]==hand]
            Xtr=np.asarray([r[3] for r in tr]); Xte=np.asarray([r[3] for r in te])
            s_tr=np.asarray([r[2] for r in tr]); s_te=np.asarray([r[2] for r in te])
            if task=="H": ytr=np.asarray([HMAP[int(v)] for v in s_tr]);yte=np.asarray([HMAP[int(v)] for v in s_te])
            elif task=="V": ytr=np.asarray([VMAP[int(v)] for v in s_tr]);yte=np.asarray([VMAP[int(v)] for v in s_te])
            else:ytr=s_tr;yte=s_te
            res=[]
            for sub,idx in subsets.items():
              for kind in ("logreg","linear","rbf","lda","extra"):
                try:
                  m=make(kind);m.fit(Xtr[:,idx],ytr);p=m.predict(Xte[:,idx])
                  res.append((accuracy_score(yte,p),f1_score(yte,p,average="macro"),sub,kind))
                except Exception:pass
            res.sort(reverse=True)
            print("  test",test_s,"best",res[0],"top3",res[:3])
