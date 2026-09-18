from pathlib import Path
import csv,sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918";sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES
S={
 "R1":(LAB/"count6_sessions/20260918_185504"/"hit_candidates.csv",True),
 "R3":(LAB/"count6_round3_fresh_r0_sessions/20260918_194947"/"predictions.csv",False),
 "R4":(LAB/"count6_round4_sessioncal_sessions/20260918_195738"/"predictions.csv",False),
}
H={0:0,1:1,2:2,3:0,4:1,5:2};V={0:0,1:0,2:0,3:1,4:1,5:1}
def load(p,counted,s):
    with Path(p).open(encoding="utf-8",newline="") as f:r=list(csv.DictReader(f))
    if counted:r=[x for x in r if x["counted"]=="1"]
    return [(s,x["hand"],int(x["stage_index"]),np.array([float(x[n]) for n in FEATURE_NAMES])) for x in r]
rows=[]
for s,(p,c) in S.items():rows+=load(p,c,s)
subs={"accel":range(0,11),"orient":range(13,31),"full":range(31)}
def model(k):
    if k=="log":return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=2000,class_weight="balanced"))])
    if k=="lin":return Pipeline([("s",StandardScaler()),("m",LinearSVC(C=1,class_weight="balanced"))])
    return Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
for hand in ("L","R"):
  print("\nHAND",hand)
  for task in ("H","V","S"):
    print(" TASK",task)
    for test in S:
      tr=[r for r in rows if r[0]!=test and r[1]==hand];te=[r for r in rows if r[0]==test and r[1]==hand]
      Xtr=np.array([r[3] for r in tr]);Xte=np.array([r[3] for r in te]);st=np.array([r[2] for r in tr]);se=np.array([r[2] for r in te])
      ytr=np.array([H[x] for x in st]) if task=="H" else np.array([V[x] for x in st]) if task=="V" else st
      yte=np.array([H[x] for x in se]) if task=="H" else np.array([V[x] for x in se]) if task=="V" else se
      best=(0,None,None)
      for sn,idx in subs.items():
        idx=list(idx)
        for mk in ("log","lin","lda"):
          try:
            m=model(mk);m.fit(Xtr[:,idx],ytr);a=accuracy_score(yte,m.predict(Xte[:,idx]))
            if a>best[0]:best=(a,sn,mk)
          except:pass
      print("  test",test,"best",round(best[0],3),best[1],best[2])
