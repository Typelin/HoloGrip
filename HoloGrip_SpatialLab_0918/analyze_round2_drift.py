from pathlib import Path
import csv,sys,json
from collections import Counter,defaultdict
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
R1=LAB/"count6_sessions/20260918_185504"
R2=LAB/"count6_round2_sessions/20260918_190558"
sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES

POS=["左上","中上","右上","左中","正中","右中"]
H=["LEFT","CENTER","RIGHT"]; V=["UP","MID"]

def rcsv(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
a=[r for r in rcsv(R1/"hit_candidates.csv") if r["counted"]=="1"]
b=rcsv(R2/"predictions.csv")

print("=== ROUND2 STAGE ACC ===")
for hand in ("L","R"):
  print("\nHAND",hand)
  for s,name in enumerate(POS):
    q=[r for r in b if r["hand"]==hand and int(r["stage_index"])==s]
    print(s,name,"n",len(q),
          "H",sum(int(r["H_ok"]) for r in q),"/",len(q),Counter(r["H_pred"] for r in q),
          "V",sum(int(r["V_ok"]) for r in q),"/",len(q),Counter(r["V_pred"] for r in q),
          "S",sum(int(r["S_ok"]) for r in q),"/",len(q),Counter(r["S_pred"] for r in q))

def r6_to_R(v):
    a=np.array([v[0],v[2],v[4]],float); b=np.array([v[1],v[3],v[5]],float)
    a/=np.linalg.norm(a); b-=a*np.dot(a,b); b/=np.linalg.norm(b); c=np.cross(a,b)
    return np.column_stack([a,b,c])
def meanR(mats): return Rot.from_matrix(np.stack(mats)).mean().as_matrix()
def ang(A,B): return float(np.degrees(Rot.from_matrix(A.T@B).magnitude()))

# same-stage mean_r6 centroid drift between rounds
C1={"L":{},"R":{}}; C2={"L":{},"R":{}}
for hand in ("L","R"):
  for s in range(6):
    x1=[r for r in a if r["hand"]==hand and int(r["stage_index"])==s]
    x2=[r for r in b if r["hand"]==hand and int(r["stage_index"])==s]
    C1[hand][s]=meanR([r6_to_R([float(r[n]) for n in FEATURE_NAMES[19:25]]) for r in x1])
    C2[hand][s]=meanR([r6_to_R([float(r[n]) for n in FEATURE_NAMES[19:25]]) for r in x2])

print("\n=== SAME POSITION CENTROID DRIFT ROUND1->ROUND2 ===")
for hand in ("L","R"):
  ds=[]
  print("\nHAND",hand)
  for s in range(6):
    d=ang(C1[hand][s],C2[hand][s]);ds.append(d)
    print(s,POS[s],round(d,1))
  print("mean",round(float(np.mean(ds)),1),"median",round(float(np.median(ds)),1),"max",round(float(np.max(ds)),1))

# Test if one left-multiplication B aligns all round2 centroids to round1.
# This approximates world-frame heading drift when old R0 is kept fixed.
def fit_left(hand,train=range(6)):
    def loss(rv):
        B=Rot.from_rotvec(rv).as_matrix()
        return np.mean([ang(B@C2[hand][s],C1[hand][s])**2 for s in train])
    best=None
    starts=[np.zeros(3)]
    for axis in np.eye(3):
      for deg in (30,-30,60,-60,90,-90,120,-120):
        starts.append(np.deg2rad(deg)*axis)
    for x0 in starts:
      res=minimize(loss,x0,method="BFGS",options={"maxiter":1000})
      if best is None or res.fun<best.fun:best=res
    B=Rot.from_rotvec(best.x).as_matrix()
    return B,best.fun

print("\n=== COMMON ROTATION ALIGNMENT ROUND2->ROUND1 ===")
for hand in ("L","R"):
  B,_=fit_left(hand)
  rv=np.degrees(Rot.from_matrix(B).as_rotvec())
  bef=[ang(C2[hand][s],C1[hand][s]) for s in range(6)]
  aft=[ang(B@C2[hand][s],C1[hand][s]) for s in range(6)]
  print("\nHAND",hand,"B rotvec deg",np.round(rv,2),"angle",round(float(np.degrees(Rot.from_matrix(B).magnitude())),2))
  for s in range(6):print(s,POS[s],"before",round(bef[s],1),"after",round(aft[s],1))
  print("mean",round(float(np.mean(bef)),1),"->",round(float(np.mean(aft)),1))

# invalid episodes in round2
raw=rcsv(R2/"raw_100hz.csv")
from datetime import datetime
for r in raw:
  r["dt"]=datetime.fromisoformat(r["wall_time"]);r["valid"]=int(r["valid"]);r["stage"]=int(r["stage_index"])
print("\n=== ROUND2 INVALID ===")
for hand in ("L","R"):
  q=sorted([r for r in raw if r["hand"]==hand and not r["valid"]],key=lambda r:r["dt"])
  print(hand,len(q),Counter(r["invalid_reason"] for r in q),Counter((r["mode"],r["stage"],r["stage_name"]) for r in q))
  eps=[];cur=[]
  for r in q:
    if cur and (r["dt"]-cur[-1]["dt"]).total_seconds()>0.2:
      eps.append(cur);cur=[]
    cur.append(r)
  if cur:eps.append(cur)
  for e in eps:
    if len(e)>=5:
      print(" ",e[0]["dt"].time(),"->",e[-1]["dt"].time(),"n",len(e),"stage",e[0]["stage"],e[0]["stage_name"],Counter(x["invalid_reason"] for x in e))
