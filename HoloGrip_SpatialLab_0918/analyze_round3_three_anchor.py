from pathlib import Path
import csv,sys,itertools
from collections import defaultdict,Counter
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
R1=LAB/"count6_sessions/20260918_185504"
R3=LAB/"count6_round3_fresh_r0_sessions/20260918_194947"
sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES

POS=["左上","中上","右上","左中","正中","右中"]
HCLS={0:0,1:1,2:2,3:0,4:1,5:2}
VCLS={0:0,1:0,2:0,3:1,4:1,5:1}

def rcsv(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
a=[r for r in rcsv(R1/"hit_candidates.csv") if r["counted"]=="1"]
b=rcsv(R3/"predictions.csv")

def r6_to_R(v):
    a=np.array([v[0],v[2],v[4]],float)
    c2=np.array([v[1],v[3],v[5]],float)
    a/=np.linalg.norm(a)
    c2-=a*np.dot(a,c2); c2/=np.linalg.norm(c2)
    c3=np.cross(a,c2)
    return np.column_stack([a,c2,c3])

def meanR(ms):
    return Rot.from_matrix(np.stack(ms)).mean().as_matrix()
def ang(A,B):
    return float(np.degrees(Rot.from_matrix(A.T@B).magnitude()))

C1={"L":{},"R":{}}; C3={"L":{},"R":{}}
HITS3={"L":defaultdict(list),"R":defaultdict(list)}
for h in ("L","R"):
    for s in range(6):
        aa=[r for r in a if r["hand"]==h and int(r["stage_index"])==s]
        bb=[r for r in b if r["hand"]==h and int(r["stage_index"])==s]
        ma=[r6_to_R([float(r[n]) for n in FEATURE_NAMES[19:25]]) for r in aa]
        mb=[r6_to_R([float(r[n]) for n in FEATURE_NAMES[19:25]]) for r in bb]
        C1[h][s]=meanR(ma); C3[h][s]=meanR(mb); HITS3[h][s]=mb

def fit_A(h,train):
    def loss(rv):
        A=Rot.from_rotvec(rv).as_matrix()
        return np.mean([ang(A@C3[h][s]@A.T,C1[h][s])**2 for s in train])
    starts=[np.zeros(3)]
    for axis in np.eye(3):
        for deg in (30,-30,60,-60,90,-90,120,-120,150,-150):
            starts.append(np.deg2rad(deg)*axis)
    best=None
    for x0 in starts:
        res=minimize(loss,x0,method="BFGS",options={"maxiter":800})
        if best is None or res.fun<best.fun: best=res
    return Rot.from_rotvec(best.x).as_matrix(),float(best.fun)

def predict_stage(h,M,A):
    Mt=A@M@A.T
    ds={s:ang(Mt,C1[h][s]) for s in range(6)}
    return min(ds,key=ds.get),ds

print("=== ROUND3 FRESH-R0 -> ROUND1 CANONICAL, 3-ANCHOR HELD-OUT ===")
allrows={}
for h in ("L","R"):
    rows=[]
    for train in itertools.combinations(range(6),3):
        A,loss=fit_A(h,train)
        tests=[s for s in range(6) if s not in train]
        n=cc=hc=vc=0
        per={}
        for s in tests:
            preds=[]
            for M in HITS3[h][s]:
                pr,_=predict_stage(h,M,A)
                preds.append(pr);n+=1
                cc+=pr==s;hc+=HCLS[pr]==HCLS[s];vc+=VCLS[pr]==VCLS[s]
            per[s]=preds
        rows.append({
            "train":train,"test":tuple(tests),"H":hc/n,"V":vc/n,"S":cc/n,
            "loss":loss,"per":per,
            "angle":float(np.degrees(Rot.from_matrix(A).magnitude()))
        })
    rows.sort(key=lambda r:(r["H"],r["S"],r["V"],-r["loss"]),reverse=True)
    allrows[h]=rows
    print("\nHAND",h,"TOP 10")
    for r in rows[:10]:
        print("train",r["train"],[POS[i] for i in r["train"]],
              "test",r["test"],"H/V/S",round(r["H"],3),round(r["V"],3),round(r["S"],3),
              "Adeg",round(r["angle"],1),"loss",round(r["loss"],1),
              "per",{POS[k]:Counter(v) for k,v in r["per"].items()})

# Joint anchor set: same 3 physical anchors for both hands. Rank min horizontal then avg S/V.
print("\n=== SAME 3 ANCHORS FOR BOTH HANDS ===")
joint=[]
for train in itertools.combinations(range(6),3):
    lr=[]
    for h in ("L","R"):
        r=next(x for x in allrows[h] if x["train"]==train)
        lr.append(r)
    joint.append((min(x["H"] for x in lr),
                  np.mean([x["H"] for x in lr]),
                  np.mean([x["S"] for x in lr]),
                  np.mean([x["V"] for x in lr]),
                  train,lr))
joint.sort(reverse=True)
for row in joint[:12]:
    mn,avh,avs,avv,train,lr=row
    print("anchors",train,[POS[i] for i in train],
          "minH",round(mn,3),"avgH",round(avh,3),"avgS",round(avs,3),"avgV",round(avv,3),
          "L",tuple(round(lr[0][k],3) for k in ("H","V","S")),
          "R",tuple(round(lr[1][k],3) for k in ("H","V","S")))
