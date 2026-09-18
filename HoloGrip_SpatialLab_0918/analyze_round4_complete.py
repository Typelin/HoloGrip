from pathlib import Path
import csv,sys,json
from collections import Counter,defaultdict
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
TR=LAB/"count6_round3_fresh_r0_sessions/20260918_194947"
TE=LAB/"count6_round4_sessioncal_sessions/20260918_195738"
sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES

POS=["左上","中上","右上","左中","正中","右中"]
HMAP={0:0,1:1,2:2,3:0,4:1,5:2}; HN=["LEFT","CENTER","RIGHT"]
VMAP={0:0,1:0,2:0,3:1,4:1,5:1}; VN=["UP","MID"]

def rcsv(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
tr=rcsv(TR/"predictions.csv")
te=rcsv(TE/"predictions.csv")
raw=rcsv(TE/"raw_100hz.csv")
stage=rcsv(TE/"stage_events.csv")

print("=== ROUND4 PER-STAGE ACC ===")
for h in ("L","R"):
    print("\nHAND",h)
    for s in range(6):
        q=[r for r in te if r["hand"]==h and int(r["stage_index"])==s]
        print(s,POS[s],
              "H",sum(int(r["H_ok"]) for r in q),"/",len(q),dict(Counter(r["H_pred"] for r in q)),
              "V",sum(int(r["V_ok"]) for r in q),"/",len(q),dict(Counter(r["V_pred"] for r in q)),
              "S",sum(int(r["S_ok"]) for r in q),"/",len(q),dict(Counter(r["S_pred"] for r in q)))

# invalid frames by stage/time
for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"]);r["stage"]=int(r["stage_index"]);r["valid"]=int(r["valid"])
print("\n=== ROUND4 INVALID ===")
for h in ("L","R"):
    q=sorted([r for r in raw if r["hand"]==h and not r["valid"]],key=lambda r:r["dt"])
    print(h,"n",len(q),"reasons",dict(Counter(r["invalid_reason"] for r in q)),
          "stage",dict(Counter((r["stage"],r["stage_name"],r["mode"]) for r in q)))
    eps=[];cur=[]
    for r in q:
        if cur and (r["dt"]-cur[-1]["dt"]).total_seconds()>0.20:
            eps.append(cur);cur=[]
        cur.append(r)
    if cur:eps.append(cur)
    for e in eps:
        if len(e)>=2:
            print(" episode",e[0]["dt"].time(),"->",e[-1]["dt"].time(),"n",len(e),
                  "stage",e[0]["stage"],e[0]["stage_name"],"mode",e[0]["mode"],
                  "reasons",dict(Counter(x["invalid_reason"] for x in e)))

def r6_to_R(v):
    c1=np.array([v[0],v[2],v[4]],float)
    c2=np.array([v[1],v[3],v[5]],float)
    c1/=np.linalg.norm(c1)
    c2-=c1*np.dot(c1,c2); c2/=np.linalg.norm(c2)
    c3=np.cross(c1,c2)
    return np.column_stack([c1,c2,c3])
def meanR(ms): return Rot.from_matrix(np.stack(ms)).mean().as_matrix()
def ang(A,B): return float(np.degrees(Rot.from_matrix(A.T@B).magnitude()))

# same-stage centroid drift based on center_r6 and mean_r6
print("\n=== SAME-POSITION ORIENTATION DRIFT R3->R4 ===")
for subset,name in [(range(13,19),"center_r6"),(range(19,25),"mean_r6")]:
    print("\nSUBSET",name)
    for h in ("L","R"):
        ds=[]
        print("HAND",h)
        for s in range(6):
            qa=[r for r in tr if r["hand"]==h and int(r["stage_index"])==s]
            qb=[r for r in te if r["hand"]==h and int(r["stage_index"])==s]
            A=meanR([r6_to_R([float(r[FEATURE_NAMES[i]]) for i in subset]) for r in qa])
            B=meanR([r6_to_R([float(r[FEATURE_NAMES[i]]) for i in subset]) for r in qb])
            d=ang(A,B);ds.append(d)
            print(s,POS[s],round(d,1))
        print(" mean/median/max",round(float(np.mean(ds)),1),round(float(np.median(ds)),1),round(float(np.max(ds)),1))

# standardized feature shift: fit scaler on R3 per hand; compare nearest same-position centroid
print("\n=== STANDARDIZED FEATURE CENTROID SHIFT ===")
for h in ("L","R"):
    xa=np.asarray([[float(r[n]) for n in FEATURE_NAMES] for r in tr if r["hand"]==h])
    ya=np.asarray([int(r["stage_index"]) for r in tr if r["hand"]==h])
    xb=np.asarray([[float(r[n]) for n in FEATURE_NAMES] for r in te if r["hand"]==h])
    yb=np.asarray([int(r["stage_index"]) for r in te if r["hand"]==h])
    sc=StandardScaler().fit(xa)
    za=sc.transform(xa);zb=sc.transform(xb)
    ca={s:za[ya==s].mean(0) for s in range(6)}
    cb={s:zb[yb==s].mean(0) for s in range(6)}
    print("\nHAND",h)
    for s in range(6):
        same=float(np.linalg.norm(cb[s]-ca[s]))
        nearest=min(range(6),key=lambda k:np.linalg.norm(cb[s]-ca[k]))
        nd=float(np.linalg.norm(cb[s]-ca[nearest]))
        print(s,POS[s],"same_zdist",round(same,2),"nearest_train",nearest,POS[nearest],"dist",round(nd,2))

# compare movement intensity / feature distributions
print("\n=== PEAK / ROT DYNAMICS R3 vs R4 ===")
for h in ("L","R"):
  print("\nHAND",h)
  for s in range(6):
    a=[r for r in tr if r["hand"]==h and int(r["stage_index"])==s]
    b=[r for r in te if r["hand"]==h and int(r["stage_index"])==s]
    def med(rr,k):return float(np.median([float(r[k]) for r in rr]))
    print(s,POS[s],
          "peak",round(med(a,"peak_g"),2),"->",round(med(b,"peak_g"),2),
          "rot_range",round(med(a,"rot_angle_range"),3),"->",round(med(b,"rot_angle_range"),3),
          "rot_std",round(med(a,"rot_angle_std"),3),"->",round(med(b,"rot_angle_std"),3))
