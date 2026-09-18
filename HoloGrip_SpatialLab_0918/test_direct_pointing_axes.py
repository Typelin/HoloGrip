from pathlib import Path
import csv,sys,itertools
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES

SESS={
 "R1":(LAB/"count6_sessions/20260918_185504"/"hit_candidates.csv","counted"),
 "R3":(LAB/"count6_round3_fresh_r0_sessions/20260918_194947"/"predictions.csv","pred"),
 "R4":(LAB/"count6_round4_sessioncal_sessions/20260918_195738"/"predictions.csv","pred"),
}
HMAP={0:0,1:1,2:2,3:0,4:1,5:2} # L,C,R
VMAP={0:0,1:0,2:0,3:1,4:1,5:1} # UP,MID
POS=["LU","CU","RU","LM","CM","RM"]

def load(p,kind,sess):
    with Path(p).open(encoding="utf-8",newline="") as f:r=list(csv.DictReader(f))
    if kind=="counted":r=[x for x in r if x["counted"]=="1"]
    out=[]
    for x in r:
        out.append({
            "sess":sess,"hand":x["hand"],"stage":int(x["stage_index"]),
            "center":np.array([float(x[n]) for n in FEATURE_NAMES[13:19]]),
            "mean":np.array([float(x[n]) for n in FEATURE_NAMES[19:25]]),
        })
    return out
rows=[]
for s,(p,k) in SESS.items():rows+=load(p,k,s)

def r6_to_R(v):
    c1=np.array([v[0],v[2],v[4]],float)
    c2=np.array([v[1],v[3],v[5]],float)
    c1/=np.linalg.norm(c1)
    c2-=c1*np.dot(c1,c2);c2/=np.linalg.norm(c2)
    c3=np.cross(c1,c2)
    return np.column_stack([c1,c2,c3])

# all proper signed permutation frames Q
Qs=[]
for perm in itertools.permutations(range(3)):
  P=np.eye(3)[:,perm]
  for signs in itertools.product([-1,1],repeat=3):
    Q=P@np.diag(signs)
    if np.linalg.det(Q)>0.5:
      Qs.append((perm,signs,Q))

def angles(R,Q):
    # express current forward direction (Q[:,0]) in neutral pointing frame Q
    w=Q.T @ R @ Q[:,0]
    h=np.degrees(np.arctan2(w[1],w[0]))
    v=np.degrees(np.arctan2(w[2],np.hypot(w[0],w[1])))
    return float(h),float(v)

def train_1d(vals,labels,ncls):
    med=[]
    for k in range(ncls):
        a=np.array([x for x,y in zip(vals,labels) if y==k],float)
        med.append(float(np.median(a)))
    return med

def pred_nearest(x,med):
    return int(np.argmin([abs(x-m) for m in med]))

def eval_frame(hand,src,Q):
    # Train only R1 medians.
    tr=[r for r in rows if r["hand"]==hand and r["sess"]=="R1"]
    if src=="center":
        Rs=[r6_to_R(r["center"]) for r in tr]
    else:Rs=[r6_to_R(r["mean"]) for r in tr]
    av=[angles(R,Q) for R in Rs]
    hmed=train_1d([x[0] for x in av],[HMAP[r["stage"]] for r in tr],3)
    vmed=train_1d([x[1] for x in av],[VMAP[r["stage"]] for r in tr],2)

    # Prefer true ordinal medians and large separation.
    monotonic=(hmed[0]<hmed[1]<hmed[2]) or (hmed[0]>hmed[1]>hmed[2])
    hsep=min(abs(hmed[0]-hmed[1]),abs(hmed[1]-hmed[2]))
    vsep=abs(vmed[0]-vmed[1])
    results={}
    for sess in ("R1","R3","R4"):
        te=[r for r in rows if r["hand"]==hand and r["sess"]==sess]
        aa=[angles(r6_to_R(r[src]),Q) for r in te]
        hp=[pred_nearest(a[0],hmed) for a in aa]
        vp=[pred_nearest(a[1],vmed) for a in aa]
        ht=[HMAP[r["stage"]] for r in te];vt=[VMAP[r["stage"]] for r in te]
        hacc=np.mean(np.array(hp)==np.array(ht));vacc=np.mean(np.array(vp)==np.array(vt))
        # exact 6 from H/V pair
        pair_to_stage={(0,0):0,(1,0):1,(2,0):2,(0,1):3,(1,1):4,(2,1):5}
        sp=[pair_to_stage[(a,b)] for a,b in zip(hp,vp)]
        sacc=np.mean(np.array(sp)==np.array([r["stage"] for r in te]))
        results[sess]=(hacc,vacc,sacc)
    # select using R1 geometry only, not test accuracy
    score=(1000 if monotonic else 0)+hsep+vsep
    return score,hmed,vmed,results

for hand in ("L","R"):
  print("\n=== HAND",hand,"===")
  candidates=[]
  for src in ("center","mean"):
    for perm,signs,Q in Qs:
      score,hmed,vmed,res=eval_frame(hand,src,Q)
      candidates.append((score,src,perm,signs,Q,hmed,vmed,res))
  candidates.sort(key=lambda x:x[0],reverse=True)
  for rank,c in enumerate(candidates[:8],1):
    score,src,perm,signs,Q,hmed,vmed,res=c
    print("rank",rank,"src",src,"perm",perm,"signs",signs,
          "Hmed",np.round(hmed,1),"Vmed",np.round(vmed,1),
          "R1/R3/R4", {k:tuple(round(z,3) for z in v) for k,v in res.items()})
  # Also plain relative ZYX Euler from center/mean.
  for src in ("center","mean"):
    tr=[r for r in rows if r["hand"]==hand and r["sess"]=="R1"]
    def ev(r):
      e=Rot.from_matrix(r6_to_R(r[src])).as_euler("ZYX",degrees=True)
      return float(e[0]),float(e[1])
    aa=[ev(r) for r in tr]
    hm=train_1d([x[0] for x in aa],[HMAP[r["stage"]] for r in tr],3)
    vm=train_1d([x[1] for x in aa],[VMAP[r["stage"]] for r in tr],2)
    print("plainZYX",src,"Hmed",np.round(hm,1),"Vmed",np.round(vm,1),end=" ")
    for sess in ("R1","R3","R4"):
      te=[r for r in rows if r["hand"]==hand and r["sess"]==sess]
      a=[ev(r) for r in te]
      hp=[pred_nearest(x[0],hm) for x in a];vp=[pred_nearest(x[1],vm) for x in a]
      ht=[HMAP[r["stage"]] for r in te];vt=[VMAP[r["stage"]] for r in te]
      pair={(0,0):0,(1,0):1,(2,0):2,(0,1):3,(1,1):4,(2,1):5}
      sp=[pair[(x,y)] for x,y in zip(hp,vp)]
      print(sess,tuple(round(z,3) for z in (np.mean(np.array(hp)==ht),np.mean(np.array(vp)==vt),np.mean(np.array(sp)==[r["stage"] for r in te]))),end=" ")
    print()
