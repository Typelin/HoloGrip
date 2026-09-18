from pathlib import Path
import csv,json,math,sys
from collections import Counter,defaultdict
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score,confusion_matrix

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
S=LAB/"count6_sessions/20260918_185504"
sys.path.insert(0,str(LAB))
from vnext_core import FEATURE_NAMES,compute_r0,make_relative_sample,feature_from_samples

POSITIONS=["左上","中上","右上","左中","正中","右中"]
HCLASS={0:0,1:1,2:2,3:0,4:1,5:2}
VCLASS={0:0,1:0,2:0,3:1,4:1,5:1} # 0 upper, 1 middle
HNAME=["LEFT","CENTER","RIGHT"]
VNAME=["UP","MID"]

def rcsv(path,enc="utf-8"):
    with Path(path).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

hits=rcsv(S/"hit_candidates.csv")
raw=rcsv(S/"raw_100hz.csv")
stages=rcsv(S/"stage_events.csv")
cal=json.loads((S/"calibration.json").read_text(encoding="utf-8"))

# counted hits only
counted=[r for r in hits if r["counted"]=="1"]
print("COUNTED",len(counted))
for h in ("L","R"):
    print("\nHAND",h)
    for idx,p in enumerate(POSITIONS):
        rr=[r for r in counted if r["hand"]==h and int(r["stage_index"])==idx]
        print(idx,p,len(rr),[r["stage_hand_count_after"] for r in rr])

# invalid frames by mode/stage and episodes
for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"])
    r["stage"]=int(r["stage_index"])
    r["valid"]=int(r["frame_valid"])
print("\n=== INVALID ===")
for h in ("L","R"):
    rr=[r for r in raw if r["hand"]==h and not r["valid"]]
    print(h,len(rr),Counter(r["invalid_reason"] for r in rr),Counter((r["mode"],r["stage"],r["stage_name"]) for r in rr))
    eps=[];cur=[]
    for r in sorted(rr,key=lambda x:x["dt"]):
        if cur and (r["dt"]-cur[-1]["dt"]).total_seconds()>0.2:
            eps.append(cur);cur=[]
        cur.append(r)
    if cur:eps.append(cur)
    for e in eps:
        if len(e)>=2:
            print(" episode",e[0]["dt"].time(),"->",e[-1]["dt"].time(),"n",len(e),"mode/stage",e[0]["mode"],e[0]["stage"],e[0]["stage_name"],Counter(x["invalid_reason"] for x in e))

# candidate stats
print("\n=== CANDIDATE STATS ===")
for h in ("L","R"):
  for idx,p in enumerate(POSITIONS):
    rr=[r for r in hits if r["hand"]==h and int(r["stage_index"])==idx and r["mode"]=="collect"]
    print(h,idx,p,"cand",len(rr),"counted",sum(r["counted"]=="1" for r in rr),"reasons",Counter(r["reason"] for r in rr))

# feature matrix counted
feat_cols=FEATURE_NAMES
X=[];y=[];hands=[];rep=[]
for r in counted:
    X.append([float(r[c]) for c in feat_cols])
    y.append(int(r["stage_index"]))
    hands.append(r["hand"])
    rep.append(int(r["stage_hand_count_after"])-1)
X=np.asarray(X,float);y=np.asarray(y,int);hands=np.asarray(hands);rep=np.asarray(rep,int)

subsets={
    "center_r6":list(range(13,19)),
    "mean_r6":list(range(19,25)),
    "orientation_all":list(range(13,31)),
    "rot+orientation":list(range(11,31)),
    "full31":list(range(31)),
}

def make(kind):
    if kind=="logreg": return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,max_iter=5000,class_weight="balanced",random_state=42))])
    if kind=="svc": return Pipeline([("s",StandardScaler()),("m",SVC(C=2,gamma="scale",kernel="rbf",random_state=42))])
    if kind=="lda": return Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])

def logo_eval(Xi,target,groups,kind):
    pred=np.full(len(target),-1,int)
    logo=LeaveOneGroupOut()
    for tr,te in logo.split(Xi,target,groups):
        m=make(kind);m.fit(Xi[tr],target[tr]);pred[te]=m.predict(Xi[te])
    return accuracy_score(target,pred),pred

print("\n=== WITHIN-SESSION LEAVE-ONE-REPETITION-OUT ===")
best={}
for h in ("L","R"):
    ix=np.where(hands==h)[0]
    print("\nHAND",h)
    for sub,ids in subsets.items():
        row=[]
        for kind in ("logreg","svc","lda"):
            try:
                a6,p6=logo_eval(X[ix][:,ids],y[ix],rep[ix],kind)
                yh=np.array([HCLASS[int(v)] for v in y[ix]])
                av,pv=logo_eval(X[ix][:,ids],yh,rep[ix],kind)
                yv=np.array([VCLASS[int(v)] for v in y[ix]])
                ar,pr=logo_eval(X[ix][:,ids],yv,rep[ix],kind)
                row.append((kind,a6,av,ar))
            except Exception as e:
                row.append((kind,np.nan,np.nan,np.nan))
        print(sub,row)

# centroid distances using mean_r6 Euclidean and reconstruct rotation from mean_r6
def r6_to_rot(v):
    # v flattened M[:,:2] row-major = [m00,m01,m10,m11,m20,m21]
    a=np.array([v[0],v[2],v[4]],float)
    b=np.array([v[1],v[3],v[5]],float)
    a=a/np.linalg.norm(a)
    b=b-a*np.dot(a,b);b=b/np.linalg.norm(b)
    c=np.cross(a,b)
    M=np.column_stack([a,b,c])
    return M
def angle(A,B):
    return float(np.degrees(Rot.from_matrix(A.T@B).magnitude()))

print("\n=== POSITION CENTROID GEODESIC ANGLES FROM mean_r6 ===")
for h in ("L","R"):
    ix=np.where(hands==h)[0]
    centers={}
    spreads={}
    print("\nHAND",h)
    for c in range(6):
        q=ix[y[ix]==c]
        mats=[r6_to_rot(X[j,19:25]) for j in q]
        C=Rot.from_matrix(np.stack(mats)).mean().as_matrix()
        ds=np.array([angle(C,M) for M in mats])
        centers[c]=C;spreads[c]=(np.median(ds),np.max(ds))
        print(c,POSITIONS[c],"within med/max",round(float(np.median(ds)),1),round(float(np.max(ds)),1))
    print("pairwise degrees:")
    print("      "+" ".join(f"{i:>6}" for i in range(6)))
    for i in range(6):
        print(f"{i:>3} "+" ".join(f"{angle(centers[i],centers[j]):6.1f}" for j in range(6)))

# Raw relative orientation distribution during each collect window.
# Use start/complete from stage_events.
events={}
for r in stages:
    if r["event"] in ("STAGE_START","STAGE_COMPLETE"):
        events[(int(r["stage_index"]),r["event"])]=datetime.fromisoformat(r["wall_time"])
R0={h:Rot.from_euler("ZYX",cal[h]["zero"],degrees=True).as_matrix() for h in ("L","R")}
print("\n=== ALL RAW RELATIVE ORIENTATION PER COLLECT STAGE ===")
for h in ("L","R"):
  print("\nHAND",h)
  for idx,p in enumerate(POSITIONS):
    a=events.get((idx,"STAGE_START"));b=events.get((idx,"STAGE_COMPLETE"))
    rr=[r for r in raw if r["hand"]==h and r["valid"] and a and b and a<=r["dt"]<=b]
    mats=[]
    if rr:
      for r in rr:
        Rc=Rot.from_euler("ZYX",[float(r["yaw"]),float(r["pitch"]),float(r["roll"])],degrees=True).as_matrix()
        mats.append(R0[h].T@Rc)
      C=Rot.from_matrix(np.stack(mats)).mean().as_matrix()
      ds=np.array([angle(C,M) for M in mats])
      e=Rot.from_matrix(C).as_euler("ZYX",degrees=True)
      print(idx,p,"n",len(rr),"center rel YPR",np.round(e,1),"spread med/p90",round(float(np.median(ds)),1),round(float(np.percentile(ds,90)),1))

