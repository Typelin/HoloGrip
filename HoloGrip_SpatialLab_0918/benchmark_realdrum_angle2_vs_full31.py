from pathlib import Path
import csv,sys,itertools,warnings
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
import joblib
warnings.filterwarnings("ignore")

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,ZONE_NAMES
from product_hit_and_zone import ZONE_MAP

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
RAW5=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
GT5=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"

SIDE={
    "Crash":0,"Hi-Hat":0,
    "高音 Tom":1,"中音 Tom":1,"小鼓":1,
    "Ride":2,"落地 Tom":2,
}
SNAME=["LEFT","CENTER","RIGHT"]

def readcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

def prep(rawp):
    rows=readcsv(rawp);out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
        out[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
                    float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
    return out
D12=prep(RAW12);D5=prep(RAW5)

def build12():
    X=[];y=[];h=[];g=[];dr=[]
    for r in readcsv(GT12,"utf-8"):
        hh=r["final_hand"];d=r["drum"]
        if hh not in ("L","R") or d not in ZONE_MAP:continue
        c=float(r["csv_center_ms"]);ft=feature_from_samples(D12[hh],c,80)
        if ft is not None:
            X.append(ft);y.append(ZONE_MAP[d]);h.append(hh);g.append(int(c//4000));dr.append(d)
    return np.asarray(X),np.asarray(y),np.asarray(h),np.asarray(g),np.asarray(dr)
def build5():
    X=[];y=[];h=[];dr=[]
    for r in readcsv(GT5):
        hh=r["assigned_hand"];d=r["zone_name"]
        if hh not in ("L","R") or d not in ZONE_MAP:continue
        c=float(r["aligned_csv_time_ms"]);ft=feature_from_samples(D5[hh],c,80)
        if ft is not None:
            X.append(ft);y.append(ZONE_MAP[d]);h.append(hh);dr.append(d)
    return np.asarray(X),np.asarray(y),np.asarray(h),np.asarray(dr)
X,y,h,g,dr=build12();X5,y5,h5,dr5=build5()

def r6_to_R(v):
    c1=np.array([v[0],v[2],v[4]],float);c2=np.array([v[1],v[3],v[5]],float)
    c1/=np.linalg.norm(c1);c2-=c1*np.dot(c1,c2);c2/=np.linalg.norm(c2);c3=np.cross(c1,c2)
    return np.column_stack([c1,c2,c3])

Qs=[]
for perm in itertools.permutations(range(3)):
    P=np.eye(3)[:,perm]
    for signs in itertools.product([-1,1],repeat=3):
        Q=P@np.diag(signs)
        if np.linalg.det(Q)>0.5:Qs.append((perm,signs,Q))

def angle2(ft,src,Q):
    v=ft[13:19] if src=="center" else ft[19:25]
    R=r6_to_R(v)
    w=Q.T@R@Q[:,0]
    hh=np.degrees(np.arctan2(w[1],w[0]))
    vv=np.degrees(np.arctan2(w[2],np.hypot(w[0],w[1])))
    return [hh,vv]

print("8/12 n",len(y),"8/5 n",len(y5),"8/5 counts",Counter(dr5.tolist()))

for hand in ("L","R"):
    idx=np.where(h==hand)[0];idx5=np.where(h5==hand)[0]
    best=None
    for src in ("center","mean"):
      for perm,signs,Q in Qs:
        Z=np.asarray([angle2(X[i],src,Q) for i in idx])
        pred=np.full(len(idx),-1,int)
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        ok=True
        try:
          for tr,te in cv.split(Z,y[idx],g[idx]):
            m=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
            m.fit(Z[tr],y[idx][tr]);pred[te]=m.predict(Z[te])
        except Exception:
          ok=False
        if not ok:continue
        acc=accuracy_score(y[idx],pred);bal=balanced_accuracy_score(y[idx],pred);f1=f1_score(y[idx],pred,average="macro")
        score=bal+0.25*acc
        if best is None or score>best[0]:
          best=(score,src,perm,signs,Q,acc,bal,f1)
    _,src,perm,signs,Q,acc,bal,f1=best
    Z=np.asarray([angle2(X[i],src,Q) for i in idx]);Z5=np.asarray([angle2(X5[i],src,Q) for i in idx5])
    m=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(Z,y[idx]);p=m.predict(Z5)
    a=accuracy_score(y5[idx5],p);b=balanced_accuracy_score(y5[idx5],p);f=f1_score(y5[idx5],p,average="macro")
    print("\nANGLE2",hand,"src",src,"perm",perm,"signs",signs,"CV",round(acc,3),round(bal,3),round(f1,3),"CROSS",round(a,3),round(b,3),round(f,3))
    print(confusion_matrix(y5[idx5],p,labels=range(7)))
    # side catastrophic
    st=[SIDE[d] for d in dr5[idx5]]
    ps=[SIDE[ZONE_NAMES[int(x)]] for x in p]
    opp=sum((aa==0 and bb==2) or (aa==2 and bb==0) for aa,bb in zip(st,ps))
    side_n=sum(aa in (0,2) for aa in st)
    print("catastrophic L<->R",opp,"/",side_n)
    # angle centroids train
    for k,nm in enumerate(ZONE_NAMES):
        q=Z[y[idx]==k]
        if len(q):print(" ",nm,"n",len(q),"med",np.round(np.median(q,axis=0),1),"p10/90 h",np.round(np.percentile(q[:,0],[10,90]),1),"v",np.round(np.percentile(q[:,1],[10,90]),1))

print("\n=== FULL31 MLP CROSS 8/12->8/5 ===")
models={hh:joblib.load(LAB/"vnext_models"/f"hologrip_vnext_{hh}_relative_rotation.joblib") for hh in ("L","R")}
pp=np.full(len(y5),-1,int)
for hh in ("L","R"):
    ii=np.where(h5==hh)[0];pp[ii]=models[hh].predict(X5[ii])
print("ACC",accuracy_score(y5,pp),"BAL",balanced_accuracy_score(y5,pp),"F1",f1_score(y5,pp,average="macro"))
print(confusion_matrix(y5,pp,labels=range(7)))
st=[SIDE[d] for d in dr5];ps=[SIDE[ZONE_NAMES[int(x)]] for x in pp]
opp=sum((aa==0 and bb==2) or (aa==2 and bb==0) for aa,bb in zip(st,ps))
side_n=sum(aa in (0,2) for aa in st)
print("catastrophic L<->R",opp,"/",side_n)
for k,nm in enumerate(ZONE_NAMES):
    ix=np.where(y5==k)[0]
    if len(ix):
        print(nm,len(ix),"correct",int(np.sum(pp[ix]==k)),"preds",Counter(ZONE_NAMES[int(x)] for x in pp[ix]))
