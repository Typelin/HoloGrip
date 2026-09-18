from pathlib import Path
import csv,sys,json,itertools
from collections import defaultdict,Counter
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
S=LAB/"count6_sessions/20260918_185504"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,FEATURE_NAMES

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"

POS=["左上","中上","右上","左中","正中","右中"]
# Current calibration positions correspond to physical 8/12 drum sectors.
TARGET_GROUPS={
    0:["Crash"],
    1:["高音 Tom","中音 Tom"],
    2:["Ride"],
    3:["Hi-Hat"],
    4:["小鼓"],
    5:["落地 Tom"],
}
HCLS={0:0,1:1,2:2,3:0,4:1,5:2}
VCLS={0:0,1:0,2:0,3:1,4:1,5:1}

def rcsv(p,enc="utf-8-sig"):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))

def r6_to_rot(v):
    a=np.array([v[0],v[2],v[4]],float)
    b=np.array([v[1],v[3],v[5]],float)
    a=a/np.linalg.norm(a)
    b=b-a*np.dot(a,b); b=b/np.linalg.norm(b)
    c=np.cross(a,b)
    return np.column_stack([a,b,c])
def ang(A,B):
    return float(np.degrees(Rot.from_matrix(A.T@B).magnitude()))
def meanR(mats):
    return Rot.from_matrix(np.stack(mats)).mean().as_matrix()

# Historical 8/12 hit mean_r6 rotations
rows=rcsv(RAW12); D={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
        float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
hist={"L":defaultdict(list),"R":defaultdict(list)}
for r in rcsv(GT12,"utf-8"):
    h=r["final_hand"];d=r["drum"]
    if h not in ("L","R"):continue
    ft=feature_from_samples(D[h],float(r["csv_center_ms"]),80)
    if ft is not None:hist[h][d].append(r6_to_rot(ft[19:25]))

# Current counted mean_r6 rotations
hits=rcsv(S/"hit_candidates.csv","utf-8")
cur={"L":defaultdict(list),"R":defaultdict(list)}
for r in hits:
    if r["counted"]!="1":continue
    h=r["hand"];st=int(r["stage_index"])
    v=np.array([float(r[n]) for n in FEATURE_NAMES[19:25]])
    cur[h][st].append(r6_to_rot(v))

# target centroids
T={"L":{},"R":{}}
C={"L":{},"R":{}}
for h in ("L","R"):
    for st,drs in TARGET_GROUPS.items():
        T[h][st]=meanR([M for d in drs for M in hist[h][d]])
        C[h][st]=meanR(cur[h][st])

# A maps current sensor-coordinate relative rotations into 8/12 sensor-coordinate:
# R_old ~= A R_cur A^T
def fit_A(hand,train_stages):
    def loss(rv):
        A=Rot.from_rotvec(rv).as_matrix()
        ds=[]
        for st in train_stages:
            M=A@C[hand][st]@A.T
            ds.append(ang(M,T[hand][st])**2)
        return np.mean(ds)
    best=None
    starts=[np.zeros(3)]
    # multiple deterministic starts
    for axis in np.eye(3):
        for deg in (45,-45,90,-90,135):
            starts.append(np.deg2rad(deg)*axis)
    for x0 in starts:
        res=minimize(loss,x0,method="BFGS",options={"maxiter":1000,"gtol":1e-10})
        if best is None or res.fun<best.fun:best=res
    return Rot.from_rotvec(best.x).as_matrix(),best.fun

def nearest_stage(hand,M):
    ds={st:ang(M,T[hand][st]) for st in range(6)}
    return min(ds,key=ds.get),ds

print("=== BEFORE / AFTER ALL-6 CONJUGATION ALIGNMENT ===")
As={}
for h in ("L","R"):
    A,loss=fit_A(h,range(6));As[h]=A
    print("\nHAND",h,"A rotvec deg",np.round(np.degrees(Rot.from_matrix(A).as_rotvec()),2),"angle",round(float(np.degrees(Rot.from_matrix(A).magnitude())),2))
    bef=[];aft=[]
    for st in range(6):
        b=ang(C[h][st],T[h][st]); a=ang(A@C[h][st]@A.T,T[h][st])
        bef.append(b);aft.append(a)
        print(st,POS[st],"before",round(b,1),"after",round(a,1))
    print("mean before",round(float(np.mean(bef)),1),"after",round(float(np.mean(aft)),1))

print("\n=== ALL-6 FIT: CLASSIFY INDIVIDUAL CURRENT HITS TO HISTORICAL 6 SECTORS ===")
for h in ("L","R"):
    A=As[h];correct=0;hc=0;vc=0;n=0;cm=np.zeros((6,6),int)
    for st in range(6):
        for M in cur[h][st]:
            Mt=A@M@A.T
            pr,ds=nearest_stage(h,Mt)
            cm[st,pr]+=1;n+=1;correct+=pr==st;hc+=HCLS[pr]==HCLS[st];vc+=VCLS[pr]==VCLS[st]
    print("\nHAND",h,"six",correct,"/",n,round(correct/n,3),"H",round(hc/n,3),"V",round(vc/n,3))
    print(cm)

print("\n=== LEAVE-ONE-POSITION-OUT ALIGNMENT ===")
for h in ("L","R"):
    correct=hc=vc=n=0
    print("\nHAND",h)
    for test in range(6):
        train=[s for s in range(6) if s!=test]
        A,_=fit_A(h,train)
        preds=[]
        for M in cur[h][test]:
            pr,_=nearest_stage(h,A@M@A.T);preds.append(pr)
            n+=1;correct+=pr==test;hc+=HCLS[pr]==HCLS[test];vc+=VCLS[pr]==VCLS[test]
        print("test",test,POS[test],"preds",preds,Counter(preds))
    print("total six",correct,"/",n,round(correct/n,3),"H",round(hc/n,3),"V",round(vc/n,3))

print("\n=== 3-ANCHOR CALIBRATION SEARCH, EVALUATE OTHER 3 ===")
for h in ("L","R"):
    rowsout=[]
    for train in itertools.combinations(range(6),3):
        A,_=fit_A(h,train)
        tests=[s for s in range(6) if s not in train]
        cc=hh=vv=nn=0
        for st in tests:
            for M in cur[h][st]:
                pr,_=nearest_stage(h,A@M@A.T)
                nn+=1;cc+=pr==st;hh+=HCLS[pr]==HCLS[st];vv+=VCLS[pr]==VCLS[st]
        rowsout.append((hh/nn,vv/nn,cc/nn,train,tests))
    rowsout.sort(reverse=True)
    print("\nHAND",h,"TOP 8 by horizontal")
    for row in rowsout[:8]:
        print("H/V/6",tuple(round(x,3) for x in row[:3]),"train",row[3],"test",row[4])

