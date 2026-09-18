from pathlib import Path
import csv,sys
from collections import Counter
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows, replay_hits, MATCH_TOL_MS, PEAK_LAG_MS, PRODUCT_DETECTOR_KWARGS

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"

with GT.open(encoding="utf-8",newline="") as f:
    gt=[r for r in csv.DictReader(f) if r["final_hand"] in ("L","R")]
rows=load_raw_rows(str(RAW))
dets=replay_hits(rows)

print("GT",len(gt),"DETS",len(dets))

# true same-hand onset gap stats
print("\n=== GT SAME-HAND GAPS ===")
for h in ("L","R"):
    ts=sorted(float(r["csv_center_ms"]) for r in gt if r["final_hand"]==h)
    gaps=np.diff(ts)
    print(h,"n",len(ts),"min",gaps.min(),"p1/5/10/25/50",np.percentile(gaps,[1,5,10,25,50]),"count<80/100/120/150/180",
          [int(np.sum(gaps<x)) for x in (80,100,120,150,180)])

def cluster_max(ds,window_ms):
    # Per hand, cluster consecutive triggers whose gap to previous is < window_ms;
    # keep highest peak_accel in each cluster.
    out=[]
    for h in ("L","R"):
        hh=sorted([d for d in ds if d["hand"]==h],key=lambda x:x["trigger_ms"])
        if not hh:continue
        cur=[hh[0]]
        for d in hh[1:]:
            if d["trigger_ms"]-cur[-1]["trigger_ms"] < window_ms:
                cur.append(d)
            else:
                out.append(max(cur,key=lambda x:float(x.get("peak_accel_g",0))))
                cur=[d]
        out.append(max(cur,key=lambda x:float(x.get("peak_accel_g",0))))
    return sorted(out,key=lambda x:x["trigger_ms"])

def refractory_first(ds,window_ms):
    out=[]
    last={"L":-1e18,"R":-1e18}
    for d in sorted(ds,key=lambda x:x["trigger_ms"]):
        h=d["hand"];t=d["trigger_ms"]
        if t-last[h]>=window_ms:
            out.append(d);last[h]=t
    return out

def match(gt,ds,tol=MATCH_TOL_MS):
    used=set();matched=[]
    for g in gt:
        h=g["final_hand"];t=float(g["csv_center_ms"])
        best=None;bd=None
        for i,d in enumerate(ds):
            if i in used or d["hand"]!=h:continue
            dt=abs(float(d["trigger_ms"])-t)
            if dt<=tol and (bd is None or dt<bd):
                best=i;bd=dt
        if best is not None:
            used.add(best);matched.append((g,ds[best],bd))
    return matched,[d for i,d in enumerate(ds) if i not in used]

def score(ds):
    m,e=match(gt,ds)
    tp=len(m);fp=len(e);fn=len(gt)-tp
    p=tp/(tp+fp) if tp+fp else 0
    r=tp/(tp+fn) if tp+fn else 0
    f=2*p*r/(p+r) if p+r else 0
    return tp,fp,fn,p,r,f,np.median([x[2] for x in m]) if m else None

print("\n=== THRESHOLDS ===")
for method in ("first","max"):
    print("\nMETHOD",method)
    for ms in (0,40,60,80,100,120,150,180,220):
        ds=dets if ms==0 else (refractory_first(dets,ms) if method=="first" else cluster_max(dets,ms))
        tp,fp,fn,p,r,f,med=score(ds)
        print(f"{ms:>3}ms det={len(ds):3} TP={tp:3} FP={fp:3} FN={fn:2} P={p:.4f} R={r:.4f} F1={f:.4f} med_dt={med:.1f}")

# Per-hand best table
print("\n=== PER HAND 100/120/150 max-cluster ===")
for ms in (100,120,150):
    ds=cluster_max(dets,ms)
    for h in ("L","R"):
        g=[x for x in gt if x["final_hand"]==h]
        d=[x for x in ds if x["hand"]==h]
        m,e=match(g,d)
        p=len(m)/len(d) if d else 0;r=len(m)/len(g) if g else 0
        f=2*p*r/(p+r) if p+r else 0
        print(ms,h,"GT",len(g),"det",len(d),"TP",len(m),"P",round(p,4),"R",round(r,4),"F1",round(f,4))
