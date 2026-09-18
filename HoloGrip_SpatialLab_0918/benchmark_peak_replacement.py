from pathlib import Path
import csv,sys
from collections import Counter
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
D=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned"
P=D/"Song2_product_hit_zone_predictions_0821.csv"
GT=D/"Song2_ground_truth_labels_final_0821.csv"

def rcsv(p,enc="utf-8-sig"):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
pred=[{"h":r["hand"],"t":float(r["peak_ms"]),"g":float(r["peak_accel_g"]),"drum":r["pred_drum"]} for r in rcsv(P)]
gt=[{"h":r["final_hand"],"t":float(r["csv_center_ms"]),"drum":r["drum"]} for r in rcsv(GT,"utf-8") if r["final_hand"] in ("L","R")]

def filter_replace(dets,window=160,ratio=1.5,delta=1.0,low_only=None):
    # Process per hand. Suppress an earlier candidate only if a later, substantially stronger
    # candidate appears soon after it. Do not chain across hands.
    out=[]
    for h in ("L","R"):
        q=[d for d in dets if d["h"]==h]
        i=0
        while i<len(q):
            a=q[i]
            if i+1<len(q):
                b=q[i+1];gap=b["t"]-a["t"]
                cond=(gap<=window and b["g"]>=a["g"]*ratio and b["g"]-a["g"]>=delta)
                if low_only is not None:cond=cond and a["g"]<=low_only
                if cond:
                    # suppress a; b remains to be considered against following candidate
                    i+=1
                    continue
            out.append(a);i+=1
    return sorted(out,key=lambda x:(x["t"],x["h"]))

def filter_refractory(dets,ref):
    out=[];last={"L":-1e18,"R":-1e18}
    for d in sorted(dets,key=lambda x:x["t"]):
        if d["t"]-last[d["h"]]>=ref:
            out.append(d);last[d["h"]]=d["t"]
    return out

def score(dets,tol=90):
    used=set();m=[]
    for gi,g in enumerate(gt):
        cand=[]
        for i,d in enumerate(dets):
            if i in used or d["h"]!=g["h"]:continue
            dt=d["t"]-g["t"]
            if abs(dt)<=tol:cand.append((abs(dt),i,dt))
        if cand:
            _,i,dt=min(cand);used.add(i);m.append((gi,i,dt))
    tp=len(m);fp=len(dets)-tp;fn=len(gt)-tp
    p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
    f=2*p*r/(p+r) if p+r else 0
    # zone accuracy using already-produced old predictions; this is only comparative.
    z=sum(gt[gi]["drum"]==dets[i]["drum"] for gi,i,_ in m)
    exact=z
    ep=exact/len(dets) if dets else 0;er=exact/len(gt)
    ef=2*ep*er/(ep+er) if ep+er else 0
    dt=np.array([x[2] for x in m]) if m else np.array([])
    return len(dets),tp,fp,fn,p,r,f,(z/tp if tp else 0),ep,er,ef,(float(np.median(dt)) if len(dt) else None)

print("BASE",score(pred))
print("\nSIMPLE REFRACTORY")
for ref in (100,120,130,140,150,160,170,180,200,220,250):
    s=score(filter_refractory(pred,ref))
    print(ref,"n/tp/fp/fn",s[:4],"P/R/F",tuple(round(x,4) for x in s[4:7]),"jointF",round(s[10],4))

print("\nREPLACE EARLY WEAKER PEAK")
best=[]
for window in (120,130,140,150,160,170,180):
  for ratio in (1.2,1.3,1.4,1.5,1.6,1.8,2.0):
    for delta in (0.5,1.0,1.5,2.0):
      d=filter_replace(pred,window,ratio,delta,None)
      s=score(d)
      best.append((s[6],s[10],-s[2],window,ratio,delta,s,d))
best.sort(reverse=True)
for x in best[:15]:
    _,_,_,w,r,dlt,s,_=x
    print("w/r/d",w,r,dlt,"n/tp/fp/fn",s[:4],"P/R/F",tuple(round(v,4) for v in s[4:7]),"jointF",round(s[10],4))

print("\nREPLACE ONLY IF EARLY PEAK <= threshold")
best2=[]
for low in (3.0,3.5,4.0,4.5,5.0):
 for window in (120,140,160,180):
  for ratio in (1.2,1.4,1.6):
   d=filter_replace(pred,window,ratio,0.5,low)
   s=score(d);best2.append((s[6],s[10],-s[2],low,window,ratio,s))
best2.sort(reverse=True)
for x in best2[:15]:
    _,_,_,low,w,r,s=x
    print("low/w/r",low,w,r,"n/tp/fp/fn",s[:4],"P/R/F",tuple(round(v,4) for v in s[4:7]),"jointF",round(s[10],4))
