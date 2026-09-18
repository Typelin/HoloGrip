from pathlib import Path
import csv
from collections import Counter,defaultdict
from datetime import datetime
import numpy as np

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918\next_person_realdrum_sessions\20260918_202322")

def rcsv(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
hits=rcsv(S/"hits.csv");raw=rcsv(S/"raw_100hz.csv")
print("HITS",len(hits),Counter(r["hand"] for r in hits))
for h in ("L","R"):
    q=[r for r in hits if r["hand"]==h]
    print("\nHAND",h)
    if not q:continue
    t=np.array([float(r["peak_ms"]) for r in q])
    dt=np.diff(t)
    print("interval counts <130",int(np.sum(dt<130)),"130-180",int(np.sum((dt>=130)&(dt<180))),"<250",int(np.sum(dt<250)))
    print("intervals",np.round(dt,1).tolist())
    print("preds",Counter(r["raw_drum"] for r in q))
    # Show short-gap pairs
    print("SHORT PAIRS <=160ms")
    for i,d in enumerate(dt):
        if d<=160:
            a,b=q[i],q[i+1]
            print(round(d,1),
                  a["raw_drum"],"->",b["raw_drum"],
                  "yaw",round(float(a["rel_yaw"]),1),"->",round(float(b["rel_yaw"]),1),
                  "pitch",round(float(a["rel_pitch"]),1),"->",round(float(b["rel_pitch"]),1),
                  "g",round(float(a["peak_g"]),1),"->",round(float(b["peak_g"]),1))

# invalid episodes
for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"]);r["valid"]=int(r["valid"])
print("\nINVALID EPISODES")
for h in ("L","R"):
    q=sorted([r for r in raw if r["hand"]==h and not r["valid"]],key=lambda x:x["dt"])
    print("\n",h,"invalid",len(q),Counter(r["invalid_reason"] for r in q))
    eps=[];cur=[]
    for r in q:
        if cur and (r["dt"]-cur[-1]["dt"]).total_seconds()>0.15:
            eps.append(cur);cur=[]
        cur.append(r)
    if cur:eps.append(cur)
    print("episodes",len(eps))
    for e in sorted(eps,key=len,reverse=True)[:20]:
        dur=(e[-1]["dt"]-e[0]["dt"]).total_seconds()
        print(e[0]["dt"].time(),"->",e[-1]["dt"].time(),"n",len(e),"dur",round(dur,3),Counter(x["invalid_reason"] for x in e))

# R valid run lengths between invalids
rraw=[r for r in raw if r["hand"]=="R"]
runs=[];n=0
for r in rraw:
    if r["valid"]:n+=1
    else:
        if n:runs.append(n);n=0
if n:runs.append(n)
print("\nR valid-run packet lengths", "n_runs",len(runs),"median",np.median(runs) if runs else None,"p10",np.percentile(runs,10) if runs else None,"min",min(runs) if runs else None)
