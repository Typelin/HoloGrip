from pathlib import Path
import csv, math
from datetime import datetime
from collections import defaultdict
import numpy as np

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918\vnext_live_sessions\20260918_152635")
with (S/"raw.csv").open(encoding="utf-8",newline="") as f: raw=list(csv.DictReader(f))
with (S/"hits.csv").open(encoding="utf-8",newline="") as f: hits=list(csv.DictReader(f))
for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"])
    for k in ("ax","ay","az","yaw","pitch","roll"): r[k]=float(r[k])
for h in hits: h["dt"]=datetime.fromisoformat(h["wall_time"])

def corrupt(r):
    # exact same raw word reused across all six values -> angle ~= accel*11.25
    acc_same=max(abs(r["ax"]-r["ay"]),abs(r["ay"]-r["az"]),abs(r["ax"]-r["az"]))<1e-6
    ang_same=max(abs(r["yaw"]-r["pitch"]),abs(r["pitch"]-r["roll"]),abs(r["yaw"]-r["roll"]))<1e-6
    scale=abs(r["yaw"]-r["ax"]*11.25)<0.02
    return acc_same and ang_same and scale and abs(r["ax"])>0.01

def zero6(r):
    return all(abs(r[k])<1e-12 for k in ("ax","ay","az","yaw","pitch","roll"))

for hand in ("L","R"):
    rr=[r for r in raw if r["hand"]==hand]
    bad=[r for r in rr if corrupt(r)]
    zz=[r for r in rr if zero6(r)]
    print("\nHAND",hand,"rows",len(rr),"corrupt",len(bad),f"{len(bad)/len(rr)*100:.3f}%","zero6",len(zz),f"{len(zz)/len(rr)*100:.3f}%")
    # episodes where corrupt/zero states separated by <=0.25 sec
    flags=sorted([r for r in rr if corrupt(r) or zero6(r)], key=lambda x:x["dt"])
    eps=[];cur=[]
    for r in flags:
        if cur and (r["dt"]-cur[-1]["dt"]).total_seconds()>0.25:
            eps.append(cur);cur=[]
        cur.append(r)
    if cur:eps.append(cur)
    for e in eps:
        dur=(e[-1]["dt"]-e[0]["dt"]).total_seconds()
        if len(e)>=3 or dur>0.1:
            nbad=sum(corrupt(x) for x in e);nz=sum(zero6(x) for x in e)
            print(" episode",e[0]["dt"].time(),"->",e[-1]["dt"].time(),"dur",round(dur,3),"rows",len(e),"corrupt",nbad,"zero6",nz)

# mark hits occurring within +/-0.25s of corrupt/zero raw
for hand in ("L","R"):
    rr=[r for r in raw if r["hand"]==hand and (corrupt(r) or zero6(r))]
    hh=[h for h in hits if h["hand"]==hand]
    contaminated=[]
    for h in hh:
        if any(abs((r["dt"]-h["dt"]).total_seconds())<=0.25 for r in rr):
            contaminated.append(h)
    print("\n",hand,"hits",len(hh),"contaminated_hits",len(contaminated))
    if contaminated:
        print(" first",contaminated[0]["dt"].time(),"last",contaminated[-1]["dt"].time())

# clean hit counts excluding around bad raw
for hand in ("L","R"):
    rr=[r for r in raw if r["hand"]==hand and (corrupt(r) or zero6(r))]
    hh=[h for h in hits if h["hand"]==hand]
    clean=[h for h in hh if not any(abs((r["dt"]-h["dt"]).total_seconds())<=0.25 for r in rr)]
    print(hand,"clean hits",len(clean),"removed",len(hh)-len(clean))

# unique corrupted values
for hand in ("L","R"):
    rr=[r for r in raw if r["hand"]==hand and corrupt(r)]
    from collections import Counter
    c=Counter((r["ax"],r["yaw"]) for r in rr)
    print(hand,"corrupt values",c.most_common(10))
