from pathlib import Path
import csv, math
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918\vnext_live_sessions\20260918_152635")
with (S/"hits.csv").open(encoding="utf-8",newline="") as f:
    hits=list(csv.DictReader(f))
with (S/"raw.csv").open(encoding="utf-8",newline="") as f:
    raw=list(csv.DictReader(f))
print("HITS",len(hits),"RAW",len(raw))
# parse wall times
for h in hits:
    h["wall_dt"]=datetime.fromisoformat(h["wall_time"])
    h["prob_f"]=float(h["prob"]); h["peak_f"]=float(h["peak_accel_g"])
for r in raw:
    r["wall_dt"]=datetime.fromisoformat(r["wall_time"])
    r["mag"]=math.sqrt(float(r["ax"])**2+float(r["ay"])**2+float(r["az"])**2)

print("\n=== TOTAL BY HAND ===")
for hand in ("L","R"):
    hh=[x for x in hits if x["hand"]==hand]
    print(hand,"n",len(hh),"labels",Counter(x["pred"] for x in hh),"median_prob",np.median([x["prob_f"] for x in hh]) if hh else None)

# segments by hit gap > 1.5 sec
print("\n=== NATURAL HIT SEGMENTS gap>1.5s ===")
for hand in ("L","R"):
    hh=sorted([x for x in hits if x["hand"]==hand],key=lambda x:x["wall_dt"])
    segs=[]; cur=[]
    for x in hh:
        if cur and (x["wall_dt"]-cur[-1]["wall_dt"]).total_seconds()>1.5:
            segs.append(cur);cur=[]
        cur.append(x)
    if cur:segs.append(cur)
    print("\nHAND",hand,"segments",len(segs))
    for i,s in enumerate(segs,1):
        dur=(s[-1]["wall_dt"]-s[0]["wall_dt"]).total_seconds()
        c=Counter(x["pred"] for x in s)
        top=c.most_common(1)[0]
        stab=top[1]/len(s)
        print(i,s[0]["wall_dt"].time(),s[-1]["wall_dt"].time(),"n",len(s),"dur",round(dur,2),"top",top,"stab",round(stab,3),"labels",dict(c))

# Combined chronological windows by gap >1.0s
allh=sorted(hits,key=lambda x:x["wall_dt"])
segs=[];cur=[]
for x in allh:
    if cur and (x["wall_dt"]-cur[-1]["wall_dt"]).total_seconds()>1.0:
        segs.append(cur);cur=[]
    cur.append(x)
if cur:segs.append(cur)
print("\n=== COMBINED SEGMENTS gap>1.0s ===")
for i,s in enumerate(segs,1):
    print("\nSEG",i,s[0]["wall_dt"].time(),s[-1]["wall_dt"].time(),"n",len(s))
    for hand in ("L","R"):
        hh=[x for x in s if x["hand"]==hand]
        if hh:
            c=Counter(x["pred"] for x in hh);top=c.most_common(1)[0]
            print(" ",hand,"n",len(hh),"top",top,"stab",round(top[1]/len(hh),3),"labels",dict(c))

# suspect tail from 15:32:20 onward
cut=datetime.fromisoformat("2026-09-18T15:32:20+08:00")
tail=[x for x in hits if x["wall_dt"]>=cut]
print("\n=== TAIL HIT INTERVALS ===")
for hand in ("L","R"):
    hh=[x for x in tail if x["hand"]==hand]
    print("\n",hand)
    prev=None
    for x in hh:
        dt=None if prev is None else (x["wall_dt"]-prev).total_seconds()
        print(x["wall_dt"].time(),"dt",dt,"pred",x["pred"],"prob",round(x["prob_f"],3),"peak",x["peak_f"])
        prev=x["wall_dt"]

# For each tail hit, inspect raw hand magnitude in +/-120ms, unique sensor packet window, max mag, max abs delta
print("\n=== HIT VS RAW MAG tail ===")
for x in tail:
    if x["hand"]!="R": continue
    t=x["wall_dt"]
    rr=[r for r in raw if r["hand"]=="R" and abs((r["wall_dt"]-t).total_seconds())<=0.18]
    if not rr: continue
    mags=np.array([r["mag"] for r in rr])
    print(t.time(),x["pred"],"logged_peak",x["peak_f"],"raw_max",round(float(mags.max()),4),"raw_med",round(float(np.median(mags)),4),"raw_n",len(rr))

# Find last time right raw magnitude > thresholds
for th in (2.3,3,5,8,10):
    rr=[r for r in raw if r["hand"]=="R" and r["mag"]>th]
    print("R last mag >",th, rr[-1]["wall_dt"].time() if rr else None, "value", round(rr[-1]["mag"],4) if rr else None)

# raw Hz-ish / backlog check: wall interarrival
for hand in ("L","R"):
    rr=sorted([r for r in raw if r["hand"]==hand],key=lambda r:r["wall_dt"])
    diffs=np.array([(rr[i]["wall_dt"]-rr[i-1]["wall_dt"]).total_seconds()*1000 for i in range(1,len(rr))])
    print(hand,"wall interarrival ms p1/50/99",np.percentile(diffs,[1,50,99]),"max",diffs.max())
