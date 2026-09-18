from pathlib import Path
import csv
from datetime import datetime
from collections import Counter

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918\vnext_live_sessions\20260918_152635")
with (S/"raw.csv").open(encoding="utf-8",newline="") as f:raw=list(csv.DictReader(f))
with (S/"hits.csv").open(encoding="utf-8",newline="") as f:hits=list(csv.DictReader(f))
for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"])
    ax,ay,az=map(float,(r["ax"],r["ay"],r["az"]))
    yaw,pitch,roll=map(float,(r["yaw"],r["pitch"],r["roll"]))
    acc_same=max(abs(ax-ay),abs(ay-az),abs(ax-az))<1e-6
    ang_same=max(abs(yaw-pitch),abs(pitch-roll),abs(yaw-roll))<1e-6
    r["bad"]=acc_same and ang_same and abs(yaw-ax*11.25)<0.02 and abs(ax)>0.01
    r["zero"]=all(abs(float(r[k]))<1e-12 for k in ("ax","ay","az","yaw","pitch","roll"))
for h in hits:h["dt"]=datetime.fromisoformat(h["wall_time"])
badR=[r for r in raw if r["hand"]=="R" and (r["bad"] or r["zero"])]
clean=[]
for h in hits:
    contaminated = h["hand"]=="R" and any(abs((r["dt"]-h["dt"]).total_seconds())<=0.25 for r in badR)
    if not contaminated: clean.append(h)

cut=datetime.fromisoformat("2026-09-18T15:31:20+08:00")
hh=sorted([h for h in clean if h["dt"]>=cut],key=lambda x:x["dt"])
# combined segments gap 1 sec
segs=[];cur=[]
for h in hh:
    if cur and (h["dt"]-cur[-1]["dt"]).total_seconds()>1.0:
        segs.append(cur);cur=[]
    cur.append(h)
if cur:segs.append(cur)
for i,s in enumerate(segs,1):
    print("\nSEG",i,s[0]["dt"].time(),"->",s[-1]["dt"].time(),"n",len(s))
    for hand in ("L","R"):
        ss=[x for x in s if x["hand"]==hand]
        if not ss:continue
        c=Counter(x["pred"] for x in ss);top=c.most_common(1)[0]
        print(hand,"n",len(ss),"top",top,"stab",round(top[1]/len(ss),3),"labels",dict(c))
