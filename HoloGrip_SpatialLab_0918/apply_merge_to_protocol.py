from pathlib import Path
import csv
from collections import Counter,defaultdict
from datetime import datetime

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918\protocol_sessions\20260918_155836")
with (S/"hits.csv").open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
for r in rows:
    r["t"]=datetime.fromisoformat(r["wall_time"]).timestamp()*1000
    r["peak"]=float(r["peak_accel_g"]);r["stage"]=int(r["stage_index"])

def merge(P,T=140,ratio=.35):
    out=[]
    for h in ("L","R"):
        pp=sorted([p for p in P if p["hand"]==h],key=lambda x:x["t"])
        k=[]
        for p in pp:
            if not k:k.append(p);continue
            prev=k[-1];dt=p["t"]-prev["t"]
            if dt<T:
                if p["peak"]<ratio*prev["peak"]:continue
                if prev["peak"]<ratio*p["peak"]:
                    k[-1]=p;continue
            k.append(p)
        out+=k
    return sorted(out,key=lambda x:x["t"])

M=merge(rows)
print("total",len(rows),"->",len(M),"removed",len(rows)-len(M))
for st in sorted(set(r["stage"] for r in rows)):
    a=[r for r in rows if r["stage"]==st];b=[r for r in M if r["stage"]==st]
    if not a:continue
    print("\nstage",st,a[0]["stage_name"],len(a),"->",len(b))
    for h in ("L","R"):
        aa=[r for r in a if r["hand"]==h];bb=[r for r in b if r["hand"]==h]
        print(h,len(aa),"->",len(bb),"labels",Counter(r["pred"] for r in bb))
