from pathlib import Path
import csv
from collections import Counter

P=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918\next_person_realdrum_sessions\20260918_202322\hits.csv")
with P.open(encoding="utf-8",newline="") as f:rows=[r for r in csv.DictReader(f) if r["hand"]=="L"]
d=[{"t":float(r["peak_ms"]),"g":float(r["peak_g"]),"drum":r["raw_drum"],"yaw":float(r["rel_yaw"]),"pitch":float(r["rel_pitch"])} for r in rows]

def filt(q,window=160,ratio=1.4,delta=.5,low=3.5):
    out=[];i=0;supp=[]
    while i<len(q):
        a=q[i]
        if i+1<len(q):
            b=q[i+1];gap=b["t"]-a["t"]
            if gap<=window and a["g"]<=low and b["g"]>=a["g"]*ratio and b["g"]-a["g"]>=delta:
                supp.append((a,b,gap));i+=1;continue
        out.append(a);i+=1
    return out,supp

print("original",len(d),Counter(x["drum"] for x in d))
for low,ratio in [(3.0,1.4),(3.5,1.4),(4.0,1.4),(4.5,1.4),(4.5,1.2)]:
    o,s=filt(d,160,ratio,.5,low)
    print("low",low,"ratio",ratio,"n",len(o),"supp",len(s),"preds",Counter(x["drum"] for x in o))
    print(" suppressed",[(round(g,0),round(a["g"],1),a["drum"],"->",round(b["g"],1),b["drum"]) for a,b,g in s])
