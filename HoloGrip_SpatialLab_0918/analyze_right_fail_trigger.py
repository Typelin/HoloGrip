from pathlib import Path
import csv,math
from collections import Counter
from datetime import datetime,timedelta
import numpy as np

LAB=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918")
sessions=[
 ("R1",LAB/"count6_sessions/20260918_185504"/"raw_100hz.csv"),
 ("R2",LAB/"count6_round2_sessions/20260918_190558"/"raw_100hz.csv"),
 ("R3",LAB/"count6_round3_fresh_r0_sessions/20260918_194947"/"raw_100hz.csv"),
 ("R4",LAB/"count6_round4_sessioncal_sessions/20260918_195738"/"raw_100hz.csv"),
 ("NP",LAB/"next_person_realdrum_sessions/20260918_202322"/"raw_100hz.csv"),
]
def rcsv(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def valid_field(r):
    if "valid" in r:return int(r["valid"])
    if "frame_valid" in r:return int(r["frame_valid"])
    return 1
for name,p in sessions:
    if not p.exists():continue
    rows=rcsv(p)
    rr=[r for r in rows if r["hand"]=="R"]
    for r in rr:
        r["_t"]=datetime.fromisoformat(r["wall_time"])
        r["_v"]=valid_field(r)
        r["_g"]=math.sqrt(float(r["ax"])**2+float(r["ay"])**2+float(r["az"])**2)
    bad=[r for r in rr if not r["_v"]]
    print("\n===",name,p.parent.name,"R total",len(rr),"bad",len(bad),Counter(r.get("invalid_reason","") for r in bad))
    eps=[];cur=[]
    for r in bad:
        if cur and (r["_t"]-cur[-1]["_t"]).total_seconds()>0.15:
            eps.append(cur);cur=[]
        cur.append(r)
    if cur:eps.append(cur)
    for e in eps:
        if len(e)<2:continue
        t0=e[0]["_t"];t1=e[-1]["_t"]
        pre=[r for r in rr if r["_v"] and t0-timedelta(seconds=.5)<=r["_t"]<t0]
        post=[r for r in rr if r["_v"] and t1<r["_t"]<=t1+timedelta(seconds=.5)]
        stage=e[0].get("stage_name","")
        mode=e[0].get("mode","")
        def stats(q):
            if not q:return None
            gs=np.array([r["_g"] for r in q])
            return (round(float(np.median(gs)),2),round(float(np.max(gs)),2),len(q))
        print(t0.time(),"->",t1.time(),"n",len(e),"dur",round((t1-t0).total_seconds(),3),
              "mode",mode,"stage",stage,
              "pre_med/max/n",stats(pre),"post",stats(post),
              "reasons",dict(Counter(r.get("invalid_reason","") for r in e)))
