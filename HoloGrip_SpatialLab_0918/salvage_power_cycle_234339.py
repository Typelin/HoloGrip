from pathlib import Path
import csv, math
from datetime import datetime, timezone
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation as Rot

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\第三次的包\Data\電源循環姿態測試\20260918_234339")
with (S/"raw.csv").open(encoding="utf-8",newline="") as f:
    rows=list(csv.DictReader(f))
for r in rows:
    r["valid"]=int(r["valid"])
    r["dt"]=datetime.fromisoformat(r["wall_time"])
    r["e"]=np.array([float(r["yaw"]),float(r["pitch"]),float(r["roll"])])
    r["a"]=np.array([float(r["ax"]),float(r["ay"]),float(r["az"])])

def center_stats(q):
    E=np.stack([r["e"] for r in q])
    A=np.stack([r["a"] for r in q])
    rs=Rot.from_euler("ZYX",E,degrees=True)
    c=rs.mean()
    spread=np.degrees((c.inv()*rs).magnitude())
    U=A/np.linalg.norm(A,axis=1,keepdims=True)
    g=U.mean(0);g/=np.linalg.norm(g)
    gspread=np.degrees(np.arccos(np.clip(U@g,-1,1)))
    norms=np.linalg.norm(A,axis=1)
    return c,g,spread,gspread,norms

def gd(a,b): return float(np.degrees((a.inv()*b).magnitude()))
def vd(a,b): return float(np.degrees(np.arccos(np.clip(a@b,-1,1))))

stages=["COLLECT_A0","COLLECT_B_AFTER","MOVE_A_AFTER_BBOOT","COLLECT_A_REBOOT","MOVE_B_BEFORE"]
print("=== VALID RUNS ===")
best={}
for st in stages:
    q=[r for r in rows if r["stage"]==st]
    runs=[];cur=[]
    for r in q:
        if r["valid"]:
            cur.append(r)
        else:
            if cur:runs.append(cur);cur=[]
    if cur:runs.append(cur)
    runs=sorted(runs,key=len,reverse=True)
    print("\n",st,"rows",len(q),"valid",sum(r["valid"] for r in q),"invalid",sum(not r["valid"] for r in q),
          "top runs",[len(x) for x in runs[:10]])
    if runs:
        rr=runs[0]
        c,g,sp,gsp,nm=center_stats(rr)
        print(" longest",len(rr),"from",rr[0]["dt"].time(),"to",rr[-1]["dt"].time(),
              "euler",np.round(c.as_euler("ZYX",degrees=True),2),
              "rot p50/p90",np.round(np.percentile(sp,[50,90]),2),
              "g",np.round(g,4),"gspread p50/p90",np.round(np.percentile(gsp,[50,90]),2),
              "norm p10/50/90",np.round(np.percentile(nm,[10,50,90]),3))
        if len(rr)>=300:
            # use last 300 of longest run, likely after transient
            z=rr[-300:]
            c,g,sp,gsp,nm=center_stats(z)
            best[st]=(c,g)
            print(" clean300 euler",np.round(c.as_euler("ZYX",degrees=True),2),
                  "rot p50/p90",np.round(np.percentile(sp,[50,90]),2),
                  "g",np.round(g,4),"gspread p50/p90",np.round(np.percentile(gsp,[50,90]),2),
                  "norm p10/50/90",np.round(np.percentile(nm,[10,50,90]),3))

print("\n=== CLEAN300 COMPARISONS ===")
pairs=[
 ("A0 vs A_reboot","COLLECT_A0","COLLECT_A_REBOOT"),
 ("B_before vs B_after","MOVE_B_BEFORE","COLLECT_B_AFTER"),
 ("A0 vs A_after_Bboot","COLLECT_A0","MOVE_A_AFTER_BBOOT"),
]
for name,a,b in pairs:
    if a in best and b in best:
        ca,ga=best[a];cb,gb=best[b]
        print(name,"rot",round(gd(ca,cb),2),"gravity",round(vd(ga,gb),2))
    else: print(name,"NO CLEAN300",a in best,b in best)
