from pathlib import Path
import csv, numpy as np
from scipy.spatial.transform import Rotation as Rot

P=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\第三次的包\Data\電源循環姿態測試\20260918_235201\raw.csv")
with P.open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))

# Use last 300 valid rows in the named stage before stage completion;
# for these completed stages this corresponds to the clean final block.
for st in ["COLLECT_A0","COLLECT_B_AFTER","MOVE_A_AFTER_BBOOT","COLLECT_A_REBOOT"]:
    q=[r for r in rows if r["stage"]==st and r["valid"]=="1"][-300:]
    if len(q)<300: continue
    E=np.array([[float(r["yaw"]),float(r["pitch"]),float(r["roll"])] for r in q])
    A=np.array([[float(r["ax"]),float(r["ay"]),float(r["az"])] for r in q])
    def meanR(z):return Rot.from_euler("ZYX",z,degrees=True).mean()
    r1=meanR(E[:50]);r2=meanR(E[-50:])
    drift=np.degrees((r1.inv()*r2).magnitude())
    e1=r1.as_euler("ZYX",degrees=True);e2=r2.as_euler("ZYX",degrees=True)
    u1=(A[:50]/np.linalg.norm(A[:50],axis=1,keepdims=True)).mean(0);u1/=np.linalg.norm(u1)
    u2=(A[-50:]/np.linalg.norm(A[-50:],axis=1,keepdims=True)).mean(0);u2/=np.linalg.norm(u2)
    gdr=np.degrees(np.arccos(np.clip(u1@u2,-1,1)))
    print(st,"orientation first->last50 drift",round(float(drift),3),"deg",
          "gravity drift",round(float(gdr),3),"deg",
          "euler first",np.round(e1,2),"last",np.round(e2,2))
