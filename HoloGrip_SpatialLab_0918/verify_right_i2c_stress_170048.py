from pathlib import Path
import csv, math
from collections import Counter
import numpy as np

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918\right_i2c_stress_sessions\20260918_170048")
with (S/"raw.csv").open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))

test=[r for r in rows if 0 <= float(r["elapsed_s"]) <= 40.2]
print("TEST_ROWS",len(test))
for hand in ("L","R"):
    rr=[r for r in test if r["hand"]==hand]
    print("\nHAND",hand,"n",len(rr))
    pids=np.array([int(r["packet_id"]) for r in rr])
    print("packet gap",int(np.sum(np.diff(pids)>1)),"back",int(np.sum(np.diff(pids)<=0)))
    print("invalid",Counter(r["invalid_reason"] for r in rr if r["valid"]=="0"))
    for idx,name in [(0,"靜止基準"),(1,"慢速大幅"),(2,"快速大幅"),(3,"猛烈"),(4,"結束靜止")]:
        q=[r for r in rr if int(r["stage_index"])==idx]
        if not q: continue
        mag=np.array([float(r["mag_g"]) for r in q])
        yaw=np.array([float(r["yaw"]) for r in q])
        pitch=np.array([float(r["pitch"]) for r in q])
        roll=np.array([float(r["roll"]) for r in q])
        print(idx,name,"n",len(q),
              "mag p50/p95/max",*[round(float(x),3) for x in np.percentile(mag,[50,95,100])],
              "motion>1.5g%",round(float(np.mean(mag>1.5)*100),1),
              "yaw_range",round(float(np.ptp(np.rad2deg(np.unwrap(np.deg2rad(yaw))))),1),
              "pitch_range",round(float(np.ptp(pitch)),1),
              "roll_range",round(float(np.ptp(np.rad2deg(np.unwrap(np.deg2rad(roll))))),1))
