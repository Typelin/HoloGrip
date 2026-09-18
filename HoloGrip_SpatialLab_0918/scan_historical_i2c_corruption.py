from pathlib import Path
import csv
from collections import Counter

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
FILES=[
("0805",ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"),
("0812",ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"),
]

def bad(r):
    ax=float(r["ax_g"]);ay=float(r["ay_g"]);az=float(r["az_g"])
    yaw=float(r["yaw_deg"]);pitch=float(r["pitch_deg"]);roll=float(r["roll_deg"])
    acc_same=max(abs(ax-ay),abs(ay-az),abs(ax-az))<1e-6
    ang_same=max(abs(yaw-pitch),abs(pitch-roll),abs(yaw-roll))<1e-6
    return acc_same and ang_same and abs(yaw-ax*11.25)<0.02 and abs(ax)>0.01

def zero(r):
    vals=[float(r[k]) for k in ("ax_g","ay_g","az_g","yaw_deg","pitch_deg","roll_deg")]
    return all(abs(v)<1e-12 for v in vals)

for name,p in FILES:
    with p.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    print("\n",name)
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        bb=[r for r in rr if bad(r)]
        zz=[r for r in rr if zero(r)]
        print(h,"rows",len(rr),"corrupt",len(bb),f"{len(bb)/len(rr)*100:.4f}%","zero6",len(zz))
        if bb:
            c=Counter((r["ax_g"],r["yaw_deg"]) for r in bb)
            print(" values",c.most_common(10))
