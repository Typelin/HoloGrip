from pathlib import Path
import csv, numpy as np
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
files=[
("0805_song1",ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"),
("0812_song1_b",ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song01_raw_100hz_20260812_111817.csv"),
("0812_song2",ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"),
]
for name,p in files:
    with p.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    print("\n",name)
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        rawp=np.array([float(r["pitch_deg"]) for r in rr])
        calp=np.array([float(r["cal_pitch_deg"]) for r in rr])
        rawy=np.array([float(r["yaw_deg"]) for r in rr])
        rawr=np.array([float(r["roll_deg"]) for r in rr])
        print(h,"raw_pitch p1/50/99",np.round(np.percentile(rawp,[1,50,99]),2),
              ">70",round(np.mean(np.abs(rawp)>70)*100,2),">80",round(np.mean(np.abs(rawp)>80)*100,2),
              "cal_pitch p1/50/99",np.round(np.percentile(calp,[1,50,99]),2),
              "raw_yaw med",round(float(np.median(rawy)),2),"raw_roll med",round(float(np.median(rawr)),2))
