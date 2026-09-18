from pathlib import Path
import csv,numpy as np
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
sessions=[
("0805",ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"),
("0812",ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"),
]
def wrap(x):return (x+180)%360-180
for name,p in sessions:
 with p.open(encoding="utf-8-sig",newline="") as f:r=list(csv.DictReader(f))
 print("\n",name)
 for h in ("L","R"):
  rr=[x for x in r if x["hand"]==h]
  t=np.array([float(x["song_time_ms"]) for x in rr])
  m=(t>=t.min())&(t<t.min()+1000)
  for c in ["yaw_deg","pitch_deg","roll_deg","cal_yaw_deg","cal_pitch_deg"]:
   a=np.array([float(x[c]) for x in rr])[m]
   print(h,c,"first1s median",round(float(np.median(a)),2),"std",round(float(np.std(a)),2),"p10-90",np.round(np.percentile(a,[10,90]),2))
  yoff=np.median([wrap(float(x["yaw_deg"])-float(x["cal_yaw_deg"])) for x in rr])
  poff=np.median([float(x["pitch_deg"])-float(x["cal_pitch_deg"]) for x in rr])
  print(h,"offset",round(float(yoff),2),round(float(poff),2))
