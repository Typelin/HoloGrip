from pathlib import Path
import csv, math
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")

sources=[
("0805",ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv","song_time_ms","yaw_deg","pitch_deg","roll_deg","ax_g","ay_g","az_g",None),
("0812",ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv","song_time_ms","yaw_deg","pitch_deg","roll_deg","ax_g","ay_g","az_g",None),
("today_protocol",ROOT/"HoloGrip_SpatialLab_0918/protocol_sessions/20260918_155836/raw_100hz.csv","protocol_elapsed_s","yaw","pitch","roll","ax","ay","az",0),
]

for item in sources:
    name,p,tcol,yc,pc,rc,axc,ayc,azc,stage=item
    with p.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    print("\n===",name,"===")
    for h in ("L","R"):
        rr=[r for r in rows if r.get("hand")==h]
        if stage is not None:
            rr=[r for r in rr if int(r["stage_index"])==stage and r.get("frame_valid","1")=="1"]
            # protocol elapsed seconds
            ts=np.array([float(r[tcol])*1000 for r in rr])
        else:
            ts=np.array([float(r[tcol]) for r in rr])
        if not len(rr):continue
        t0=ts.min()
        # first .5 second
        idx=np.where(ts<t0+500)[0]
        e=np.array([[float(rr[i][yc]),float(rr[i][pc]),float(rr[i][rc])] for i in idx])
        A=np.array([[float(rr[i][axc]),float(rr[i][ayc]),float(rr[i][azc])] for i in idx])
        rots=Rot.from_euler("ZYX",e,degrees=True)
        mean=rots.mean()
        dev=np.rad2deg((mean.inv()*rots).magnitude())
        mags=np.linalg.norm(A,axis=1)
        print(h,"n",len(idx),
              "rotdev med/p90/max",*[round(float(x),3) for x in np.percentile(dev,[50,90,100])],
              "mag mean/std/p95err",round(float(mags.mean()),4),round(float(mags.std()),4),round(float(np.percentile(np.abs(mags-1),95)),4))
