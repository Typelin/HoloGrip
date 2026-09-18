from pathlib import Path
import csv, numpy as np, json, math

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
sessions=[
 ("0805", ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv", 16000.0),
 ("0812", ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv", 6027.0),
]

def cwrap(x): return (x+180)%360-180

for name,p,first_midi in sessions:
    with p.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    print("\n===",name,"===")
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        rawy=np.array([float(r["yaw_deg"]) for r in rr])
        caly=np.array([float(r["cal_yaw_deg"]) for r in rr])
        rawp=np.array([float(r["pitch_deg"]) for r in rr])
        calp=np.array([float(r["cal_pitch_deg"]) for r in rr])
        roll=np.array([float(r["roll_deg"]) for r in rr])
        t=np.array([float(r["song_time_ms"]) for r in rr])
        yawoff=np.array([cwrap(a-b) for a,b in zip(rawy,caly)])
        pitchoff=rawp-calp
        print(h,"yaw_offset med/std/min/max",*[round(float(x),4) for x in [np.median(yawoff),np.std(yawoff),np.min(yawoff),np.max(yawoff)]])
        print(h,"pitch_offset med/std/min/max",*[round(float(x),4) for x in [np.median(pitchoff),np.std(pitchoff),np.min(pitchoff),np.max(pitchoff)]])
        pre=(t>=max(0, first_midi-5000))&(t<first_midi-500) # last 4.5s before first MIDI
        if np.sum(pre)<50: pre=t<first_midi
        print(h,"PRE n",int(np.sum(pre)))
        for label,a in [("cal_yaw",caly),("cal_pitch",calp),("raw_yaw",rawy),("raw_pitch",rawp),("roll",roll)]:
            q=np.percentile(a[pre],[1,10,50,90,99])
            print(" ",label,"p1/10/50/90/99",*[round(float(x),2) for x in q],"std",round(float(np.std(a[pre])),2))
        # first 1 sec after start vs last pre 1 sec drift
        p1=(t>=max(0,first_midi-5000))&(t<max(0,first_midi-4000))
        p2=(t>=first_midi-1500)&(t<first_midi-500)
        if p1.sum()>20 and p2.sum()>20:
            y1=np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(caly[p1])))))
            y2=np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(caly[p2])))))
            dy=cwrap(y2-y1)
            dp=np.mean(calp[p2])-np.mean(calp[p1])
            print(" PRE drift over ~4s cal yaw",round(float(dy),2),"pitch",round(float(dp),2))
